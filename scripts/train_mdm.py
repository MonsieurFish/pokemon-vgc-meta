"""Train the Meta Distribution Model (MDM) v1 and backtest it vs persistence.

    python scripts/train_mdm.py --epochs 8 --n-models 3

Embeds every team once through the frozen encoder, builds leave-one-out
masked-team examples, trains a deep ensemble of anchor-distribution predictors,
runs a retrain-on-past rolling backtest (KL vs persistence), then trains a final
ensemble on all data and prints the forecast for the next week.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.tournaments import load_tournament_teams
from vgc_team.data.vgc_bench import MA_FILES, download_vgc_bench_file, load_vgc_bench_teams
from vgc_team.meta.mdm.dataset import MDMDatasetConfig, build_team_features
from vgc_team.meta.mdm.evaluate import backtest
from vgc_team.meta.mdm.interpret import (
    anchor_to_cluster_distribution,
    cluster_shift_table,
)
from vgc_team.meta.mdm.predict import forecast_next_week
from vgc_team.meta.mdm.train import MDMTrainConfig, save_ensemble, train_ensemble
from vgc_team.models.frozen_encoder import (
    ReferenceSpace,
    load_frozen_encoder,
    select_device,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    data_dir: Annotated[
        Path, typer.Option(help="VGC-Bench JSON folder.")
    ] = DATA_DIR / "raw" / "vgc_bench",
    reference_npz: Annotated[
        Path, typer.Option(help="team_embeddings.npz with centroids/anchors.")
    ] = DATA_DIR / "processed" / "team_knn" / "team_embeddings.npz",
    include_bo3: Annotated[bool, typer.Option(help="Include the BO3 dataset.")] = True,
    download: Annotated[bool, typer.Option(help="Download missing files.")] = True,
    tournament_teams: Annotated[
        Path | None,
        typer.Option(help="ma_tournament_teams.json to merge in (upweighted via its weights)."),
    ] = None,
    max_battles: Annotated[int | None, typer.Option(help="Cap for a fast smoke run.")] = None,
    min_week: Annotated[
        str | None,
        typer.Option(help="Drop weeks before this ISO label (e.g. 2026-W22) to skip the break."),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Maximum epochs (early stopping may end sooner).")] = 100,
    early_stop_min_delta: Annotated[
        float | None,
        typer.Option(help="Stop when epoch CE improves by less than this fraction (e.g. 0.01 = 1%)."),
    ] = None,
    early_stop_patience: Annotated[
        int, typer.Option(help="Consecutive low-improvement epochs before stopping.")
    ] = 1,
    n_models: Annotated[int, typer.Option(help="Ensemble size.")] = 3,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 128,
    full_mask_prob: Annotated[
        float, typer.Option(help="Probability the current week is fully masked (forecast regime).")
    ] = 0.5,
    holdout_weeks: Annotated[int, typer.Option(help="Weeks held out for the backtest.")] = 2,
    output_dir: Annotated[
        Path, typer.Option(help="Where the trained ensemble is saved.")
    ] = PROJECT_ROOT / "models" / "mdm",
) -> None:
    filenames = [MA_FILES["ma"]] + ([MA_FILES["ma_bo3"]] if include_bo3 else [])
    paths: list[Path] = []
    for filename in filenames:
        path = data_dir / filename
        if download and not path.exists():
            path = download_vgc_bench_file(filename, data_dir)
        paths.append(path)

    reference = ReferenceSpace.load(reference_npz)
    device = select_device()
    console.print("Loading teams + frozen encoder...")
    teams = load_vgc_bench_teams(paths, max_battles=max_battles)
    weights = np.ones(len(teams), dtype=np.float32)

    if tournament_teams is not None:
        t_teams, t_weights = load_tournament_teams(tournament_teams)
        console.print(
            f"Merging {len(t_teams)} tournament teams (mean weight {float(t_weights.mean()):.1f}x)."
        )
        teams = teams + t_teams
        weights = np.concatenate([weights, t_weights])

    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)

    console.print("Embedding teams + assigning anchors...")
    features = build_team_features(
        teams, reference, model_enc, kind_vocab, token_vocab, weights=weights, device=device
    )
    if min_week is not None:
        kept_weeks = [w for w in features.weeks if w >= min_week]
        old_to_new = {week: new for new, week in enumerate(kept_weeks)}
        row_labels = np.array(features.weeks)[features.week_index]
        row_mask = row_labels >= min_week
        features.embeddings = features.embeddings[row_mask]
        features.anchors = features.anchors[row_mask]
        features.weight = features.weight[row_mask]
        features.week_index = np.array(
            [old_to_new[week] for week in row_labels[row_mask]], dtype=np.int64
        )
        features.weeks = kept_weeks
    console.print(f"{features.n_weeks} weeks, {len(features.anchors)} teams.")

    cfg = MDMTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_patience=early_stop_patience,
        dataset=MDMDatasetConfig(full_mask_prob=full_mask_prob),
        device=device,
        verbose=True,
    )

    def train_fn(sub):
        return train_ensemble(sub, reference.n_anchors, cfg, n_models=n_models)

    console.print(f"\n[bold]Backtest (retrain on past, {holdout_weeks} held-out weeks)[/bold]")
    bt = backtest(features, reference.n_anchors, train_fn, holdout_weeks=holdout_weeks, device=device)
    console.print(bt.to_string(index=False))
    if not bt.empty:
        console.print(
            f"\nmean KL: model={bt['kl_model'].mean():.4f}  "
            f"persistence={bt['kl_persistence'].mean():.4f}  "
            f"model beats persistence in {int(bt['model_beats_persistence'].sum())}/{len(bt)} weeks"
        )

    console.print("\n[bold]Training final ensemble on all weeks...[/bold]")
    models = train_fn(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_ensemble(
        models, output_dir / "mdm.pt",
        n_anchors=reference.n_anchors, d_in=features.embeddings.shape[1], d_model=cfg.d_model,
    )
    console.print(f"Saved ensemble to {output_dir / 'mdm.pt'}")

    anchor_pred, _ = forecast_next_week(models, features, device=device)
    predicted_clusters = anchor_to_cluster_distribution(anchor_pred, reference)

    last_week = features.n_weeks - 1
    last_mask = features.week_index == last_week
    from vgc_team.meta.mdm.evaluate import anchor_histogram

    current_anchor = anchor_histogram(
        features.anchors[last_mask], features.weight[last_mask], reference.n_anchors
    )
    current_clusters = anchor_to_cluster_distribution(current_anchor, reference)

    reps = {}
    rep_path = reference_npz.parent / "cluster_representatives.csv"
    if rep_path.exists():
        reps = {int(r.cluster_id): str(r.species) for r in pd.read_csv(rep_path).itertuples()}

    shift = cluster_shift_table(current_clusters, predicted_clusters, representatives=reps)
    console.print(f"\n[bold]Predicted archetype shifts for the week after {features.weeks[-1]}[/bold]")
    console.print(shift.head(12).to_string(index=False))


if __name__ == "__main__":
    app()
