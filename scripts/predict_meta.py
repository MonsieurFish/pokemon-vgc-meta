"""Forecast the next week's meta from a (possibly thin) sample of recent weeks.

Demonstrates the deployment scenario: plug in the last few weeks of teams --
even a thin 3x60 sample -- and get the predicted next-meta archetype shifts with
ensemble uncertainty.

    python scripts/predict_meta.py --weeks 3 --sample-per-week 60
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.vgc_bench import MA_FILES, download_vgc_bench_file, load_vgc_bench_teams
from vgc_team.meta.mdm.dataset import TeamFeatures, build_team_features
from vgc_team.meta.mdm.evaluate import anchor_histogram
from vgc_team.meta.mdm.interpret import anchor_to_cluster_distribution
from vgc_team.meta.mdm.predict import forecast_next_week, predict_anchor_samples, relative_time
from vgc_team.meta.mdm.train import load_ensemble
from vgc_team.models.frozen_encoder import ReferenceSpace, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()


def _recent_slice(
    features: TeamFeatures, n_weeks: int, sample_per_week: int, seed: int
) -> TeamFeatures:
    rng = np.random.default_rng(seed)
    first = max(0, features.n_weeks - n_weeks)
    kept_weeks = features.weeks[first:]
    old_to_new = {week: new for new, week in enumerate(kept_weeks)}
    row_labels = np.array(features.weeks)[features.week_index]

    keep_idx = []
    for week in kept_weeks:
        members = np.flatnonzero(row_labels == week)
        if sample_per_week and len(members) > sample_per_week:
            members = rng.choice(members, size=sample_per_week, replace=False)
        keep_idx.append(members)
    keep_idx = np.concatenate(keep_idx)

    return TeamFeatures(
        embeddings=features.embeddings[keep_idx],
        anchors=features.anchors[keep_idx],
        week_index=np.array([old_to_new[w] for w in row_labels[keep_idx]], dtype=np.int64),
        weight=features.weight[keep_idx],
        weeks=kept_weeks,
    )


@app.command()
def main(
    mdm_path: Annotated[
        Path, typer.Option(help="Trained MDM ensemble.")
    ] = PROJECT_ROOT / "models" / "mdm" / "mdm.pt",
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    reference_npz: Annotated[
        Path, typer.Option(help="team_embeddings.npz with centroids/anchors.")
    ] = DATA_DIR / "processed" / "team_knn" / "team_embeddings.npz",
    data_dir: Annotated[
        Path, typer.Option(help="VGC-Bench JSON folder (source of recent weeks).")
    ] = DATA_DIR / "raw" / "vgc_bench",
    include_bo3: Annotated[bool, typer.Option(help="Include the BO3 dataset.")] = True,
    download: Annotated[bool, typer.Option(help="Download missing files.")] = True,
    weeks: Annotated[int, typer.Option(help="How many recent weeks to use as context.")] = 3,
    sample_per_week: Annotated[
        int, typer.Option(help="Subsample to N teams/week (0 = all) — simulate a thin sample.")
    ] = 60,
    seed: Annotated[int, typer.Option(help="Subsampling seed.")] = 0,
    top: Annotated[int, typer.Option(help="How many archetype rows to display.")] = 12,
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
    teams = load_vgc_bench_teams(paths)
    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
    full = build_team_features(teams, reference, model_enc, kind_vocab, token_vocab, device=device)

    recent = _recent_slice(full, weeks, sample_per_week, seed)
    console.print(
        f"Context: weeks {recent.weeks[0]}..{recent.weeks[-1]} "
        f"({len(recent.anchors)} teams, ~{sample_per_week or 'all'}/week)"
    )

    models = load_ensemble(mdm_path, device)

    query_week = recent.n_weeks
    context_time = relative_time(recent.week_index, query_week)
    per_model = predict_anchor_samples(
        models, recent.embeddings, context_time, recent.anchors, device=device
    )
    cluster_samples = np.stack(
        [anchor_to_cluster_distribution(p, reference) for p in per_model], axis=0
    )
    predicted = cluster_samples.mean(axis=0)
    uncertainty = cluster_samples.std(axis=0)

    last_mask = recent.week_index == (recent.n_weeks - 1)
    current = anchor_to_cluster_distribution(
        anchor_histogram(recent.anchors[last_mask], recent.weight[last_mask], reference.n_anchors),
        reference,
    )

    reps = {}
    rep_path = reference_npz.parent / "cluster_representatives.csv"
    if rep_path.exists():
        reps = {int(r.cluster_id): str(r.species) for r in pd.read_csv(rep_path).itertuples()}

    rows = []
    for cluster_id in range(reference.n_clusters):
        delta = float(predicted[cluster_id] - current[cluster_id])
        rows.append(
            {
                "cluster_id": cluster_id,
                "current": float(current[cluster_id]),
                "predicted": float(predicted[cluster_id]),
                "delta": delta,
                "uncertainty": float(uncertainty[cluster_id]),
                "representative": reps.get(cluster_id, ""),
            }
        )
    table = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)

    _ = forecast_next_week  # (available for the full leave-all-out helper)
    console.print(f"\n[bold]Predicted archetype shifts for the week after {recent.weeks[-1]}[/bold]")
    console.print("(delta = predicted - current; uncertainty = ensemble std)")
    console.print(table.head(top).to_string(index=False))


if __name__ == "__main__":
    app()
