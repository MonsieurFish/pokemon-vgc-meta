"""Train the Meta Distribution Model across many regulations.

Each regulation is an independent meta trajectory. Context never crosses
regulations, weeks are per-regulation, and regulations are balanced so no single
one dominates the loss. Optionally runs a leave-one-regulation-out transfer test
(train on the others, forecast the held-out reg's next meta vs persistence).

    # 1) build the balanced pooled codebook once
    python scripts/build_multi_reg_anchors.py
    # 2) train
    python scripts/train_multi_reg_mdm.py --epochs 50
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.tournaments import load_tournament_teams
from vgc_team.meta.mdm.dataset import (
    MDMDatasetConfig,
    build_multi_reg_features,
    single_reg_view,
)
from vgc_team.meta.mdm.evaluate import (
    anchor_histogram,
    backtest_leave_one_reg_out,
    slice_before,
)
from vgc_team.meta.mdm.interpret import anchor_to_cluster_distribution, cluster_shift_table
from vgc_team.meta.mdm.predict import forecast_next_week
from vgc_team.meta.mdm.train import MDMTrainConfig, save_ensemble, train_ensemble
from vgc_team.models.frozen_encoder import ReferenceSpace, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    reg_dir: Annotated[
        Path, typer.Option(help="Directory of per-regulation team JSON files.")
    ] = DATA_DIR / "processed" / "tournaments" / "multi_reg",
    reference_npz: Annotated[
        Path, typer.Option(help="Pooled balanced reference from build_multi_reg_anchors.py.")
    ] = DATA_DIR / "processed" / "multi_reg" / "reference.npz",
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    epochs: Annotated[int, typer.Option(help="Training epochs (fixed).")] = 50,
    n_models: Annotated[int, typer.Option(help="Ensemble size.")] = 5,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 128,
    dropout: Annotated[float, typer.Option(help="Dropout in the MDM head/tokens.")] = 0.1,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay.")] = 0.01,
    correction_l2: Annotated[
        float, typer.Option(help="L2 penalty on the residual correction (pulls toward persistence).")
    ] = 0.0,
    balance: Annotated[
        str, typer.Option(help="Per-regulation balancing: equal | sqrt | none.")
    ] = "equal",
    full_mask_prob: Annotated[
        float, typer.Option(help="P(current week fully masked) — high = emphasize forecasting.")
    ] = 0.9,
    loro: Annotated[
        bool, typer.Option(help="Run leave-one-regulation-out transfer test (expensive).")
    ] = False,
    loro_epochs: Annotated[int, typer.Option(help="Epochs per LORO fold (cheaper).")] = 20,
    loro_models: Annotated[int, typer.Option(help="Ensemble size per LORO fold.")] = 2,
    output_dir: Annotated[
        Path, typer.Option(help="Where the trained ensemble is saved.")
    ] = PROJECT_ROOT / "models" / "mdm_multireg",
) -> None:
    if not reference_npz.exists():
        raise typer.BadParameter(
            f"Missing {reference_npz}. Run scripts/build_multi_reg_anchors.py first."
        )
    files = sorted(reg_dir.glob("reg*.json"))
    if not files:
        raise typer.BadParameter(f"No reg*.json in {reg_dir}.")

    reference = ReferenceSpace.load(reference_npz)
    device = select_device()
    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)

    regulations = []
    for f in files:
        teams, weights = load_tournament_teams(f)
        regulations.append((f.stem, teams, weights))
    console.print(f"Loaded {len(regulations)} regulations: {[r[0] for r in regulations]}")

    features = build_multi_reg_features(
        regulations, reference, model_enc, kind_vocab, token_vocab, balance=balance, device=device
    )
    console.print(
        f"{features.n_regulations} regulations, {len(features.anchors)} teams, "
        f"balance={balance!r} (per-reg total weight equalized)"
    )

    cfg = MDMTrainConfig(
        epochs=epochs, batch_size=batch_size, dropout=dropout,
        weight_decay=weight_decay, correction_l2=correction_l2,
        dataset=MDMDatasetConfig(full_mask_prob=full_mask_prob), device=device, verbose=True,
    )

    if loro:
        console.print("\n[bold]Leave-one-regulation-out transfer test[/bold]")
        loro_cfg = MDMTrainConfig(
            epochs=loro_epochs, batch_size=batch_size, dropout=dropout,
            weight_decay=weight_decay, correction_l2=correction_l2,
            dataset=MDMDatasetConfig(full_mask_prob=full_mask_prob), device=device, verbose=False,
        )
        table = backtest_leave_one_reg_out(
            features, reference.n_anchors,
            lambda f: train_ensemble(f, reference.n_anchors, loro_cfg, n_models=loro_models),
            device=device,
        )
        console.print(table.to_string(index=False))
        if not table.empty:
            console.print(
                f"mean KL: model={table['kl_model'].mean():.4f} "
                f"persistence={table['kl_persistence'].mean():.4f} "
                f"| beats persistence {int(table['model_beats_persistence'].sum())}/{len(table)} regs"
            )

    console.print(f"\n[bold]Training final ensemble on all {features.n_regulations} regs "
                  f"({epochs} epochs x {n_models} models)...[/bold]")
    models = train_ensemble(features, reference.n_anchors, cfg, n_models=n_models)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_ensemble(models, output_dir / "mdm.pt", n_anchors=reference.n_anchors,
                  d_in=features.embeddings.shape[1], d_model=cfg.d_model)
    console.print(f"Saved ensemble to {output_dir / 'mdm.pt'}")

    # demo: forecast the most recent regulation's next week
    recent_id = features.n_regulations - 1
    recent = single_reg_view(features, recent_id)
    anchor_pred, _ = forecast_next_week(models, recent, device=device)
    predicted = anchor_to_cluster_distribution(anchor_pred, reference)
    last = recent.n_weeks - 1
    current = anchor_to_cluster_distribution(
        anchor_histogram(recent.anchors[recent.week_index == last],
                         recent.weight[recent.week_index == last], reference.n_anchors),
        reference,
    )
    reps = {}
    rep_path = reference_npz.parent / "cluster_representatives.csv"
    if rep_path.exists():
        reps = {int(r.cluster_id): str(r.species) for r in pd.read_csv(rep_path).itertuples()}
    shift = cluster_shift_table(current, predicted, representatives=reps)
    label = features.reg_labels[recent_id] if features.reg_labels else str(recent_id)
    console.print(f"\n[bold]Predicted next-meta shifts for {label}[/bold]")
    console.print(shift.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
