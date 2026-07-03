"""Leave-one-regulation-out hyperparameter sweep for MDM regularization.

Embeds all teams ONCE, then evaluates each regularization config by leave-one-
regulation-out CV (train on the other regs, forecast the held-out reg's next week
vs persistence). The many small trainings run in parallel across CPU processes
(the MDM is tiny, so CPU-parallel is faster than serial MPS). Picks the config
with the best mean margin over persistence and (optionally) trains a final model.

    python scripts/sweep_mdm_regularization.py --epochs 5
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.tournaments import load_tournament_teams
from vgc_team.meta.mdm.dataset import build_multi_reg_features
from vgc_team.models.frozen_encoder import ReferenceSpace, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()

# Regularization grid. correction_l2 is the primary lever (shrinks the residual
# correction toward persistence); dropout / weight_decay are secondary.
GRID = [
    {"dropout": 0.1, "weight_decay": 0.05, "correction_l2": 0.0},
    {"dropout": 0.1, "weight_decay": 0.05, "correction_l2": 0.3},
    {"dropout": 0.1, "weight_decay": 0.05, "correction_l2": 1.0},
    {"dropout": 0.1, "weight_decay": 0.05, "correction_l2": 3.0},
    {"dropout": 0.3, "weight_decay": 0.05, "correction_l2": 0.3},
    {"dropout": 0.3, "weight_decay": 0.05, "correction_l2": 1.0},
    {"dropout": 0.3, "weight_decay": 0.10, "correction_l2": 1.0},
    {"dropout": 0.3, "weight_decay": 0.10, "correction_l2": 3.0},
]

# --- worker globals (populated per process) ---
_F = None
_N_ANCHORS = None
_EPOCHS = None
_FULL_MASK = None


def _load_features(npz_path: str, meta_path: str):
    from vgc_team.meta.mdm.dataset import TeamFeatures

    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(Path(meta_path).read_text())
    return TeamFeatures(
        embeddings=z["embeddings"], anchors=z["anchors"], week_index=z["week_index"],
        weight=z["weight"], weeks=[], regulation=z["regulation"],
        reg_weeks=meta["reg_weeks"], reg_labels=meta["reg_labels"],
    )


def _init_worker(npz_path: str, meta_path: str, n_anchors: int, epochs: int, full_mask: float):
    import torch

    torch.set_num_threads(1)  # single-thread each process -> clean cross-process parallelism
    global _F, _N_ANCHORS, _EPOCHS, _FULL_MASK
    _F = _load_features(npz_path, meta_path)
    _N_ANCHORS, _EPOCHS, _FULL_MASK = n_anchors, epochs, full_mask


def _run_fold(args: tuple[int, int]) -> dict | None:
    config_idx, fold = args
    from vgc_team.meta.mdm.dataset import (
        MDMDatasetConfig,
        single_reg_view,
        subset_regulations,
    )
    from vgc_team.meta.mdm.evaluate import anchor_histogram, kl_divergence, slice_before
    from vgc_team.meta.mdm.predict import forecast_next_week
    from vgc_team.meta.mdm.train import MDMTrainConfig, train_single

    cfg_params = GRID[config_idx]
    train_ids = [r for r in range(_F.n_regulations) if r != fold]
    train_feat = subset_regulations(_F, train_ids)

    cfg = MDMTrainConfig(
        epochs=_EPOCHS, dropout=cfg_params["dropout"], weight_decay=cfg_params["weight_decay"],
        correction_l2=cfg_params["correction_l2"],
        dataset=MDMDatasetConfig(full_mask_prob=_FULL_MASK), device="cpu", verbose=False,
    )
    model = train_single(train_feat, _N_ANCHORS, cfg, seed=0)

    held = single_reg_view(_F, fold)
    if held.n_weeks < 2:
        return None
    last = held.n_weeks - 1
    pred, _ = forecast_next_week([model], slice_before(held, last), device="cpu")
    actual = anchor_histogram(
        held.anchors[held.week_index == last], held.weight[held.week_index == last], _N_ANCHORS
    )
    persist = anchor_histogram(
        held.anchors[held.week_index == last - 1],
        held.weight[held.week_index == last - 1], _N_ANCHORS,
    )
    return {
        "config_idx": config_idx,
        "fold": _F.reg_labels[fold],
        "kl_model": kl_divergence(actual, pred),
        "kl_persistence": kl_divergence(actual, persist),
    }


@app.command()
def main(
    reg_dir: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "tournaments" / "multi_reg",
    reference_npz: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "multi_reg" / "reference.npz",
    checkpoint_path: Annotated[Path, typer.Option()] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    epochs: Annotated[int, typer.Option(help="Epochs per fold training.")] = 5,
    full_mask_prob: Annotated[float, typer.Option()] = 0.9,
    balance: Annotated[str, typer.Option()] = "equal",
    workers: Annotated[int, typer.Option(help="Parallel processes (default cpu-2).")] = 0,
    n_folds: Annotated[
        int, typer.Option(help="Use the N largest regs as held-out folds (0 = all).")
    ] = 5,
    reuse_features: Annotated[
        bool, typer.Option(help="Reuse cached embedded features in --scratch if present.")
    ] = True,
    scratch: Annotated[Path, typer.Option(help="Temp dir for shared feature arrays.")] = Path(
        os.environ.get("SCRATCH_DIR", "/tmp")
    ),
) -> None:
    reference = ReferenceSpace.load(reference_npz)
    scratch.mkdir(parents=True, exist_ok=True)
    npz_path = str(scratch / "sweep_features.npz")
    meta_path = str(scratch / "sweep_features.json")

    if reuse_features and Path(npz_path).exists() and Path(meta_path).exists():
        print("Reusing cached embedded features.", flush=True)
        features = _load_features(npz_path, meta_path)
    else:
        device = select_device()
        model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
        regulations = []
        for f in sorted(reg_dir.glob("reg*.json")):
            teams, weights = load_tournament_teams(f)
            regulations.append((f.stem, teams, weights))
        print(f"Embedding {sum(len(t) for _, t, _ in regulations)} teams once...", flush=True)
        features = build_multi_reg_features(
            regulations, reference, model_enc, kind_vocab, token_vocab, balance=balance, device=device
        )
        np.savez(npz_path, embeddings=features.embeddings, anchors=features.anchors,
                 week_index=features.week_index, weight=features.weight, regulation=features.regulation)
        Path(meta_path).write_text(
            json.dumps({"reg_weeks": features.reg_weeks, "reg_labels": features.reg_labels})
        )

    # choose held-out folds: the N largest regulations (most reliable to evaluate)
    counts = np.bincount(features.reg_ids(), minlength=features.n_regulations)
    fold_ids = list(range(features.n_regulations))
    if n_folds and n_folds < features.n_regulations:
        fold_ids = sorted(fold_ids, key=lambda r: counts[r], reverse=True)[:n_folds]
    print(f"Folds ({len(fold_ids)}): {[features.reg_labels[r] for r in fold_ids]}", flush=True)

    jobs = [(ci, fold) for ci in range(len(GRID)) for fold in fold_ids]
    n_workers = workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"Running {len(jobs)} fold-trainings ({len(GRID)} configs x {len(fold_ids)} folds) "
          f"on {n_workers} workers, {epochs} epochs each...", flush=True)

    results = []
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(npz_path, meta_path, reference.n_anchors, epochs, full_mask_prob),
    ) as ex:
        futs = {ex.submit(_run_fold, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r:
                results.append(r)
                print(f"  [{done}/{len(jobs)}] cfg{r['config_idx']} {r['fold']}: "
                      f"kl_model={r['kl_model']:.3f} kl_persist={r['kl_persistence']:.3f}", flush=True)
            else:
                print(f"  [{done}/{len(jobs)}] skipped (fold too short)", flush=True)

    df = pd.DataFrame(results)
    df["margin"] = df["kl_model"] - df["kl_persistence"]  # negative => beats persistence
    summary = df.groupby("config_idx").agg(
        mean_kl_model=("kl_model", "mean"),
        mean_kl_persist=("kl_persistence", "mean"),
        mean_margin=("margin", "mean"),
        beats=("margin", lambda s: int((s < 0).sum())),
        folds=("margin", "size"),
    ).reset_index()
    for col in ("dropout", "weight_decay", "correction_l2"):
        summary[col] = summary["config_idx"].map(lambda i: GRID[i][col])
    summary = summary.sort_values("mean_margin").reset_index(drop=True)

    console.print("\n[bold]Sweep results (sorted by mean margin vs persistence; negative = better)[/bold]")
    console.print(summary[[
        "dropout", "weight_decay", "correction_l2",
        "mean_kl_model", "mean_kl_persist", "mean_margin", "beats", "folds",
    ]].to_string(index=False))

    best = summary.iloc[0]
    console.print(
        f"\n[bold]Best config[/bold]: dropout={best.dropout} weight_decay={best.weight_decay} "
        f"correction_l2={best.correction_l2}  "
        f"(mean margin {best.mean_margin:+.4f}, beats persistence {int(best.beats)}/{int(best.folds)})"
    )
    out = DATA_DIR / "processed" / "multi_reg" / "sweep_results.csv"
    summary.to_csv(out, index=False)
    console.print(f"Saved full results to {out}")


if __name__ == "__main__":
    app()
