"""Horse race: UCSV-style forecasters vs last-week persistence (and the MDM).

Races every method over ALL forecastable weeks in ALL regulations (~111 points),
with leave-one-regulation-out hyperparameter selection so nothing is overfit.
Reports mean KL and paired %-beating-persistence per method — the direct test of
whether the inflation playbook (UCSV / glide / combination) cracks the meta.

    python scripts/eval_forecasters.py
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
from vgc_team.meta import ucsv
from vgc_team.meta.mdm.dataset import build_multi_reg_features, single_reg_view
from vgc_team.meta.mdm.evaluate import kl_divergence
from vgc_team.meta.mdm.interpret import anchor_to_cluster_distribution, cluster_shift_table
from vgc_team.models.frozen_encoder import ReferenceSpace, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()


def _forecast_points(features, reference) -> tuple[list, np.ndarray]:
    """(reg_id, history (t,K), actual (K,)) for every forecastable week t>=1."""

    points = []
    reg_ids = []
    for reg_id in range(features.n_regulations):
        view = single_reg_view(features, reg_id)
        P = ucsv.weekly_anchor_matrix(view, reference.n_anchors)
        for t in range(1, P.shape[0]):
            points.append((reg_id, P[:t], P[t]))
            reg_ids.append(reg_id)
    return points, np.array(reg_ids)


def _kl_for_candidate(forecaster, points) -> np.ndarray:
    return np.array([kl_divergence(actual, forecaster.forecast(hist)) for _, hist, actual in points])


def _loro_select(kl_matrix: np.ndarray, point_reg: np.ndarray, labels: list[str]):
    """Per held-out reg, pick the candidate best on the *other* regs -> honest per-point KL."""

    honest = np.empty(kl_matrix.shape[1])
    chosen = []
    for r in np.unique(point_reg):
        test = point_reg == r
        train_mean = kl_matrix[:, ~test].mean(axis=1)
        best = int(train_mean.argmin())
        honest[test] = kl_matrix[best, test]
        chosen.append(labels[best])
    return honest, chosen


@app.command()
def main(
    reg_dir: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "tournaments" / "multi_reg",
    reference_npz: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "multi_reg" / "reference.npz",
    checkpoint_path: Annotated[Path, typer.Option()] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    include_mdm: Annotated[bool, typer.Option(help="Also race the trained multi-reg MDM if present.")] = True,
    mdm_path: Annotated[Path, typer.Option()] = PROJECT_ROOT / "models" / "mdm_multireg" / "mdm.pt",
) -> None:
    reference = ReferenceSpace.load(reference_npz)
    device = select_device()
    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
    regulations = [(f.stem, *load_tournament_teams(f)) for f in sorted(reg_dir.glob("reg*.json"))]
    console.print(f"Embedding {sum(len(t) for _, t, _ in regulations)} teams once...")
    features = build_multi_reg_features(
        regulations, reference, model_enc, kind_vocab, token_vocab, balance="equal", device=device
    )

    points, point_reg = _forecast_points(features, reference)
    console.print(f"{len(points)} forecastable (reg, week) points across {features.n_regulations} regs.")

    ref_idx = ucsv.most_common_anchor(features, reference.n_anchors)

    # persistence baseline (no hyperparameter)
    kl_persist = _kl_for_candidate(ucsv.Persistence(), points)

    # candidate families (LORO-selected)
    families: dict[str, list] = {
        "ucsv_local_level": [ucsv.LocalLevel(q, ref_idx) for q in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)],
        "ucsv_stoch_vol": [
            ucsv.StochVol(qb, g, ref_idx) for qb in (0.1, 0.5, 1.0, 2.0) for g in (0.5, 0.8)
        ],
        "glide_to_anchor": [ucsv.Glide(lam) for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8)],
        "combination": [
            ucsv.Combination(ucsv.LocalLevel(q, ref_idx), w)
            for q in (0.2, 0.5, 1.0) for w in (0.25, 0.5, 0.75)
        ],
    }

    rows = [{
        "method": "persistence", "mean_kl": float(kl_persist.mean()),
        "beats_persist_%": 0.0, "selected": "-",
    }]
    for name, candidates in families.items():
        kl_matrix = np.stack([_kl_for_candidate(c, points) for c in candidates])
        honest, chosen = _loro_select(kl_matrix, point_reg, [c.name for c in candidates])
        rows.append({
            "method": name, "mean_kl": float(honest.mean()),
            "beats_persist_%": float(100.0 * np.mean(honest < kl_persist)),
            "selected": pd.Series(chosen).value_counts().index[0],
        })

    if include_mdm and mdm_path.exists():
        try:
            from vgc_team.meta.mdm.evaluate import slice_before
            from vgc_team.meta.mdm.predict import forecast_next_week
            from vgc_team.meta.mdm.train import load_ensemble

            models = load_ensemble(mdm_path, device)
            views = {r: single_reg_view(features, r) for r in range(features.n_regulations)}
            kl_mdm = []
            for reg_id, hist, actual in points:
                t = hist.shape[0]
                pred, _ = forecast_next_week(models, slice_before(views[reg_id], t), device=device)
                kl_mdm.append(kl_divergence(actual, pred))
            kl_mdm = np.array(kl_mdm)
            rows.append({
                "method": "mdm_multireg", "mean_kl": float(kl_mdm.mean()),
                "beats_persist_%": float(100.0 * np.mean(kl_mdm < kl_persist)), "selected": "-",
            })
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Skipping MDM in race: {exc}[/yellow]")

    table = pd.DataFrame(rows).sort_values("mean_kl").reset_index(drop=True)
    console.print("\n[bold]Forecaster horse race (LORO-honest; lower mean KL is better)[/bold]")
    console.print(table.to_string(index=False))
    console.print(f"\npersistence mean KL = {kl_persist.mean():.4f}  (the bar to beat)")

    # best method -> next-week forecast for the most recent regulation
    best_name = table.iloc[0]["method"]
    if best_name in families:
        candidates = families[best_name]
        kl_matrix = np.stack([_kl_for_candidate(c, points) for c in candidates])
        best_cand = candidates[int(kl_matrix.mean(axis=1).argmin())]
        recent = single_reg_view(features, features.n_regulations - 1)
        P = ucsv.weekly_anchor_matrix(recent, reference.n_anchors)
        predicted = anchor_to_cluster_distribution(best_cand.forecast(P), reference)
        current = anchor_to_cluster_distribution(P[-1], reference)
        reps = {}
        rep_path = reference_npz.parent / "cluster_representatives.csv"
        if rep_path.exists():
            reps = {int(r.cluster_id): str(r.species) for r in pd.read_csv(rep_path).itertuples()}
        shift = cluster_shift_table(current, predicted, representatives=reps)
        label = features.reg_labels[-1] if features.reg_labels else "recent"
        console.print(f"\n[bold]Next-meta shifts for {label} via {best_cand.name}[/bold]")
        console.print(shift.head(10).to_string(index=False))


if __name__ == "__main__":
    app()
