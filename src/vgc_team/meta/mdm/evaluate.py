"""Rolling backtest of the MDM against the persistence baseline.

For each held-out week ``h`` we retrain on weeks ``< h`` only, forecast week
``h`` in the leave-all-out regime, and compare the predicted anchor distribution
to the truth via KL — against persistence ("week h looks like week h-1"). Beating
persistence is the bar that proves the model learned dynamics, not stickiness.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from vgc_team.meta.mdm.dataset import (
    TeamFeatures,
    single_reg_view,
    subset_regulations,
)
from vgc_team.meta.mdm.model import MetaDistributionModel
from vgc_team.meta.mdm.predict import forecast_next_week


def slice_before(features: TeamFeatures, h: int) -> TeamFeatures:
    """Sub-features containing only weeks strictly before ``h`` (self-consistent)."""

    mask = features.week_index < h
    return TeamFeatures(
        embeddings=features.embeddings[mask],
        anchors=features.anchors[mask],
        week_index=features.week_index[mask],
        weight=features.weight[mask],
        weeks=features.weeks[:h],
    )


def anchor_histogram(
    anchors: np.ndarray, weights: np.ndarray, n_anchors: int, *, smoothing: float = 1.0
) -> np.ndarray:
    counts = np.bincount(anchors, weights=weights, minlength=n_anchors).astype(np.float64)
    counts += smoothing
    return (counts / counts.sum()).astype(np.float64)


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-9) -> float:
    """KL(p || q) in bits."""

    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def backtest(
    features: TeamFeatures,
    n_anchors: int,
    train_fn: Callable[[TeamFeatures], list[MetaDistributionModel]],
    *,
    holdout_weeks: int = 2,
    device=None,
) -> pd.DataFrame:
    """Retrain-on-past rolling backtest. Returns one row per held-out week."""

    W = features.n_weeks
    first = max(1, W - holdout_weeks)
    rows = []
    for h in range(first, W):
        sub = slice_before(features, h)
        models = train_fn(sub)

        anchor_pred, _ = forecast_next_week(models, sub, device=device)

        actual = anchor_histogram(
            features.anchors[features.week_index == h],
            features.weight[features.week_index == h],
            n_anchors,
        )
        persistence = anchor_histogram(
            features.anchors[features.week_index == (h - 1)],
            features.weight[features.week_index == (h - 1)],
            n_anchors,
        )

        rows.append(
            {
                "week": features.weeks[h],
                "kl_model": kl_divergence(actual, anchor_pred),
                "kl_persistence": kl_divergence(actual, persistence),
                "n_teams": int((features.week_index == h).sum()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["model_beats_persistence"] = table["kl_model"] < table["kl_persistence"]
    return table


def backtest_leave_one_reg_out(
    features: TeamFeatures,
    n_anchors: int,
    train_fn: Callable[[TeamFeatures], list[MetaDistributionModel]],
    *,
    holdout_reg_ids: list[int] | None = None,
    device=None,
) -> pd.DataFrame:
    """Leave-one-regulation-out transfer test.

    For each held-out regulation, train on the *other* regulations, then forecast
    the held-out regulation's final week from its own earlier weeks (deployment:
    you have the new meta's recent weeks, predict next) — vs persistence. This is
    the honest test of whether the model learned transferable dynamics.
    """

    n_reg = features.n_regulations
    holdout_reg_ids = holdout_reg_ids if holdout_reg_ids is not None else list(range(n_reg))
    labels = features.reg_labels or [str(r) for r in range(n_reg)]
    rows = []
    for h in holdout_reg_ids:
        train_ids = [r for r in range(n_reg) if r != h]
        models = train_fn(subset_regulations(features, train_ids))

        held = single_reg_view(features, h)
        if held.n_weeks < 2:
            continue
        last = held.n_weeks - 1
        context = slice_before(held, last)
        anchor_pred, _ = forecast_next_week(models, context, device=device)

        actual = anchor_histogram(
            held.anchors[held.week_index == last], held.weight[held.week_index == last], n_anchors
        )
        persistence = anchor_histogram(
            held.anchors[held.week_index == (last - 1)],
            held.weight[held.week_index == (last - 1)],
            n_anchors,
        )
        rows.append({
            "held_out_reg": labels[h],
            "kl_model": kl_divergence(actual, anchor_pred),
            "kl_persistence": kl_divergence(actual, persistence),
            "n_teams": int((features.reg_ids() == h).sum()),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["model_beats_persistence"] = table["kl_model"] < table["kl_persistence"]
    return table
