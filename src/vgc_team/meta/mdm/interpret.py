"""Turn an anchor-level distribution into interpretable per-archetype shifts.

The MDM predicts over the fine anchor codebook; humans want archetype clusters.
Each anchor is mapped to its nearest interpretable cluster, anchor mass is summed
per cluster, and the predicted meta is compared to the current meta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vgc_team.models.frozen_encoder import ReferenceSpace


def anchor_to_cluster_distribution(
    anchor_dist: np.ndarray, reference: ReferenceSpace
) -> np.ndarray:
    """Sum anchor probability mass into cluster shares (n_clusters,)."""

    mapping = reference.anchor_to_cluster()
    cluster_dist = np.zeros(reference.n_clusters, dtype=np.float64)
    np.add.at(cluster_dist, mapping, anchor_dist)
    total = cluster_dist.sum()
    return (cluster_dist / total).astype(np.float32) if total > 0 else cluster_dist.astype(np.float32)


def cluster_shift_table(
    current_cluster_dist: np.ndarray,
    predicted_cluster_dist: np.ndarray,
    *,
    representatives: dict[int, str] | None = None,
    min_share_for_pct: float = 0.01,
) -> pd.DataFrame:
    """Per-cluster current vs predicted share, absolute and relative change.

    ``pct_change`` is only reported for clusters with a non-trivial current share
    (>= ``min_share_for_pct``); on near-zero shares a percent change explodes and
    is meaningless, so it is left as NaN — read ``delta`` (absolute) there.
    """

    representatives = representatives or {}
    rows = []
    for cluster_id, (current, predicted) in enumerate(
        zip(current_cluster_dist, predicted_cluster_dist, strict=True)
    ):
        delta = float(predicted - current)
        pct = float(delta / current) if current >= min_share_for_pct else float("nan")
        rows.append(
            {
                "cluster_id": cluster_id,
                "current_share": float(current),
                "predicted_share": float(predicted),
                "delta": delta,
                "pct_change": pct,
                "representative": representatives.get(cluster_id, ""),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("delta", ascending=False).reset_index(drop=True)
