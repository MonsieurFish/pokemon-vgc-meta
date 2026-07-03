import numpy as np
import pandas as pd

from vgc_team.meta import timeseries as ts


def _frame() -> pd.DataFrame:
    # week 0: clusters [0,0,1], week 1: clusters [1,1,1]
    return pd.DataFrame(
        [
            {"week": "2026-W01", "cluster_id": 0, "won": True, "weight": 1.0,
             "source": "ladder", "species": ("A", "B")},
            {"week": "2026-W01", "cluster_id": 0, "won": False, "weight": 1.0,
             "source": "ladder", "species": ("A", "C")},
            {"week": "2026-W01", "cluster_id": 1, "won": None, "weight": 1.0,
             "source": "ladder", "species": ("B", "C")},
            {"week": "2026-W02", "cluster_id": 1, "won": True, "weight": 1.0,
             "source": "ladder", "species": ("A", "B")},
            {"week": "2026-W02", "cluster_id": 1, "won": True, "weight": 1.0,
             "source": "ladder", "species": ("A", "B")},
            {"week": "2026-W02", "cluster_id": 1, "won": False, "weight": 1.0,
             "source": "ladder", "species": ("C", "C")},
        ]
    )


def test_cluster_panel_shares_sum_to_one_per_week() -> None:
    panel = ts.cluster_panel(_frame())
    sums = panel.groupby("week")["share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_cluster_win_rate_uses_only_labeled_teams() -> None:
    panel = ts.cluster_panel(_frame())
    # week 1 cluster 0: one win, one loss -> 0.5
    row = panel[(panel["week"] == "2026-W01") & (panel["cluster_id"] == 0)].iloc[0]
    assert row["win_rate"] == 0.5
    # week 1 cluster 1: only an unlabeled team -> NaN
    row1 = panel[(panel["week"] == "2026-W01") & (panel["cluster_id"] == 1)].iloc[0]
    assert np.isnan(row1["win_rate"])


def test_species_panel_dedupes_within_team_and_weights_usage() -> None:
    panel = ts.species_panel(_frame())
    # species C appears once in the W02 'C,C' team -> counted once (de-duped)
    c_w2 = panel[(panel["week"] == "2026-W02") & (panel["species"] == "C")].iloc[0]
    assert c_w2["n_teams"] == 1
    # usage = teams containing species / total teams that week (3 teams in W02)
    assert np.isclose(c_w2["usage"], 1 / 3)


def test_weekly_entropy_and_drift_shapes() -> None:
    frame = _frame()
    panel = ts.cluster_panel(frame)
    entropy = ts.weekly_entropy(panel)
    assert set(entropy["week"]) == {"2026-W01", "2026-W02"}
    # week 2 is single-cluster -> zero entropy
    assert entropy[entropy["week"] == "2026-W02"]["entropy_bits"].iloc[0] == 0.0

    embeddings = np.random.default_rng(0).normal(size=(len(frame), 4)).astype(np.float32)
    drift = ts.weekly_centroid_drift(frame, embeddings)
    assert list(drift["week"]) == ["2026-W02"]  # one transition
