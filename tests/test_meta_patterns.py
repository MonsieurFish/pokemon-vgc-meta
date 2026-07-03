import numpy as np
import pandas as pd

from vgc_team.meta import patterns


def test_ols_recovers_known_line() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.01, size=500)
    result = patterns.ols(y, x, ["x"])
    assert abs(result.coef[0] - 2.0) < 0.05  # intercept
    assert abs(result.coef[1] - 3.0) < 0.05  # slope
    assert result.r2 > 0.99


def _panel_with_momentum(coef: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(20):
        delta = 0.0
        value = 0.5
        for week in range(12):
            delta = coef * delta + rng.normal(scale=0.01)
            value = max(value + delta, 0.0)
            rows.append(
                {
                    "cluster_id": unit,
                    "week": f"2026-W{week:02d}",
                    "share": value,
                    "win_rate": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_test_detects_injected_signal() -> None:
    strong = patterns.momentum_test(_panel_with_momentum(0.8, seed=1))
    assert strong.coef[1] > 0.3  # delta_lag1 coefficient
    assert strong.pvalue[1] < 0.01


def test_momentum_test_null_on_white_noise() -> None:
    null = patterns.momentum_test(_panel_with_momentum(0.0, seed=2))
    assert null.pvalue[1] > 0.05


def test_persistence_baseline_reports_transitions() -> None:
    panel = _panel_with_momentum(0.5, seed=3)
    baseline = patterns.persistence_baseline(panel)
    assert baseline["n_transitions"] > 0
    assert baseline["persistence_mae"] >= 0.0
