"""Falsifiable 'exploitable pattern' tests + the persistence baseline.

These are deliberately simple, fully transparent statistics on the weekly panels.
They serve two purposes:

1. Tell us whether there is *any* learnable temporal signal (momentum,
   winners-lead-usage) before we invest in the MDM.
2. Provide the **persistence baseline** the MDM must beat — "next week looks like
   this week." Metas are sticky, so this is a strong opponent.

Implemented with numpy + scipy only (no statsmodels dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pvalue: np.ndarray
    r2: float
    n: int

    def table(self) -> str:
        lines = [f"  n={self.n}  R^2={self.r2:.4f}"]
        lines.append(f"  {'term':<16}{'coef':>12}{'se':>12}{'t':>10}{'p':>10}")
        for name, c, se, t, p in zip(
            self.names, self.coef, self.se, self.tstat, self.pvalue, strict=True
        ):
            lines.append(f"  {name:<16}{c:>12.5f}{se:>12.5f}{t:>10.3f}{p:>10.4f}")
        return "\n".join(lines)


def ols(y: np.ndarray, X: np.ndarray, names: list[str], *, add_intercept: bool = True) -> OLSResult:
    """Ordinary least squares with t-stats / p-values."""

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["intercept", *names]

    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    dof = max(n - k, 1)
    sigma2 = float(residuals @ residuals) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(residuals @ residuals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return OLSResult(names, beta, se, tstat, pvalue, r2, n)


def _sorted_with_lags(
    panel: pd.DataFrame, unit_col: str, value_col: str
) -> pd.DataFrame:
    df = panel.sort_values([unit_col, "week"]).copy()
    df["value"] = df[value_col]
    df["value_lag1"] = df.groupby(unit_col)["value"].shift(1)
    df["delta"] = df["value"] - df["value_lag1"]
    df["delta_lag1"] = df.groupby(unit_col)["delta"].shift(1)
    df["winrate_lag1"] = df.groupby(unit_col)["win_rate"].shift(1)
    return df


def momentum_test(
    panel: pd.DataFrame,
    *,
    unit_col: str = "cluster_id",
    value_col: str = "share",
    fixed_effects: bool = True,
) -> OLSResult:
    """Does this week's share change predict next week's? delta_t ~ delta_{t-1}."""

    df = _sorted_with_lags(panel, unit_col, value_col).dropna(subset=["delta", "delta_lag1"])
    y = df["delta"].to_numpy()
    x = df["delta_lag1"].to_numpy()
    if fixed_effects:
        groups = df[unit_col].to_numpy()
        y = _within_demean(y, groups)
        x = _within_demean(x, groups)
    return ols(y, x, ["delta_lag1"])


def winners_lead_usage_test(
    panel: pd.DataFrame,
    *,
    unit_col: str = "cluster_id",
    value_col: str = "share",
) -> OLSResult:
    """Does over-performing this week predict next week's share?

    value_t ~ winrate_{t-1} + value_{t-1}. A positive winrate coefficient after
    controlling for current share is the "meta follows winners" signal.
    """

    df = _sorted_with_lags(panel, unit_col, value_col).dropna(
        subset=["value", "winrate_lag1", "value_lag1"]
    )
    y = df["value"].to_numpy()
    X = df[["winrate_lag1", "value_lag1"]].to_numpy()
    return ols(y, X, ["winrate_lag1", "value_lag1"])


def _within_demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    for g in np.unique(groups):
        mask = groups == g
        out[mask] -= out[mask].mean()
    return out


def persistence_baseline(
    panel: pd.DataFrame,
    *,
    unit_col: str = "cluster_id",
    value_col: str = "share",
) -> dict[str, float]:
    """Mean abs error of 'next = current' vs 'next = unit's mean share'.

    The MDM is only worthwhile if it beats persistence_mae.
    """

    df = _sorted_with_lags(panel, unit_col, value_col).dropna(subset=["value", "value_lag1"])
    persistence_mae = float(np.abs(df["value"] - df["value_lag1"]).mean())

    unit_mean = df.groupby(unit_col)["value"].transform("mean")
    climatology_mae = float(np.abs(df["value"] - unit_mean).mean())

    return {
        "persistence_mae": persistence_mae,
        "climatology_mae": climatology_mae,
        "n_transitions": int(len(df)),
    }
