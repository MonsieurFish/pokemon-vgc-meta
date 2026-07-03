"""Compositional UCSV-style forecasters for the weekly meta distribution.

Motivation: across leave-one-regulation-out, the neural MDM never beats a
last-week persistence baseline — at the weekly level the meta is close to a
random walk. This is the classic inflation-forecasting situation, where the one
thing that reliably beats a random walk is **UCSV** (Stock-Watson 2007): a
random-walk *trend* + transitory noise whose variances are time-varying, i.e.
adaptive smoothing.

Here the "series" is the weekly anchor distribution p_t (K anchors). We work in
additive-log-ratio (alr) space so the simplex becomes unconstrained R^(K-1),
run a local-level Kalman filter per dimension, and map the filtered trend back
through the softmax. No neural weights, no encoder gradients — the frozen encoder
+ anchor codebook are used only upstream to produce p_t.
"""

from __future__ import annotations

import numpy as np

from vgc_team.meta.mdm.dataset import TeamFeatures, single_reg_view
from vgc_team.meta.mdm.evaluate import anchor_histogram


def weekly_anchor_matrix(
    reg_view: TeamFeatures, n_anchors: int, *, smoothing: float = 1.0
) -> np.ndarray:
    """(W, K) matrix: one smoothed anchor distribution per within-regulation week."""

    W = reg_view.n_weeks
    out = np.zeros((W, n_anchors), dtype=np.float64)
    for w in range(W):
        mask = reg_view.week_index == w
        out[w] = anchor_histogram(reg_view.anchors[mask], reg_view.weight[mask], n_anchors,
                                  smoothing=smoothing)
    return out


def most_common_anchor(features: TeamFeatures, n_anchors: int) -> int:
    """Global reference anchor for the alr transform (avoids a rare denominator)."""

    return int(np.bincount(features.anchors, minlength=n_anchors).argmax())


# --- compositional transform (additive log-ratio with a fixed reference) ---

def alr(P: np.ndarray, ref_idx: int) -> np.ndarray:
    """(W, K) simplex rows -> (W, K-1) unconstrained (log p_i / p_ref for i != ref)."""

    non_ref = [i for i in range(P.shape[1]) if i != ref_idx]
    return np.log(P[:, non_ref]) - np.log(P[:, ref_idx : ref_idx + 1])


def inv_alr(y: np.ndarray, ref_idx: int, n_anchors: int) -> np.ndarray:
    """(K-1,) alr coords -> (K,) simplex (softmax with 0 at the reference)."""

    full = np.zeros(n_anchors, dtype=np.float64)
    non_ref = [i for i in range(n_anchors) if i != ref_idx]
    full[non_ref] = y
    full[ref_idx] = 0.0
    full -= full.max()
    e = np.exp(full)
    return e / e.sum()


# --- Kalman local-level filters (vectorized across the K-1 alr dimensions) ---

def _kalman_local_level(Y: np.ndarray, q: float) -> np.ndarray:
    """Filtered level at the last step of each dim (forecast of the next step).

    Local level: state a_t = a_{t-1} + N(0, Q); obs y_t = a_t + N(0, R).
    R fixed to 1, Q = q is the signal-to-noise ratio. q -> inf => track last obs
    (persistence); q -> 0 => heavy smoothing toward the mean.
    """

    W, D = Y.shape
    a = Y[0].copy()
    P = np.ones(D)  # R
    for t in range(1, W):
        P_pred = P + q
        v = Y[t] - a
        F = P_pred + 1.0
        gain = P_pred / F
        a = a + gain * v
        P = P_pred * (1.0 - gain)
    return a


def _kalman_stoch_vol(Y: np.ndarray, q_base: float, gamma: float) -> np.ndarray:
    """Local level with a time-varying, shared level-variance (SV-lite).

    Q_t = q_base * vol_t, where vol_t is an EWMA of the mean squared innovation
    across dimensions -> smooth when the meta is calm, track when it churns.
    """

    W, D = Y.shape
    a = Y[0].copy()
    P = np.ones(D)
    vol = 1.0
    for t in range(1, W):
        P_pred = P + q_base * vol
        v = Y[t] - a
        F = P_pred + 1.0
        gain = P_pred / F
        a = a + gain * v
        P = P_pred * (1.0 - gain)
        vol = gamma * vol + (1.0 - gamma) * float(np.mean(v**2))
    return a


# --- forecasters: each maps a (W, K) history to a (K,) next-week prediction ---

class Forecaster:
    name = "base"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:  # (W, K) -> (K,)
        raise NotImplementedError


class Persistence(Forecaster):
    name = "persistence"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:
        return P_history[-1]


class LocalLevel(Forecaster):
    def __init__(self, q: float, ref_idx: int) -> None:
        self.q = q
        self.ref_idx = ref_idx
        self.name = f"ucsv_ll(q={q:g})"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:
        K = P_history.shape[1]
        trend = _kalman_local_level(alr(P_history, self.ref_idx), self.q)
        return inv_alr(trend, self.ref_idx, K)


class StochVol(Forecaster):
    def __init__(self, q_base: float, gamma: float, ref_idx: int) -> None:
        self.q_base = q_base
        self.gamma = gamma
        self.ref_idx = ref_idx
        self.name = f"ucsv_sv(q={q_base:g},g={gamma:g})"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:
        K = P_history.shape[1]
        trend = _kalman_stoch_vol(alr(P_history, self.ref_idx), self.q_base, self.gamma)
        return inv_alr(trend, self.ref_idx, K)


class Glide(Forecaster):
    """Faust-Wright glide: blend last week toward the regulation's running mean."""

    def __init__(self, lam: float) -> None:
        self.lam = lam
        self.name = f"glide(λ={lam:g})"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:
        blended = (1.0 - self.lam) * P_history[-1] + self.lam * P_history.mean(axis=0)
        return blended / blended.sum()


class Combination(Forecaster):
    """Forecast combination: convex mix of persistence and a base forecaster."""

    def __init__(self, base: Forecaster, w: float) -> None:
        self.base = base
        self.w = w
        self.name = f"combo(w={w:g},{base.name})"

    def forecast(self, P_history: np.ndarray) -> np.ndarray:
        mixed = (1.0 - self.w) * P_history[-1] + self.w * self.base.forecast(P_history)
        return mixed / mixed.sum()
