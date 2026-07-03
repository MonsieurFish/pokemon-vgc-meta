"""Load a trained matchup model and score team-vs-team win probabilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vgc_team.meta.matchup.model import MatchupModel


def load_matchup(path: Path, device=None) -> tuple[MatchupModel, np.ndarray, np.ndarray]:
    # our own checkpoint (stores numpy mean/std) — safe to load fully
    blob = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = MatchupModel(blob["d_in"], blob["hidden"], blob["dropout"], interactions=True)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["mean"], blob["std"]


def standardize(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((raw - mean) / std).astype(np.float32)


def win_probability(model: MatchupModel, a_std: np.ndarray, b_std: np.ndarray) -> float:
    """P(team a beats team b) from standardized team embeddings."""

    with torch.no_grad():
        p = model.win_prob(
            torch.tensor(a_std[None], dtype=torch.float32),
            torch.tensor(b_std[None], dtype=torch.float32),
        )
    return float(p[0])


def expected_winrate_vs(
    model: MatchupModel, a_std: np.ndarray, opponents_std: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Expected win rate of team a against a field of opponents (e.g. the meta),
    optionally weighted by each opponent's usage share."""

    with torch.no_grad():
        A = torch.tensor(np.repeat(a_std[None], len(opponents_std), axis=0), dtype=torch.float32)
        B = torch.tensor(opponents_std, dtype=torch.float32)
        p = model.win_prob(A, B).numpy()
    if weights is None:
        return float(p.mean())
    w = np.asarray(weights, dtype=np.float64)
    return float((p * w).sum() / w.sum())
