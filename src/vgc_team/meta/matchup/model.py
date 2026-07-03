"""Antisymmetric team-vs-team win-probability head on frozen embeddings.

The output logit for "A beats B" is built to be **antisymmetric**:

    logit(a, b) = f([a, b]) - f([b, a])

so P(A beats B) = 1 - P(B beats A) by construction, and P(A beats A) = 0.5.
This removes any side bias and makes the win rate a pure function of the two
team compositions. The linear variant (``interactions=False``) reduces to a
Bradley-Terry / Elo-style scalar strength s(a) - s(b) — a baseline that tells us
whether real rock-paper-scissors *interactions* exist beyond raw team strength.
"""

from __future__ import annotations

import torch
from torch import nn


class MatchupModel(nn.Module):
    def __init__(
        self, d_in: int = 128, hidden: int = 256, dropout: float = 0.2, interactions: bool = True
    ) -> None:
        super().__init__()
        self.interactions = interactions
        if interactions:
            self.net = nn.Sequential(
                nn.Linear(2 * d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.strength = nn.Linear(d_in, 1, bias=False)

    def logit(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """(B, d), (B, d) -> (B,) logit that team a beats team b."""

        if self.interactions:
            forward = self.net(torch.cat([a, b], dim=-1))
            reverse = self.net(torch.cat([b, a], dim=-1))
            return (forward - reverse).squeeze(-1)
        return (self.strength(a) - self.strength(b)).squeeze(-1)

    def win_prob(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(a, b))
