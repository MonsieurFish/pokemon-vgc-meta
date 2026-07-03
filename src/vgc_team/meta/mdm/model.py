"""The Meta Distribution Model: time-aware set encoder + anchor head.

Reads a variable-size set of recent team embeddings, each tagged with its
*weeks-ago* offset, and predicts a distribution over the anchor codebook for the
week immediately after the most recent context (horizon-1 forecast).

The set encoder is **attention pooling**: a learned query attends over the
context teams (whose keys carry recency via the time encoding), so recent teams
can be weighted more and the pooled "meta embedding" summarizes the recent meta.
This is the same set-attention idea as the existing ``team_encoder``, one level
up (teams -> week). Time is encoded *relative to the queried week*, so any
context window size is in-distribution.
"""

from __future__ import annotations

import torch
from torch import nn


class MetaDistributionModel(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        n_anchors: int,
        d_model: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(d_in, d_model)
        self.time_enc = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.query = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_anchors),
        )
        # Zero-init the final layer so the initial correction is 0 -> the model
        # starts *at* the persistence baseline and only learns the drift on top.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model**0.5

    def forward(
        self,
        context_emb: torch.Tensor,  # (B, L, d_in)
        context_time: torch.Tensor,  # (B, L) weeks-ago offsets (<= 0)
        context_pad_mask: torch.Tensor,  # (B, L) True = pad
        baseline_logprob: torch.Tensor,  # (B, n_anchors) log persistence distribution
    ) -> torch.Tensor:
        tokens = self.token_proj(context_emb) + self.time_enc(context_time.unsqueeze(-1))
        tokens = self.dropout(tokens)
        keys = self.key(tokens)
        values = self.value(tokens)

        query = self.query(self.query_token).expand(context_emb.shape[0], -1)  # (B, d_model)

        scores = (keys @ query.unsqueeze(-1)).squeeze(-1) / self.scale  # (B, L)
        scores = scores.masked_fill(context_pad_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # (B, L)
        summary = (attn.unsqueeze(-1) * values).sum(dim=1)  # (B, d_model)

        # residual on persistence: logits = log(persistence) + learned correction
        return baseline_logprob + self.head(summary)
