"""Hierarchical transformer for masked VGC team modeling."""

from __future__ import annotations

import torch
from torch import nn


class PokemonSetTransformer(nn.Module):
    """Encode one Pokemon set from unordered attribute tokens."""

    def __init__(
        self,
        *,
        n_kinds: int,
        n_values: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(n_kinds, d_model)
        self.value_embedding = nn.Embedding(n_values, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        kind_ids: torch.Tensor,
        value_ids: torch.Tensor,
        attr_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = kind_ids.shape[0]
        attr_tokens = self.kind_embedding(kind_ids) + self.value_embedding(value_ids)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, attr_tokens], dim=1)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=attr_mask.device)
        keep_mask = torch.cat([cls_mask, attr_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~keep_mask)
        return self.norm(encoded[:, 0])


class HierarchicalTeamTransformer(nn.Module):
    """Shared Pokemon-set transformer followed by positionless team transformer."""

    def __init__(
        self,
        *,
        n_kinds: int,
        n_values: int,
        d_model: int = 128,
        n_heads: int = 4,
        pokemon_layers: int = 2,
        team_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pokemon_encoder = PokemonSetTransformer(
            n_kinds=n_kinds,
            n_values=n_values,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=pokemon_layers,
            dropout=dropout,
        )
        self.team_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        team_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.team_encoder = nn.TransformerEncoder(team_layer, num_layers=team_layers)
        self.kind_embedding = nn.Embedding(n_kinds, d_model)
        self.output = nn.Linear(d_model, n_values)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        kind_ids: torch.Tensor,
        value_ids: torch.Tensor,
        attr_mask: torch.Tensor,
        target_pokemon_idx: torch.Tensor,
        target_kind_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, team_size, n_attrs = kind_ids.shape
        flat_embeddings = self.pokemon_encoder(
            kind_ids.reshape(batch_size * team_size, n_attrs),
            value_ids.reshape(batch_size * team_size, n_attrs),
            attr_mask.reshape(batch_size * team_size, n_attrs),
        )
        pokemon_tokens = flat_embeddings.reshape(batch_size, team_size, -1)

        team_cls = self.team_cls.expand(batch_size, -1, -1)
        team_tokens = torch.cat([team_cls, pokemon_tokens], dim=1)
        team_keep_mask = torch.ones(
            (batch_size, team_size + 1),
            dtype=torch.bool,
            device=kind_ids.device,
        )
        team_encoded = self.team_encoder(team_tokens, src_key_padding_mask=~team_keep_mask)
        contextual_pokemon = team_encoded[:, 1:]

        batch_index = torch.arange(batch_size, device=kind_ids.device)
        target_pokemon = contextual_pokemon[batch_index, target_pokemon_idx]
        target_kind = self.kind_embedding(target_kind_id)
        return self.output(self.norm(target_pokemon + target_kind))

    def team_embedding(
        self,
        kind_ids: torch.Tensor,
        value_ids: torch.Tensor,
        attr_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, team_size, n_attrs = kind_ids.shape
        flat_embeddings = self.pokemon_encoder(
            kind_ids.reshape(batch_size * team_size, n_attrs),
            value_ids.reshape(batch_size * team_size, n_attrs),
            attr_mask.reshape(batch_size * team_size, n_attrs),
        )
        pokemon_tokens = flat_embeddings.reshape(batch_size, team_size, -1)
        team_cls = self.team_cls.expand(batch_size, -1, -1)
        team_tokens = torch.cat([team_cls, pokemon_tokens], dim=1)
        team_encoded = self.team_encoder(team_tokens)
        return self.norm(team_encoded[:, 0])
