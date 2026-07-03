"""Structured team objects.

Species names are kept as metadata so we can inspect and debug teams, but the
first model does not feed a species-id token directly. It uses battle-relevant
attributes such as base stats, types, item, ability, nature, and moves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PokemonSet:
    species: str
    item: str
    ability: str
    moves: tuple[str, ...]
    nature: str
    level: int | None = None
    gender: str | None = None


@dataclass(frozen=True)
class Team:
    pokemon: tuple[PokemonSet, ...]
    format_id: str
    source_battle_id: str | None = None
    side: str | None = None
    player: str | None = None
    timestamp: int | None = None
    won: bool | None = None

    @property
    def is_full_team(self) -> bool:
        return len(self.pokemon) == 6
