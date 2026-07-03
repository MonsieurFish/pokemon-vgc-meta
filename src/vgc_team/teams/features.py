"""Convert parsed teams into six-token transformer inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from vgc_team.teams.schema import PokemonSet, Team
from vgc_team.teams.vocab import KindVocab, TokenVocab

STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
MASKABLE_KINDS = {"item", "ability", "nature", "move"}
MAX_TEAM_SIZE = 6
MAX_ATTR_TOKENS = 15


@dataclass(frozen=True)
class AttributeToken:
    kind: str
    token: str
    maskable: bool = False


@dataclass(frozen=True)
class EncodedTeam:
    kind_ids: torch.Tensor
    value_ids: torch.Tensor
    attr_mask: torch.Tensor
    maskable_positions: tuple[tuple[int, int, str, int], ...]


def _canonical(value: str) -> str:
    return to_id_str(value) if value else "unknown"


def pokemon_attribute_tokens(pokemon: PokemonSet, gen: int = 9) -> list[AttributeToken]:
    """Build species-id-free attribute tokens for one Pokemon set."""

    gen_data = GenData.from_gen(gen)
    pokedex_entry = gen_data.pokedex.get(_canonical(pokemon.species), {})
    base_stats = pokedex_entry.get("baseStats", {})
    types = pokedex_entry.get("types", [])

    tokens: list[AttributeToken] = []
    for type_name in types[:2]:
        tokens.append(AttributeToken("type", f"type:{_canonical(type_name)}"))

    for stat_name in STAT_NAMES:
        stat_value = int(base_stats.get(stat_name, 0))
        tokens.append(AttributeToken("base_stat", f"base_stat:{stat_name}:{stat_value}"))

    tokens.append(AttributeToken("item", f"item:{_canonical(pokemon.item)}", maskable=True))
    tokens.append(
        AttributeToken("ability", f"ability:{_canonical(pokemon.ability)}", maskable=True)
    )
    tokens.append(AttributeToken("nature", f"nature:{_canonical(pokemon.nature)}", maskable=True))

    for move in pokemon.moves[:4]:
        tokens.append(AttributeToken("move", f"move:{_canonical(move)}", maskable=True))

    return tokens[:MAX_ATTR_TOKENS]


def build_vocabs(teams: list[Team], gen: int = 9) -> tuple[KindVocab, TokenVocab]:
    kind_vocab = KindVocab()
    token_vocab = TokenVocab()

    for team in teams:
        for pokemon in team.pokemon[:MAX_TEAM_SIZE]:
            for attr in pokemon_attribute_tokens(pokemon, gen=gen):
                kind_vocab.add(attr.kind)
                token_vocab.add(attr.token)

    return kind_vocab, token_vocab


def encode_team(
    team: Team,
    kind_vocab: KindVocab,
    token_vocab: TokenVocab,
    *,
    gen: int = 9,
) -> EncodedTeam:
    kind_ids = torch.full(
        (MAX_TEAM_SIZE, MAX_ATTR_TOKENS),
        fill_value=kind_vocab.pad_id,
        dtype=torch.long,
    )
    value_ids = torch.full(
        (MAX_TEAM_SIZE, MAX_ATTR_TOKENS),
        fill_value=token_vocab.pad_id,
        dtype=torch.long,
    )
    attr_mask = torch.zeros((MAX_TEAM_SIZE, MAX_ATTR_TOKENS), dtype=torch.bool)
    maskable_positions: list[tuple[int, int, str, int]] = []

    for pokemon_idx, pokemon in enumerate(team.pokemon[:MAX_TEAM_SIZE]):
        attrs = pokemon_attribute_tokens(pokemon, gen=gen)
        for attr_idx, attr in enumerate(attrs):
            kind_ids[pokemon_idx, attr_idx] = kind_vocab.id(attr.kind)
            value_id = token_vocab.id(attr.token)
            value_ids[pokemon_idx, attr_idx] = value_id
            attr_mask[pokemon_idx, attr_idx] = True
            if attr.maskable:
                maskable_positions.append((pokemon_idx, attr_idx, attr.kind, value_id))

    return EncodedTeam(
        kind_ids=kind_ids,
        value_ids=value_ids,
        attr_mask=attr_mask,
        maskable_positions=tuple(maskable_positions),
    )
