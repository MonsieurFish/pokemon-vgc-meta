"""Parse Showdown-export / pokepaste text into structured teams.

This is the human-facing counterpart to ``parse_packed_showteam`` in
``showdown.py``: tournament team lists (VGCPastes, Limitless decklists) and
user-supplied teams come as Showdown export blocks like::

    Charizard @ Charizardite Y
    Ability: Blaze
    Level: 50
    Tera Type: Fire
    EVs: 4 HP / 252 SpA / 252 Spe
    Timid Nature
    - Heat Wave
    - Protect
    - Weather Ball
    - Solar Beam

We keep species/item/ability/move *display* names verbatim; downstream
``features.encode_team`` canonicalizes them with ``to_id_str``. EVs/IVs/Tera/
Shiny lines are intentionally ignored (EVs are not in the schema; see project
notes), matching the attributes the frozen encoder was trained on.
"""

from __future__ import annotations

import re

from vgc_team.teams.schema import PokemonSet

_GENDER_RE = re.compile(r"\((M|F|N)\)\s*$")
_PAREN_SPECIES_RE = re.compile(r"^(?P<nick>.*?)\((?P<species>[^()]+)\)\s*$")


def _parse_header(line: str) -> tuple[str, str, str | None]:
    """Return (species, item, gender) from a Pokemon block's first line."""

    item = ""
    if "@" in line:
        left, item = line.split("@", 1)
        item = item.strip()
    else:
        left = line
    left = left.strip()

    gender: str | None = None
    gender_match = _GENDER_RE.search(left)
    if gender_match:
        gender = gender_match.group(1)
        left = left[: gender_match.start()].strip()

    species_match = _PAREN_SPECIES_RE.match(left)
    if species_match:
        species = species_match.group("species").strip()
    else:
        species = left.strip()

    return species, item, gender


def parse_pokepaste(text: str) -> tuple[PokemonSet, ...]:
    """Parse one Showdown-export team into a tuple of ``PokemonSet``.

    Blocks are separated by blank lines. Robust to missing item, nickname,
    gender, level, and nature fields.
    """

    pokemon: list[PokemonSet] = []
    for raw_block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.rstrip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue

        species, item, gender = _parse_header(lines[0])
        if not species:
            continue

        ability = ""
        nature = ""
        level: int | None = None
        moves: list[str] = []

        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("-\t"):
                moves.append(stripped[1:].strip())
            elif stripped == "-":
                continue
            elif stripped.lower().startswith("ability:"):
                ability = stripped.split(":", 1)[1].strip()
            elif stripped.lower().startswith("level:"):
                value = stripped.split(":", 1)[1].strip()
                if value.isdigit():
                    level = int(value)
            elif stripped.lower().endswith("nature"):
                nature = stripped[: -len("nature")].strip()
            # EVs / IVs / Tera Type / Shiny / Happiness / etc. are ignored.

        pokemon.append(
            PokemonSet(
                species=species,
                item=item,
                ability=ability,
                moves=tuple(moves[:4]),
                nature=nature,
                level=level,
                gender=gender,
            )
        )

    return tuple(pokemon)
