"""Parsers for Pokemon Showdown open-team-sheet logs."""

from __future__ import annotations

from vgc_team.teams.schema import PokemonSet, Team


def _clean(value: str) -> str:
    return value.strip()


def parse_packed_showteam(packed: str) -> tuple[PokemonSet, ...]:
    """Parse a Showdown `|showteam|...` packed team string.

    The packed format is compact and only semi-documented. For the VGC-Bench
    open-team-sheet logs we need these fields:

    species | nickname | item | ability | moves | nature | evs | gender | ivs
    | shiny | level | ...
    """

    pokemon: list[PokemonSet] = []
    for raw_mon in packed.split("]"):
        raw_mon = raw_mon.strip()
        if not raw_mon:
            continue

        fields = raw_mon.split("|")
        species = _clean(fields[0]) if len(fields) > 0 else ""
        item = _clean(fields[2]) if len(fields) > 2 else ""
        ability = _clean(fields[3]) if len(fields) > 3 else ""
        moves = (
            tuple(_clean(move) for move in fields[4].split(",") if _clean(move))
            if len(fields) > 4
            else ()
        )
        nature = _clean(fields[5]) if len(fields) > 5 else ""
        gender = _clean(fields[7]) or None if len(fields) > 7 else None

        level: int | None = None
        if len(fields) > 10 and fields[10].strip().isdigit():
            level = int(fields[10])

        pokemon.append(
            PokemonSet(
                species=species,
                item=item,
                ability=ability,
                moves=moves,
                nature=nature,
                level=level,
                gender=gender,
            )
        )

    return tuple(pokemon)


def iter_showdown_log_lines(log: str):
    for line in log.splitlines():
        line = line.strip()
        if line:
            yield line


def extract_open_sheet_teams(
    *,
    battle_id: str,
    timestamp: int,
    format_id: str,
    log: str,
) -> list[Team]:
    """Extract p1/p2 teams and outcome labels from one open-team-sheet log."""

    players: dict[str, str] = {}
    showteams: dict[str, tuple[PokemonSet, ...]] = {}
    winner: str | None = None

    for line in iter_showdown_log_lines(log):
        if line.startswith("|showteam|"):
            _, event, side, packed = line.split("|", maxsplit=3)
            showteams[side] = parse_packed_showteam(packed)
            continue

        parts = line.removeprefix("|").split("|")
        if not parts:
            continue

        event = parts[0]
        if event == "player" and len(parts) >= 3:
            players[parts[1]] = parts[2]
        elif event == "win" and len(parts) >= 2:
            winner = parts[1]

    teams: list[Team] = []
    for side, pokemon in showteams.items():
        player = players.get(side)
        teams.append(
            Team(
                pokemon=pokemon,
                format_id=format_id,
                source_battle_id=battle_id,
                side=side,
                player=player,
                timestamp=timestamp,
                won=(player == winner) if winner and player else None,
            )
        )

    return teams
