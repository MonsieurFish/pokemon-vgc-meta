"""Extract labelled team-vs-team matchups from VGC-Bench battle logs.

Each open-team-sheet battle has two full teams and a winner, giving a
(team_A, team_B, A_won) example. We return a flat team list plus index pairs so
the teams can be embedded once through the frozen encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

from vgc_team.teams.schema import Team
from vgc_team.teams.showdown import extract_open_sheet_teams


def extract_matchup_pairs(
    paths: list[Path], *, max_battles: int | None = None
) -> tuple[list[Team], list[tuple[int, int, int]]]:
    """Return (teams, pairs) where pairs are (idx_a, idx_b, label), label=1 if a beat b."""

    teams: list[Team] = []
    pairs: list[tuple[int, int, int]] = []
    n_battles = 0

    for path in paths:
        format_id = path.stem.removeprefix("logs_")
        with path.open(encoding="utf-8") as handle:
            logs = json.load(handle)

        for battle_id, row in logs.items():
            timestamp, log = row
            sides = extract_open_sheet_teams(
                battle_id=battle_id, timestamp=int(timestamp), format_id=format_id, log=log
            )
            full = [team for team in sides if team.is_full_team]
            if len(full) != 2:
                continue
            a, b = full
            if a.won is None or a.won == b.won:  # need a decided winner
                continue

            idx_a, idx_b = len(teams), len(teams) + 1
            teams.extend([a, b])
            pairs.append((idx_a, idx_b, 1 if a.won else 0))

            n_battles += 1
            if max_battles is not None and n_battles >= max_battles:
                return teams, pairs

    return teams, pairs
