"""Limitless VGC tournament scraping -> structured teams.

The Limitless API (https://play.limitlesstcg.com/api) is keyless for tournament
listing and standings. Standings already return *structured* decklists, so we
map them straight to ``PokemonSet`` (using the canonical ``id`` field, e.g.
``ninetales-alola``, which matches poke_env's pokedex keys).

Tournament teams are a stronger signal than ladder teams, so each carries
``source="tournament"`` and an upweighting ``weight`` (``source_weight`` times an
optional placement weight). Weights — not row duplication — are how the
upweighting reaches the analytics and the MDM (avoids dedup-collapse / leakage).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from vgc_team.teams.schema import PokemonSet, Team

API_BASE = "https://play.limitlesstcg.com/api"
_HEADERS = {"User-Agent": "vgc-team-research/0.1"}


def _get(path: str, *, params: dict | None = None, retries: int = 6) -> object:
    url = f"{API_BASE}/{path}"
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=_HEADERS, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                # exponential backoff for rate limiting
                time.sleep(min(5.0 * (2**attempt), 60.0))
                continue
            response.raise_for_status()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def _to_timestamp(iso_date: str) -> int:
    dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp())


@dataclass(frozen=True)
class TournamentInfo:
    id: str
    name: str
    date: str
    format: str
    players: int

    @property
    def timestamp(self) -> int:
        return _to_timestamp(self.date)


# Limitless's `format` field is unreliable for VGC (many M-B-named events are
# tagged "M-A"), so we identify a regulation by its legal **date window** plus
# name-based exclusions of off-format events run during that window.
DEFAULT_EXCLUDE = (
    "UU", "M-B", "CUSTOM", "SVI", "BSS",
    "Reg B", "Reg C", "Reg D", "Reg E", "Reg F", "Reg G", "Reg H",
    "Reg I", "Reg J", "Reg K", "Reg L",
)


def list_vgc_tournaments(
    *,
    start_ts: int,
    end_ts: int,
    exclude_keywords: tuple[str, ...] = DEFAULT_EXCLUDE,
    max_pages: int = 80,
    per_page: int = 50,
) -> list[TournamentInfo]:
    """List VGC tournaments within a date window (newest first).

    The window defines the regulation (e.g. Reg M-A = 2026-04-08..2026-06-17).
    Results are date-descending, so paging stops once it passes ``start_ts``.
    """

    found: list[TournamentInfo] = []
    for page in range(1, max_pages + 1):
        rows = _get("tournaments", params={"game": "VGC", "limit": per_page, "page": page})
        if not rows:
            break
        oldest_ts = None
        for row in rows:
            ts = _to_timestamp(row["date"])
            oldest_ts = ts
            if ts > end_ts or ts < start_ts:
                continue
            name = row.get("name", "")
            if any(keyword.lower() in name.lower() for keyword in exclude_keywords):
                continue
            found.append(
                TournamentInfo(
                    id=row["id"],
                    name=name,
                    date=row["date"],
                    format=row.get("format", ""),
                    players=int(row.get("players", 0)),
                )
            )
        if oldest_ts is not None and oldest_ts < start_ts:
            break
        time.sleep(0.3)
    return found


def fetch_all_vgc_rows(*, max_pages: int = 200, per_page: int = 50, on_page=None) -> list[dict]:
    """Page the whole VGC tournament index once (newest first) -> raw row dicts.

    Far cheaper than re-paging per date window; callers bucket locally.
    """

    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        page_rows = _get("tournaments", params={"game": "VGC", "limit": per_page, "page": page})
        if not page_rows:
            break
        rows.extend(page_rows)
        if on_page:
            on_page(page, page_rows[-1].get("date", "")[:10])
        time.sleep(0.6)
    return rows


def rows_to_infos(rows: list[dict]) -> list[TournamentInfo]:
    return [
        TournamentInfo(
            id=r["id"], name=r.get("name", ""), date=r["date"],
            format=r.get("format", ""), players=int(r.get("players", 0)),
        )
        for r in rows
    ]


def get_standings(tournament_id: str) -> list[dict]:
    try:
        standings = _get(f"tournaments/{tournament_id}/standings")
    except requests.RequestException:
        return []  # transient failure on one event shouldn't abort a bulk scrape
    return standings if isinstance(standings, list) else []


def _placement_weight(placing) -> float:
    """Light extra weight for better placements (1.0 if unknown)."""

    if placing is None:
        return 1.0
    try:
        rank = int(placing)
    except (TypeError, ValueError):
        return 1.0
    if rank <= 0:
        return 1.0
    return 1.0 + 1.0 / rank  # 1st -> 2.0, 2nd -> 1.5, ... fades to 1.0


def standing_to_team(
    standing: dict, info: TournamentInfo, *, format_id: str = "gen9championsvgc2026regma"
) -> Team | None:
    decklist = standing.get("decklist")
    if not decklist:
        return None
    pokemon = []
    for entry in decklist:
        species = entry.get("id") or entry.get("name") or ""
        if not species:
            continue
        pokemon.append(
            PokemonSet(
                species=species,
                item=entry.get("item") or "",
                ability=entry.get("ability") or "",
                moves=tuple(entry.get("attacks") or ()),
                nature=entry.get("nature") or "",
            )
        )
    if not pokemon:
        return None
    return Team(
        pokemon=tuple(pokemon),
        format_id=format_id,
        source_battle_id=info.id,
        player=standing.get("player") or standing.get("name"),
        timestamp=info.timestamp,
        won=None,
    )


def scrape_tournament_teams(
    tournaments: list[TournamentInfo],
    *,
    source_weight: float = 8.0,
    use_placement_weight: bool = True,
    format_id: str = "gen9championsvgc2026regma",
    polite_delay: float = 0.5,
    on_progress=None,
) -> tuple[list[Team], list[float], list[dict]]:
    """Fetch standings for each tournament -> (teams, weights, raw_standings)."""

    teams: list[Team] = []
    weights: list[float] = []
    raw: list[dict] = []
    for info in tournaments:
        standings = get_standings(info.id)
        raw.append({"tournament": info.__dict__, "standings": standings})
        for standing in standings:
            team = standing_to_team(standing, info, format_id=format_id)
            if team is None:
                continue
            placement = _placement_weight(standing.get("placing")) if use_placement_weight else 1.0
            teams.append(team)
            weights.append(source_weight * placement)
        if on_progress:
            on_progress(info, len(standings))
        time.sleep(polite_delay)
    return teams, weights, raw


def save_teams_json(teams: list[Team], weights: list[float], path: Path) -> None:
    """Persist tournament teams + weights so analytics/MDM can reload them."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for team, weight in zip(teams, weights, strict=True):
        payload.append(
            {
                "pokemon": [
                    {
                        "species": p.species,
                        "item": p.item,
                        "ability": p.ability,
                        "nature": p.nature,
                        "moves": list(p.moves),
                    }
                    for p in team.pokemon
                ],
                "format_id": team.format_id,
                "source_battle_id": team.source_battle_id,
                "player": team.player,
                "timestamp": team.timestamp,
                "weight": float(weight),
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_tournament_teams(path: Path) -> tuple[list[Team], np.ndarray]:
    """Reload tournament teams + weights saved by ``save_teams_json``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    teams: list[Team] = []
    weights: list[float] = []
    for row in payload:
        pokemon = tuple(
            PokemonSet(
                species=p["species"],
                item=p["item"],
                ability=p["ability"],
                moves=tuple(p["moves"]),
                nature=p["nature"],
            )
            for p in row["pokemon"]
        )
        teams.append(
            Team(
                pokemon=pokemon,
                format_id=row["format_id"],
                source_battle_id=row.get("source_battle_id"),
                player=row.get("player"),
                timestamp=row.get("timestamp"),
                won=None,
            )
        )
        weights.append(float(row["weight"]))
    return teams, np.array(weights, dtype=np.float32)
