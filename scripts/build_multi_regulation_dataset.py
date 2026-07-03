"""Construct a multi-regulation tournament dataset from the Limitless API.

Past-regulation full teams only exist in tournament lists (old ladders were not
open-team-sheet — see project notes), so we scrape Limitless one date window per
regulation. The **date window is the regulation label** (the API `format` field
is unreliable); off-format and throwback-to-another-reg events are excluded by
name. Each regulation's teams are tagged via `format_id` and saved to its own
JSON (loadable with `tournaments.load_tournament_teams`), plus a summary.

    python scripts/build_multi_regulation_dataset.py --dry-run
    python scripts/build_multi_regulation_dataset.py --max-tournaments 60
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vgc_team.config import DATA_DIR
from vgc_team.data import tournaments as tr

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class Regulation:
    key: str          # short id, also the output filename stem
    format_id: str    # stamped onto every team (the regulation label)
    token: str        # this reg's name token (kept; other regs excluded)
    start: str        # ISO date (inclusive)
    end: str          # ISO date (inclusive)


# Non-overlapping windows by the primary ranked regulation, newest first.
# "as far back as reasonable": Limitless VGC volume is solid from ~late 2023;
# earlier windows are attempted but tend to be sparse.
REGULATIONS: list[Regulation] = [
    Regulation("regmb26", "gen9championsvgc2026regmb", "M-B", "2026-06-18", "2026-07-15"),
    Regulation("regma26", "gen9championsvgc2026regma", "M-A", "2026-04-08", "2026-06-17"),
    Regulation("regj25", "gen9vgc2025regj", "Reg J", "2025-09-01", "2026-01-04"),
    Regulation("regi25", "gen9vgc2025regi", "Reg I", "2025-05-01", "2025-08-31"),
    Regulation("regg25", "gen9vgc2025regg", "Reg G", "2025-01-06", "2025-04-30"),
    Regulation("regh24", "gen9vgc2024regh", "Reg H", "2024-09-01", "2025-01-05"),
    Regulation("regg24", "gen9vgc2024regg", "Reg G", "2024-05-01", "2024-08-31"),
    Regulation("regf24", "gen9vgc2024regf", "Reg F", "2024-01-04", "2024-04-30"),
    Regulation("rege23", "gen9vgc2023rege", "Reg E", "2023-10-01", "2024-01-03"),
    Regulation("regd23", "gen9vgc2023regd", "Reg D", "2023-07-01", "2023-09-30"),
]

BASE_OFF_FORMAT = (
    "UU", "CUSTOM", "SVI", "BSS", "2v2", "Little Cup", "Metronome", "Gen 8", "Gen8", "LGPE",
)
ALL_REG_TOKENS = ["M-A", "M-B"] + [f"Reg {c}" for c in "ABCDEFGHIJKL"] + [
    f"Regulation {c}" for c in "ABCDEFGHIJKL"
]


def _exclude_for(reg: Regulation) -> tuple[str, ...]:
    """Off-format markers + every other regulation's name tokens."""

    own = {reg.token, reg.token.replace("Reg ", "Regulation ")}
    others = [tok for tok in ALL_REG_TOKENS if tok not in own]
    return tuple(BASE_OFF_FORMAT) + tuple(others)


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())


def _in_window_events(rows: list[dict], reg: Regulation) -> list[tr.TournamentInfo]:
    start, end = _ts(reg.start), _ts(reg.end)
    exclude = _exclude_for(reg)
    kept = []
    for r in rows:
        ts = int(datetime.fromisoformat(r["date"].replace("Z", "+00:00")).timestamp())
        if ts < start or ts > end:
            continue
        name = r.get("name", "")
        if any(k.lower() in name.lower() for k in exclude):
            continue
        kept.append(r)
    return tr.rows_to_infos(kept)


@app.command()
def main(
    dry_run: Annotated[bool, typer.Option(help="List tournament counts per reg without scraping.")] = False,
    max_tournaments: Annotated[
        int, typer.Option(help="Per regulation, keep the N largest events (0 = all).")
    ] = 60,
    source_weight: Annotated[float, typer.Option(help="Tournament upweight factor.")] = 8.0,
    max_pages: Annotated[int, typer.Option(help="Max API pages to page through once.")] = 200,
    out_dir: Annotated[
        Path, typer.Option(help="Where per-regulation team files are written.")
    ] = DATA_DIR / "processed" / "tournaments" / "multi_reg",
    raw_dir: Annotated[
        Path, typer.Option(help="Where raw standings JSON is written.")
    ] = DATA_DIR / "raw" / "tournaments" / "limitless",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    console.print("Fetching the full VGC tournament index once...")
    all_rows = tr.fetch_all_vgc_rows(
        max_pages=max_pages,
        on_page=lambda p, d: console.print(f"  page {p}: back to {d}", highlight=False)
        if p % 20 == 0 else None,
    )
    console.print(f"Indexed {len(all_rows)} VGC tournaments "
                  f"({all_rows[-1]['date'][:10]} .. {all_rows[0]['date'][:10]}).")

    for reg in REGULATIONS:
        tournaments = _in_window_events(all_rows, reg)
        raw_in_window = sum(
            1 for r in all_rows
            if _ts(reg.start) <= int(datetime.fromisoformat(r["date"].replace("Z", "+00:00")).timestamp()) <= _ts(reg.end)
        )
        console.print(
            f"\n[bold]{reg.key}[/bold] ({reg.format_id}) {reg.start}..{reg.end}: "
            f"{raw_in_window} in window, {len(tournaments)} after off-format/other-reg filter"
        )
        if max_tournaments and len(tournaments) > max_tournaments:
            tournaments = sorted(tournaments, key=lambda t: t.players, reverse=True)[:max_tournaments]

        if dry_run:
            players = sorted((t.players for t in tournaments), reverse=True)
            summary.append({"reg": reg.key, "events": len(tournaments), "raw_in_window": raw_in_window,
                            "top_players": players[:3]})
            continue

        out_path = out_dir / f"{reg.key}.json"
        if out_path.exists():
            existing, _ = tr.load_tournament_teams(out_path)
            console.print(f"  already scraped ({len(existing)} teams) -> skipping")
            summary.append({"reg": reg.key, "events": len(tournaments), "teams": len(existing),
                            "skipped": True})
            continue

        if not tournaments:
            summary.append({"reg": reg.key, "events": 0, "teams": 0})
            continue

        try:
            teams, weights, raw = tr.scrape_tournament_teams(
                tournaments, source_weight=source_weight, format_id=reg.format_id,
            )
        except Exception as exc:  # noqa: BLE001 - one reg failing shouldn't kill the rest
            console.print(f"  [red]failed: {exc}[/red]")
            summary.append({"reg": reg.key, "events": len(tournaments), "error": str(exc)[:120]})
            continue

        for entry in raw:
            tid = entry["tournament"]["id"]
            (raw_dir / f"{tid}.json").write_text(json.dumps(entry), encoding="utf-8")
        tr.save_teams_json(teams, weights, out_path)
        span = (
            f"{min(t.timestamp for t in teams)}..{max(t.timestamp for t in teams)}" if teams else "-"
        )
        console.print(f"  saved {len(teams)} teams -> {out_path}")
        summary.append({"reg": reg.key, "events": len(tournaments), "teams": len(teams), "ts_span": span})

    console.print("\n[bold]Summary[/bold]")
    for row in summary:
        console.print(f"  {row}")
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"\nManifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    app()
