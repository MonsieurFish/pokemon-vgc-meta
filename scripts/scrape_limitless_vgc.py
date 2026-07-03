"""Scrape Reg M-A VGC tournaments from Limitless into upweighted teams.

    python scripts/scrape_limitless_vgc.py --dry-run        # list tournaments only
    python scripts/scrape_limitless_vgc.py --source-weight 8

Saves raw standings JSON to data/raw/tournaments/limitless/ and a consolidated
teams file (with upweighting weights) to
data/processed/tournaments/ma_tournament_teams.json, which build_meta_timeseries
and train_mdm can load via --tournament-teams.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vgc_team.config import DATA_DIR
from vgc_team.data import tournaments as tr

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    start_date: Annotated[
        str, typer.Option(help="Regulation window start (ISO). Reg M-A = 2026-04-08.")
    ] = "2026-04-08",
    end_date: Annotated[
        str, typer.Option(help="Regulation window end (ISO). Reg M-A = 2026-06-17.")
    ] = "2026-06-17",
    format_id: Annotated[
        str, typer.Option(help="format_id to stamp on scraped teams.")
    ] = "gen9championsvgc2026regma",
    dry_run: Annotated[bool, typer.Option(help="List tournaments without scraping standings.")] = False,
    source_weight: Annotated[
        float, typer.Option(help="Upweight factor for tournament teams vs ladder.")
    ] = 8.0,
    max_pages: Annotated[int, typer.Option(help="Max API pages to scan.")] = 80,
    max_tournaments: Annotated[
        int, typer.Option(help="Cap to the N largest in-window events (0 = all).")
    ] = 60,
    raw_dir: Annotated[
        Path, typer.Option(help="Where raw standings JSON is written.")
    ] = DATA_DIR / "raw" / "tournaments" / "limitless",
    out_path: Annotated[
        Path, typer.Option(help="Consolidated teams output.")
    ] = DATA_DIR / "processed" / "tournaments" / "ma_tournament_teams.json",
) -> None:
    start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
    console.print(
        f"Listing VGC tournaments in {start_date}..{end_date} "
        "(date window; Limitless 'format' field is unreliable)..."
    )
    tournaments = tr.list_vgc_tournaments(
        start_ts=start_ts, end_ts=end_ts, max_pages=max_pages
    )
    console.print(f"Found {len(tournaments)} in-window tournaments (off-format names excluded).")
    if max_tournaments and len(tournaments) > max_tournaments:
        tournaments = sorted(tournaments, key=lambda t: t.players, reverse=True)[:max_tournaments]
        console.print(f"Keeping the {max_tournaments} largest by player count.")
    for info in tournaments[:20]:
        console.print(f"  {info.date[:10]}  {info.players:>4}p  {info.name}")
    if len(tournaments) > 20:
        console.print(f"  ... and {len(tournaments) - 20} more")

    if dry_run:
        console.print("\n[yellow]--dry-run: not fetching standings.[/yellow]")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    def progress(info: tr.TournamentInfo, n_standings: int) -> None:
        console.print(f"  scraped {n_standings:>3} standings from {info.name}")

    teams, weights, raw = tr.scrape_tournament_teams(
        tournaments, source_weight=source_weight, format_id=format_id, on_progress=progress
    )
    for entry in raw:
        tid = entry["tournament"]["id"]
        (raw_dir / f"{tid}.json").write_text(json.dumps(entry), encoding="utf-8")

    tr.save_teams_json(teams, weights, out_path)
    console.print(
        f"\nSaved {len(teams)} tournament teams (source_weight={source_weight}) to {out_path}"
    )
    console.print(f"Raw standings under {raw_dir}")


if __name__ == "__main__":
    app()
