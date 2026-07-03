"""VGC-Bench log loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import requests
from tqdm import tqdm

from vgc_team.teams.schema import Team
from vgc_team.teams.showdown import extract_open_sheet_teams

VGC_BENCH_BASE_URL = "https://huggingface.co/datasets/cameronangliss/vgc-battle-logs/resolve/main"
MA_FILES = {
    "ma": "logs_gen9championsvgc2026regma.json",
    "ma_bo3": "logs_gen9championsvgc2026regmabo3.json",
}


def download_vgc_bench_file(filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if output_path.exists():
        return output_path

    url = f"{VGC_BENCH_BASE_URL}/{filename}"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output_path.open("wb") as handle:
            progress = tqdm(total=total, unit="B", unit_scale=True, desc=filename)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
            progress.close()
    return output_path


def load_vgc_bench_teams(
    paths: list[Path],
    *,
    max_battles: int | None = None,
    full_teams_only: bool = True,
) -> list[Team]:
    teams: list[Team] = []
    n_battles = 0

    for path in paths:
        format_id = path.stem.removeprefix("logs_")
        with path.open(encoding="utf-8") as handle:
            logs = json.load(handle)

        for battle_id, row in logs.items():
            timestamp, log = row
            for team in extract_open_sheet_teams(
                battle_id=battle_id,
                timestamp=int(timestamp),
                format_id=format_id,
                log=log,
            ):
                if not full_teams_only or team.is_full_team:
                    teams.append(team)
            n_battles += 1
            if max_battles is not None and n_battles >= max_battles:
                return teams

    return teams
