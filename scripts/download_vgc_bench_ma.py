"""Download VGC-Bench Reg M-A open-team-sheet logs.

The M-A file is small enough for quick local experimentation. The BO3 file is
larger but still manageable on a normal disk. Both contain Showdown logs with
`|showteam|` lines and battle winners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from vgc_team.config import DATA_DIR
from vgc_team.data.vgc_bench import MA_FILES, download_vgc_bench_file

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option(help="Where raw VGC-Bench files should be stored."),
    ] = DATA_DIR / "raw" / "vgc_bench",
    include_bo3: Annotated[
        bool,
        typer.Option(help="Also download the larger Reg M-A BO3 file."),
    ] = False,
) -> None:
    filenames = [MA_FILES["ma"]]
    if include_bo3:
        filenames.append(MA_FILES["ma_bo3"])

    for filename in filenames:
        path = download_vgc_bench_file(filename, output_dir)
        console.print(f"Ready: {path}")


if __name__ == "__main__":
    app()
