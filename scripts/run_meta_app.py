"""Launch the live meta-forecast web app.

    python scripts/run_meta_app.py               # dev mode: auto-reload on edits
    python scripts/run_meta_app.py --no-reload   # production-style, no reloader

In the default (reload) mode:
  - editing the HTML/JS template shows up on a plain browser refresh (no restart);
  - editing backend Python auto-restarts the server (the model re-initialises,
    a few seconds, then the next request is served with your changes).
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console

from vgc_team.app import server

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[
        bool, typer.Option(help="Auto-reload on file edits (dev). --no-reload preloads for fast first paint."),
    ] = True,
) -> None:
    if not reload:
        console.print("Loading frozen encoder + cached meta (first paint will be instant)...")
        server.get_state()
        console.print(f"[bold green]Meta Forecaster at http://{host}:{port} (no auto-reload)[/bold green]")
        server.app.run(host=host, port=port, debug=False)
        return

    # Preload only in the process that actually serves (the reloader's child),
    # so the model isn't loaded twice and the first paint is still fast.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        console.print("Loading frozen encoder + cached meta...")
        server.get_state()
    console.print(
        f"[bold green]Meta Forecaster (auto-reload) at http://{host}:{port}[/bold green]\n"
        "template edits: refresh browser · Python edits: auto-restart"
    )
    server.app.run(
        host=host, port=port, debug=True, use_reloader=True,
        exclude_patterns=["*/site-packages/*", "*/.venv/*"],
    )


if __name__ == "__main__":
    app()
