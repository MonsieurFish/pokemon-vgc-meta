"""Run the falsifiable meta-pattern tests + persistence baseline on the panels.

    python scripts/test_meta_patterns.py

These are the baselines the MDM must beat. Run build_meta_timeseries.py first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR
from vgc_team.meta import patterns

app = typer.Typer(add_completion=False)
console = Console()


def _run_unit(name: str, panel: pd.DataFrame, unit_col: str, value_col: str) -> None:
    console.print(f"\n[bold]=== {name} (unit={unit_col}, value={value_col}) ===[/bold]")
    if panel.empty:
        console.print("  (empty panel)")
        return

    baseline = patterns.persistence_baseline(panel, unit_col=unit_col, value_col=value_col)
    console.print(
        f"  persistence MAE={baseline['persistence_mae']:.5f}  "
        f"climatology MAE={baseline['climatology_mae']:.5f}  "
        f"(n_transitions={baseline['n_transitions']})"
    )

    console.print("\n  [Momentum]  delta_t ~ delta_{t-1}  (unit fixed effects)")
    console.print(
        patterns.momentum_test(panel, unit_col=unit_col, value_col=value_col).table()
    )

    console.print("\n  [Winners-lead-usage]  value_t ~ winrate_{t-1} + value_{t-1}")
    console.print(
        patterns.winners_lead_usage_test(panel, unit_col=unit_col, value_col=value_col).table()
    )


@app.command()
def main(
    meta_dir: Annotated[
        Path, typer.Option(help="Directory with weekly_cluster.csv / weekly_species.csv.")
    ] = DATA_DIR / "processed" / "meta",
    min_species_weight: Annotated[
        float,
        typer.Option(help="Drop species whose total weight is below this (reduce noise)."),
    ] = 200.0,
) -> None:
    cluster_path = meta_dir / "weekly_cluster.csv"
    species_path = meta_dir / "weekly_species.csv"
    if not cluster_path.exists():
        raise typer.BadParameter(f"Missing {cluster_path}; run build_meta_timeseries.py first.")

    clusters = pd.read_csv(cluster_path)
    _run_unit("Archetype clusters", clusters, "cluster_id", "share")

    if species_path.exists():
        species = pd.read_csv(species_path)
        totals = species.groupby("species")["weight"].sum()
        keep = totals[totals >= min_species_weight].index
        species = species[species["species"].isin(keep)]
        _run_unit(
            f"Species (>= {min_species_weight:g} total weight)", species, "species", "usage"
        )


if __name__ == "__main__":
    app()
