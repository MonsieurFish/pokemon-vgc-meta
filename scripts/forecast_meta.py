"""Production meta-shift forecaster: glide-to-anchor.

This is the forecaster that actually beats last-week persistence out-of-sample
(see scripts/eval_forecasters.py): a Faust-Wright glide that blends last week
toward the regulation's running-mean meta. No neural net is needed to forecast —
the frozen encoder + anchor codebook are used only to turn recent teams into
weekly anchor distributions; the forecast itself is ~one line of NumPy.

    python scripts/forecast_meta.py                       # forecast the most recent regulation
    python scripts/forecast_meta.py --teams-json <file>   # any tournament-teams JSON

Output: the predicted next-week archetype-cluster shares vs the current week,
sorted by absolute change, plus a saved CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.tournaments import load_tournament_teams
from vgc_team.meta import ucsv
from vgc_team.meta.mdm.dataset import build_team_features
from vgc_team.meta.mdm.interpret import anchor_to_cluster_distribution, cluster_shift_table
from vgc_team.models.frozen_encoder import ReferenceSpace, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_MULTI_REG = DATA_DIR / "processed" / "tournaments" / "multi_reg"


@app.command()
def main(
    teams_json: Annotated[
        Path | None,
        typer.Option(help="Tournament-teams JSON to forecast from (default: most recent regulation)."),
    ] = None,
    reference_npz: Annotated[
        Path, typer.Option(help="Pooled anchor codebook + clusters.")
    ] = DATA_DIR / "processed" / "multi_reg" / "reference.npz",
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    lam: Annotated[
        float, typer.Option(help="Glide weight toward the running mean (LORO-validated = 0.3).")
    ] = 0.3,
    top: Annotated[int, typer.Option(help="How many archetype rows to display.")] = 15,
    output_dir: Annotated[Path, typer.Option()] = DATA_DIR / "processed" / "meta",
) -> None:
    if teams_json is None:
        candidates = sorted(DEFAULT_MULTI_REG.glob("reg*.json"))
        if not candidates:
            raise typer.BadParameter(f"No reg*.json in {DEFAULT_MULTI_REG}; pass --teams-json.")
        teams_json = candidates[-1]  # most recent regulation (naming sorts chronologically)
    label = teams_json.stem

    reference = ReferenceSpace.load(reference_npz)
    device = select_device()
    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)

    teams, weights = load_tournament_teams(teams_json)
    console.print(f"Forecasting [bold]{label}[/bold] from {len(teams)} teams...")
    features = build_team_features(
        teams, reference, model_enc, kind_vocab, token_vocab, weights=weights,
        device=device, show_progress=True,
    )
    P = ucsv.weekly_anchor_matrix(features, reference.n_anchors)
    if P.shape[0] < 2:
        console.print("[yellow]Only one week of data — forecast equals the current meta.[/yellow]")

    predicted_anchor = ucsv.Glide(lam).forecast(P)
    predicted = anchor_to_cluster_distribution(predicted_anchor, reference)
    current = anchor_to_cluster_distribution(P[-1], reference)

    reps = {}
    rep_path = reference_npz.parent / "cluster_representatives.csv"
    if rep_path.exists():
        reps = {int(r.cluster_id): str(r.species) for r in pd.read_csv(rep_path).itertuples()}

    shift = cluster_shift_table(current, predicted, representatives=reps)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"forecast_{label}.csv"
    shift.to_csv(out_csv, index=False)

    console.print(
        f"\n[bold]Predicted next-week archetype shifts for {label}[/bold] "
        f"(glide-to-anchor λ={lam}, {P.shape[0]} weeks of history)"
    )
    console.print("(delta = predicted − current share; read delta, not pct_change, on small shares)")
    rising = shift.head(top // 2)
    falling = shift.tail(top - top // 2)
    console.print("\n[green]Rising:[/green]")
    console.print(rising.to_string(index=False))
    console.print("\n[red]Falling:[/red]")
    console.print(falling.to_string(index=False))
    console.print(f"\nSaved full table to {out_csv}")


if __name__ == "__main__":
    app()
