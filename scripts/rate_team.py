"""Rate how an unseen team fits each known archetype cluster.

Embeds a user-supplied team through the frozen encoder, places it in the
reference archetype space, and prints a sorted "archetype fingerprint":
how close the team is to each cluster, with that cluster's representative team
for context.

    python scripts/rate_team.py --paste my_team.txt
    python scripts/rate_team.py --showteam "Charizard||CharizarditeY|Blaze|..."

This is idea-2's no-new-data quick win. The matchup-vs-meta rating is a later
phase.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.models.frozen_encoder import (
    ReferenceSpace,
    embed_teams,
    load_frozen_encoder,
    select_device,
)
from vgc_team.teams.pokepaste import parse_pokepaste
from vgc_team.teams.schema import Team
from vgc_team.teams.showdown import parse_packed_showteam

app = typer.Typer(add_completion=False)
console = Console()


def _representatives(npz_dir: Path) -> dict[int, str]:
    path = npz_dir / "cluster_representatives.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(r.cluster_id): str(r.species) for r in df.itertuples()}


@app.command()
def main(
    paste: Annotated[
        Path | None, typer.Option(help="Path to a Showdown-export / pokepaste team file.")
    ] = None,
    showteam: Annotated[
        str | None, typer.Option(help="A packed |showteam| string (without the prefix).")
    ] = None,
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    reference_npz: Annotated[
        Path, typer.Option(help="team_embeddings.npz with persisted centroids.")
    ] = DATA_DIR / "processed" / "team_knn" / "team_embeddings.npz",
    temperature: Annotated[
        float, typer.Option(help="Softmax temperature for the fit score (higher = softer).")
    ] = 2.0,
    top: Annotated[int, typer.Option(help="How many clusters to display.")] = 8,
) -> None:
    if paste is not None:
        pokemon = parse_pokepaste(paste.read_text(encoding="utf-8"))
    elif showteam is not None:
        pokemon = parse_packed_showteam(showteam)
    elif not sys.stdin.isatty():
        pokemon = parse_pokepaste(sys.stdin.read())
    else:
        raise typer.BadParameter("Provide --paste FILE, --showteam STRING, or pipe a paste via stdin.")

    if not pokemon:
        raise typer.BadParameter("Could not parse any Pokemon from the input.")
    team = Team(pokemon=pokemon, format_id="user")
    console.print(f"Parsed team: {' | '.join(p.species for p in team.pokemon)}")

    reference = ReferenceSpace.load(reference_npz)
    device = select_device()
    model, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
    raw = embed_teams(model, [team], kind_vocab, token_vocab, device=device)

    fit = reference.cluster_fit_scores(raw, temperature=temperature)[0]
    standardized = reference.standardize(raw)
    distances = np.linalg.norm(standardized - reference.cluster_centroids, axis=1)
    reps = _representatives(reference_npz.parent)

    order = np.argsort(-fit)[:top]
    console.print("\n[bold]Archetype fingerprint[/bold] (best matches first):")
    console.print(f"  {'cluster':>7}{'fit':>9}{'distance':>10}   representative")
    for cluster_id in order:
        rep = reps.get(int(cluster_id), "")
        console.print(
            f"  {int(cluster_id):>7}{fit[cluster_id]:>9.3f}{distances[cluster_id]:>10.2f}   {rep}"
        )
    best = int(order[0])
    console.print(
        f"\nNearest archetype: cluster {best} (fit {fit[best]:.3f}) — {reps.get(best, '')}"
    )


if __name__ == "__main__":
    app()
