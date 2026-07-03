"""Fit a balanced anchor codebook + interpretable clusters on the pooled
multi-regulation corpus.

The M-A-only codebook over-represents M-A archetypes, so older regulations (with
restricted legendaries etc.) map poorly. Here we embed every regulation's teams,
**subsample equally per regulation**, and fit KMeans so the codebook spans all
metas. Output is a reference.npz (feature_mean/std + cluster_centroids +
anchor_centroids) loadable by ReferenceSpace, plus cluster_representatives.csv.

    python scripts/build_multi_reg_anchors.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sklearn.cluster import KMeans

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.tournaments import load_tournament_teams
from vgc_team.models.frozen_encoder import (
    embed_teams,
    load_frozen_encoder,
    select_device,
    standardize_embeddings,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    reg_dir: Annotated[
        Path, typer.Option(help="Directory of per-regulation team JSON files.")
    ] = DATA_DIR / "processed" / "tournaments" / "multi_reg",
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    n_clusters: Annotated[int, typer.Option(help="Interpretable clusters.")] = 24,
    n_anchors: Annotated[int, typer.Option(help="Anchor codebook size.")] = 128,
    per_reg_sample: Annotated[
        int, typer.Option(help="Teams sampled per regulation for a balanced KMeans fit.")
    ] = 3500,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    out_dir: Annotated[
        Path, typer.Option(help="Where reference.npz is written.")
    ] = DATA_DIR / "processed" / "multi_reg",
) -> None:
    files = sorted(reg_dir.glob("reg*.json"))
    if not files:
        raise typer.BadParameter(f"No reg*.json in {reg_dir}; run build_multi_regulation_dataset.py.")

    device = select_device()
    model, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)

    all_teams = []
    reg_of = []
    for reg_id, f in enumerate(files):
        teams, _ = load_tournament_teams(f)
        teams = [t for t in teams if t.timestamp is not None]
        all_teams.extend(teams)
        reg_of.extend([reg_id] * len(teams))
        console.print(f"  {f.stem}: {len(teams)} teams")
    reg_of = np.array(reg_of)

    console.print(f"Embedding {len(all_teams)} teams across {len(files)} regulations...")
    raw = embed_teams(model, all_teams, kind_vocab, token_vocab, device=device, show_progress=True)
    standardized, feature_mean, feature_std = standardize_embeddings(raw)

    # balanced subsample: equal teams per regulation for the KMeans fit
    rng = np.random.default_rng(seed)
    pick = []
    for r in np.unique(reg_of):
        idx = np.flatnonzero(reg_of == r)
        if len(idx) > per_reg_sample:
            idx = rng.choice(idx, size=per_reg_sample, replace=False)
        pick.append(idx)
    pick = np.concatenate(pick)
    console.print(f"Fitting KMeans on {len(pick)} balanced teams "
                  f"(~{per_reg_sample}/reg): {n_clusters} clusters, {n_anchors} anchors...")
    fit_X = standardized[pick]

    cluster_km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(fit_X)
    anchor_km = KMeans(n_clusters=n_anchors, random_state=seed, n_init=10).fit(fit_X)

    # representatives: nearest pooled team to each cluster centroid (for readability)
    reps = []
    for c in range(n_clusters):
        d = np.linalg.norm(standardized - cluster_km.cluster_centers_[c], axis=1)
        best = int(d.argmin())
        reps.append({
            "cluster_id": c,
            "species": " | ".join(p.species for p in all_teams[best].pokemon),
            "regulation": files[reg_of[best]].stem,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "reference.npz",
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        cluster_centroids=cluster_km.cluster_centers_.astype(np.float32),
        anchor_centroids=anchor_km.cluster_centers_.astype(np.float32),
    )
    pd.DataFrame(reps).to_csv(out_dir / "cluster_representatives.csv", index=False)
    console.print(f"\nWrote {out_dir / 'reference.npz'} and cluster_representatives.csv")
    console.print("\nCluster representatives:")
    for r in reps:
        console.print(f"  c{r['cluster_id']:>2} [{r['regulation']}] {r['species']}")


if __name__ == "__main__":
    app()
