"""Embed VGC teams once through a trained model and build nearest neighbors.

This script is the bridge from "I trained a representation model" to "I can
inspect what the representation thinks is similar."

Pipeline:

1. Load a trained masked-feature transformer checkpoint (frozen).
2. Parse teams from VGC-Bench open-team-sheet logs.
3. Encode each full team with the same vocab used during training.
4. Run every team through ``model.team_embedding(...)`` exactly once.
5. Standardize each embedding dimension across the dataset.
6. Fit an exact k-nearest-neighbor index and write neighbor tables.
7. Cluster the same normalized embeddings and write one representative real
   team per cluster.

It also persists, into ``team_embeddings.npz``:

- ``cluster_centroids``: the ``n_clusters`` interpretable archetype centers, and
- ``anchor_centroids``: a finer codebook (``n_anchors``) used as the categorical
  output vocabulary of the meta-distribution model (MDM).

Both live in the standardized embedding space and let any new team be placed in
the existing archetype space via ``vgc_team.models.frozen_encoder.ReferenceSpace``.

By default the script deduplicates exact teams before KNN. Public ladder data
contains many repeated rental or copied teams, and those exact duplicates tend
to dominate nearest-neighbor results without teaching us much about archetypes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.vgc_bench import MA_FILES, download_vgc_bench_file, load_vgc_bench_teams
from vgc_team.models.frozen_encoder import (
    embed_teams,
    load_frozen_encoder,
    select_device,
    standardize_embeddings,
)
from vgc_team.teams.schema import PokemonSet, Team

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True)
class TeamRecord:
    """One team plus stable inspection metadata."""

    row_index: int
    team_hash: str
    team: Team
    occurrence_count: int


def _pokemon_signature(pokemon: PokemonSet) -> dict[str, object]:
    """Return the order-insensitive identity of one Pokemon set."""

    return {
        "species": pokemon.species,
        "item": pokemon.item,
        "ability": pokemon.ability,
        "nature": pokemon.nature,
        "moves": sorted(pokemon.moves),
    }


def team_signature(team: Team) -> tuple[dict[str, object], ...]:
    """Canonical team signature that ignores team-slot order."""

    return tuple(
        sorted(
            (_pokemon_signature(pokemon) for pokemon in team.pokemon),
            key=lambda pokemon: json.dumps(pokemon, sort_keys=True),
        )
    )


def team_hash(team: Team) -> str:
    """Stable short hash for joining embeddings, metadata, and neighbors."""

    payload = json.dumps(team_signature(team), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _team_records(teams: list[Team], *, dedupe: bool) -> list[TeamRecord]:
    counts: dict[str, int] = {}
    first_seen: dict[str, Team] = {}
    ordered_hashes: list[str] = []

    for team in teams:
        key = team_hash(team)
        counts[key] = counts.get(key, 0) + 1
        if key not in first_seen:
            first_seen[key] = team
            ordered_hashes.append(key)

    if dedupe:
        selected = [(key, first_seen[key], counts[key]) for key in ordered_hashes]
    else:
        running_counts: dict[str, int] = {}
        selected = []
        for team in teams:
            key = team_hash(team)
            running_counts[key] = running_counts.get(key, 0) + 1
            row_key = f"{key}-{running_counts[key]}"
            selected.append((row_key, team, counts[key]))

    return [
        TeamRecord(
            row_index=row_index,
            team_hash=key,
            team=team,
            occurrence_count=occurrence_count,
        )
        for row_index, (key, team, occurrence_count) in enumerate(selected)
    ]


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Scale each team vector to unit length, mostly useful with cosine distance."""

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (embeddings / norms).astype(np.float32)


def _metadata_rows(records: list[TeamRecord]) -> list[dict[str, object]]:
    return [_record_inspection_fields(record) for record in records]


def _record_inspection_fields(record: TeamRecord) -> dict[str, object]:
    team = record.team
    return {
        "row_index": record.row_index,
        "team_hash": record.team_hash,
        "occurrence_count": record.occurrence_count,
        "format_id": team.format_id,
        "source_battle_id": team.source_battle_id,
        "side": team.side,
        "player": team.player,
        "timestamp": team.timestamp,
        "won": team.won,
        "species": " | ".join(pokemon.species for pokemon in team.pokemon),
        "items": " | ".join(pokemon.item for pokemon in team.pokemon),
        "abilities": " | ".join(pokemon.ability for pokemon in team.pokemon),
        "natures": " | ".join(pokemon.nature for pokemon in team.pokemon),
        "moves": " | ".join(
            f"{pokemon.species}: {', '.join(pokemon.moves)}" for pokemon in team.pokemon
        ),
    }


def _neighbor_rows(
    records: list[TeamRecord],
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_index, (query_distances, query_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        neighbor_rank = 1
        for distance, neighbor_index in zip(query_distances, query_indices, strict=True):
            if int(neighbor_index) == query_index:
                continue
            query = records[query_index]
            neighbor = records[int(neighbor_index)]
            rows.append(
                {
                    "query_index": query.row_index,
                    "neighbor_rank": neighbor_rank,
                    "neighbor_index": neighbor.row_index,
                    "distance": float(distance),
                    "query_team_hash": query.team_hash,
                    "neighbor_team_hash": neighbor.team_hash,
                    "query_species": " | ".join(
                        pokemon.species for pokemon in query.team.pokemon
                    ),
                    "neighbor_species": " | ".join(
                        pokemon.species for pokemon in neighbor.team.pokemon
                    ),
                    "neighbor_occurrence_count": neighbor.occurrence_count,
                }
            )
            neighbor_rank += 1
            if neighbor_rank > k:
                break
    return rows


def representative_indices_by_cluster(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | int]]:
    """Find the real team closest to each cluster's mean embedding."""

    representatives: list[dict[str, float | int]] = []
    for cluster_id in sorted(int(cluster_id) for cluster_id in np.unique(labels)):
        member_indices = np.flatnonzero(labels == cluster_id)
        cluster_embeddings = embeddings[member_indices]
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        local_index = int(distances.argmin())
        representatives.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(len(member_indices)),
                "representative_index": int(member_indices[local_index]),
                "centroid_distance": float(distances[local_index]),
            }
        )
    return representatives


def _cluster_assignment_rows(
    records: list[TeamRecord],
    labels: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record, label in zip(records, labels, strict=True):
        rows.append(
            {
                "row_index": record.row_index,
                "team_hash": record.team_hash,
                "cluster_id": int(label),
                "species": " | ".join(pokemon.species for pokemon in record.team.pokemon),
            }
        )
    return rows


def _cluster_representative_rows(
    records: list[TeamRecord],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representative in representative_indices_by_cluster(embeddings, labels):
        record = records[int(representative["representative_index"])]
        rows.append(
            {
                "cluster_id": representative["cluster_id"],
                "cluster_size": representative["cluster_size"],
                "centroid_distance": representative["centroid_distance"],
                **_record_inspection_fields(record),
            }
        )
    return rows


@app.command()
def main(
    checkpoint_path: Annotated[
        Path,
        typer.Option(help="Trained model checkpoint to embed teams with."),
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    data_dir: Annotated[
        Path,
        typer.Option(help="Folder containing VGC-Bench JSON log files."),
    ] = DATA_DIR / "raw" / "vgc_bench",
    include_bo3: Annotated[
        bool,
        typer.Option(help="Include the larger Reg M-A BO3 dataset."),
    ] = False,
    download: Annotated[
        bool,
        typer.Option(help="Download missing VGC-Bench files before embedding."),
    ] = True,
    max_battles: Annotated[
        int | None,
        typer.Option(help="Optional cap for a fast smoke run."),
    ] = None,
    max_teams: Annotated[
        int | None,
        typer.Option(help="Optional cap after parsing/deduplication."),
    ] = None,
    dedupe: Annotated[
        bool,
        typer.Option(help="Collapse exact duplicate teams before nearest-neighbor search."),
    ] = True,
    k: Annotated[int, typer.Option(help="Number of neighbors to keep per team.")] = 10,
    metric: Annotated[
        str,
        typer.Option(help="NearestNeighbors metric. Good defaults: cosine or euclidean."),
    ] = "cosine",
    n_clusters: Annotated[
        int,
        typer.Option(help="Number of interpretable archetype clusters."),
    ] = 24,
    n_anchors: Annotated[
        int,
        typer.Option(help="Size of the finer anchor codebook (MDM output vocabulary)."),
    ] = 128,
    cluster_seed: Annotated[int, typer.Option(help="Random seed for KMeans clustering.")] = 7,
    l2_normalize: Annotated[
        bool,
        typer.Option(help="L2-normalize each standardized team vector before KNN."),
    ] = False,
    batch_size: Annotated[int, typer.Option(help="Inference batch size.")] = 512,
    n_jobs: Annotated[int, typer.Option(help="CPU workers used by scikit-learn KNN.")] = -1,
    output_dir: Annotated[
        Path,
        typer.Option(help="Where embeddings, metadata, and neighbors should be written."),
    ] = DATA_DIR / "processed" / "team_knn",
) -> None:
    if k < 1:
        raise typer.BadParameter("k must be at least 1.")
    if n_clusters < 1:
        raise typer.BadParameter("n_clusters must be at least 1.")

    filenames = [MA_FILES["ma"]]
    if include_bo3:
        filenames.append(MA_FILES["ma_bo3"])

    paths: list[Path] = []
    for filename in filenames:
        path = data_dir / filename
        if download and not path.exists():
            path = download_vgc_bench_file(filename, data_dir)
        paths.append(path)

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise typer.BadParameter(f"Missing data files: {missing}")
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Missing checkpoint: {checkpoint_path}")

    console.print("Loading teams...")
    teams = load_vgc_bench_teams(paths, max_battles=max_battles)
    console.print(f"Parsed {len(teams)} full team-side examples.")

    records = _team_records(teams, dedupe=dedupe)
    if max_teams is not None:
        records = records[:max_teams]
    if len(records) <= k:
        raise typer.BadParameter(f"Need more than k={k} teams; got {len(records)}.")
    if len(records) < n_clusters:
        raise typer.BadParameter(
            f"Need at least n_clusters={n_clusters} teams; got {len(records)}."
        )
    effective_anchors = min(n_anchors, len(records))
    console.print(f"Using {len(records)} teams for embedding/KNN.")

    device = select_device()
    console.print(f"Loading frozen checkpoint on {device}...")
    model, kind_vocab, token_vocab, config = load_frozen_encoder(checkpoint_path, device)
    console.print(
        "Checkpoint config: "
        f"d_model={config.get('d_model', 128)}, "
        f"pokemon_layers={config.get('pokemon_layers', 2)}, "
        f"team_layers={config.get('team_layers', 2)}"
    )

    raw_embeddings = embed_teams(
        model,
        [record.team for record in records],
        kind_vocab,
        token_vocab,
        batch_size=batch_size,
        device=device,
        show_progress=True,
    )
    normalized_embeddings, feature_mean, feature_std = standardize_embeddings(raw_embeddings)
    if l2_normalize:
        normalized_embeddings = l2_normalize_embeddings(normalized_embeddings)

    console.print(f"Fitting exact {k}-nearest-neighbor index with metric={metric!r}...")
    neighbor_model = NearestNeighbors(
        n_neighbors=min(k + 1, len(records)),
        metric=metric,
        n_jobs=n_jobs,
    )
    neighbor_model.fit(normalized_embeddings)
    distances, indices = neighbor_model.kneighbors(normalized_embeddings)

    console.print(f"Clustering into {n_clusters} interpretable clusters...")
    cluster_model = KMeans(n_clusters=n_clusters, random_state=cluster_seed, n_init=10)
    labels = cluster_model.fit_predict(normalized_embeddings)

    console.print(f"Fitting anchor codebook with {effective_anchors} anchors...")
    anchor_model = KMeans(n_clusters=effective_anchors, random_state=cluster_seed, n_init=10)
    anchor_model.fit(normalized_embeddings)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(_metadata_rows(records))
    neighbors = pd.DataFrame(_neighbor_rows(records, distances, indices, k=k))
    assignments = pd.DataFrame(_cluster_assignment_rows(records, labels))
    representatives = pd.DataFrame(
        _cluster_representative_rows(records, normalized_embeddings, labels)
    )
    metadata.to_csv(output_dir / "team_metadata.csv", index=False)
    neighbors.to_csv(output_dir / "team_neighbors.csv", index=False)
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    representatives.to_csv(output_dir / "cluster_representatives.csv", index=False)
    np.savez_compressed(
        output_dir / "team_embeddings.npz",
        raw_embeddings=raw_embeddings,
        normalized_embeddings=normalized_embeddings,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        team_hashes=np.array([record.team_hash for record in records]),
        row_indices=np.array([record.row_index for record in records], dtype=np.int64),
        cluster_labels=labels.astype(np.int64),
        cluster_centroids=cluster_model.cluster_centers_.astype(np.float32),
        anchor_centroids=anchor_model.cluster_centers_.astype(np.float32),
    )

    console.print(f"Wrote metadata to {output_dir / 'team_metadata.csv'}")
    console.print(f"Wrote neighbors to {output_dir / 'team_neighbors.csv'}")
    console.print(f"Wrote cluster assignments to {output_dir / 'cluster_assignments.csv'}")
    console.print(f"Wrote cluster representatives to {output_dir / 'cluster_representatives.csv'}")
    console.print(f"Wrote embeddings to {output_dir / 'team_embeddings.npz'}")
    console.print("\nRepresentative teams:")
    console.print(
        representatives[
            [
                "cluster_id",
                "cluster_size",
                "centroid_distance",
                "species",
                "items",
                "abilities",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    app()
