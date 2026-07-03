"""Build weekly meta time-series panels and plots from VGC-Bench teams.

Embeds every team once through the frozen encoder, assigns it to the reference
archetype clusters, bins by ISO week, and writes weight-aware cluster/species
panels plus diagnostic plots (share trajectories, diversity, drift, novelty).

    python scripts/build_meta_timeseries.py --no-download
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402
from rich.console import Console  # noqa: E402

import numpy as np  # noqa: E402

from vgc_team.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from vgc_team.data.tournaments import load_tournament_teams  # noqa: E402
from vgc_team.data.vgc_bench import (  # noqa: E402
    MA_FILES,
    download_vgc_bench_file,
    load_vgc_bench_teams,
)
from vgc_team.meta import timeseries as ts  # noqa: E402
from vgc_team.models.frozen_encoder import (  # noqa: E402
    ReferenceSpace,
    embed_teams,
    load_frozen_encoder,
    select_device,
)

app = typer.Typer(add_completion=False)
console = Console()


def _plot_cluster_trajectories(panel, weeks, out_path: Path, *, top_k: int = 8) -> None:
    totals = panel.groupby("cluster_id")["weight"].sum().sort_values(ascending=False)
    top = list(totals.head(top_k).index)
    fig, ax = plt.subplots(figsize=(11, 6))
    for cluster_id in top:
        sub = panel[panel["cluster_id"] == cluster_id].set_index("week").reindex(weeks)
        ax.plot(weeks, sub["share"].to_numpy(), marker="o", label=f"cluster {cluster_id}")
    ax.set_title(f"Weekly share of top-{top_k} archetype clusters")
    ax.set_xlabel("ISO week")
    ax.set_ylabel("within-week share")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_series(df, x, y, title, ylabel, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x].to_numpy(), df[y].to_numpy(), marker="o")
    ax.set_title(title)
    ax.set_xlabel("ISO week")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@app.command()
def main(
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen checkpoint to embed teams with.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    data_dir: Annotated[
        Path, typer.Option(help="Folder containing VGC-Bench JSON log files.")
    ] = DATA_DIR / "raw" / "vgc_bench",
    reference_npz: Annotated[
        Path, typer.Option(help="team_embeddings.npz with persisted centroids/anchors.")
    ] = DATA_DIR / "processed" / "team_knn" / "team_embeddings.npz",
    include_bo3: Annotated[bool, typer.Option(help="Include the BO3 dataset.")] = True,
    download: Annotated[bool, typer.Option(help="Download missing files.")] = True,
    tournament_teams: Annotated[
        Path | None,
        typer.Option(help="Optional ma_tournament_teams.json to merge (upweighted)."),
    ] = None,
    max_battles: Annotated[
        int | None, typer.Option(help="Optional cap for a fast smoke run.")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Inference batch size.")] = 512,
    output_dir: Annotated[
        Path, typer.Option(help="Where panels and plots are written.")
    ] = DATA_DIR / "processed" / "meta",
) -> None:
    filenames = [MA_FILES["ma"]] + ([MA_FILES["ma_bo3"]] if include_bo3 else [])
    paths: list[Path] = []
    for filename in filenames:
        path = data_dir / filename
        if download and not path.exists():
            path = download_vgc_bench_file(filename, data_dir)
        paths.append(path)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise typer.BadParameter(f"Missing data files: {missing}")

    console.print("Loading reference archetype space...")
    reference = ReferenceSpace.load(reference_npz)

    console.print("Loading teams...")
    teams = load_vgc_bench_teams(paths, max_battles=max_battles)
    teams = [team for team in teams if team.timestamp is not None]
    weights = np.ones(len(teams), dtype=np.float32)
    sources = ["ladder"] * len(teams)

    if tournament_teams is not None:
        t_teams, t_weights = load_tournament_teams(tournament_teams)
        kept = [
            (team, weight)
            for team, weight in zip(t_teams, t_weights, strict=True)
            if team.timestamp is not None
        ]
        t_teams = [team for team, _ in kept]
        t_weights = np.array([weight for _, weight in kept], dtype=np.float32)
        console.print(
            f"Merging {len(t_teams)} tournament teams "
            f"(mean weight {float(t_weights.mean()) if len(t_weights) else 0:.1f}x)."
        )
        teams = teams + t_teams
        weights = np.concatenate([weights, t_weights])
        sources = sources + ["tournament"] * len(t_teams)

    console.print(f"Embedding {len(teams)} teams...")
    device = select_device()
    model, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
    raw = embed_teams(
        model, teams, kind_vocab, token_vocab, batch_size=batch_size, device=device,
        show_progress=True,
    )
    standardized = reference.standardize(raw)
    cluster_ids = reference.assign_clusters(raw)

    frame = ts.build_base_frame(teams, cluster_ids, weights=weights, sources=sources)
    weeks = ts.ordered_weeks(frame)
    console.print(f"{len(weeks)} weeks: {weeks[0]} .. {weeks[-1]}")

    clusters = ts.cluster_panel(frame)
    species = ts.species_panel(frame)
    entropy = ts.weekly_entropy(clusters)
    drift = ts.weekly_centroid_drift(frame, standardized)
    novelty = ts.weekly_novelty(frame, standardized)

    output_dir.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output_dir / "weekly_cluster.csv", index=False)
    species.to_csv(output_dir / "weekly_species.csv", index=False)
    entropy.to_csv(output_dir / "weekly_entropy.csv", index=False)
    drift.to_csv(output_dir / "weekly_centroid_drift.csv", index=False)
    novelty.to_csv(output_dir / "weekly_novelty.csv", index=False)

    _plot_cluster_trajectories(clusters, weeks, output_dir / "cluster_trajectories.png")
    _plot_series(
        entropy, "week", "entropy_bits",
        "Meta diversity (cluster-share entropy)", "bits",
        output_dir / "diversity.png",
    )
    if not drift.empty:
        _plot_series(
            drift, "week", "centroid_drift",
            "Week-to-week meta centroid drift", "L2 distance",
            output_dir / "centroid_drift.png",
        )
    if not novelty.empty:
        _plot_series(
            novelty, "week", "novelty",
            "Weekly novelty vs previous meta", "mean distance to prior centroid",
            output_dir / "novelty.png",
        )

    console.print(f"Wrote panels and plots to {output_dir}")
    console.print("\nWeekly diversity:")
    console.print(entropy.to_string(index=False))


if __name__ == "__main__":
    app()
