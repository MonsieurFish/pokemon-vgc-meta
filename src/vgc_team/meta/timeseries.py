"""Weekly meta time-series panels from embedded teams.

Pure pandas/numpy logic so it is unit-testable without torch: the heavy lifting
(parsing, embedding, cluster assignment) happens in ``scripts/build_meta_timeseries.py``,
which hands this module a tidy per-team frame.

A "panel" is a tidy table with one row per (week, unit), where a unit is either
an archetype cluster or a species. Every aggregation is **weight-aware**: ladder
teams have weight 1 and tournament teams carry ``source_weight`` (see Phase 4),
so shares and win-rates are weighted sums, never raw counts.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from vgc_team.teams.schema import Team


def iso_week_label(timestamp: int) -> str:
    """Unix timestamp -> sortable ISO year-week label, e.g. '2026-W15'."""

    iso = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def build_base_frame(
    teams: list[Team],
    cluster_ids: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """One row per team: week, cluster, outcome, weight, species list."""

    if len(teams) != len(cluster_ids):
        raise ValueError("teams and cluster_ids must align")
    if weights is None:
        weights = np.ones(len(teams), dtype=np.float64)
    if sources is None:
        sources = ["ladder"] * len(teams)

    rows = []
    for team, cluster_id, weight, source in zip(
        teams, cluster_ids, weights, sources, strict=True
    ):
        if team.timestamp is None:
            continue
        rows.append(
            {
                "week": iso_week_label(team.timestamp),
                "timestamp": int(team.timestamp),
                "cluster_id": int(cluster_id),
                "won": team.won,
                "weight": float(weight),
                "source": source,
                "species": tuple(pokemon.species for pokemon in team.pokemon),
            }
        )
    return pd.DataFrame(rows)


def _weighted_win_rate(group: pd.DataFrame) -> float:
    labeled = group[group["won"].notna()]
    total = labeled["weight"].sum()
    if total <= 0:
        return float("nan")
    wins = (labeled["won"].astype(float) * labeled["weight"]).sum()
    return float(wins / total)


def cluster_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (week, cluster): team count, weight, within-week share, win-rate."""

    if frame.empty:
        return pd.DataFrame(
            columns=["week", "cluster_id", "n_teams", "weight", "share", "win_rate"]
        )

    week_weight = frame.groupby("week")["weight"].sum().rename("week_weight")
    records = []
    for (week, cluster_id), group in frame.groupby(["week", "cluster_id"]):
        records.append(
            {
                "week": week,
                "cluster_id": int(cluster_id),
                "n_teams": int(len(group)),
                "weight": float(group["weight"].sum()),
                "win_rate": _weighted_win_rate(group),
            }
        )
    panel = pd.DataFrame(records)
    panel = panel.merge(week_weight, on="week")
    panel["share"] = panel["weight"] / panel["week_weight"]
    panel = panel.drop(columns="week_weight")
    return panel.sort_values(["week", "cluster_id"]).reset_index(drop=True)


def species_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (week, species): usage share (teams containing it) and win-rate."""

    if frame.empty:
        return pd.DataFrame(
            columns=["week", "species", "n_teams", "weight", "usage", "win_rate"]
        )

    exploded = frame.copy()
    # de-duplicate species within a team so a team is counted once per species
    exploded["species"] = exploded["species"].apply(lambda names: tuple(set(names)))
    exploded = exploded.explode("species").dropna(subset=["species"])

    week_weight = frame.groupby("week")["weight"].sum().rename("week_weight")
    records = []
    for (week, species), group in exploded.groupby(["week", "species"]):
        records.append(
            {
                "week": week,
                "species": species,
                "n_teams": int(len(group)),
                "weight": float(group["weight"].sum()),
                "win_rate": _weighted_win_rate(group),
            }
        )
    panel = pd.DataFrame(records)
    panel = panel.merge(week_weight, on="week")
    panel["usage"] = panel["weight"] / panel["week_weight"]
    panel = panel.drop(columns="week_weight")
    return panel.sort_values(["week", "weight"], ascending=[True, False]).reset_index(drop=True)


def ordered_weeks(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["week"].unique().tolist())


def weekly_entropy(panel: pd.DataFrame, *, unit: str = "cluster_id") -> pd.DataFrame:
    """Shannon entropy (bits) of the share distribution per week — meta diversity."""

    records = []
    for week, group in panel.groupby("week"):
        shares = group["share"] if "share" in group else group["usage"]
        shares = shares[shares > 0].to_numpy()
        shares = shares / shares.sum()
        entropy = float(-(shares * np.log2(shares)).sum())
        records.append({"week": week, "entropy_bits": entropy, "n_units": int(len(group))})
    return pd.DataFrame(records).sort_values("week").reset_index(drop=True)


def weekly_centroid_drift(
    frame: pd.DataFrame, embeddings: np.ndarray
) -> pd.DataFrame:
    """Distance between consecutive weeks' mean embedding — how fast the meta moves.

    ``embeddings`` must be row-aligned with ``frame`` (standardized space).
    """

    weeks = ordered_weeks(frame)
    means = {}
    for week in weeks:
        mask = (frame["week"] == week).to_numpy()
        means[week] = embeddings[mask].mean(axis=0)

    records = []
    for prev, cur in zip(weeks[:-1], weeks[1:], strict=False):
        drift = float(np.linalg.norm(means[cur] - means[prev]))
        records.append({"week": cur, "prev_week": prev, "centroid_drift": drift})
    return pd.DataFrame(records)


def weekly_novelty(frame: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Mean distance of each week's teams to the *previous* week's centroid.

    A spike means the week introduced teams unlike the prior meta.
    """

    weeks = ordered_weeks(frame)
    centroids = {
        week: embeddings[(frame["week"] == week).to_numpy()].mean(axis=0) for week in weeks
    }
    records = []
    for prev, cur in zip(weeks[:-1], weeks[1:], strict=False):
        mask = (frame["week"] == cur).to_numpy()
        distances = np.linalg.norm(embeddings[mask] - centroids[prev], axis=1)
        records.append({"week": cur, "prev_week": prev, "novelty": float(distances.mean())})
    return pd.DataFrame(records)
