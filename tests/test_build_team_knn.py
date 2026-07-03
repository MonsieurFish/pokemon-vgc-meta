import numpy as np

from scripts.build_team_knn import (
    representative_indices_by_cluster,
    standardize_embeddings,
    team_hash,
)
from vgc_team.teams.schema import PokemonSet, Team


def _team(pokemon: tuple[PokemonSet, ...]) -> Team:
    return Team(pokemon=pokemon, format_id="gen9vgc2026regma")


def test_team_hash_ignores_team_slot_order() -> None:
    flutter_mane = PokemonSet(
        species="Flutter Mane",
        item="Booster Energy",
        ability="Protosynthesis",
        moves=("Moonblast", "Shadow Ball", "Protect", "Icy Wind"),
        nature="Timid",
    )
    incineroar = PokemonSet(
        species="Incineroar",
        item="Safety Goggles",
        ability="Intimidate",
        moves=("Fake Out", "Flare Blitz", "Parting Shot", "Knock Off"),
        nature="Careful",
    )

    assert team_hash(_team((flutter_mane, incineroar))) == team_hash(
        _team((incineroar, flutter_mane))
    )


def test_standardize_embeddings_scales_feature_space() -> None:
    embeddings = np.array(
        [
            [1.0, 10.0, 4.0],
            [2.0, 20.0, 4.0],
            [3.0, 30.0, 4.0],
        ],
        dtype=np.float32,
    )

    standardized, mean, std = standardize_embeddings(embeddings)

    assert np.allclose(mean, np.array([2.0, 20.0, 4.0], dtype=np.float32))
    assert std[2] == 1.0
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(standardized[:, :2].std(axis=0), 1.0, atol=1e-6)


def test_representative_indices_are_closest_to_cluster_mean() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 1])

    representatives = representative_indices_by_cluster(embeddings, labels)

    assert representatives[0]["representative_index"] == 0
    assert representatives[0]["cluster_size"] == 2
    assert representatives[1]["representative_index"] == 3
    assert representatives[1]["cluster_size"] == 3
