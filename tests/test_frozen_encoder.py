import numpy as np
import pytest

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.models.frozen_encoder import (
    DEFAULT_REFERENCE_NPZ,
    ReferenceSpace,
    embed_teams,
    load_frozen_encoder,
)
from vgc_team.teams.schema import PokemonSet, Team

CHECKPOINT = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt"
NPZ = DEFAULT_REFERENCE_NPZ

requires_artifacts = pytest.mark.skipif(
    not (CHECKPOINT.exists() and NPZ.exists()),
    reason="trained checkpoint / reference npz not present",
)


def _team() -> Team:
    return Team(
        pokemon=(
            PokemonSet("Incineroar", "Safety Goggles", "Intimidate",
                       ("Fake Out", "Flare Blitz", "Parting Shot", "Knock Off"), "Careful"),
            PokemonSet("Flutter Mane", "Booster Energy", "Protosynthesis",
                       ("Moonblast", "Shadow Ball", "Protect", "Icy Wind"), "Timid"),
        ),
        format_id="test",
    )


@requires_artifacts
def test_reference_space_geometry() -> None:
    ref = ReferenceSpace.load(NPZ)
    assert ref.cluster_centroids.shape[1] == ref.anchor_centroids.shape[1]
    assert ref.n_clusters >= 2 and ref.n_anchors >= ref.n_clusters

    z = np.load(NPZ, allow_pickle=True)
    raw = z["raw_embeddings"][:500]
    saved = z["cluster_labels"][:500]
    assert float((ref.assign_clusters(raw) == saved).mean()) == 1.0

    fit = ref.cluster_fit_scores(raw[:10])
    assert np.allclose(fit.sum(axis=1), 1.0, atol=1e-5)
    # each anchor maps to a valid cluster id
    mapping = ref.anchor_to_cluster()
    assert mapping.min() >= 0 and mapping.max() < ref.n_clusters


@requires_artifacts
def test_encoder_is_frozen_and_deterministic() -> None:
    model, kind_vocab, token_vocab, _ = load_frozen_encoder(CHECKPOINT)
    assert all(not p.requires_grad for p in model.parameters())

    a = embed_teams(model, [_team()], kind_vocab, token_vocab)
    b = embed_teams(model, [_team()], kind_vocab, token_vocab)
    assert a.shape[0] == 1
    assert np.allclose(a, b)
