import torch

from vgc_team.models.hierarchical_team_transformer import HierarchicalTeamTransformer


def test_hierarchical_model_forward_shape() -> None:
    model = HierarchicalTeamTransformer(
        n_kinds=8,
        n_values=32,
        d_model=16,
        n_heads=4,
        pokemon_layers=1,
        team_layers=1,
    )

    kind_ids = torch.ones((2, 6, 15), dtype=torch.long)
    value_ids = torch.ones((2, 6, 15), dtype=torch.long)
    attr_mask = torch.ones((2, 6, 15), dtype=torch.bool)
    target_pokemon_idx = torch.tensor([0, 5], dtype=torch.long)
    target_kind_id = torch.tensor([1, 2], dtype=torch.long)

    logits = model(kind_ids, value_ids, attr_mask, target_pokemon_idx, target_kind_id)
    embedding = model.team_embedding(kind_ids, value_ids, attr_mask)

    assert logits.shape == (2, 32)
    assert embedding.shape == (2, 16)
