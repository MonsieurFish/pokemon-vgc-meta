import torch

from vgc_team.meta.mdm.model import MetaDistributionModel


def _model() -> MetaDistributionModel:
    torch.manual_seed(0)
    model = MetaDistributionModel(d_in=8, n_anchors=11, d_model=16, dropout=0.0)
    model.eval()
    return model


def test_forward_shape() -> None:
    model = _model()
    emb = torch.randn(3, 5, 8)
    time = torch.zeros(3, 5)
    pad = torch.zeros(3, 5, dtype=torch.bool)
    baseline = torch.zeros(3, 11)
    out = model(emb, time, pad, baseline)
    assert out.shape == (3, 11)


def test_starts_at_persistence_baseline() -> None:
    # zero-init head => initial output equals the baseline exactly
    model = _model()
    emb = torch.randn(2, 4, 8)
    time = torch.randn(2, 4)
    pad = torch.zeros(2, 4, dtype=torch.bool)
    baseline = torch.randn(2, 11)
    out = model(emb, time, pad, baseline)
    assert torch.allclose(out, baseline, atol=1e-6)


def test_permutation_invariance() -> None:
    model = _model()
    emb = torch.randn(1, 6, 8)
    time = torch.randn(1, 6)
    pad = torch.zeros(1, 6, dtype=torch.bool)
    baseline = torch.randn(1, 11)
    out = model(emb, time, pad, baseline)

    perm = torch.randperm(6)
    out_perm = model(emb[:, perm], time[:, perm], pad[:, perm], baseline)
    assert torch.allclose(out, out_perm, atol=1e-5)


def test_padding_invariance() -> None:
    model = _model()
    emb = torch.randn(1, 4, 8)
    time = torch.randn(1, 4)
    pad = torch.zeros(1, 4, dtype=torch.bool)
    baseline = torch.randn(1, 11)
    out = model(emb, time, pad, baseline)

    # append two padded (masked) tokens; output must not change
    emb_pad = torch.cat([emb, torch.randn(1, 2, 8)], dim=1)
    time_pad = torch.cat([time, torch.randn(1, 2)], dim=1)
    pad_pad = torch.cat([pad, torch.ones(1, 2, dtype=torch.bool)], dim=1)
    out_pad = model(emb_pad, time_pad, pad_pad, baseline)
    assert torch.allclose(out, out_pad, atol=1e-5)
