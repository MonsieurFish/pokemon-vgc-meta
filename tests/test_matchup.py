import numpy as np
import torch

from vgc_team.meta.matchup.model import MatchupModel
from vgc_team.meta.matchup.predict import expected_winrate_vs, win_probability


def _model(interactions=True):
    torch.manual_seed(0)
    m = MatchupModel(d_in=8, hidden=16, dropout=0.0, interactions=interactions)
    m.eval()
    return m


def test_antisymmetry_and_self_play():
    for interactions in (True, False):
        m = _model(interactions)
        a, b = torch.randn(4, 8), torch.randn(4, 8)
        assert torch.allclose(m.logit(a, b), -m.logit(b, a), atol=1e-6)
        assert torch.allclose(m.win_prob(a, b) + m.win_prob(b, a), torch.ones(4), atol=1e-6)
        assert torch.allclose(m.win_prob(a, a), torch.full((4,), 0.5), atol=1e-6)


def test_predict_helpers_consistent():
    m = _model()
    a, b = np.random.default_rng(1).normal(size=8), np.random.default_rng(2).normal(size=8)
    assert abs(win_probability(m, a, b) + win_probability(m, b, a) - 1.0) < 1e-6
    # expected win rate vs a field is a mean of pairwise probs
    opp = np.random.default_rng(3).normal(size=(5, 8))
    wr = expected_winrate_vs(m, a, opp)
    assert 0.0 <= wr <= 1.0


def test_strength_model_learns_separable_matchups():
    # synthetic: team "strength" = first embedding dim; a beats b iff a[0] > b[0].
    rng = np.random.default_rng(0)
    N = 400
    emb = rng.normal(size=(N, 8)).astype(np.float32)
    A_idx = rng.integers(0, N, size=2000)
    B_idx = rng.integers(0, N, size=2000)
    y = (emb[A_idx, 0] > emb[B_idx, 0]).astype(np.float32)
    A = torch.tensor(emb[A_idx]); B = torch.tensor(emb[B_idx]); Y = torch.tensor(y)

    model = MatchupModel(8, interactions=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(300):
        opt.zero_grad()
        loss_fn(model.logit(A, B), Y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = ((model.win_prob(A, B) > 0.5).float() == Y).float().mean().item()
    assert acc > 0.9  # recovers the strength ordering
