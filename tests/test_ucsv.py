import numpy as np

from vgc_team.meta import ucsv


def _simplex(rng, W, K):
    P = rng.random((W, K)) + 0.05
    return P / P.sum(axis=1, keepdims=True)


def test_alr_inv_alr_round_trips() -> None:
    rng = np.random.default_rng(0)
    P = _simplex(rng, 5, 8)
    Y = ucsv.alr(P, ref_idx=3)
    recon = np.stack([ucsv.inv_alr(Y[w], 3, 8) for w in range(5)])
    assert np.allclose(recon, P, atol=1e-10)


def test_kalman_beats_last_obs_on_noisy_constant() -> None:
    # true level constant; observations = level + transitory noise.
    # a smoothing filter should estimate the level better than the last observation.
    rng = np.random.default_rng(1)
    filt_err, last_err = [], []
    for _ in range(200):
        level = 2.0
        y = level + rng.normal(scale=1.0, size=(8, 1))
        a = ucsv._kalman_local_level(y, q=0.1)
        filt_err.append(abs(float(a[0]) - level))
        last_err.append(abs(float(y[-1, 0]) - level))
    assert np.mean(filt_err) < np.mean(last_err)


def test_high_q_collapses_to_persistence() -> None:
    rng = np.random.default_rng(2)
    P = _simplex(rng, 6, 10)
    pred = ucsv.LocalLevel(q=1e9, ref_idx=0).forecast(P)
    assert np.allclose(pred, P[-1], atol=1e-4)


def test_glide_endpoints() -> None:
    rng = np.random.default_rng(3)
    P = _simplex(rng, 4, 6)
    assert np.allclose(ucsv.Glide(0.0).forecast(P), P[-1])
    mean_dist = P.mean(axis=0)
    assert np.allclose(ucsv.Glide(1.0).forecast(P), mean_dist / mean_dist.sum())


def test_combination_endpoints() -> None:
    rng = np.random.default_rng(4)
    P = _simplex(rng, 5, 7)
    base = ucsv.LocalLevel(q=0.5, ref_idx=1)
    assert np.allclose(ucsv.Combination(base, 0.0).forecast(P), P[-1])
    b = base.forecast(P)
    assert np.allclose(ucsv.Combination(base, 1.0).forecast(P), b / b.sum())


def test_weekly_anchor_matrix_shape_and_normalized() -> None:
    from vgc_team.meta.mdm.dataset import TeamFeatures

    # 2 weeks x a few teams, single-reg view style
    feats = TeamFeatures(
        embeddings=np.zeros((6, 4), dtype=np.float32),
        anchors=np.array([0, 0, 1, 2, 1, 1]),
        week_index=np.array([0, 0, 0, 1, 1, 1]),
        weight=np.ones(6, dtype=np.float32),
        weeks=["w0", "w1"],
    )
    P = ucsv.weekly_anchor_matrix(feats, n_anchors=3)
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
