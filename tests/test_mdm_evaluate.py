import numpy as np

from vgc_team.meta.mdm.dataset import TeamFeatures
from vgc_team.meta.mdm.evaluate import anchor_histogram, kl_divergence, slice_before


def _features() -> TeamFeatures:
    week_index = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    return TeamFeatures(
        embeddings=np.eye(6, dtype=np.float32),
        anchors=np.array([0, 1, 0, 2, 1, 2], dtype=np.int64),
        week_index=week_index,
        weight=np.ones(6, dtype=np.float32),
        weeks=["2026-W01", "2026-W02", "2026-W03"],
    )


def test_slice_before_keeps_only_earlier_weeks() -> None:
    sub = slice_before(_features(), 2)
    assert sub.n_weeks == 2
    assert set(sub.week_index.tolist()) == {0, 1}
    assert len(sub.anchors) == 4


def test_anchor_histogram_normalizes_and_smooths() -> None:
    hist = anchor_histogram(np.array([0, 0, 1]), np.array([1.0, 1.0, 1.0]), 4, smoothing=1.0)
    assert np.isclose(hist.sum(), 1.0)
    assert (hist > 0).all()  # smoothing keeps zero-count anchors positive


def test_kl_divergence_zero_for_identical() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) < 1e-9
    assert kl_divergence(p, np.array([0.5, 0.3, 0.2])) > 0.0
