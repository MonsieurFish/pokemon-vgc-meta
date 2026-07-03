import numpy as np

from vgc_team.meta.mdm.dataset import (
    MDMDataset,
    MDMDatasetConfig,
    TeamFeatures,
    collate_mdm,
    relative_time,
)


def _features(n_per_week: int = 5, n_weeks: int = 4) -> TeamFeatures:
    n = n_per_week * n_weeks
    # identity embeddings so each context row maps back to a unique team index
    embeddings = np.eye(n, dtype=np.float32)
    anchors = (np.arange(n) % 7).astype(np.int64)
    week_index = np.repeat(np.arange(n_weeks), n_per_week).astype(np.int64)
    weight = np.ones(n, dtype=np.float32)
    weeks = [f"2026-W{w:02d}" for w in range(n_weeks)]
    return TeamFeatures(embeddings, anchors, week_index, weight, weeks)


def test_relative_time_is_weeks_ago() -> None:
    assert relative_time(np.array([3, 2, 0]), 4).tolist() == [-1.0, -2.0, -4.0]


def test_target_never_leaks_into_context() -> None:
    f = _features()
    ds = MDMDataset(f, MDMDatasetConfig(full_mask_prob=0.0, seed=1))
    for i in range(len(ds)):
        target_idx = int(ds.target_indices[i])
        item = ds[i]
        context_rows = item["context_emb"].numpy().argmax(axis=1)
        assert target_idx not in context_rows.tolist()


def test_full_mask_drops_all_current_week_context() -> None:
    f = _features()
    ds = MDMDataset(f, MDMDatasetConfig(full_mask_prob=1.0, seed=2))
    for i in range(len(ds)):
        target_idx = int(ds.target_indices[i])
        t = int(f.week_index[target_idx])
        item = ds[i]
        context_rows = item["context_emb"].numpy().argmax(axis=1)
        assert all(f.week_index[row] < t for row in context_rows)


def test_context_budget_is_respected() -> None:
    f = _features(n_per_week=50, n_weeks=4)
    ds = MDMDataset(f, MDMDatasetConfig(context_budgets=(8,), max_context=8, seed=3))
    item = ds[len(ds) - 1]
    assert item["context_emb"].shape[0] <= 8


def _multi_reg_features() -> TeamFeatures:
    # 2 regulations, each 3 weeks x 4 teams; identity embeddings for traceability
    n = 24
    reg = np.repeat([0, 1], 12).astype(np.int64)
    week = np.tile(np.repeat([0, 1, 2], 4), 2).astype(np.int64)
    return TeamFeatures(
        embeddings=np.eye(n, dtype=np.float32),
        anchors=(np.arange(n) % 5).astype(np.int64),
        week_index=week,
        weight=np.ones(n, dtype=np.float32),
        weeks=[],
        regulation=reg,
        reg_weeks=[["w0", "w1", "w2"], ["w0", "w1", "w2"]],
        reg_labels=["regA", "regB"],
    )


def test_context_never_crosses_regulations() -> None:
    f = _multi_reg_features()
    ds = MDMDataset(f, MDMDatasetConfig(full_mask_prob=0.5, seed=1))
    for i in range(len(ds)):
        target_idx = int(ds.target_indices[i])
        target_reg = int(f.regulation[target_idx])
        context_rows = ds[i]["context_emb"].numpy().argmax(axis=1)
        assert all(int(f.regulation[row]) == target_reg for row in context_rows)


def test_single_reg_view_and_subset() -> None:
    from vgc_team.meta.mdm.dataset import single_reg_view, subset_regulations

    f = _multi_reg_features()
    view = single_reg_view(f, 1)
    assert view.regulation is None  # standalone single-reg
    assert view.n_weeks == 3
    assert len(view.anchors) == 12

    sub = subset_regulations(f, [1])
    assert sub.n_regulations == 1
    assert sub.reg_labels == ["regB"]
    assert len(sub.anchors) == 12


def test_collate_pads_and_masks() -> None:
    f = _features()
    ds = MDMDataset(f, MDMDatasetConfig(seed=4))
    batch = collate_mdm([ds[0], ds[1], ds[2]], n_anchors=8)
    b, length = batch["context_pad_mask"].shape
    assert b == 3
    assert batch["baseline_logprob"].shape == (3, 8)
    # padded positions are masked True; real positions False
    for row in range(3):
        n_real = int((~batch["context_pad_mask"][row]).sum())
        assert n_real >= 1
        assert batch["context_emb"][row, n_real:].abs().sum() == 0
