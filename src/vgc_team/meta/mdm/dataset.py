"""Training-example construction for the Meta Distribution Model (MDM).

The MDM learns ``p(team's anchor | recent meta context, query time)`` so that,
queried at a future week with no current-week teams, it predicts that week's
anchor distribution. The data multiplier is **leave-one-out over every team**:
each team is a target whose anchor we predict from a *context set* of other
teams, turning ~10 weeks into ~50k examples.

Key regimes baked in here:
- **mask curriculum** on the current week (weighted toward fully masking it), so
  the model learns to *forecast* from past weeks rather than impute from
  same-week neighbours; 100%-mask matches deployment.
- **variable / small context sizes**, so thin-context deployment (e.g. 3x60
  teams) is in-distribution.
- **continuous time** tags, so the model can be queried at week+1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from vgc_team.models.frozen_encoder import ReferenceSpace
from vgc_team.teams.schema import Team
from vgc_team.teams.vocab import KindVocab, TokenVocab


# One ISO week = one unit of time. Context teams are encoded by how many weeks
# *before the queried week* they occurred (0 = same week, -1 = last week, ...),
# so the representation is independent of absolute season position and of how
# many weeks the context spans. This is what makes a thin 3-week deployment
# in-distribution with 10-week training.
TIME_STEP = 1.0


def relative_time(week_index: np.ndarray | int, query_week: int) -> np.ndarray:
    """Weeks-ago offset of each token relative to the queried week (<= 0 for past)."""

    return (np.asarray(week_index, dtype=np.float32) - float(query_week)) * TIME_STEP


def build_baseline_logprob(
    context_anchors: np.ndarray,
    context_time: np.ndarray,
    n_anchors: int,
    *,
    decay: float = 1.0,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Log of the recency-weighted context anchor distribution — the PERSISTENCE
    baseline the MDM predicts a residual on top of.

    Context teams are weighted by ``exp(decay * weeks_ago)`` (weeks_ago <= 0), so
    the most recent week dominates -> baseline ~= "last week's meta". The model's
    output is added to this in log-space, so a zero correction == persistence.
    """

    weights = np.exp(decay * np.asarray(context_time, dtype=np.float64))
    hist = np.bincount(
        np.asarray(context_anchors, dtype=np.int64), weights=weights, minlength=n_anchors
    ).astype(np.float64)
    hist += smoothing
    hist /= hist.sum()
    return np.log(hist).astype(np.float32)


@dataclass
class TeamFeatures:
    """Per-team arrays the MDM consumes (row-aligned).

    Single-regulation: ``regulation`` is None and ``week_index`` is global.
    Multi-regulation: ``regulation`` is an int id per team, ``week_index`` is
    *within-regulation* (each reg 0..W_reg-1), and ``reg_weeks``/``reg_labels``
    describe each regulation. Context in the MDM never crosses regulations.
    """

    embeddings: np.ndarray  # (N, d) standardized
    anchors: np.ndarray  # (N,) int anchor id
    week_index: np.ndarray  # (N,) int within-regulation week
    weight: np.ndarray  # (N,) float
    weeks: list[str]  # single-reg: ordered week labels; multi-reg: unused ([])
    regulation: np.ndarray | None = None  # (N,) int reg id, or None => single reg
    reg_weeks: list[list[str]] | None = None  # per-reg ordered week labels
    reg_labels: list[str] | None = None  # per-reg name (e.g. format_id)

    @property
    def n_weeks(self) -> int:
        return len(self.weeks)

    @property
    def n_regulations(self) -> int:
        return 1 if self.regulation is None else int(self.regulation.max()) + 1

    def reg_ids(self) -> np.ndarray:
        return np.zeros(len(self.anchors), dtype=np.int64) if self.regulation is None else self.regulation


def build_team_features(
    teams: list[Team],
    reference: ReferenceSpace,
    model,
    kind_vocab: KindVocab,
    token_vocab: TokenVocab,
    *,
    weights: np.ndarray | None = None,
    batch_size: int = 512,
    device=None,
    show_progress: bool = True,
) -> TeamFeatures:
    """Embed teams, standardize, assign anchors, and index weeks."""

    from vgc_team.meta.timeseries import iso_week_label
    from vgc_team.models.frozen_encoder import embed_teams

    if weights is None:
        weights = np.ones(len(teams), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    # drop teams without a timestamp, keeping weights aligned
    kept = [(team, weight) for team, weight in zip(teams, weights, strict=True)
            if team.timestamp is not None]
    teams = [team for team, _ in kept]
    weights = np.array([weight for _, weight in kept], dtype=np.float32)

    raw = embed_teams(
        model, teams, kind_vocab, token_vocab,
        batch_size=batch_size, device=device, show_progress=show_progress,
    )
    standardized = reference.standardize(raw)
    anchors = reference.assign_anchors(raw)

    labels = [iso_week_label(team.timestamp) for team in teams]
    ordered = sorted(set(labels))
    week_to_idx = {week: idx for idx, week in enumerate(ordered)}
    week_index = np.array([week_to_idx[label] for label in labels], dtype=np.int64)

    return TeamFeatures(
        embeddings=standardized.astype(np.float32),
        anchors=anchors.astype(np.int64),
        week_index=week_index,
        weight=weights.astype(np.float32),
        weeks=ordered,
    )


def _balance_factors(reg_total_weight: np.ndarray, mode: str) -> np.ndarray:
    """Per-regulation multiplier so regulations contribute comparably to the loss.

    - "equal": every regulation gets the same total weight (mean of totals).
    - "sqrt":  softer — total weight scaled to sqrt (down-weights big regs less).
    - "none":  no rebalancing.
    """

    totals = np.asarray(reg_total_weight, dtype=np.float64)
    if mode == "none":
        return np.ones_like(totals)
    if mode == "sqrt":
        target = np.sqrt(totals) * (totals.mean() / np.sqrt(totals).mean())
        return target / totals
    # equal
    return totals.mean() / totals


def build_multi_reg_features(
    regulations: list[tuple[str, list[Team], np.ndarray]],
    reference: ReferenceSpace,
    model,
    kind_vocab: KindVocab,
    token_vocab: TokenVocab,
    *,
    balance: str = "equal",
    batch_size: int = 512,
    device=None,
    show_progress: bool = True,
) -> TeamFeatures:
    """Embed teams from many regulations into one regulation-aware TeamFeatures.

    ``regulations`` is a list of (label, teams, base_weights). Week indices are
    reset per regulation; a per-regulation balance factor is folded into weights.
    """

    from vgc_team.meta.timeseries import iso_week_label
    from vgc_team.models.frozen_encoder import embed_teams

    all_teams: list[Team] = []
    reg_of: list[int] = []
    within_week: list[int] = []
    base_weight: list[float] = []
    reg_weeks: list[list[str]] = []
    reg_labels: list[str] = []

    for reg_id, (label, teams, weights) in enumerate(regulations):
        weights = np.asarray(weights, dtype=np.float32)
        kept = [(t, w) for t, w in zip(teams, weights, strict=True) if t.timestamp is not None]
        if not kept:
            continue
        teams_r = [t for t, _ in kept]
        weights_r = [float(w) for _, w in kept]
        labels = [iso_week_label(t.timestamp) for t in teams_r]
        ordered = sorted(set(labels))
        w2i = {w: i for i, w in enumerate(ordered)}

        reg_labels.append(label)
        reg_weeks.append(ordered)
        for team, weight, wk in zip(teams_r, weights_r, labels, strict=True):
            all_teams.append(team)
            reg_of.append(len(reg_labels) - 1)
            within_week.append(w2i[wk])
            base_weight.append(weight)

    reg_of = np.array(reg_of, dtype=np.int64)
    base_weight = np.array(base_weight, dtype=np.float32)

    # per-regulation balance
    n_reg = len(reg_labels)
    reg_totals = np.array([base_weight[reg_of == r].sum() for r in range(n_reg)], dtype=np.float64)
    factors = _balance_factors(reg_totals, balance)
    weight = base_weight * factors[reg_of].astype(np.float32)

    raw = embed_teams(
        model, all_teams, kind_vocab, token_vocab,
        batch_size=batch_size, device=device, show_progress=show_progress,
    )
    return TeamFeatures(
        embeddings=reference.standardize(raw).astype(np.float32),
        anchors=reference.assign_anchors(raw).astype(np.int64),
        week_index=np.array(within_week, dtype=np.int64),
        weight=weight.astype(np.float32),
        weeks=[],
        regulation=reg_of,
        reg_weeks=reg_weeks,
        reg_labels=reg_labels,
    )


def single_reg_view(features: TeamFeatures, reg_id: int) -> TeamFeatures:
    """Extract one regulation as a standalone single-reg TeamFeatures (for forecasting)."""

    mask = features.reg_ids() == reg_id
    weeks = features.reg_weeks[reg_id] if features.reg_weeks else features.weeks
    return TeamFeatures(
        embeddings=features.embeddings[mask],
        anchors=features.anchors[mask],
        week_index=features.week_index[mask],
        weight=features.weight[mask],
        weeks=weeks,
    )


def subset_regulations(features: TeamFeatures, keep_ids: list[int]) -> TeamFeatures:
    """Multi-reg features restricted to ``keep_ids`` (reg ids remapped to 0..k-1)."""

    reg = features.reg_ids()
    mask = np.isin(reg, keep_ids)
    remap = {old: new for new, old in enumerate(keep_ids)}
    new_reg = np.array([remap[r] for r in reg[mask]], dtype=np.int64)
    return TeamFeatures(
        embeddings=features.embeddings[mask],
        anchors=features.anchors[mask],
        week_index=features.week_index[mask],
        weight=features.weight[mask],
        weeks=[],
        regulation=new_reg,
        reg_weeks=[features.reg_weeks[r] for r in keep_ids] if features.reg_weeks else None,
        reg_labels=[features.reg_labels[r] for r in keep_ids] if features.reg_labels else None,
    )


@dataclass
class MDMDatasetConfig:
    full_mask_prob: float = 0.5  # probability the current week is fully masked
    context_budgets: tuple[int, ...] = (64, 128, 256)  # variable context sizes
    max_context: int = 256
    min_prev_weeks: int = 1  # targets need this many prior weeks
    seed: int = 0


class MDMDataset(Dataset):
    """Leave-one-out targets with on-the-fly mask curriculum + context sampling."""

    def __init__(self, features: TeamFeatures, config: MDMDatasetConfig | None = None) -> None:
        self.f = features
        self.cfg = config or MDMDatasetConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        reg = features.reg_ids()
        week = features.week_index
        # Context pools are keyed by (regulation, week) so context never crosses
        # regulations — each regulation is its own temporal sequence.
        self.week_to_members: dict[tuple[int, int], np.ndarray] = {}
        self.prev_pool: dict[tuple[int, int], np.ndarray] = {}
        for r in np.unique(reg):
            reg_mask = reg == r
            for w in np.unique(week[reg_mask]):
                self.week_to_members[(int(r), int(w))] = np.flatnonzero(reg_mask & (week == w))
                self.prev_pool[(int(r), int(w))] = np.flatnonzero(reg_mask & (week < w))

        self._reg = reg
        self.target_indices = np.flatnonzero(week >= self.cfg.min_prev_weeks)

    def __len__(self) -> int:
        return len(self.target_indices)

    def _sample(self, pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(pool) == 0:
            return np.empty(0, dtype=np.int64)
        if count >= len(pool):
            return pool
        return self._rng.choice(pool, size=count, replace=False)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.target_indices[i])
        t = int(self.f.week_index[idx])
        r = int(self._reg[idx])

        mask_ratio = (
            1.0 if self._rng.random() < self.cfg.full_mask_prob else float(self._rng.random())
        )
        budget = int(self._rng.choice(self.cfg.context_budgets))
        budget = min(budget, self.cfg.max_context)

        current_pool = self.week_to_members[(r, t)]
        current_pool = current_pool[current_pool != idx]
        keep_current = int(round((1.0 - mask_ratio) * len(current_pool)))
        keep_current = min(keep_current, budget)
        current_ctx = self._sample(current_pool, keep_current)

        remaining = budget - len(current_ctx)
        past_ctx = self._sample(self.prev_pool[(r, t)], remaining)

        context = np.concatenate([current_ctx, past_ctx]) if remaining or len(current_ctx) else past_ctx
        if len(context) == 0:
            # degenerate (no prior teams kept) — fall back to one past team
            context = self._sample(self.prev_pool[(r, t)], 1)

        ctx_emb = torch.from_numpy(self.f.embeddings[context])
        ctx_time = torch.from_numpy(relative_time(self.f.week_index[context], t))
        return {
            "context_emb": ctx_emb,
            "context_time": ctx_time.float(),
            "context_anchors": torch.from_numpy(self.f.anchors[context]).long(),
            "target": torch.tensor(int(self.f.anchors[idx]), dtype=torch.long),
            "weight": torch.tensor(float(self.f.weight[idx]), dtype=torch.float32),
        }


def collate_mdm(
    batch: list[dict[str, torch.Tensor]],
    *,
    n_anchors: int,
    decay: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Pad variable-length contexts; build a key padding mask + persistence baseline."""

    max_len = max(item["context_emb"].shape[0] for item in batch)
    dim = batch[0]["context_emb"].shape[1]
    bsz = len(batch)

    context_emb = torch.zeros(bsz, max_len, dim, dtype=torch.float32)
    context_time = torch.zeros(bsz, max_len, dtype=torch.float32)
    pad_mask = torch.ones(bsz, max_len, dtype=torch.bool)  # True where padded
    baseline = torch.zeros(bsz, n_anchors, dtype=torch.float32)
    for row, item in enumerate(batch):
        n = item["context_emb"].shape[0]
        context_emb[row, :n] = item["context_emb"]
        context_time[row, :n] = item["context_time"]
        pad_mask[row, :n] = False
        baseline[row] = torch.from_numpy(
            build_baseline_logprob(
                item["context_anchors"].numpy(), item["context_time"].numpy(),
                n_anchors, decay=decay,
            )
        )

    return {
        "context_emb": context_emb,
        "context_time": context_time,
        "context_pad_mask": pad_mask,
        "baseline_logprob": baseline,
        "target": torch.stack([item["target"] for item in batch]),
        "weight": torch.stack([item["weight"] for item in batch]),
    }
