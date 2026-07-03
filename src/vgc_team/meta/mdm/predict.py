"""MDM inference: roll the recent meta forward into a predicted distribution."""

from __future__ import annotations

import numpy as np
import torch

from vgc_team.meta.mdm.dataset import TeamFeatures, build_baseline_logprob, relative_time
from vgc_team.meta.mdm.model import MetaDistributionModel
from vgc_team.models.frozen_encoder import select_device


def _n_anchors(model: MetaDistributionModel) -> int:
    return model.head[-1].out_features


def predict_anchor_samples(
    models: list[MetaDistributionModel],
    context_emb: np.ndarray,  # (C, d) standardized
    context_time: np.ndarray,  # (C,) weeks-ago offsets
    context_anchors: np.ndarray,  # (C,) anchor id per context team
    *,
    decay: float = 1.0,
    device=None,
) -> np.ndarray:
    """Per-model softmax over anchors: (n_models, n_anchors) — for epistemic spread."""

    device = device or select_device()
    n_anchors = _n_anchors(models[0])
    emb = torch.from_numpy(np.asarray(context_emb, dtype=np.float32)).unsqueeze(0).to(device)
    time = torch.from_numpy(np.asarray(context_time, dtype=np.float32)).unsqueeze(0).to(device)
    pad = torch.zeros(1, emb.shape[1], dtype=torch.bool, device=device)
    baseline = torch.from_numpy(
        build_baseline_logprob(context_anchors, context_time, n_anchors, decay=decay)
    ).unsqueeze(0).to(device)

    probs = []
    with torch.no_grad():
        for model in models:
            logits = model(emb, time, pad, baseline)
            probs.append(torch.softmax(logits, dim=-1)[0].cpu().numpy())
    return np.stack(probs, axis=0).astype(np.float32)


def predict_anchor_distribution(
    models: list[MetaDistributionModel],
    context_emb: np.ndarray,
    context_time: np.ndarray,
    context_anchors: np.ndarray,
    *,
    decay: float = 1.0,
    device=None,
) -> np.ndarray:
    """Ensemble-averaged softmax over anchors for one context set."""

    return predict_anchor_samples(
        models, context_emb, context_time, context_anchors, decay=decay, device=device
    ).mean(axis=0)


def forecast_next_week(
    models: list[MetaDistributionModel],
    features: TeamFeatures,
    *,
    decay: float = 1.0,
    device=None,
) -> tuple[np.ndarray, int]:
    """Leave-all-out forecast for the week after the last observed one.

    Uses every team in ``features`` as past context, queried at the next week.
    Returns (anchor_distribution, query_week_index).
    """

    query_week = features.n_weeks  # one past the last observed week
    context_time = relative_time(features.week_index, query_week)
    anchor_dist = predict_anchor_distribution(
        models, features.embeddings, context_time, features.anchors, decay=decay, device=device
    )
    return anchor_dist, query_week
