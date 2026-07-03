"""Training loop for the MDM (weighted cross-entropy; deep-ensemble support)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from vgc_team.meta.mdm.dataset import (
    MDMDataset,
    MDMDatasetConfig,
    TeamFeatures,
    collate_mdm,
)
from vgc_team.meta.mdm.model import MetaDistributionModel
from vgc_team.models.frozen_encoder import select_device


@dataclass
class MDMTrainConfig:
    d_model: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    epochs: int = 10  # maximum epoch cap (early stopping may end sooner)
    batch_size: int = 128
    baseline_decay: float = 1.0  # recency decay for the persistence baseline
    weight_decay: float = 0.01  # AdamW L2 on weights
    correction_l2: float = 0.0  # penalty on the residual correction (0 => off; pulls toward persistence)
    early_stop_min_delta: float | None = None  # stop when rel. CE improvement < this
    early_stop_patience: int = 1
    dataset: MDMDatasetConfig = field(default_factory=MDMDatasetConfig)
    device: torch.device | None = None
    verbose: bool = True


def relative_loss_improvement(previous_loss: float, current_loss: float) -> float:
    """Fractional epoch-to-epoch loss improvement (>=0 means it got better)."""

    if previous_loss <= 0:
        return 0.0
    return (previous_loss - current_loss) / previous_loss


def train_single(
    features: TeamFeatures,
    n_anchors: int,
    config: MDMTrainConfig,
    *,
    seed: int = 0,
) -> MetaDistributionModel:
    device = config.device or select_device()
    torch.manual_seed(seed)

    ds_config = MDMDatasetConfig(**{**vars(config.dataset), "seed": seed})
    dataset = MDMDataset(features, ds_config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_mdm, n_anchors=n_anchors, decay=config.baseline_decay),
        num_workers=0,
        drop_last=False,
    )

    model = MetaDistributionModel(
        d_in=features.embeddings.shape[1],
        n_anchors=n_anchors,
        d_model=config.d_model,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    model.train()
    previous_epoch_loss: float | None = None
    low_improvement_epochs = 0
    for epoch in range(config.epochs):
        total_loss = 0.0
        total_weight = 0.0
        for batch in loader:
            context_emb = batch["context_emb"].to(device)
            context_time = batch["context_time"].to(device)
            pad_mask = batch["context_pad_mask"].to(device)
            baseline = batch["baseline_logprob"].to(device)
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)

            logits = model(context_emb, context_time, pad_mask, baseline)
            per_example = F.cross_entropy(logits, target, reduction="none")
            loss = (per_example * weight).sum() / weight.sum().clamp_min(1e-8)
            if config.correction_l2 > 0:
                # logits = baseline + correction; shrink the correction toward 0
                correction = logits - baseline
                loss = loss + config.correction_l2 * correction.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float((per_example * weight).sum().item())
            total_weight += float(weight.sum().item())

        epoch_loss = total_loss / max(total_weight, 1e-8)
        if config.verbose:
            print(f"  [seed {seed}] epoch {epoch + 1}/{config.epochs}  ce={epoch_loss:.4f}")

        if config.early_stop_min_delta is not None and previous_epoch_loss is not None:
            improvement = relative_loss_improvement(previous_epoch_loss, epoch_loss)
            if improvement < config.early_stop_min_delta:
                low_improvement_epochs += 1
            else:
                low_improvement_epochs = 0
            if low_improvement_epochs >= config.early_stop_patience:
                if config.verbose:
                    print(
                        f"  [seed {seed}] early stop at epoch {epoch + 1}: "
                        f"improvement < {config.early_stop_min_delta:.1%} "
                        f"for {config.early_stop_patience} epoch(s)."
                    )
                break
        previous_epoch_loss = epoch_loss

    model.eval()
    return model


def train_ensemble(
    features: TeamFeatures,
    n_anchors: int,
    config: MDMTrainConfig,
    *,
    n_models: int = 5,
) -> list[MetaDistributionModel]:
    return [train_single(features, n_anchors, config, seed=seed) for seed in range(n_models)]


def save_ensemble(
    models: list[MetaDistributionModel],
    path,
    *,
    n_anchors: int,
    d_in: int,
    d_model: int,
) -> None:
    import torch as _torch

    _torch.save(
        {
            "state_dicts": [m.state_dict() for m in models],
            "n_anchors": n_anchors,
            "d_in": d_in,
            "d_model": d_model,
        },
        path,
    )


def load_ensemble(path, device: torch.device | None = None) -> list[MetaDistributionModel]:
    device = device or select_device()
    blob = torch.load(path, map_location=device)
    models = []
    for state in blob["state_dicts"]:
        model = MetaDistributionModel(
            d_in=blob["d_in"], n_anchors=blob["n_anchors"], d_model=blob["d_model"]
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models
