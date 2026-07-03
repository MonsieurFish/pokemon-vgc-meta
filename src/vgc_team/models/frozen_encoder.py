"""Frozen team-encoder utilities for downstream models.

The masked-feature transformer is trained once and then *frozen*: every later
task (meta-shift prediction, unseen-team rating, ...) consumes the same fixed
128-d team embedding instead of re-learning a representation. This module is the
single place that:

1. loads a checkpoint and freezes the encoder (``load_frozen_encoder``),
2. turns ``Team`` objects into embeddings (``embed_teams``),
3. standardizes embeddings the same way the reference corpus was standardized,
4. exposes the reference cluster/anchor geometry (``ReferenceSpace``) so any new
   team can be placed in the existing archetype space.

``scripts/build_team_knn.py`` imports the embedding/standardization helpers from
here so there is exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vgc_team.config import DATA_DIR
from vgc_team.models.hierarchical_team_transformer import HierarchicalTeamTransformer
from vgc_team.teams.features import encode_team
from vgc_team.teams.schema import Team
from vgc_team.teams.vocab import KindVocab, TokenVocab

DEFAULT_CHECKPOINT = (
    DATA_DIR.parent / "models" / "masked_team_transformer_ma" / "checkpoint.pt"
)
DEFAULT_REFERENCE_NPZ = DATA_DIR / "processed" / "team_knn" / "team_embeddings.npz"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_encoder(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[HierarchicalTeamTransformer, KindVocab, TokenVocab, dict[str, object]]:
    """Load a checkpoint, build the model, freeze every parameter, eval mode.

    Downstream heads never backprop into this model; they consume its
    ``team_embedding`` output under ``torch.no_grad``.
    """

    device = device or select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    kind_vocab = KindVocab.from_json_dict(checkpoint["kind_vocab"])
    token_vocab = TokenVocab.from_json_dict(checkpoint["token_vocab"])

    model = HierarchicalTeamTransformer(
        n_kinds=len(kind_vocab.kind_to_id),
        n_values=len(token_vocab.token_to_id),
        d_model=int(config.get("d_model", 128)),
        n_heads=int(config.get("n_heads", 4)),
        pokemon_layers=int(config.get("pokemon_layers", 2)),
        team_layers=int(config.get("team_layers", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, kind_vocab, token_vocab, config


def embed_teams(
    model: HierarchicalTeamTransformer,
    teams: list[Team],
    kind_vocab: KindVocab,
    token_vocab: TokenVocab,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed teams through the frozen encoder, returning raw (N, d_model)."""

    if not teams:
        return np.zeros((0, model.output.in_features), dtype=np.float32)

    device = device or next(model.parameters()).device
    encoded = [
        encode_team(team, kind_vocab=kind_vocab, token_vocab=token_vocab)
        for team in teams
    ]

    batches: list[np.ndarray] = []
    index_range = range(0, len(encoded), batch_size)
    if show_progress:
        from tqdm import tqdm

        index_range = tqdm(index_range, desc="embedding teams")

    with torch.no_grad():
        for start in index_range:
            chunk = encoded[start : start + batch_size]
            kind_ids = torch.stack([example.kind_ids for example in chunk]).to(device)
            value_ids = torch.stack([example.value_ids for example in chunk]).to(device)
            attr_mask = torch.stack([example.attr_mask for example in chunk]).to(device)
            embeddings = model.team_embedding(kind_ids, value_ids, attr_mask)
            batches.append(embeddings.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(batches, axis=0)


def standardize_embeddings(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale every embedding dimension to mean 0 and standard deviation 1."""

    feature_mean = embeddings.mean(axis=0, keepdims=True)
    feature_std = embeddings.std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    standardized = (embeddings - feature_mean) / feature_std
    return standardized.astype(np.float32), feature_mean.squeeze(0), feature_std.squeeze(0)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


@dataclass
class ReferenceSpace:
    """The frozen archetype geometry shared by every downstream consumer.

    All vectors live in the *standardized* embedding space (same space the
    reference KMeans was fit in). New teams must be standardized with
    ``feature_mean``/``feature_std`` from the reference corpus before they are
    compared to centroids or anchors -- never re-standardized per batch.
    """

    feature_mean: np.ndarray  # (d_model,)
    feature_std: np.ndarray  # (d_model,)
    cluster_centroids: np.ndarray  # (n_clusters, d_model) interpretable archetypes
    anchor_centroids: np.ndarray  # (n_anchors, d_model) finer MDM vocabulary

    @classmethod
    def load(cls, npz_path: Path = DEFAULT_REFERENCE_NPZ) -> "ReferenceSpace":
        data = np.load(npz_path, allow_pickle=True)
        if "cluster_centroids" not in data:
            raise KeyError(
                f"{npz_path} has no 'cluster_centroids'. Re-run scripts/build_team_knn.py "
                "to persist centroids and the anchor codebook."
            )
        anchors = (
            data["anchor_centroids"]
            if "anchor_centroids" in data
            else data["cluster_centroids"]
        )
        return cls(
            feature_mean=data["feature_mean"].astype(np.float32),
            feature_std=data["feature_std"].astype(np.float32),
            cluster_centroids=data["cluster_centroids"].astype(np.float32),
            anchor_centroids=anchors.astype(np.float32),
        )

    @property
    def n_clusters(self) -> int:
        return self.cluster_centroids.shape[0]

    @property
    def n_anchors(self) -> int:
        return self.anchor_centroids.shape[0]

    def standardize(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self.feature_mean) / self.feature_std).astype(np.float32)

    @staticmethod
    def _distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # (N, K) euclidean distances without forming the full diff tensor.
        sq = (
            (points**2).sum(axis=1, keepdims=True)
            - 2.0 * points @ centers.T
            + (centers**2).sum(axis=1)[None, :]
        )
        return np.sqrt(np.maximum(sq, 0.0))

    def assign_clusters(self, raw: np.ndarray) -> np.ndarray:
        """Nearest interpretable cluster id for each raw embedding."""

        return self._distances(self.standardize(raw), self.cluster_centroids).argmin(axis=1)

    def assign_anchors(self, raw: np.ndarray) -> np.ndarray:
        """Nearest anchor id (MDM categorical target) for each raw embedding."""

        return self._distances(self.standardize(raw), self.anchor_centroids).argmin(axis=1)

    def anchor_to_cluster(self) -> np.ndarray:
        """Map each anchor (fine codebook) to its nearest interpretable cluster.

        Lets an anchor-level meta distribution be summed into archetype shares.
        """

        return self._distances(self.anchor_centroids, self.cluster_centroids).argmin(axis=1)

    def cluster_fit_scores(self, raw: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
        """Soft membership over interpretable clusters: (N, n_clusters), rows sum to 1."""

        distances = self._distances(self.standardize(raw), self.cluster_centroids)
        return _softmax(-distances / max(temperature, 1e-6), axis=1)

    def anchor_distribution(
        self,
        raw: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        smoothing: float = 1.0,
    ) -> np.ndarray:
        """Weighted, smoothed histogram over anchors -> a meta distribution (n_anchors,)."""

        labels = self.assign_anchors(raw)
        if weights is None:
            weights = np.ones(len(labels), dtype=np.float64)
        counts = np.bincount(labels, weights=weights, minlength=self.n_anchors).astype(np.float64)
        counts += smoothing
        return (counts / counts.sum()).astype(np.float32)
