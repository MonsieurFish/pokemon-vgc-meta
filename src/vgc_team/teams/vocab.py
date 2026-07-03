"""Vocabulary utilities for masked team modeling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

PAD = "<PAD>"
MASK = "<MASK>"
UNK = "<UNK>"
PKMN_CLS = "<PKMN_CLS>"
TEAM_CLS = "<TEAM_CLS>"


@dataclass
class TokenVocab:
    token_to_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for token in [PAD, MASK, UNK, PKMN_CLS, TEAM_CLS]:
            self.add(token)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
        return self.token_to_id[token]

    def update(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self.add(token)

    def id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK])

    def token(self, token_id: int) -> str:
        id_to_token = {value: key for key, value in self.token_to_id.items()}
        return id_to_token.get(token_id, UNK)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK]

    def to_json_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_json_dict(cls, data: dict[str, int]) -> TokenVocab:
        vocab = cls()
        vocab.token_to_id = dict(data)
        return vocab


@dataclass
class KindVocab:
    kind_to_id: dict[str, int] = field(default_factory=lambda: {"pad": 0})

    def add(self, kind: str) -> int:
        if kind not in self.kind_to_id:
            self.kind_to_id[kind] = len(self.kind_to_id)
        return self.kind_to_id[kind]

    def id(self, kind: str) -> int:
        return self.kind_to_id.get(kind, self.kind_to_id["pad"])

    @property
    def pad_id(self) -> int:
        return self.kind_to_id["pad"]

    def to_json_dict(self) -> dict[str, int]:
        return dict(self.kind_to_id)

    @classmethod
    def from_json_dict(cls, data: dict[str, int]) -> KindVocab:
        vocab = cls()
        vocab.kind_to_id = dict(data)
        return vocab
