"""Train the hierarchical masked-feature team model.

This is the bootstrap pretraining task:

- parse open-team-sheet teams
- mask one item/ability/nature/move token
- predict the original token from the rest of the Pokemon set and team

It intentionally ignores win/loss labels for now.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.vgc_bench import MA_FILES, download_vgc_bench_file, load_vgc_bench_teams
from vgc_team.models.hierarchical_team_transformer import HierarchicalTeamTransformer
from vgc_team.teams.features import build_vocabs, encode_team
from vgc_team.teams.schema import Team
from vgc_team.teams.vocab import KindVocab, TokenVocab

app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class TrainingConfig:
    dataset_files: list[str]
    max_battles: int | None
    epochs: int
    early_stop_min_delta: float | None
    early_stop_patience: int
    batch_size: int
    learning_rate: float
    d_model: int
    n_heads: int
    pokemon_layers: int
    team_layers: int
    seed: int


class MaskedTeamDataset(Dataset):
    def __init__(
        self,
        teams: list[Team],
        kind_vocab: KindVocab,
        token_vocab: TokenVocab,
        *,
        seed: int = 0,
    ) -> None:
        self.examples = [
            encode_team(team, kind_vocab, token_vocab)
            for team in teams
            if team.is_full_team
        ]
        self.examples = [example for example in self.examples if example.maskable_positions]
        self.kind_vocab = kind_vocab
        self.token_vocab = token_vocab
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        value_ids = example.value_ids.clone()
        pokemon_idx, attr_idx, kind, target_value_id = self.rng.choice(
            example.maskable_positions
        )
        value_ids[pokemon_idx, attr_idx] = self.token_vocab.mask_id
        return {
            "kind_ids": example.kind_ids,
            "value_ids": value_ids,
            "attr_mask": example.attr_mask,
            "target_pokemon_idx": torch.tensor(pokemon_idx, dtype=torch.long),
            "target_kind_id": torch.tensor(self.kind_vocab.id(kind), dtype=torch.long),
            "target_value_id": torch.tensor(target_value_id, dtype=torch.long),
        }


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _relative_loss_improvement(previous_loss: float, current_loss: float) -> float:
    """Return fractional loss improvement from previous to current epoch."""

    if previous_loss <= 0:
        return 0.0
    return (previous_loss - current_loss) / previous_loss


def _write_metadata(
    output_dir: Path,
    config: TrainingConfig,
    kind_vocab: KindVocab,
    token_vocab: TokenVocab,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (output_dir / "kind_vocab.json").write_text(
        json.dumps(kind_vocab.to_json_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "token_vocab.json").write_text(
        json.dumps(token_vocab.to_json_dict(), indent=2),
        encoding="utf-8",
    )


@app.command()
def main(
    data_dir: Annotated[
        Path,
        typer.Option(help="Folder containing VGC-Bench JSON log files."),
    ] = DATA_DIR / "raw" / "vgc_bench",
    include_bo3: Annotated[
        bool,
        typer.Option(help="Use the larger Reg M-A BO3 file as well."),
    ] = False,
    download: Annotated[
        bool,
        typer.Option(help="Download missing VGC-Bench files before training."),
    ] = True,
    max_battles: Annotated[
        int | None,
        typer.Option(help="Optional cap for fast smoke runs."),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Number of passes over team examples.")] = 3,
    early_stop_min_delta: Annotated[
        float | None,
        typer.Option(
            help=(
                "Stop if relative epoch-loss improvement is below this value. "
                "Use 0.01 for a 1% threshold."
            ),
        ),
    ] = None,
    early_stop_patience: Annotated[
        int,
        typer.Option(help="Consecutive low-improvement epochs before stopping."),
    ] = 1,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 128,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 3e-4,
    d_model: Annotated[int, typer.Option(help="Transformer hidden size.")] = 128,
    n_heads: Annotated[int, typer.Option(help="Attention heads.")] = 4,
    pokemon_layers: Annotated[int, typer.Option(help="Pokemon-set transformer layers.")] = 2,
    team_layers: Annotated[int, typer.Option(help="Team transformer layers.")] = 2,
    output_dir: Annotated[
        Path,
        typer.Option(help="Where checkpoint and vocab files should be saved."),
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    filenames = [MA_FILES["ma"]]
    if include_bo3:
        filenames.append(MA_FILES["ma_bo3"])

    paths: list[Path] = []
    for filename in filenames:
        path = data_dir / filename
        if download and not path.exists():
            path = download_vgc_bench_file(filename, data_dir)
        paths.append(path)

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise typer.BadParameter(f"Missing data files: {missing}")

    console.print("Loading open-team-sheet teams...")
    teams = load_vgc_bench_teams(paths, max_battles=max_battles)
    console.print(f"Loaded {len(teams)} team-side examples.")

    kind_vocab, token_vocab = build_vocabs(teams)
    dataset = MaskedTeamDataset(teams, kind_vocab, token_vocab, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    console.print(f"Training examples: {len(dataset)}")
    console.print(f"Kinds: {len(kind_vocab.kind_to_id)} | Tokens: {len(token_vocab.token_to_id)}")

    device = _device()
    model = HierarchicalTeamTransformer(
        n_kinds=len(kind_vocab.kind_to_id),
        n_values=len(token_vocab.token_to_id),
        d_model=d_model,
        n_heads=n_heads,
        pokemon_layers=pokemon_layers,
        team_layers=team_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    config = TrainingConfig(
        dataset_files=filenames,
        max_battles=max_battles,
        epochs=epochs,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_patience=early_stop_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        d_model=d_model,
        n_heads=n_heads,
        pokemon_layers=pokemon_layers,
        team_layers=team_layers,
        seed=seed,
    )
    _write_metadata(output_dir, config, kind_vocab, token_vocab)

    model.train()
    previous_epoch_loss: float | None = None
    low_improvement_epochs = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                batch["kind_ids"],
                batch["value_ids"],
                batch["attr_mask"],
                batch["target_pokemon_idx"],
                batch["target_kind_id"],
            )
            loss = loss_fn(logits, batch["target_value_id"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == batch["target_value_id"]).sum().item()
                total_correct += correct
                total_seen += preds.numel()
                total_loss += loss.item() * preds.numel()
                progress.set_postfix(
                    loss=total_loss / max(total_seen, 1),
                    acc=total_correct / max(total_seen, 1),
                )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "kind_vocab": kind_vocab.to_json_dict(),
            "token_vocab": token_vocab.to_json_dict(),
        }
        torch.save(checkpoint, output_dir / "checkpoint.pt")

        epoch_loss = total_loss / max(total_seen, 1)
        epoch_accuracy = total_correct / max(total_seen, 1)
        console.print(
            f"epoch {epoch}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}"
        )

        if early_stop_min_delta is not None and previous_epoch_loss is not None:
            improvement = _relative_loss_improvement(previous_epoch_loss, epoch_loss)
            console.print(
                f"relative loss improvement vs previous epoch: {improvement:.4%}"
            )
            if improvement < early_stop_min_delta:
                low_improvement_epochs += 1
            else:
                low_improvement_epochs = 0

            if low_improvement_epochs >= early_stop_patience:
                console.print(
                    "Early stopping: relative loss improvement stayed below "
                    f"{early_stop_min_delta:.2%} for {early_stop_patience} epoch(s)."
                )
                break

        previous_epoch_loss = epoch_loss

    console.print(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    app()
