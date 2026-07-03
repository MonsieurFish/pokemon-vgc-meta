"""Train a team-vs-team win-probability model on frozen team embeddings.

    python scripts/train_matchup.py                 # Reg M-A ladder battles
    python scripts/train_matchup.py --include-bo3    # + the larger BO3 file

Embeds every battle team once through the frozen encoder, then trains an
antisymmetric matchup head. Reports val AUC / accuracy / log-loss against a
Bradley-Terry strength baseline and a coin-flip, so we can see whether the model
learns real matchups and whether interactions beat raw team strength.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from rich.console import Console
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vgc_team.config import DATA_DIR, PROJECT_ROOT
from vgc_team.data.vgc_bench import MA_FILES
from vgc_team.meta.matchup.data import extract_matchup_pairs
from vgc_team.meta.matchup.model import MatchupModel
from vgc_team.models.frozen_encoder import embed_teams, load_frozen_encoder, select_device

app = typer.Typer(add_completion=False)
console = Console()


def _train(model, A_tr, B_tr, y_tr, *, epochs, batch_size, lr, wd, device):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(A_tr, B_tr, y_tr), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for a, b, y in loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            loss = loss_fn(model.logit(a, b), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _evaluate(model, A, B, y, device) -> dict:
    with torch.no_grad():
        logits = model.logit(A.to(device), B.to(device)).cpu()
    probs = torch.sigmoid(logits).numpy()
    yv = y.numpy()
    logloss = float(nn.functional.binary_cross_entropy(torch.tensor(probs), torch.tensor(yv)))
    return {
        "auc": float(roc_auc_score(yv, probs)),
        "accuracy": float(((probs > 0.5) == (yv > 0.5)).mean()),
        "log_loss": logloss,
        "brier": float(np.mean((probs - yv) ** 2)),
    }


@app.command()
def main(
    checkpoint_path: Annotated[
        Path, typer.Option(help="Frozen encoder checkpoint.")
    ] = PROJECT_ROOT / "models" / "masked_team_transformer_ma" / "checkpoint.pt",
    data_dir: Annotated[Path, typer.Option()] = DATA_DIR / "raw" / "vgc_bench",
    include_bo3: Annotated[bool, typer.Option(help="Include the larger BO3 battle file.")] = False,
    max_battles: Annotated[int | None, typer.Option(help="Cap for a fast run.")] = None,
    epochs: Annotated[int, typer.Option()] = 40,
    batch_size: Annotated[int, typer.Option()] = 256,
    lr: Annotated[float, typer.Option()] = 1e-3,
    weight_decay: Annotated[float, typer.Option()] = 1e-4,
    hidden: Annotated[int, typer.Option()] = 256,
    dropout: Annotated[float, typer.Option()] = 0.2,
    val_frac: Annotated[float, typer.Option()] = 0.15,
    seed: Annotated[int, typer.Option()] = 0,
    output_dir: Annotated[Path, typer.Option()] = PROJECT_ROOT / "models" / "matchup",
) -> None:
    paths = [data_dir / MA_FILES["ma"]]
    if include_bo3:
        paths.append(data_dir / MA_FILES["ma_bo3"])

    console.print("Extracting matchup pairs...")
    teams, pairs = extract_matchup_pairs(paths, max_battles=max_battles)
    console.print(f"{len(pairs)} labelled matchups over {len(teams)} team instances "
                  f"(p1 win rate {np.mean([p[2] for p in pairs]):.3f}).")

    device = select_device()
    model_enc, kind_vocab, token_vocab, _ = load_frozen_encoder(checkpoint_path, device)
    console.print("Embedding teams...")
    raw = embed_teams(model_enc, teams, kind_vocab, token_vocab, device=device, show_progress=True)
    mean, std = raw.mean(0), raw.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    emb = ((raw - mean) / std).astype(np.float32)

    pairs = np.array(pairs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs))
    n_val = int(len(pairs) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def tensors(idx):
        p = pairs[idx]
        return (torch.from_numpy(emb[p[:, 0]]), torch.from_numpy(emb[p[:, 1]]),
                torch.from_numpy(p[:, 2].astype(np.float32)))

    A_tr, B_tr, y_tr = tensors(tr_idx)
    A_va, B_va, y_va = tensors(val_idx)
    console.print(f"train {len(tr_idx)} / val {len(val_idx)}")

    common = dict(epochs=epochs, batch_size=batch_size, lr=lr, wd=weight_decay, device=device)
    torch.manual_seed(seed)
    interact = _train(MatchupModel(emb.shape[1], hidden, dropout, interactions=True),
                      A_tr, B_tr, y_tr, **common)
    torch.manual_seed(seed)
    strength = _train(MatchupModel(emb.shape[1], interactions=False), A_tr, B_tr, y_tr, **common)

    ev_i = _evaluate(interact, A_va, B_va, y_va, device)
    ev_s = _evaluate(strength, A_va, B_va, y_va, device)
    coin = float(np.log(2))
    console.print("\n[bold]Validation (higher AUC / lower log-loss = better)[/bold]")
    console.print(f"  {'model':<22}{'AUC':>8}{'acc':>8}{'log_loss':>10}{'brier':>8}")
    console.print(f"  {'coin flip':<22}{0.5:>8.3f}{max(np.mean(y_va.numpy()),1-np.mean(y_va.numpy())):>8.3f}"
                  f"{coin:>10.4f}{0.25:>8.3f}")
    console.print(f"  {'strength (Bradley-Terry)':<22}{ev_s['auc']:>8.3f}{ev_s['accuracy']:>8.3f}"
                  f"{ev_s['log_loss']:>10.4f}{ev_s['brier']:>8.3f}")
    console.print(f"  {'+ interactions (MLP)':<22}{ev_i['auc']:>8.3f}{ev_i['accuracy']:>8.3f}"
                  f"{ev_i['log_loss']:>10.4f}{ev_i['brier']:>8.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": interact.state_dict(), "mean": mean.astype(np.float32),
                "std": std.astype(np.float32), "d_in": emb.shape[1], "hidden": hidden,
                "dropout": dropout}, output_dir / "matchup.pt")
    console.print(f"\nSaved matchup model to {output_dir / 'matchup.pt'}")


if __name__ == "__main__":
    app()
