# Team Representation Learning Plan

The project has pivoted from battle-policy RL to VGC teambuilding.

The first model is a hierarchical masked-feature transformer trained on Reg M-A
open-team-sheet data from VGC-Bench.

## Representation

Each team is represented as six Pokemon-set tokens.

Each Pokemon set is represented as unordered attribute tokens:

- type tokens
- base stat tokens
- item token
- ability token
- nature token
- move tokens

Species names are kept as metadata for inspection, but the first model does not
feed a direct species-id token. This makes the representation lean toward
battle-relevant structure instead of pure species memorization.

## Architecture

```text
Pokemon attribute tokens
        |
shared Pokemon-set transformer
        |
Pokemon-set embedding

six Pokemon-set embeddings
        |
positionless team transformer
        |
contextualized Pokemon embeddings / team embedding
```

There is one shared Pokemon-set transformer, not six separate ones. It is
applied independently to each Pokemon. Then one team transformer lets the six
Pokemon interact through attention.

No Pokemon slot positional embeddings are used at the team level, so team order
does not matter.

## Pretraining Task

The first task is masked-feature prediction:

1. Pick one team.
2. Pick one Pokemon.
3. Mask one item, ability, nature, or move token.
4. Predict the original token from the rest of the Pokemon set and team.

This bootstraps both:

- individual set coherence
- team-level synergy context

It ignores win/loss labels for now.

## Data

Use VGC-Bench Reg M-A:

- `logs_gen9championsvgc2026regma.json`
- optionally `logs_gen9championsvgc2026regmabo3.json`

The data contains open team sheets, so each battle has `|showteam|` lines.

## Commands

Download only non-BO3 M-A:

```bash
source .venv/bin/activate
python scripts/download_vgc_bench_ma.py
```

Train a quick smoke model:

```bash
python scripts/train_masked_team_model.py --max-battles 200 --epochs 1
```

Train on all non-BO3 M-A:

```bash
python scripts/train_masked_team_model.py --epochs 5
```

Train until epoch-to-epoch loss improvement is below 1%, with `--epochs` acting
as a maximum cap:

```bash
python scripts/train_masked_team_model.py --epochs 100 --early-stop-min-delta 0.01
```

Include BO3:

```bash
python scripts/download_vgc_bench_ma.py --include-bo3
python scripts/train_masked_team_model.py --include-bo3 --epochs 5
```

Outputs are saved under:

```text
models/masked_team_transformer_ma/
```

This folder contains:

- `checkpoint.pt`
- `config.json`
- `kind_vocab.json`
- `token_vocab.json`

## Next Step After Training

Use `model.team_embedding(...)` to embed teams, normalize those embeddings, and
run k-nearest neighbors. That should be the first qualitative check before any
winrate modeling.

## Frozen-Encoder Extensions

The encoder is frozen after pretraining; new heads consume the fixed 128-d team
embedding. Two extensions live on top of it.

### Meta Distribution Model (MDM)

A third level of the hierarchy — Pokemon -> team -> **week** — via a permutation-
invariant, time-aware set encoder (attention pooling) over recent teams. It is
trained as **conditional masked-team prediction**: hold out one team and predict
its anchor (a point in a fixed global codebook of ~128 k-means centroids) from a
context set of other teams, each tagged with its *weeks-ago* offset. Leave-one-
out over every team turns ~10 weeks into ~50k examples.

Key regimes:

- **Mask curriculum** on the current week (often fully masked) so the model
  learns to *forecast* from past weeks, not impute from same-week neighbours;
  full-mask matches deployment (no teams from the unseen future week).
- **Relative (weeks-ago) time** so any context-window size — including a thin
  3-week, 60-teams-per-week sample — is in-distribution.
- **Variable / small context sizes** during training for the same reason.
- **Deep ensemble** (a handful of seeds) for epistemic uncertainty, which is
  large given how little temporal data exists.

At inference the model is queried at the next week with the current week fully
masked; the predicted anchor distribution is mapped onto interpretable clusters
to report per-archetype shifts. It is always benchmarked against a **persistence
baseline** (next = this week) via held-out-week KL.

Clustering is **not** the training target — the MDM predicts a team-level
distribution; per-cluster shifts are a post-hoc interpretation. Predicting
cluster-share deltas directly would collapse the dataset back to ~10 points.

### Honest scope

Within-week leave-one-out multiplies the *conditional* signal and gradient
volume, but the independent *temporal-dynamics* signal is still ~10 weeks (a
single regulation). So single-regulation training is reliable only for a thin
sample of the *same* meta it learned; forecasting a *new* meta from a thin recent
sample is the use case that needs **multi-regulation training** (the end-state
unlock). The Reg M-A VGC-Bench data also shows a structural break around week
W22 (cluster count jumps 8 -> 24), almost certainly a data-collection change.

### Team rating (idea 2)

`scripts/rate_team.py` embeds an unseen team and scores its fit to each archetype
cluster (cosine/softmax over centroids). The matchup-vs-meta head (a frozen-
encoder siamese win-probability model on the labelled p1-vs-p2 pairs) is the
deferred next step.
