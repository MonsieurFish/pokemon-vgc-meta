# Pokemon VGC Teambuilding

This folder is focused on representation learning for Pokemon VGC teambuilding.

## Live demo

A static snapshot of the meta-projection dashboard — interactive Plotly charts
(current-vs-predicted scatters, matchup-advantage graph, rotatable 3D PCA) — is
hosted on GitHub Pages:
**https://monsieurfish.github.io/pokemon-vgc-meta/**

GitHub Pages serves static files only, so the interactive **team rater**, **team
completer**, and live tournament scraping (which run model inference on a Python
backend) are not available there — run the full app locally with
`python scripts/run_meta_app.py`. Regenerate the static snapshot with
`python scripts/build_static_site.py`.

The current workflow is:

1. Download open-team-sheet Reg M-A data from VGC-Bench.
2. Parse Showdown `showteam` logs into structured teams.
3. Train a hierarchical transformer with masked-feature prediction.
4. Use the learned team embeddings for k-nearest-neighbor exploration.

## Setup

Use Python 3.11. This machine has it installed as `python3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
```

## Download VGC-Bench M-A Data

```bash
source .venv/bin/activate
python scripts/download_vgc_bench_ma.py
```

To include the larger BO3 file:

```bash
python scripts/download_vgc_bench_ma.py --include-bo3
```

## Train The First Hierarchical Transformer

Smoke run:

```bash
source .venv/bin/activate
python scripts/train_masked_team_model.py --max-battles 200 --epochs 1
```

Full non-BO3 M-A run:

```bash
python scripts/train_masked_team_model.py --epochs 5
```

The model predicts masked item, ability, nature, or move tokens from the rest of
the Pokemon set and the team. To train until loss improvement drops below 1%:

```bash
python scripts/train_masked_team_model.py \
  --no-download \
  --epochs 100 \
  --early-stop-min-delta 0.01
```

See `docs/team_representation_learning.md` for the design.

## Build Team Embeddings And KNN

After training, embed each team once and build an exact nearest-neighbor table:

```bash
source .venv/bin/activate
python scripts/build_team_knn.py --no-download --include-bo3 --n-clusters 12
```

Fast smoke run:

```bash
python scripts/build_team_knn.py --no-download --max-battles 200 --k 5
```

Outputs are written to `data/processed/team_knn/`:

- `team_embeddings.npz`: raw transformer embeddings, standardized embeddings,
  feature means, feature standard deviations, and row ids.
- `team_metadata.csv`: one row per embedded team, with species/items/abilities/moves
  for inspection.
- `team_neighbors.csv`: one row per query-neighbor pair.
- `cluster_assignments.csv`: the cluster id assigned to each embedded team.
- `cluster_representatives.csv`: one real team per cluster, chosen by taking the
  cluster's mean embedding and selecting the team embedding closest to it.

The default normalization is feature-wise standardization: every embedding
dimension is centered to mean 0 and scaled to standard deviation 1 across the
teams in that run. Exact duplicate teams are collapsed by default so KNN returns
nearby variants rather than repeated copies. The `--n-clusters` value controls
how many representative teams you get.

## Meta-Shift Prediction And Team Rating

Everything below freezes the trained encoder and tacks new heads on top of the
128-d team embedding. First make sure `build_team_knn.py` has been run so
`team_embeddings.npz` carries cluster centroids and the anchor codebook.

### Live web app

An interactive web app that scrapes the most recent tournaments, runs them
through the frozen encoder + glide-to-anchor forecaster, and shows the predicted
next-week meta shifts plus an interactive **meta-space map** (which archetypes
are predicted to expand / contract):

```bash
python scripts/run_meta_app.py            # http://127.0.0.1:8000
```

First paint uses a cached regulation; click **"Fetch latest tournaments"** to
scrape live Limitless data. The λ slider re-forecasts instantly. Backend is
Flask (`src/vgc_team/app/`); the frontend is Plotly.js from CDN (no build step).

Two tabs:
- **Meta projection** — current-vs-predicted scatters (pokémon cores + team
  archetypes), a matchup-advantage directed graph (who beats whom), and a
  rotatable 3D PCA meta-space.
- **Team rater** — paste a `pokepast.es` link or a Showdown export; the matchup
  model scores your win rate against each archetype and reports a share-weighted
  aggregate meta-matchup score for the current and predicted-next meta.
- **Team completer** — paste an *incomplete* team (missing the 6th Pokémon,
  items, or moves — any level); it finds the team's worst archetype matchups and
  recommends fills, drawn from real current-meta sets, that most improve them.

### Team-vs-team win probability

An antisymmetric matchup model on the frozen embeddings predicts P(team A beats
team B) for any two teams (`logit(a,b) = f([a,b]) − f([b,a])`, so it's side-
agnostic and `P(A>A)=0.5`). Trained on labelled p1-vs-p2 battles from VGC-Bench:

```bash
python scripts/train_matchup.py --include-bo3
```

Reports validation AUC / log-loss against a Bradley-Terry strength baseline and a
coin flip. `vgc_team.meta.matchup.predict` exposes `win_probability(a, b)` and
`expected_winrate_vs(team, field, weights)` (e.g. a team's expected win rate
against the current meta, weighted by usage share).

### Meta-shift forecast (production)

The forecaster that actually beats a last-week persistence baseline
out-of-sample is **glide-to-anchor** (a Faust–Wright glide that blends last week
toward the regulation's running-mean meta) — see `scripts/eval_forecasters.py`
for the horse race across ~111 week-transitions where UCSV / glide / combination
all beat persistence. No neural net is needed to forecast; the frozen encoder is
used only to turn recent teams into weekly anchor distributions.

```bash
python scripts/forecast_meta.py                      # forecast the most recent regulation
python scripts/forecast_meta.py --teams-json <file>  # any tournament-teams JSON
```

The neural MDM (`scripts/train_multi_reg_mdm.py`, `scripts/predict_meta.py`) is
retained as a heavier alternative but does not beat glide-to-anchor honestly.

### Building blocks

Weekly meta time series + diagnostics (cluster and species panels, diversity,
drift, novelty):

```bash
python scripts/build_meta_timeseries.py --no-download
```

Falsifiable baselines the model must beat (persistence + momentum +
winners-lead-usage panel regressions):

```bash
python scripts/test_meta_patterns.py
```

Rate how an unseen team fits each archetype cluster:

```bash
python scripts/rate_team.py --paste my_team.txt
```

Train the Meta Distribution Model (MDM): a frozen-encoder set-encoder that, from
recent weeks of teams, predicts next week's anchor distribution. Trained with
leave-one-out masked-team examples (turns ~10 weeks into ~50k examples), it
backtests against a persistence baseline:

```bash
python scripts/train_mdm.py --epochs 8 --n-models 3
```

Forecast the next meta from a thin sample of recent weeks (the deployment shape),
with ensemble uncertainty:

```bash
python scripts/predict_meta.py --weeks 3 --sample-per-week 60
```

Scrape Reg M-A tournaments from Limitless and upweight them as a stronger signal
than ladder data (Limitless's `format` field is unreliable, so events are
selected by date window):

```bash
python scripts/scrape_limitless_vgc.py --dry-run            # list events
python scripts/scrape_limitless_vgc.py --source-weight 8    # scrape + save
# then merge into the analytics (upweighted):
python scripts/build_meta_timeseries.py --no-download \
  --tournament-teams data/processed/tournaments/ma_tournament_teams.json
```

See `docs/team_representation_learning.md` for the design and the honest scope
limits (a ~10-week single regulation bounds temporal signal; multi-regulation
training is the end-state unlock).

## Key Folders

- `scripts/`: command-line entry points (download, train, KNN, meta time series,
  pattern tests, MDM train/predict, team rating, tournament scraping).
- `src/vgc_team/teams/`: team schema, Showdown parser, pokepaste parser, tokenization.
- `src/vgc_team/models/`: hierarchical transformer + `frozen_encoder` (frozen
  encoder loader, embedding helpers, `ReferenceSpace` archetype geometry).
- `src/vgc_team/meta/`: meta-shift analysis — `timeseries`, `patterns`, and the
  `mdm/` meta-distribution model (dataset, model, train, predict, interpret, evaluate).
- `src/vgc_team/data/`: VGC-Bench loaders and Limitless tournament scraping.
- `docs/`: design notes.
- `models/masked_team_transformer_ma/`: frozen encoder checkpoint and vocab files.
- `models/mdm/`: trained MDM ensemble.
