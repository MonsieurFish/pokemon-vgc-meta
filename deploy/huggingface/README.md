---
title: VGC Meta Forecaster
emoji: 🔮
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VGC Meta Forecaster

Interactive Pokémon VGC metagame forecaster and team tools — a frozen
hierarchical team encoder, a glide-to-anchor meta forecaster, and an
antisymmetric matchup model, served as a Flask + Plotly web app with three tabs
(meta projection, team rater, team completer).

This Space builds from the public repo
**https://github.com/MonsieurFish/pokemon-vgc-meta** (see its README for the
method). The first request loads the frozen encoder and embeds the cached meta
(~30–60 s); subsequent requests are fast.
