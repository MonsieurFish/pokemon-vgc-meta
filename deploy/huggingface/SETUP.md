# Deploying the app to Hugging Face Spaces (free, ~16 GB RAM)

The Space is a Docker Space whose `Dockerfile` clones this public GitHub repo at
build time, so the Space itself only needs the two files in this folder
(`Dockerfile` and `README.md`).

## Web UI (no credentials to share)

1. Go to https://huggingface.co/new-space
2. **Owner**: your account. **Space name**: `vgc-meta-forecaster`.
3. **SDK**: choose **Docker** → **Blank**. **Hardware**: `CPU basic` (free).
   Visibility: Public.
4. Create the Space, then add two files (Files → Add file → Create/Upload):
   - `README.md`  ← copy from `deploy/huggingface/README.md`
   - `Dockerfile` ← copy from `deploy/huggingface/Dockerfile`
5. The Space builds automatically (a few minutes — mostly installing torch) and
   goes live at `https://huggingface.co/spaces/<you>/vgc-meta-forecaster`.

To pick up later GitHub changes: on the Space, **Settings → Factory rebuild**
(the Dockerfile re-clones the repo).

## Updating the app

Push changes to GitHub `main`, then Factory-rebuild the Space (or edit the
`CACHEBUST` arg in the Dockerfile to force a fresh clone).
