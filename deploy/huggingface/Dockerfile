# Hugging Face Space (Docker SDK) for the VGC Meta Forecaster.
#
# The application, frozen encoder, and cached meta data are pulled from the
# public GitHub repo at build time, so this Space is just this Dockerfile + a
# README.md. To update the deployed app after pushing to GitHub, hit
# "Factory rebuild" on the Space (or bump the CACHEBUST arg below).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONWARNINGS=ignore
WORKDIR /home/user/app

# Pull the application (public repo). Bump CACHEBUST to force a fresh clone.
ARG CACHEBUST=1
RUN git clone --depth 1 https://github.com/MonsieurFish/pokemon-vgc-meta.git .

# CPU-only torch (the default PyPI wheel bundles CUDA, ~2.5 GB) + runtime deps
# + the vgc_team package.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-render.txt \
    && pip install --no-cache-dir -e . --no-deps

# HF Spaces expect the app on port 7860.
EXPOSE 7860
CMD ["gunicorn", "vgc_team.app.server:app", \
     "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
