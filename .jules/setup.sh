#!/usr/bin/env bash
set -eo pipefail

echo "==> Setting up environment for Jules AI..."

# Install system compilation packages and container tools
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    podman \
    libgomp1 \
    ca-certificates

# Upgrade pip and install linting/testing harnesses
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install ruff pytest pytest-mock datasets pydantic transformers requests streamlit huggingface-hub

# Install local repo in editable mode if pyproject.toml exists
python3 -m pip install -e . 2>/dev/null || true

echo "==> Setup completed successfully."
