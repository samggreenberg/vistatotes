#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Upgrade setuptools first — the system version (68.x) has a broken
# install_layout attribute that prevents building wheels for some packages
# (progressbar, wget) needed by laion_clap.
pip install --upgrade setuptools -q

# Install Python dependencies (CPU-only PyTorch).
# --ignore-installed blinker: the system blinker (debian-managed) has no
# RECORD file so pip cannot uninstall it; force-installing a fresh copy
# lets Flask pick it up cleanly.
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  flask \
  "numpy<2" \
  "torch>=2.0.0" \
  requests \
  tqdm \
  scikit-learn \
  transformers \
  laion_clap \
  librosa \
  "opencv-python-headless<4.10" \
  Pillow \
  ultralytics \
  sentence-transformers \
  --ignore-installed blinker \
  -q

# Install dev tools (linter + test runner)
pip install pytest ruff -q
