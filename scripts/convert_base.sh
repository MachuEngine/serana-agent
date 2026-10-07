#!/usr/bin/env bash
# Quantize Qwen3-8B (Hugging Face cache) to MLX 4bit. Takes a few minutes and ~17GB of RAM.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_PATH="${1:-Qwen/Qwen3-8B}"
OUT="${2:-models/qwen3-8b-4bit}"

uv run python -m mlx_lm convert --hf-path "$HF_PATH" --mlx-path "$OUT" -q --q-bits 4 --q-group-size 64
