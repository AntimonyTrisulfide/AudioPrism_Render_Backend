#!/bin/sh
set -eu

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_FILENAME="${MODEL_FILENAME:-model_render.pth}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_FILENAME}"
PORT="${PORT:-8001}"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ] && [ -n "${MODEL_URL:-}" ]; then
  echo "Downloading model from MODEL_URL to $MODEL_PATH"
  curl -L --fail --retry 3 --retry-delay 5 -o "$MODEL_PATH" "$MODEL_URL"
fi

exec uvicorn app:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips "*"
