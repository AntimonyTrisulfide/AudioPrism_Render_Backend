# syntax = docker/dockerfile:1.5

FROM python:3.11-slim AS base

# Keep the runtime predictable on small Render instances.
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PYTHONMALLOC=malloc \
	MALLOC_ARENA_MAX=1 \
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	TORCH_NUM_THREADS=1 \
	TORCH_NUM_INTEROP_THREADS=1 \
	MODEL_FILENAME=model_render.pth \
	PORT=8001 \
	MAX_UPLOAD_MB=25 \
	MAX_AUDIO_SECONDS=0 \
	INFERENCE_CHUNK_SECONDS=0.5 \
	INFERENCE_FREQ_TILE_BINS=256 \
	INFERENCE_FREQ_OVERLAP_BINS=64 \
	OUTPUT_TTL_MINUTES=120

RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl ca-certificates libsndfile1 ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /
ENV MODEL_DIR=/models
RUN mkdir -p "${MODEL_DIR}"

# Install Python dependencies before copying the rest of the app for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy FastAPI source code
COPY . /


FROM base AS final
ENV MODEL_DIR=/models
WORKDIR /

EXPOSE 8001

CMD ["sh","scripts/start_render.sh"]
