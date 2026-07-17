from __future__ import annotations

import os
import pathlib
import sys

import torch


BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(BACKEND_ROOT)
os.environ["MONGODB_URI"] = ""
os.environ["MONGO_REQUIRED"] = "0"
os.environ["AUTH_REQUIRED"] = "0"
os.environ["STORAGE_BACKEND"] = "local"
print(f"pid={os.getpid()}", flush=True)

import app  # noqa: E402


def main() -> None:
    runtime = app.build_model_runtime()
    frames = 1 + runtime.chunk_samples // runtime.hop_length
    sample = torch.zeros((1, 1, runtime.n_fft // 2 + 1, frames), dtype=torch.float32)
    with torch.inference_mode():
        output = app.predict_masks(runtime.model, sample)
    print(
        f"ok input={tuple(sample.shape)} output={tuple(output.shape)} "
        f"chunk_seconds={runtime.chunk_duration:.3f}"
    )


if __name__ == "__main__":
    main()
