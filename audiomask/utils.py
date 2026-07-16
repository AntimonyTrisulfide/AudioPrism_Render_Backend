from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    return torchaudio.load(path)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    torchaudio.save(path, audio, sample_rate)
