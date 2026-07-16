from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ExternalPreprocessedDataset(Dataset):
    def __init__(self, preprocessed_dir: str | Path, source_names: list[str]):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.track_dirs = sorted(directory for directory in self.preprocessed_dir.iterdir() if directory.is_dir())
        self.all_source_names = source_names
        self.track_metadata: list[dict[str, Any]] = []

        for track_dir in self.track_dirs:
            metadata_path = track_dir / "metadata.pt"
            if metadata_path.exists():
                metadata = torch.load(metadata_path)
            else:
                mix_data = torch.load(track_dir / "mix.pt")
                metadata = {
                    "track_name": track_dir.name,
                    "n_chunks": int(mix_data["spectrogram"].shape[0]),
                }
            self.track_metadata.append(metadata)

    def __len__(self) -> int:
        return sum(metadata["n_chunks"] for metadata in self.track_metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("ExternalPreprocessedDataset is only iterated by track directory during inference.")
