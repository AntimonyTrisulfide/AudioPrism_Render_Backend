from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strip an AudioMask training checkpoint down to the tensors and metadata "
            "needed by the Render inference API."
        )
    )
    parser.add_argument("input", type=Path, help="HPC/training checkpoint, for example epoch_0100.pth")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("models/model_render.pth"),
        help="Deployment checkpoint path. Default: models/model_render.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    if not isinstance(checkpoint, dict):
        payload = checkpoint
    else:
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint.get("model")
        if state_dict is None:
            state_dict = checkpoint
        payload = {
            "model_state": state_dict,
            "train_metadata": checkpoint.get("train_metadata", {}),
            "config": checkpoint.get("config", {}),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved Render checkpoint to {args.output}")


if __name__ == "__main__":
    main()
