from __future__ import annotations

import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T

from audiomask.utils import load_audio


class ExternalPreprocessor:
    def __init__(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        chunk_duration: float = 4.0,
        sr: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration_seconds: float | None = None,
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sr = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration_seconds = max_duration_seconds
        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

    def _validate_duration_before_load(self) -> None:
        if not self.max_duration_seconds:
            return
        try:
            metadata = torchaudio.info(self.input_file)
        except Exception:
            return
        if not metadata.sample_rate or metadata.num_frames <= 0:
            return
        duration = metadata.num_frames / metadata.sample_rate
        if duration > self.max_duration_seconds:
            raise ValueError(
                f"Audio is {duration:.1f}s long; maximum allowed is {self.max_duration_seconds:.1f}s."
            )

    def _load_audio(self) -> torch.Tensor:
        self._validate_duration_before_load()
        mix, sample_rate = load_audio(self.input_file)
        if self.max_duration_seconds and sample_rate and mix.shape[1] / sample_rate > self.max_duration_seconds:
            duration = mix.shape[1] / sample_rate
            raise ValueError(
                f"Audio is {duration:.1f}s long; maximum allowed is {self.max_duration_seconds:.1f}s."
            )
        if sample_rate != self.sr:
            mix = T.Resample(sample_rate, self.sr)(mix)
        if mix.shape[0] > 1:
            mix = mix.mean(dim=0, keepdim=True)
        return mix

    def preprocess(self) -> Path:
        track_output_dir = self.output_dir / self.input_file.stem
        track_output_dir.mkdir(exist_ok=True)

        n_chunks = 0
        spec_shape: list[int] | None = None
        total_samples = 0
        for index, magnitude, phase, valid_samples in self._iter_specs_from_file():
            spec_shape = list(magnitude.shape)
            torch.save(
                {
                    "spectrogram": magnitude,
                    "phase": phase,
                    "valid_samples": valid_samples,
                    "track_name": self.input_file.stem,
                    "chunk_index": index,
                    "type": "mix_chunk",
                },
                track_output_dir / f"chunk_{index:05d}.pt",
            )
            n_chunks += 1
            total_samples += valid_samples

        torch.save(
            {
                "track_name": self.input_file.stem,
                "n_chunks": n_chunks,
                "spec_shape": spec_shape or [],
                "mix_path": str(self.input_file),
                "sr": self.sr,
                "chunk_samples": self.chunk_samples,
                "total_samples": total_samples,
            },
            track_output_dir / "metadata.pt",
        )
        return track_output_dir

    def _prepare_chunk(self, audio: torch.Tensor, sample_rate: int, expected_valid_samples: int | None = None) -> tuple[torch.Tensor, int]:
        if sample_rate != self.sr:
            audio = T.Resample(sample_rate, self.sr)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if expected_valid_samples is None:
            valid_samples = min(int(audio.shape[1]), self.chunk_samples)
        else:
            valid_samples = min(max(0, int(expected_valid_samples)), self.chunk_samples)
            if audio.shape[1] < valid_samples:
                valid_samples = int(audio.shape[1])

        if audio.shape[1] < self.chunk_samples:
            audio = torch.nn.functional.pad(audio, (0, self.chunk_samples - int(audio.shape[1])))
        else:
            audio = audio[:, : self.chunk_samples]

        return audio, valid_samples

    def _iter_audio_chunks_from_known_length(self, sample_rate: int, total_frames: int):
        total_target_samples = max(1, math.ceil(total_frames * self.sr / sample_rate))
        n_chunks = max(1, math.ceil(total_target_samples / self.chunk_samples))

        for index in range(n_chunks):
            target_start = index * self.chunk_samples
            target_end = min((index + 1) * self.chunk_samples, total_target_samples)
            source_start = math.floor(target_start * sample_rate / self.sr)
            source_end = math.ceil(target_end * sample_rate / self.sr)
            num_frames = max(1, source_end - source_start)
            audio, loaded_sample_rate = torchaudio.load(
                self.input_file,
                frame_offset=source_start,
                num_frames=num_frames,
            )
            expected_valid_samples = target_end - target_start
            yield index, *self._prepare_chunk(audio, loaded_sample_rate, expected_valid_samples)

    def _iter_audio_chunks_from_full_load(self):
        audio = self._load_audio()
        total_samples = int(audio.shape[1])
        n_chunks = max(1, math.ceil(total_samples / self.chunk_samples))
        for index in range(n_chunks):
            start = index * self.chunk_samples
            end = min(start + self.chunk_samples, total_samples)
            chunk = audio[:, start:end]
            yield index, *self._prepare_chunk(chunk, self.sr)

    def _iter_audio_chunks(self):
        self._validate_duration_before_load()
        try:
            metadata = torchaudio.info(self.input_file)
        except Exception:
            metadata = None

        if metadata and metadata.sample_rate and metadata.num_frames > 0:
            try:
                yield from self._iter_audio_chunks_from_known_length(metadata.sample_rate, metadata.num_frames)
                return
            except Exception:
                pass

        yield from self._iter_audio_chunks_from_full_load()

    def _iter_specs_from_file(self):
        for index, chunk, valid_samples in self._iter_audio_chunks():
            complex_spec = self.stft(chunk)
            yield index, torch.abs(complex_spec[0]), torch.angle(complex_spec[0]), valid_samples

    def _iter_specs(self, audio: torch.Tensor):
        total_samples = int(audio.shape[1])
        n_chunks = max(1, math.ceil(total_samples / self.chunk_samples))

        for index in range(n_chunks):
            start = index * self.chunk_samples
            end = min(start + self.chunk_samples, total_samples)
            chunk = audio[:, start:end]
            valid_samples = int(chunk.shape[1])
            if valid_samples < self.chunk_samples:
                chunk = torch.nn.functional.pad(chunk, (0, self.chunk_samples - valid_samples))

            complex_spec = self.stft(chunk)
            yield index, torch.abs(complex_spec[0]), torch.angle(complex_spec[0]), valid_samples

    def _audio_to_specs(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        specs = []
        true_phases = []
        for _index, magnitude, phase, _valid_samples in self._iter_specs(audio):
            specs.append(magnitude)
            true_phases.append(phase)

        return torch.stack(specs), torch.stack(true_phases)
