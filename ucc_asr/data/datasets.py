from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from ..audio import concat_with_silence, load_audio_mono
from .manifest import ManifestItem


@dataclass(frozen=True)
class UCCSpec:
    enabled: bool = False
    target_n: int = 1
    schedule: str = "fixed"       # "fixed" | "ramp"
    rampup_fraction: float = 0.0
    boundary_silence_sec: float = 0.0


class UtteranceDataset(Dataset):
    """Map-style dataset returning single utterances."""

    def __init__(self, items: List[ManifestItem], target_sr: int) -> None:
        self.items = items
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        wav, _ = load_audio_mono(it.audio_path, self.target_sr)
        return {"waveform": wav, "transcript": it.transcript, "duration": it.duration}


class RandomConcatIterableDataset(IterableDataset):
    """
    Iterable dataset that yields single utterances or random concatenations (UCC).

    Each iteration returns `max_steps * batch_size` items so that the outer DataLoader
    sees enough samples to fill the requested number of steps.
    """

    def __init__(
        self,
        items: List[ManifestItem],
        target_sr: int,
        ucc: UCCSpec,
        max_steps: int,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.target_sr = target_sr
        self.ucc = ucc
        self.max_steps = max_steps
        self.seed = seed

    def _n_for_step(self, step: int) -> int:
        if not self.ucc.enabled or self.ucc.target_n <= 1:
            return 1
        if self.ucc.schedule == "fixed":
            return self.ucc.target_n
        # Ramp schedule.
        ramp_end = int(self.max_steps * self.ucc.rampup_fraction)
        if ramp_end <= 0 or step >= ramp_end:
            return self.ucc.target_n
        frac = step / float(ramp_end)
        return max(1, int(1 + frac * (self.ucc.target_n - 1)))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)
        # Produce enough samples for max_steps (assuming batch_size 1; DataLoader batches them).
        for step in range(self.max_steps * 4):  # generous buffer
            n = self._n_for_step(step)
            if n <= 1:
                it = self.items[rng.randrange(len(self.items))]
                wav, _ = load_audio_mono(it.audio_path, self.target_sr)
                yield {"waveform": wav, "transcript": it.transcript}
            else:
                chosen = [self.items[rng.randrange(len(self.items))] for _ in range(n)]
                wavs = []
                texts = []
                for it in chosen:
                    w, _ = load_audio_mono(it.audio_path, self.target_sr)
                    wavs.append(w)
                    texts.append(it.transcript.strip())
                waveform = concat_with_silence(wavs, self.ucc.boundary_silence_sec, self.target_sr)
                transcript = " ".join(texts)
                yield {"waveform": waveform, "transcript": transcript}
