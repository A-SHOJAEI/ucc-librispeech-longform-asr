from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio_mono(path: str | Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """
    Returns (waveform [T] float32, sample_rate).

    Uses soundfile for broad codec support (including FLAC on most systems).
    """
    path = Path(path)
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # [T, C] -> mono
    wav = wav.mean(axis=1)
    x = torch.from_numpy(np.ascontiguousarray(wav))
    if sr != target_sr:
        x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return x, sr


def concat_with_silence(wavs: list[torch.Tensor], silence_sec: float, sr: int) -> torch.Tensor:
    if not wavs:
        return torch.zeros(0, dtype=torch.float32)
    if silence_sec <= 0:
        return torch.cat(wavs, dim=0)
    pad = torch.zeros(int(round(silence_sec * sr)), dtype=torch.float32)
    parts: list[torch.Tensor] = []
    for i, w in enumerate(wavs):
        if i > 0:
            parts.append(pad)
        parts.append(w)
    return torch.cat(parts, dim=0)

