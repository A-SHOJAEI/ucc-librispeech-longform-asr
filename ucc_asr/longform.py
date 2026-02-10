from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

from .audio import concat_with_silence, load_audio_mono
from .data.manifest import ManifestItem
from .text import words


@dataclass(frozen=True)
class LongformExample:
    id: str
    waveform: torch.Tensor
    sample_rate: int
    transcript: str
    seam_word_indices: List[int]  # seam after word i


def make_longform_examples(
    items: List[ManifestItem],
    k: int,
    seed: int,
    target_sr: int,
    boundary_silence_sec: float = 0.0,
    max_examples: int | None = None,
) -> List[LongformExample]:
    """
    Creates synthetic long-form examples by concatenating K utterances.
    seam_word_indices are computed in reference word space.
    """
    if k <= 1:
        raise ValueError("k must be >= 2 for long-form concatenation")
    rng = random.Random(seed)
    if not items:
        return []

    # Heuristic: build about len(items)//k examples unless max_examples is set.
    n = max(1, len(items) // k)
    if max_examples is not None:
        n = min(n, int(max_examples))

    out: List[LongformExample] = []
    for ex_i in range(n):
        chosen = [items[rng.randrange(0, len(items))] for _ in range(k)]
        wavs = []
        texts: List[str] = []
        seam_word_indices: List[int] = []
        word_count = 0
        for idx, it in enumerate(chosen):
            w, _ = load_audio_mono(it.audio_path, target_sr)
            wavs.append(w)
            t = it.transcript.strip()
            texts.append(t)
            # Seam after this utterance (except last): seam is after last word of current ref.
            if idx < k - 1:
                wc = len(words(t))
                word_count += wc
                seam_word_indices.append(max(0, word_count - 1))
            else:
                word_count += len(words(t))

        waveform = concat_with_silence(wavs, boundary_silence_sec, target_sr)
        ref = " ".join(texts).strip()
        out.append(
            LongformExample(
                id=f"longform-{ex_i:04d}",
                waveform=waveform,
                sample_rate=target_sr,
                transcript=ref,
                seam_word_indices=seam_word_indices,
            )
        )
    return out

