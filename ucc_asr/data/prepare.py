from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch

from .manifest import ManifestItem, write_jsonl


def _generate_smoke_data(cfg: Dict[str, Any]) -> None:
    """Generate a tiny synthetic audio dataset for pipeline validation."""
    smoke_cfg = cfg["data"]["smoke"]
    root = Path(smoke_cfg["root"])
    sr = int(smoke_cfg.get("sample_rate", 16000))
    clip_sec = float(smoke_cfg.get("clip_seconds", 1.2))
    vocab = smoke_cfg.get("vocab", ["hello", "world", "speech", "test"])
    n_train = int(smoke_cfg.get("n_train", 24))
    n_dev = int(smoke_cfg.get("n_dev", 8))
    n_test = int(smoke_cfg.get("n_test", 8))

    rng = random.Random(42)
    samples = int(sr * clip_sec)

    for split_name, n_items in [("train", n_train), ("dev", n_dev), ("test", n_test)]:
        audio_dir = root / "audio" / split_name
        audio_dir.mkdir(parents=True, exist_ok=True)
        items: List[ManifestItem] = []
        for i in range(n_items):
            # Generate synthetic audio: tones + noise.
            t = np.linspace(0, clip_sec, samples, dtype=np.float32)
            freq = rng.uniform(200, 800)
            wav = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(samples).astype(np.float32)
            wav = np.clip(wav, -1.0, 1.0)

            audio_path = audio_dir / f"{split_name}_{i:04d}.wav"
            sf.write(str(audio_path), wav, sr)

            n_words = rng.randint(1, 3)
            transcript = " ".join(rng.choice(vocab) for _ in range(n_words))

            items.append(ManifestItem(
                audio_path=str(audio_path.resolve()),
                transcript=transcript,
                duration=clip_sec,
                extra={},
            ))

        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest_dir / f"{split_name}.jsonl", items)


def prepare_data(cfg: Dict[str, Any]) -> None:
    kind = cfg["data"]["kind"]
    if kind == "smoke":
        _generate_smoke_data(cfg)
    elif kind == "librispeech":
        # LibriSpeech download/extraction is handled by separate scripts.
        pass
    else:
        raise ValueError(f"Unknown data.kind: {kind}")
