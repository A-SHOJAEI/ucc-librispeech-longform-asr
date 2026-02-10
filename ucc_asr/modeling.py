from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


@dataclass(frozen=True)
class ModelBundle:
    processor: Wav2Vec2Processor
    model: Wav2Vec2ForCTC


def _waveform_to_1d_numpy(w: torch.Tensor) -> np.ndarray:
    """
    Transformers' Wav2Vec2* feature extractors reliably handle:
      - a single example as 1D array-like [T]
      - a batch as list of 1D arrays

    Passing a list of torch tensors can be misinterpreted as multi-channel audio and
    produce input_values shaped [1, B, T] (channel-first) instead of [B, T].
    Convert each example to a 1D NumPy array to avoid that ambiguity.
    """
    if not isinstance(w, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor waveform, got {type(w)}")
    w = w.detach().to(device="cpu")
    w = w.squeeze()
    if w.ndim == 2:
        # Heuristic downmix: treat the smaller dimension as channels.
        if w.shape[0] <= 8 and w.shape[1] > w.shape[0]:
            w = w.mean(dim=0)
        elif w.shape[1] <= 8 and w.shape[0] > w.shape[1]:
            w = w.mean(dim=1)
        else:
            raise ValueError(f"Ambiguous 2D waveform shape {tuple(w.shape)}; expected [T] or [C,T]/[T,C].")
    if w.ndim != 1:
        raise ValueError(f"Expected 1D waveform after squeeze/downmix, got shape {tuple(w.shape)}")
    w = w.to(dtype=torch.float32).contiguous()
    return w.numpy()


def load_wav2vec2_ctc(pretrained_name: str, cfg: Dict[str, Any]) -> ModelBundle:
    processor = Wav2Vec2Processor.from_pretrained(pretrained_name)
    model = Wav2Vec2ForCTC.from_pretrained(pretrained_name)

    # Optional masking hyperparams (SpecAugment-style on latent features).
    mask_time_prob = cfg.get("mask_time_prob", None)
    mask_time_length = cfg.get("mask_time_length", None)
    if mask_time_prob is not None:
        model.config.mask_time_prob = float(mask_time_prob)
    if mask_time_length is not None:
        model.config.mask_time_length = int(mask_time_length)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.gradient_checkpointing = True

    return ModelBundle(processor=processor, model=model)


def prepare_batch(
    processor: Wav2Vec2Processor,
    waveforms: list[torch.Tensor],
    transcripts: list[str],
    sampling_rate: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if len(waveforms) != len(transcripts):
        # Defensive: if a pre-collated batch tensor slips through, split it.
        if len(waveforms) == 1 and isinstance(waveforms[0], torch.Tensor) and waveforms[0].ndim == 2:
            w0 = waveforms[0]
            if w0.shape[0] == len(transcripts):
                waveforms = [w0[i] for i in range(w0.shape[0])]
            else:
                raise ValueError(
                    f"Batch size mismatch: {len(waveforms)=} vs {len(transcripts)=} with waveform[0].shape={tuple(w0.shape)}"
                )
        else:
            raise ValueError(f"Batch size mismatch: {len(waveforms)=} vs {len(transcripts)=}")

    # Inputs.
    wav_np = [_waveform_to_1d_numpy(w) for w in waveforms]
    inputs = processor(
        wav_np,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    # Targets.
    with processor.as_target_processor():
        labels = processor(transcripts, return_tensors="pt", padding=True).input_ids
    # Transformers expects -100 for masked label padding.
    labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)
    return dict(inputs), labels
