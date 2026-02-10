from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, load_experiments
from .data.collate import collate_keep_lists
from .data.datasets import UtteranceDataset
from .data.manifest import read_jsonl
from .longform import make_longform_examples
from .metrics import cer, seam_local_error, wer
from .modeling import load_wav2vec2_ctc, prepare_batch
from .text import normalize_text
from .utils import device_from_config, now_utc_iso, set_seed


def _manifest_items(cfg: Dict[str, Any], split_key: str):
    kind = cfg["data"]["kind"]
    if kind == "smoke":
        root = Path(cfg["data"]["smoke"]["root"])
        mf = root / "manifests" / f"{split_key}.jsonl"
        return read_jsonl(mf)
    if kind == "librispeech":
        mf = Path(cfg["data"]["librispeech"]["manifests_dir"]) / f"{split_key}.jsonl"
        return read_jsonl(mf)
    raise ValueError(f"Unknown data.kind={kind}")


@torch.no_grad()
def _decode_split(
    model,
    processor,
    device: torch.device,
    items,
    sr: int,
    batch_size: int,
    num_workers: int,
    max_batches: int | None = None,
) -> Tuple[List[str], List[str], float, float]:
    ds = UtteranceDataset(items, target_sr=sr)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_keep_lists,
    )

    model.eval()
    refs: List[str] = []
    hyps: List[str] = []
    audio_sec = 0.0
    t0 = time.perf_counter()
    for b_i, batch in enumerate(dl):
        if max_batches is not None and b_i >= max_batches:
            break
        wavs = [w.to(torch.float32) for w in batch["waveform"]]
        texts = [normalize_text(t) for t in batch["transcript"]]
        for w in wavs:
            audio_sec += float(w.numel()) / float(sr)
        inputs, _labels = prepare_batch(processor, wavs, texts, sampling_rate=sr)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)
        refs.extend(texts)
        hyps.extend([normalize_text(x) for x in pred])
    t1 = time.perf_counter()
    wall = max(1e-9, t1 - t0)
    rtf = float(wall / max(1e-9, audio_sec))
    return refs, hyps, audio_sec, rtf


@torch.no_grad()
def _decode_longform(
    model,
    processor,
    device: torch.device,
    examples,
    sr: int,
    batch_size: int,
    seam_window_words: int,
) -> Dict[str, Any]:
    model.eval()
    refs: List[str] = []
    hyps: List[str] = []
    seam_counts = []

    # Batch examples by padding waveforms.
    audio_sec = 0.0
    t0 = time.perf_counter()
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        wavs = [ex.waveform.to(torch.float32) for ex in chunk]
        texts = [normalize_text(ex.transcript) for ex in chunk]
        for w in wavs:
            audio_sec += float(w.numel()) / float(sr)
        # Use the shared batch-prep helper to avoid HF Processor ambiguity with list[torch.Tensor].
        inputs, _labels = prepare_batch(processor, wavs, texts, sampling_rate=sr)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)
        for ex, ref, hyp in zip(chunk, texts, pred):
            hyp_n = normalize_text(hyp)
            refs.append(ref)
            hyps.append(hyp_n)
            seam_counts.append(seam_local_error(ref, hyp_n, ex.seam_word_indices, window_words=seam_window_words))
    t1 = time.perf_counter()
    wall = max(1e-9, t1 - t0)
    rtf = float(wall / max(1e-9, audio_sec))

    # Aggregate seam-local metrics.
    subs = sum(c.substitutions for c in seam_counts)
    dels = sum(c.deletions for c in seam_counts)
    ins = sum(c.insertions for c in seam_counts)
    refw = sum(c.ref_words for c in seam_counts)
    seam_err = float((subs + dels + ins) / refw) if refw > 0 else 0.0

    return {
        "n_examples": len(examples),
        "audio_seconds": audio_sec,
        "rtf": rtf,
        "wer": float(wer("\n".join(refs), "\n".join(hyps))) if refs else 1.0,
        "cer": float(cer("\n".join(refs), "\n".join(hyps))) if refs else 1.0,
        "seam_local": {
            "window_words": seam_window_words,
            "ref_words": refw,
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "error_rate": seam_err,
        },
    }


def eval_one(exp: ExperimentConfig) -> Dict[str, Any]:
    cfg = exp.cfg
    device = device_from_config(cfg.get("train", {}).get("device", "auto") if "train" in cfg else "auto")
    seed = int(cfg.get("train", {}).get("seed", 0)) if "train" in cfg else 0
    set_seed(seed)

    out_dir = Path(cfg["project"]["output_dir"]) / exp.name
    ckpt = out_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        ckpt = out_dir / "checkpoints" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {exp.name}: {out_dir}/checkpoints/(best.pt|last.pt)")

    bundle = load_wav2vec2_ctc(cfg["model"]["pretrained_name"], cfg.get("model", {}))
    processor, model = bundle.processor, bundle.model
    st = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(st["model"], strict=True)
    model.to(device)
    train_cfg = st.get("config", {}) if isinstance(st, dict) else {}

    sr = int(cfg["data"].get("sample_rate", 16000))
    if cfg["data"]["kind"] == "smoke":
        sr = int(cfg["data"]["smoke"].get("sample_rate", sr))

    batch_size = int(cfg.get("train", {}).get("batch_size", 2))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))

    results: Dict[str, Any] = {
        "name": exp.name,
        "checkpoint": str(ckpt),
        "evaluated_at": now_utc_iso(),
        "device": str(device),
        "splits": {},
    }

    # Standard split evaluation.
    if cfg["data"]["kind"] == "smoke":
        split_keys = ["dev", "test"]
    else:
        split_keys = ["dev_clean", "dev_other", "test_clean", "test_other"]

    for sk in split_keys:
        items = _manifest_items(cfg, sk)
        refs, hyps, audio_sec, rtf = _decode_split(
            model=model,
            processor=processor,
            device=device,
            items=items,
            sr=sr,
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=2 if cfg["data"]["kind"] == "smoke" else None,
        )
        results["splits"][sk] = {
            "n_utterances": len(refs),
            "audio_seconds": audio_sec,
            "rtf": rtf,
            "wer": float(wer("\n".join(refs), "\n".join(hyps))) if refs else 1.0,
            "cer": float(cer("\n".join(refs), "\n".join(hyps))) if refs else 1.0,
        }

    # Long-form synthetic evaluation.
    lf_cfg = cfg.get("eval", {}).get("longform", {})
    if bool(lf_cfg.get("enabled", True)):
        k = int(lf_cfg.get("k", 8))
        seam_window_words = int(lf_cfg.get("seam_window_words", 3))
        boundary_silence_sec = 0.0
        # Prefer the training boundary setting stored in the checkpoint.
        ucc_cfg = (train_cfg.get("train", {}) if isinstance(train_cfg, dict) else {}).get("ucc", {})
        if isinstance(ucc_cfg, dict) and bool(ucc_cfg.get("enabled", False)):
            boundary_silence_sec = float(ucc_cfg.get("boundary_silence_sec", 0.0))

        test_items = _manifest_items(cfg, "test" if cfg["data"]["kind"] == "smoke" else "test_clean")
        examples = make_longform_examples(
            items=test_items,
            k=k,
            seed=seed + 123,
            target_sr=sr,
            boundary_silence_sec=boundary_silence_sec,
            max_examples=4 if cfg["data"]["kind"] == "smoke" else None,
        )

        # Patch seam window into decoder result (seam_local_error currently uses fixed window=3)
        # Keep the window_words in results for traceability.
        lf_res = _decode_longform(
            model=model,
            processor=processor,
            device=device,
            examples=examples,
            sr=sr,
            batch_size=batch_size,
            seam_window_words=seam_window_words,
        )
        results["longform"] = lf_res

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()

    _base, exps = load_experiments(args.config)
    match = [e for e in exps if e.name == args.experiment]
    if not match:
        raise SystemExit(f"Experiment not found: {args.experiment}")
    res = eval_one(match[0])
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
