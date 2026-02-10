from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig, load_experiments
from .data.collate import collate_keep_lists
from .data.datasets import RandomConcatIterableDataset, UCCSpec, UtteranceDataset
from .data.manifest import read_jsonl
from .metrics import wer
from .modeling import load_wav2vec2_ctc, prepare_batch
from .text import normalize_text
from .utils import (
    atomic_write_json,
    configure_determinism,
    device_from_config,
    env_snapshot,
    mkdirp,
    now_utc_iso,
    set_seed,
)


def _read_manifest(cfg: Dict[str, Any], split_key: str):
    kind = cfg["data"]["kind"]
    if kind == "smoke":
        root = Path(cfg["data"]["smoke"]["root"])
        mf = root / "manifests" / f"{split_key}.jsonl"
        return read_jsonl(mf)
    if kind == "librispeech":
        mf = Path(cfg["data"]["librispeech"]["manifests_dir"]) / f"{split_key}.jsonl"
        return read_jsonl(mf)
    raise ValueError(f"Unknown data.kind={kind}")


def _eval_wer(
    model,
    processor,
    device: torch.device,
    items,
    batch_size: int,
    num_workers: int,
    sr: int,
    max_batches: int | None = None,
) -> float:
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
    with torch.no_grad():
        for b_i, batch in enumerate(dl):
            if max_batches is not None and b_i >= max_batches:
                break
            wavs = [w.to(torch.float32) for w in batch["waveform"]]
            texts = [normalize_text(t) for t in batch["transcript"]]
            inputs, _labels = prepare_batch(processor, wavs, texts, sampling_rate=sr)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred = processor.batch_decode(pred_ids)
            refs.extend(texts)
            hyps.extend([normalize_text(x) for x in pred])

    if not refs:
        return 1.0
    # Aggregate WER across all utterances by concatenating with newlines (stable for jiwer).
    return float(wer("\n".join(refs), "\n".join(hyps)))


def train_one(exp: ExperimentConfig) -> Path:
    cfg = exp.cfg
    out_root = Path(cfg["project"]["output_dir"]) / exp.name
    ckpt_dir = mkdirp(out_root / "checkpoints")
    log_path = out_root / "train_log.jsonl"
    cfg_path = out_root / "config.json"
    env_path = out_root / "env.json"

    mkdirp(out_root)

    # Reproducibility controls.
    seed = int(cfg["train"].get("seed", 0))
    set_seed(seed)
    configure_determinism(cfg["train"].get("determinism", "warn"))

    device = device_from_config(cfg["train"].get("device", "auto"))

    bundle = load_wav2vec2_ctc(cfg["model"]["pretrained_name"], cfg.get("model", {}))
    processor, model = bundle.processor, bundle.model
    model.to(device)

    sr = int(cfg["data"].get("sample_rate", 16000))
    if cfg["data"]["kind"] == "smoke":
        sr = int(cfg["data"]["smoke"].get("sample_rate", sr))

    # Data.
    train_items = _read_manifest(cfg, "train")
    dev_key = "dev"
    if cfg["data"]["kind"] == "librispeech":
        # Prefer dev-clean for selection.
        dev_key = "dev_clean"
    dev_items = _read_manifest(cfg, dev_key)
    sel_metric_key = f"{dev_key}_wer"

    ucc_cfg = cfg["train"].get("ucc", {"enabled": False})
    ucc = UCCSpec(
        enabled=bool(ucc_cfg.get("enabled", False)),
        target_n=int(ucc_cfg.get("target_n", 1)),
        schedule=str(ucc_cfg.get("schedule", "fixed")),
        rampup_fraction=float(ucc_cfg.get("rampup_fraction", 0.0)),
        boundary_silence_sec=float(ucc_cfg.get("boundary_silence_sec", 0.0)),
    )

    max_steps = int(cfg["train"]["max_steps"])
    it_ds = RandomConcatIterableDataset(
        items=train_items,
        target_sr=sr,
        ucc=ucc,
        max_steps=max_steps,
        seed=seed,
    )

    batch_size = int(cfg["train"]["batch_size"])
    dl = DataLoader(
        it_ds,
        batch_size=batch_size,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_keep_lists,
    )
    it = iter(dl)

    opt = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    warmup_steps = int(cfg["train"].get("warmup_steps", 0))
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    grad_accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    best_metric = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    # Save config + env snapshot once.
    atomic_write_json(cfg_path, {"name": exp.name, "config": cfg, "saved_at": now_utc_iso()})
    atomic_write_json(env_path, env_snapshot())

    def lr_for_step(step: int) -> float:
        base = float(cfg["train"]["lr"])
        if warmup_steps <= 0:
            return base
        return base * min(1.0, (step + 1) / float(warmup_steps))

    model.train()
    pbar = tqdm(range(max_steps), desc=f"train {exp.name}", unit="step")
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for micro in range(grad_accum):
            batch = next(it)
            wavs = [w.to(torch.float32) for w in batch["waveform"]]
            texts = [normalize_text(t) for t in batch["transcript"]]
            inputs, labels = prepare_batch(processor, wavs, texts, sampling_rate=sr)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                out = model(**inputs, labels=labels)
                loss = out.loss / float(grad_accum)
            total_loss += float(loss.detach().cpu().item())

            scaler.scale(loss).backward()

        # Optimizer step.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        for pg in opt.param_groups:
            pg["lr"] = lr_for_step(step)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % int(cfg["train"].get("log_every_steps", 50)) == 0 or step == 0:
            rec = {
                "time": now_utc_iso(),
                "step": step + 1,
                "loss": total_loss,
                "lr": opt.param_groups[0]["lr"],
                "ucc_enabled": ucc.enabled,
                "ucc_target_n": ucc.target_n,
                "ucc_schedule": ucc.schedule,
                "ucc_boundary_silence_sec": ucc.boundary_silence_sec,
            }
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            pbar.set_postfix(loss=f"{total_loss:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        do_eval = (step + 1) % int(cfg["train"].get("eval_every_steps", 1000)) == 0 or (step + 1) == max_steps
        if do_eval:
            dev_wer = _eval_wer(
                model=model,
                processor=processor,
                device=device,
                items=dev_items,
                batch_size=batch_size,
                num_workers=0,
                sr=sr,
                max_batches=2 if cfg["data"]["kind"] == "smoke" else None,
            )
            rec = {
                "time": now_utc_iso(),
                "step": step + 1,
                sel_metric_key: dev_wer,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

            # Save last + best.
            torch.save({"model": model.state_dict(), "step": step + 1, "config": cfg}, last_path)
            if dev_wer < best_metric:
                best_metric = dev_wer
                torch.save({"model": model.state_dict(), "step": step + 1, "config": cfg}, best_path)
            model.train()

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--experiment", required=True, help="Experiment name from config")
    args = ap.parse_args()

    _base, exps = load_experiments(args.config)
    match = [e for e in exps if e.name == args.experiment]
    if not match:
        raise SystemExit(f"Experiment not found: {args.experiment}")
    train_one(match[0])


if __name__ == "__main__":
    main()
