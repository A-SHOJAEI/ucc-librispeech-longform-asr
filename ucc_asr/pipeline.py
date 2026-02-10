from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .config import load_experiments
from .data.prepare import prepare_data
from .eval import eval_one
from .train import train_one
from .utils import atomic_write_json, mkdirp, now_utc_iso


def run_stage(config_path: str, stage: str) -> None:
    _base, exps = load_experiments(config_path)
    stage = stage.lower()
    if stage not in {"data", "train", "eval"}:
        raise ValueError("stage must be one of: data, train, eval")

    # Data stage runs once with the base config (shared inputs).
    if stage == "data":
        prepare_data(exps[0].cfg)
        return

    if stage == "train":
        for exp in exps:
            train_one(exp)
        return

    if stage == "eval":
        # Write a single aggregated results file.
        cfg = exps[0].cfg
        artifacts_dir = mkdirp(Path(cfg["project"]["artifacts_dir"]))
        out_path = artifacts_dir / "results.json"
        results = {
            "generated_at": now_utc_iso(),
            "config_path": str(Path(config_path).resolve()),
            "experiments": [],
        }
        for exp in exps:
            results["experiments"].append(eval_one(exp))
        atomic_write_json(out_path, results)
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, help="data|train|eval")
    args = ap.parse_args()
    run_stage(args.config, args.stage)


if __name__ == "__main__":
    main()

