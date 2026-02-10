from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def mkdirp(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    def default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    text = json.dumps(obj, indent=2, sort_keys=True, default=default) + "\n"
    atomic_write_text(path, text)


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism(mode: str) -> None:
    """
    mode:
      - off: fastest, nondeterministic allowed
      - warn: prefer determinism, warn-only where possible
      - strict: error on nondeterministic ops
    """
    mode = str(mode).lower()
    if mode not in {"off", "warn", "strict"}:
        raise ValueError("determinism must be one of: off, warn, strict")

    if mode == "off":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if mode == "strict":
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # warn mode: deterministic where feasible without hard-failing.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[arg-type]
        except TypeError:
            torch.use_deterministic_algorithms(True)


def env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        snap["gpu_count"] = torch.cuda.device_count()
        snap["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return snap


def device_from_config(device: str) -> torch.device:
    d = str(device).lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d in {"cpu", "cuda"}:
        if d == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device(d)
    raise ValueError("device must be one of: auto, cpu, cuda")


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    return float(x)

