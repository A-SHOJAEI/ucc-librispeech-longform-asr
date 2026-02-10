from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)} in {p}")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    stack = [(out, updates)]
    while stack:
        dst, src = stack.pop()
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                stack.append((dst[k], v))
            else:
                dst[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    cfg: Dict[str, Any]


def load_experiments(config_path: str | Path) -> tuple[Dict[str, Any], list[ExperimentConfig]]:
    cfg = load_yaml(config_path)
    exps = cfg.get("experiments", [])
    if not isinstance(exps, list):
        raise ValueError("`experiments` must be a list")

    base = copy.deepcopy(cfg)
    base.pop("experiments", None)
    out: list[ExperimentConfig] = []
    for e in exps:
        if not isinstance(e, dict) or "name" not in e:
            raise ValueError("Each experiment must be a mapping with a `name`")
        name = str(e["name"])
        overrides = copy.deepcopy(e)
        overrides.pop("name", None)
        merged = deep_update(base, overrides)
        out.append(ExperimentConfig(name=name, cfg=merged))
    if not out:
        # Allow config without experiments: treat as single experiment called "default".
        out.append(ExperimentConfig(name="default", cfg=base))
    return base, out

