from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ManifestItem:
    audio_path: str
    transcript: str
    duration: float
    extra: Dict[str, Any]


def read_jsonl(path: str | Path) -> List[ManifestItem]:
    p = Path(path)
    items: List[ManifestItem] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                ManifestItem(
                    audio_path=str(obj["audio_path"]),
                    transcript=str(obj.get("transcript", obj.get("text", ""))),
                    duration=float(obj.get("duration", 0.0)),
                    extra={k: v for k, v in obj.items() if k not in ("audio_path", "transcript", "text", "duration")},
                )
            )
    return items


def write_jsonl(path: str | Path, items: List[ManifestItem]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            obj = {"audio_path": it.audio_path, "transcript": it.transcript, "duration": it.duration}
            obj.update(it.extra)
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
