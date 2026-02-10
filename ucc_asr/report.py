from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import atomic_write_text, now_utc_iso


def _find_exp(results: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for e in results.get("experiments", []):
        if e.get("name") == name:
            return e
    return None


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int,)):
        return str(x)
    try:
        xf = float(x)
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def render_report(results: Dict[str, Any]) -> str:
    exps = results.get("experiments", [])
    if not exps:
        return "# Report\n\nNo experiments found in results.\n"

    # Baseline is expected to be named "baseline" by convention in this repo.
    baseline = _find_exp(results, "baseline")

    lines: List[str] = []
    lines.append("# UCC LibriSpeech Long-Form ASR Report")
    lines.append("")
    lines.append(f"Generated at: `{now_utc_iso()}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Baseline: per-utterance fine-tuning (N=1, no concatenation).")
    lines.append("- UCC: random concatenation of N utterances with optional boundary silence (curriculum ramp).")
    lines.append("- Ablation: boundary condition (200ms silence vs no silence).")
    lines.append("")

    def split_row(exp: Dict[str, Any], split: str) -> Dict[str, Any]:
        s = exp.get("splits", {}).get(split, {})
        return {
            "wer": s.get("wer", None),
            "cer": s.get("cer", None),
            "rtf": s.get("rtf", None),
            "audio_seconds": s.get("audio_seconds", None),
            "n_utterances": s.get("n_utterances", None),
        }

    # Choose split keys based on what exists.
    split_keys = []
    for k in ["dev_clean", "dev_other", "test_clean", "test_other", "dev", "test"]:
        if any(k in e.get("splits", {}) for e in exps):
            split_keys.append(k)

    lines.append("## Standard Evaluation")
    lines.append("")
    header = ["experiment"] + [f"{k}_wer" for k in split_keys]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for e in exps:
        row = [e["name"]]
        for k in split_keys:
            row.append(_fmt(split_row(e, k)["wer"], 4))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if baseline is not None:
        lines.append("## Deltas vs Baseline (WER)")
        lines.append("")
        header = ["experiment"] + [f"delta_{k}_wer" for k in split_keys]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for e in exps:
            row = [e["name"]]
            for k in split_keys:
                b = split_row(baseline, k)["wer"]
                v = split_row(e, k)["wer"]
                if b is None or v is None:
                    row.append("n/a")
                else:
                    row.append(_fmt(float(v) - float(b), 4))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Long-form section if present.
    if any("longform" in e for e in exps):
        lines.append("## Long-Form Synthetic Evaluation")
        lines.append("")
        header = ["experiment", "longform_wer", "longform_cer", "seam_local_error_rate", "rtf"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for e in exps:
            lf = e.get("longform", {})
            seam = lf.get("seam_local", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        e["name"],
                        _fmt(lf.get("wer", None), 4),
                        _fmt(lf.get("cer", None), 4),
                        _fmt(seam.get("error_rate", None), 4),
                        _fmt(lf.get("rtf", None), 4),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Repro Notes")
    lines.append("")
    lines.append("- `artifacts/results.json` is the source of truth for the numbers above.")
    lines.append("- Per-run configs and environment snapshots are saved under `runs/<experiment>/`.")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to artifacts/results.json")
    ap.add_argument("--out", required=True, help="Path to artifacts/report.md")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    md = render_report(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, md)


if __name__ == "__main__":
    main()

