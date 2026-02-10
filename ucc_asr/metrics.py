from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import jiwer

from .text import normalize_text, words


def wer(ref: str, hyp: str) -> float:
    return float(jiwer.wer(normalize_text(ref), normalize_text(hyp)))


def cer(ref: str, hyp: str) -> float:
    return float(jiwer.cer(normalize_text(ref), normalize_text(hyp)))


@dataclass(frozen=True)
class AlignmentCounts:
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int

    @property
    def error_rate(self) -> float:
        if self.ref_words <= 0:
            return 0.0
        return float(self.substitutions + self.deletions + self.insertions) / float(self.ref_words)


def _align_words(ref_words: List[str], hyp_words: List[str]) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Classic Levenshtein DP with backpointers.
    Returns (cost_dp, op_dp) where op_dp stores 'ok'|'sub'|'del'|'ins' for best path.
    """
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "ins"
    op[0][0] = "ok"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                best = (dp[i - 1][j - 1], "ok")
            else:
                best = (dp[i - 1][j - 1] + 1, "sub")
            cand_del = (dp[i - 1][j] + 1, "del")
            cand_ins = (dp[i][j - 1] + 1, "ins")
            best = min(best, cand_del, cand_ins, key=lambda x: x[0])
            dp[i][j], op[i][j] = best
    return dp, op


def seam_local_error(
    ref: str,
    hyp: str,
    seam_word_indices: List[int],
    window_words: int = 3,
) -> AlignmentCounts:
    """
    Seam-local error rate: count edits within +/- window_words of each seam, based on alignment.

    seam_word_indices: indices in ref word sequence where a seam occurs *after* that word.
      Example: if ref words are [w0,w1,w2,w3] and seam between w1 and w2 => seam index 1.
    """
    rw = words(ref)
    hw = words(hyp)
    if not rw:
        return AlignmentCounts(0, 0, 0, 0)

    _, op = _align_words(rw, hw)
    i, j = len(rw), len(hw)

    # Backtrace to assign ops to ref positions; insertions are attached to previous ref word (or 0).
    per_ref = [{"sub": 0, "del": 0, "ins": 0} for _ in range(len(rw))]
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur in {"ok", "sub"}:
            ref_idx = i - 1
            if cur == "sub":
                per_ref[ref_idx]["sub"] += 1
            i -= 1
            j -= 1
        elif cur == "del":
            ref_idx = i - 1
            per_ref[ref_idx]["del"] += 1
            i -= 1
        elif cur == "ins":
            ref_attach = max(0, i - 1)
            if len(per_ref) > 0:
                per_ref[ref_attach]["ins"] += 1
            j -= 1
        else:
            # Shouldn't happen, but avoid infinite loops.
            break

    marks = [False] * len(rw)
    w = max(0, int(window_words))
    for seam in seam_word_indices:
        # seam after word seam => affects [seam-w, seam+w+1]
        lo = max(0, seam - w)
        hi = min(len(rw) - 1, seam + w + 1)
        for k in range(lo, hi + 1):
            marks[k] = True

    subs = dels = ins = ref_cnt = 0
    for k, use in enumerate(marks):
        if not use:
            continue
        ref_cnt += 1
        subs += per_ref[k]["sub"]
        dels += per_ref[k]["del"]
        ins += per_ref[k]["ins"]

    return AlignmentCounts(substitutions=subs, deletions=dels, insertions=ins, ref_words=ref_cnt)

