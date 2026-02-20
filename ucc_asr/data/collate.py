from __future__ import annotations

from typing import Any, Dict, List


def collate_keep_lists(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Collate a list of sample dicts by grouping values by key (no padding/stacking)."""
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: [sample[k] for sample in batch] for k in keys}
