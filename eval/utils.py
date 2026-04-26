"""Common utilities for evaluation scripts."""
import os
import json
from typing import List, Dict, Optional, Tuple

def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def pair_by_id(a: List[Dict], b: List[Dict]) -> List[Tuple[Dict, Dict]]:
    amap = {it["id"]: it for it in a}
    pairs = []
    for it in b:
        key = it.get("id")
        if key in amap:
            pairs.append((amap[key], it))
    return pairs

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
