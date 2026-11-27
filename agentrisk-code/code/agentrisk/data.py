"""Loading AgentRisk data and building grouped cross-validation folds."""

from __future__ import annotations

import json
from pathlib import Path

HF_DATASET = "AgentRisk/agentrisk-data-2027"


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(split: str, data_dir: str | Path | None = None) -> list[dict]:
    """Load a split ("train", "dev", or a test file later).

    If `data_dir` contains `<split>.jsonl` it is used; otherwise the split is
    downloaded from Hugging Face with the `datasets` library.
    """
    if data_dir:
        path = Path(data_dir) / f"{split}.jsonl"
        if path.exists():
            return read_jsonl(path)
    from datasets import load_dataset  # imported lazily

    ds = load_dataset(HF_DATASET, split=split)
    return [dict(r) for r in ds]


def download_all(data_dir: str | Path) -> None:
    """Save train/dev (and the split manifest, if reachable) as local JSONL files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        rows = load_split(split)
        write_jsonl(data_dir / f"{split}.jsonl", rows)
        print(f"saved {len(rows)} rows -> {data_dir / f'{split}.jsonl'}")
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(HF_DATASET, "metadata/split_manifest.jsonl", repo_type="dataset")
        rows = read_jsonl(p)
        write_jsonl(data_dir / "split_manifest.jsonl", rows)
        print(f"saved manifest ({len(rows)} rows)")
    except Exception as exc:  # manifest is optional
        print(f"could not download split manifest: {exc}")


def load_groups(data_dir: str | Path | None) -> dict[str, str]:
    """example_id -> scenario_group (for grouped cross-validation)."""
    if not data_dir:
        return {}
    path = Path(data_dir) / "split_manifest.jsonl"
    if not path.exists():
        return {}
    return {r["example_id"]: r["scenario_group"] for r in read_jsonl(path)}


def grouped_folds(rows: list[dict], groups: dict[str, str], n_folds: int = 5,
                  seed: int = 13) -> list[tuple[list[int], list[int]]]:
    """Fold indices that keep each scenario group in a single fold.

    Items from the same scenario (e.g. a risky trajectory and its safe
    counterpart) must not be split across train/validation, or scores are
    inflated. Falls back to one group per item if no manifest is available.
    """
    from sklearn.model_selection import GroupKFold

    group_ids = [groups.get(r["example_id"], r["example_id"]) for r in rows]
    # Shuffle group order reproducibly, then let GroupKFold balance fold sizes.
    import random

    unique = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    rank = {g: i for i, g in enumerate(unique)}
    numeric = [rank[g] for g in group_ids]
    gkf = GroupKFold(n_splits=n_folds)
    return [(list(tr), list(va)) for tr, va in gkf.split(rows, groups=numeric)]
