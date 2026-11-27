"""Turn a prediction file into a Codabench ZIP and validate it.

    python scripts/make_submission.py runs/s1/qwen9b/dev.jsonl --split dev --out submissions/s1_dev.zip

The ZIP contains `predictions.jsonl` with example_id + risk_severity,
risk_domains and risk_drivers, so the same file can be uploaded to the
Subtask 1, 2a and 2b competitions. Use --tasks to restrict the fields.
Then check it with the official evaluator from the Starting Kit:
    python3 evaluate.py submissions/s1_dev.zip --task 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import load_split, read_jsonl  # noqa: E402
from agentrisk.submission import validate, write_zip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tasks", nargs="+", default=["1", "2a", "2b"], choices=["1", "2a", "2b"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = read_jsonl(args.preds)
    expected = [r["example_id"] for r in load_split(args.split, args.data_dir)]
    problems = validate(preds, expected, tasks=args.tasks)
    if problems:
        print("NOT written - problems found:")
        for p in problems[:20]:
            print("  -", p)
        sys.exit(1)
    path = write_zip(preds, args.out, tasks=args.tasks)
    print(f"OK: {len(preds)} predictions -> {path}")


if __name__ == "__main__":
    main()
