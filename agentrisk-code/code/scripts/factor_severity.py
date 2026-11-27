"""S1 with a trained severity step: predict severity from the S1 factor ratings.

Trains small models on factor ratings of the TRAIN split (from
`run_llm_judge.py --method factors --split train`), picks the one with the
lowest grouped cross-validated MAE on train, refits it on all of train, and
uses it to replace the severity in S1 predictions for the target split.
Domains and drivers come from the S1 decision step; items rated severity 1
get "No Material Harm" / "No Material Risk". Dev is never used for fitting
or model choice.

    python scripts/factor_severity.py --data-dir data \
        --train-factors runs/factors/qwen3.5-9b-v3/train.jsonl \
        --target-preds runs/s1/qwen3.5-9b-v3/dev.jsonl --out-dir runs/s1f/qwen3.5-9b-v3

Outputs in --out-dir: <split>.jsonl (predictions; the LLM's own severity is
kept as severity_llm), cv_metrics.json, run_info.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.data import grouped_folds, load_groups, load_split, read_jsonl, write_jsonl  # noqa: E402
from agentrisk.factor_model import MODEL_NAMES, apply_consistency, factor_vector, make_model  # noqa: E402
from agentrisk.metrics import mae  # noqa: E402


def training_matrix(train: list[dict], factor_rows: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Train rows that have complete factor ratings, with features X and gold severity y."""
    by_id = {r["example_id"]: factor_vector(r["factors"]) for r in factor_rows}
    kept = [r for r in train if by_id.get(r["example_id"])]
    X = np.array([by_id[r["example_id"]] for r in kept])
    y = np.array([r["risk_severity"] for r in kept])
    return kept, X, y


def cross_val_mae(name: str, X: np.ndarray, y: np.ndarray, folds) -> float:
    pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        pred[va] = make_model(name).fit(X[tr], y[tr]).predict(X[va])
    return mae(list(y), list(pred))


def predict_target(model, target_preds: list[dict]) -> tuple[list[dict], int]:
    """Replace severity where factor ratings are complete; keep the LLM severity otherwise."""
    out, n_fallback = [], 0
    for p in target_preds:
        vec = factor_vector(p.get("factors"))
        conf = None
        if vec is None:
            n_fallback += 1
            sev = int(p["risk_severity"])
        else:
            pred, confs = model.predict_with_confidence(np.array([vec]))
            sev = int(pred[0])
            conf = None if confs is None else round(float(confs[0]), 4)
        out.append(apply_consistency({**p, "risk_severity": sev, "severity_llm": p["risk_severity"],
                                      "severity_confidence": conf}))
    return out, n_fallback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-factors", required=True, help="factor ratings for the train split")
    ap.add_argument("--target-preds", required=True, help="S1 predictions (with factors) to re-score")
    ap.add_argument("--split", default="dev", help="name of the target split, for the output file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    train = load_split("train", args.data_dir)
    kept, X, y = training_matrix(train, read_jsonl(args.train_factors))
    print(f"train items with complete factor ratings: {len(kept)}/{len(train)}")
    folds = grouped_folds(kept, load_groups(args.data_dir), args.folds, args.seed)

    cv = {name: cross_val_mae(name, X, y, folds) for name in MODEL_NAMES}
    chosen = min(cv, key=cv.get)
    for name, score in cv.items():
        print(f"  CV MAE on train  {name:14s} {score:.3f}" + ("  <- chosen" if name == chosen else ""))

    model = make_model(chosen).fit(X, y)
    preds, n_fallback = predict_target(model, read_jsonl(args.target_preds))
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / f"{args.split}.jsonl", preds)
    (out_dir / "cv_metrics.json").write_text(json.dumps({"cv_mae": cv, "chosen": chosen,
                                                         "n_train": len(kept)}, indent=2))
    info = vars(args) | {"chosen": chosen, "finished": datetime.now(timezone.utc).isoformat()}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {len(preds)} predictions -> {out_dir / f'{args.split}.jsonl'} "
          f"({n_fallback} kept the LLM severity because factor ratings were missing)")


if __name__ == "__main__":
    main()
