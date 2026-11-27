"""Metrics for AgentRisk, matching the official definitions on Codabench.

- Subtask 1: MAE (primary). Also accuracy, RMSE, quadratic weighted kappa.
- Subtasks 2a/2b: mean of per-example Jaccard (primary), plus exact-set match.
- Subtask 3: macro F1 (primary).
Bootstrap confidence intervals resample examples with replacement.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def mae(gold: Sequence[int], pred: Sequence[int]) -> float:
    return sum(abs(g - p) for g, p in zip(gold, pred)) / len(gold)


def rmse(gold: Sequence[int], pred: Sequence[int]) -> float:
    return math.sqrt(sum((g - p) ** 2 for g, p in zip(gold, pred)) / len(gold))


def accuracy(gold: Sequence, pred: Sequence) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def quadratic_weighted_kappa(gold: Sequence[int], pred: Sequence[int],
                             k_min: int = 1, k_max: int = 5) -> float:
    n = k_max - k_min + 1
    conf = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        conf[g - k_min][p - k_min] += 1
    total = len(gold)
    hist_g = [sum(row) for row in conf]
    hist_p = [sum(conf[i][j] for i in range(n)) for j in range(n)]
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            w = ((i - j) ** 2) / ((n - 1) ** 2)
            expected = hist_g[i] * hist_p[j] / total
            num += w * conf[i][j]
            den += w * expected
    return 1.0 - num / den if den else 0.0


def jaccard(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    return len(gold & pred) / len(gold | pred)


def mean_jaccard(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> float:
    return sum(jaccard(set(g), set(p)) for g, p in zip(gold, pred)) / len(gold)


def exact_set_match(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> float:
    return sum(set(g) == set(p) for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] | None = None) -> float:
    labels = list(labels) if labels else sorted(set(gold) | set(pred))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def bootstrap_ci(metric: Callable[[list, list], float], gold: list, pred: list,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 13) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(gold)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(metric([gold[i] for i in idx], [pred[i] for i in idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def paired_bootstrap_pvalue(metric: Callable[[list, list], float], gold: list,
                            pred_a: list, pred_b: list, higher_is_better: bool,
                            n_boot: int = 2000, seed: int = 13) -> float:
    """One-sided p-value that system A is NOT better than B (small = A better)."""
    rng = random.Random(seed)
    n = len(gold)
    worse_or_equal = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        a = metric(g, [pred_a[i] for i in idx])
        b = metric(g, [pred_b[i] for i in idx])
        a_better = a > b if higher_is_better else a < b
        worse_or_equal += not a_better
    return worse_or_equal / n_boot


def score_all(gold_rows: list[dict], pred_rows: list[dict], with_ci: bool = True) -> dict:
    """Score predictions for Subtasks 1, 2a, 2b (those present in pred_rows)."""
    pred_by_id = {r["example_id"]: r for r in pred_rows}
    missing = [r["example_id"] for r in gold_rows if r["example_id"] not in pred_by_id]
    if missing:
        raise ValueError(f"{len(missing)} gold examples have no prediction, e.g. {missing[:3]}")
    out: dict = {"n": len(gold_rows)}
    first = pred_by_id[gold_rows[0]["example_id"]]

    if "risk_severity" in first:
        g = [r["risk_severity"] for r in gold_rows]
        p = [int(pred_by_id[r["example_id"]]["risk_severity"]) for r in gold_rows]
        out["st1_mae"] = mae(g, p)
        out["st1_acc"] = accuracy(g, p)
        out["st1_rmse"] = rmse(g, p)
        out["st1_qwk"] = quadratic_weighted_kappa(g, p)
        if with_ci:
            out["st1_mae_ci"] = bootstrap_ci(mae, g, p)
    for key, name in (("risk_domains", "st2a"), ("risk_drivers", "st2b")):
        if key in first:
            g = [r[key] for r in gold_rows]
            p = [pred_by_id[r["example_id"]][key] for r in gold_rows]
            out[f"{name}_jaccard"] = mean_jaccard(g, p)
            out[f"{name}_exact"] = exact_set_match(g, p)
            if with_ci:
                out[f"{name}_jaccard_ci"] = bootstrap_ci(mean_jaccard, g, p)
    return out
