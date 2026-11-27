"""Combine several sampled judgments of the same example into one prediction.

Severity: median of the samples (rounded), plus the spread as a confidence
signal. Labels: the most frequent labels across samples (at most 2), with the
"No Material ..." rule applied.
"""

from __future__ import annotations

import statistics
from collections import Counter

from .labels import repair_label_set


def aggregate(samples: list[dict]) -> dict:
    sevs = [s["risk_severity"] for s in samples]
    out = {
        "risk_severity": int(round(statistics.median(sevs))),
        "severity_std": statistics.pstdev(sevs) if len(sevs) > 1 else 0.0,
        "severity_samples": sevs,
        "n_samples": len(samples),
    }
    for key, kind in (("risk_domains", "domains"), ("risk_drivers", "drivers")):
        counts = Counter(l for s in samples for l in s[key])
        # keep labels chosen in at least half the samples; always keep the top one
        half = len(samples) / 2
        chosen = [l for l, c in counts.most_common() if c >= half] or [counts.most_common(1)[0][0]]
        labels, _ = repair_label_set(chosen, kind, scores=dict(counts))
        out[key] = labels
        out[f"{key}_votes"] = dict(counts)
    return out
