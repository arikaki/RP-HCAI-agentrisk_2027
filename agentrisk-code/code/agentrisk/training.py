"""Detecting fine-tuning runs that failed to learn.

Large encoders sometimes fail to train with an unlucky seed and predict
roughly the same output for every input. A fold counts as failed when its
validation MAE is not clearly better than always predicting the training
median severity.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from .metrics import mae

COLLAPSE_RATIO = 0.7


def constant_baseline_mae(train_sev: Sequence[int], val_sev: Sequence[int]) -> float:
    median = statistics.median_low(train_sev)
    return mae(list(val_sev), [median] * len(val_sev))


def fold_collapsed(val_sev: Sequence[int], val_pred: Sequence[int], train_sev: Sequence[int],
                   ratio: float = COLLAPSE_RATIO) -> bool:
    return mae(list(val_sev), list(val_pred)) >= ratio * constant_baseline_mae(train_sev, val_sev)
