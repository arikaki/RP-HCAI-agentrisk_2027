"""Checks for detecting fine-tuning folds that failed to learn.

Run:  python tests/test_training.py      (or: pytest tests/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.training import constant_baseline_mae, fold_collapsed  # noqa: E402


def test_constant_baseline_uses_train_median():
    # train median is 2 -> errors on val [1, 4] are 1 and 2
    assert constant_baseline_mae(train_sev=[1, 1, 2, 3, 4], val_sev=[1, 4]) == 1.5


def test_fold_collapsed():
    train = [1] * 5 + [2] + [3] * 2 + [4] * 2
    val = [1, 1, 3, 4]  # constant-2 baseline MAE = (1 + 1 + 1 + 2) / 4 = 1.25
    assert not fold_collapsed(val, [1, 1, 3, 4], train)
    assert not fold_collapsed(val, [1, 2, 3, 3], train)   # MAE 0.5 = 0.4 x baseline
    assert fold_collapsed(val, [2, 2, 2, 2], train)
    assert fold_collapsed(val, [2, 2, 2, 3], train)       # MAE 1.0 = 0.8 x baseline


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
