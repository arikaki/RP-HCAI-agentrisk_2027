"""Checks for predicting severity from S1 factor ratings.

Run:  python tests/test_factor_model.py      (or: pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.factor_model import (FACTOR_KEYS, MODEL_NAMES, apply_consistency, factor_vector,  # noqa: E402
                                    make_model, median_class)


def _factors(rating: int) -> dict:
    return {k: {"rating": rating, "reason": "r"} for k in FACTOR_KEYS}


def test_factor_vector():
    assert factor_vector(_factors(3)) == [3.0] * 7
    assert factor_vector(None) is None
    partial = _factors(2)
    del partial["oversight"]
    assert factor_vector(partial) is None


def test_median_class():
    classes = np.array([1, 2, 3, 4, 5])
    assert median_class(np.array([0.6, 0.1, 0.1, 0.1, 0.1]), classes) == 1
    assert median_class(np.array([0.3, 0.1, 0.1, 0.2, 0.3]), classes) == 3
    assert median_class(np.array([0.3, 0.1, 0.05, 0.25, 0.3]), classes) == 4
    assert median_class(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), classes) == 5


def test_models_learn_a_monotonic_mapping():
    rng = np.random.default_rng(0)
    y = rng.choice([1, 3, 4], size=150)
    X = np.repeat(y[:, None], 7, axis=1) + rng.normal(0, 0.3, size=(150, 7))
    for name in MODEL_NAMES:
        model = make_model(name).fit(X, y)
        assert list(model.predict(np.array([[1.0] * 7, [3.0] * 7, [4.0] * 7]))) == [1, 3, 4], name


def test_apply_consistency_does_not_mutate_input():
    pred = {"risk_severity": 1, "risk_domains": ["Privacy / Confidentiality"],
            "risk_drivers": ["Authorization / Consent Failure"]}
    out = apply_consistency(pred)
    assert out["risk_domains"] == ["No Material Harm"] and out["risk_drivers"] == ["No Material Risk"]
    assert pred["risk_domains"] == ["Privacy / Confidentiality"]
    risky = {"risk_severity": 3, "risk_domains": ["Financial / Assets"], "risk_drivers": ["Execution / Action Hazard"]}
    assert apply_consistency(risky) == risky


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
