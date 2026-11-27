"""Predict severity from the seven S1 factor ratings with a small trained model.

Models (all take the seven ratings as features):
  logreg-argmax   multinomial logistic regression, most likely class
  logreg-median   same model, median of the predicted distribution, which
                  minimises the expected absolute error (the ST1 metric)
  ridge-round     ridge regression, rounded and clipped to 1-5
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from .labels import NO_HARM, NO_RISK, RISK_FACTORS

FACTOR_KEYS = list(RISK_FACTORS)
MODEL_NAMES = ("logreg-argmax", "logreg-median", "ridge-round")


def factor_vector(factors: dict | None) -> list[float] | None:
    """The seven ratings in a fixed order, or None if any rating is missing."""
    if not factors:
        return None
    ratings = [(factors.get(k) or {}).get("rating") for k in FACTOR_KEYS]
    if any(r is None for r in ratings):
        return None
    return [float(r) for r in ratings]


def median_class(probs: np.ndarray, classes: np.ndarray) -> int:
    """The class at which the cumulative probability first reaches 0.5."""
    idx = int(np.searchsorted(np.cumsum(probs), 0.5 - 1e-9))
    return int(classes[min(idx, len(classes) - 1)])


class _LogReg:
    def __init__(self, use_median: bool):
        self.use_median = use_median
        self.model = LogisticRegression(max_iter=2000)

    def fit(self, X, y) -> "_LogReg":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_with_confidence(X)[0]

    def predict_with_confidence(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predicted severities and the model's probability for each of them."""
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        if self.use_median:
            pred = np.array([median_class(p, classes) for p in probs])
        else:
            pred = classes[np.argmax(probs, axis=1)].astype(int)
        conf = probs[np.arange(len(pred)), np.searchsorted(classes, pred)]
        return pred, conf


class _RidgeRound:
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, X, y) -> "_RidgeRound":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.clip(np.rint(self.model.predict(X)), 1, 5).astype(int)

    def predict_with_confidence(self, X) -> tuple[np.ndarray, None]:
        return self.predict(X), None


def make_model(name: str):
    if name == "logreg-argmax":
        return _LogReg(use_median=False)
    if name == "logreg-median":
        return _LogReg(use_median=True)
    if name == "ridge-round":
        return _RidgeRound()
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def apply_consistency(pred: dict) -> dict:
    """Give severity-1 predictions the labels 'No Material Harm' and 'No Material Risk'."""
    if pred["risk_severity"] != 1:
        return pred
    return {**pred, "risk_domains": [NO_HARM], "risk_drivers": [NO_RISK]}
