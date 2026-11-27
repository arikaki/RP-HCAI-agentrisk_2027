"""Quick checks for parsing, label repair, metrics and aggregation.

Run:  python tests/test_core.py      (or: pytest tests/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentrisk.aggregate import aggregate
from agentrisk.labels import normalize_label, normalize_severity, repair_label_set
from agentrisk.llm import extract_json
from agentrisk.metrics import macro_f1, mae, mean_jaccard, quadratic_weighted_kappa


def test_extract_json():
    assert extract_json('<think>{"x": 1}</think> Answer: {"risk_severity": 3}') == {"risk_severity": 3}
    assert extract_json('```json\n{"a": "b { c"}\n```') == {"a": "b { c"}
    assert extract_json("no json here") is None
    assert extract_json('{"a": {"b": 2}} trailing') == {"a": {"b": 2}}


def test_severity():
    assert normalize_severity(3) == 3
    assert normalize_severity("4") == 4
    assert normalize_severity(" High ") == 4
    assert normalize_severity(6) is None
    assert normalize_severity(True) is None
    assert normalize_severity(2.0) == 2


def test_labels():
    assert normalize_label("privacy", "domains") == "Privacy / Confidentiality"
    assert normalize_label("Security_System-Integrity", "domains") == "Security / System Integrity"
    assert normalize_label("nmh", "domains") == "No Material Harm"
    assert normalize_label("made up", "domains") is None
    labs, rep = repair_label_set(["No Material Harm", "Financial / Assets"], "domains")
    assert labs == ["Financial / Assets"] and rep
    labs, rep = repair_label_set(["bogus"], "drivers")
    assert labs == ["No Material Risk"] and rep
    labs, _ = repair_label_set(["privacy", "financial", "legal"], "domains",
                               scores={"Legal / Compliance": 3, "Privacy / Confidentiality": 2,
                                       "Financial / Assets": 1})
    assert labs == ["Legal / Compliance", "Privacy / Confidentiality"]
    labs, rep = repair_label_set(["Privacy / Confidentiality"], "domains")
    assert labs == ["Privacy / Confidentiality"] and not rep


def test_metrics():
    assert mae([4, 5, 3], [3, 5, 1]) == 1.0
    assert mean_jaccard([["A", "B"]], [["A"]]) == 0.5
    assert mean_jaccard([["A", "B"]], [["B", "A"]]) == 1.0
    assert abs(mean_jaccard([["A", "B"]], [["A", "C"]]) - 1 / 3) < 1e-9
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0


def test_aggregate():
    samples = [
        {"risk_severity": 3, "risk_domains": ["Financial / Assets"], "risk_drivers": ["Scope / Safeguard Failure"]},
        {"risk_severity": 4, "risk_domains": ["Financial / Assets", "Privacy / Confidentiality"],
         "risk_drivers": ["Scope / Safeguard Failure"]},
        {"risk_severity": 3, "risk_domains": ["Financial / Assets"], "risk_drivers": ["Execution / Action Hazard"]},
    ]
    out = aggregate(samples)
    assert out["risk_severity"] == 3
    assert out["risk_domains"] == ["Financial / Assets"]
    assert out["risk_drivers"] == ["Scope / Safeguard Failure"]
    assert out["severity_std"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
