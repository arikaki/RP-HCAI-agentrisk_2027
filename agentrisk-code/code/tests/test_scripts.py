"""End-to-end checks for the command-line scripts on a tiny synthetic split.

Run:  python tests/test_scripts.py      (or: pytest tests/)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GOLD = [
    {"example_id": "X-1", "risk_severity": 1, "risk_domains": ["No Material Harm"],
     "risk_drivers": ["No Material Risk"]},
    {"example_id": "X-2", "risk_severity": 4, "risk_domains": ["Financial / Assets"],
     "risk_drivers": ["Scope / Safeguard Failure"]},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_evaluate_runs_on_non_utf8_console():
    """evaluate.py must not crash when stdout is a legacy Windows code page (cp1252)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_jsonl(tmp / "dev.jsonl", GOLD)
        _write_jsonl(tmp / "preds.jsonl", GOLD)
        env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data-dir", str(tmp),
             str(tmp / "preds.jsonl")],
            capture_output=True, env=env,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        out = proc.stdout.decode("utf-8")
        assert "| System |" in out
        assert "0.000 [0.000, 0.000]" in out  # perfect predictions -> MAE 0


def test_factors_method_saves_only_factor_ratings():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_jsonl(tmp / "train.jsonl", [{**g, "formatted_interaction": f"USER: task {g['example_id']}"}
                                           for g in GOLD])
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_llm_judge.py"), "--method", "factors",
             "--model", "mock", "--tag", "t", "--data-dir", str(tmp), "--split", "train",
             "--runs-dir", str(tmp / "runs")],
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        rows = [json.loads(l) for l in (tmp / "runs" / "factors" / "t" / "train.jsonl").read_text().splitlines()]
        assert [r["example_id"] for r in rows] == ["X-1", "X-2"]
        assert all(len(r["factors"]) == 7 and "risk_severity" not in r for r in rows)


def test_factor_severity_replaces_llm_severity():
    keys = ["likelihood", "harm_intensity", "scale", "target_susceptibility", "system_dependency",
            "oversight", "reversibility"]

    def factors(level: int) -> dict:
        return {k: {"rating": level, "reason": ""} for k in keys}

    labels = {1: (["No Material Harm"], ["No Material Risk"]),
              4: (["Financial / Assets"], ["Scope / Safeguard Failure"])}
    train = [{"example_id": f"T-{i}", "risk_severity": s, "risk_domains": labels[s][0],
              "risk_drivers": labels[s][1]} for i, s in enumerate([1, 4] * 20)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_jsonl(tmp / "train.jsonl", train)
        _write_jsonl(tmp / "dev.jsonl", GOLD)
        _write_jsonl(tmp / "train_factors.jsonl",
                     [{"example_id": r["example_id"], "factors": factors(r["risk_severity"])} for r in train])
        # The LLM said 5 for both; the factor ratings say 1 and 4.
        _write_jsonl(tmp / "s1_dev.jsonl", [
            {"example_id": "X-1", "risk_severity": 5, "risk_domains": ["Financial / Assets"],
             "risk_drivers": ["Execution / Action Hazard"], "factors": factors(1)},
            {"example_id": "X-2", "risk_severity": 5, "risk_domains": ["Financial / Assets"],
             "risk_drivers": ["Scope / Safeguard Failure"], "factors": factors(4)},
        ])
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "factor_severity.py"), "--data-dir", str(tmp),
             "--train-factors", str(tmp / "train_factors.jsonl"), "--target-preds", str(tmp / "s1_dev.jsonl"),
             "--out-dir", str(tmp / "out"), "--folds", "2"],
            capture_output=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        rows = {r["example_id"]: r for r in map(json.loads, (tmp / "out" / "dev.jsonl").read_text().splitlines())}
        assert rows["X-1"]["risk_severity"] == 1 and rows["X-1"]["risk_domains"] == ["No Material Harm"]
        assert rows["X-2"]["risk_severity"] == 4 and rows["X-2"]["severity_llm"] == 5
        assert json.loads((tmp / "out" / "cv_metrics.json").read_text())["chosen"]


def test_confidence_summary_counts_misses_caught():
    sys.path.insert(0, str(ROOT / "scripts"))
    from confidence import summarise

    gold = [{"example_id": f"X-{i}", "risk_severity": s} for i, s in enumerate([1, 1, 4, 4, 3])]
    preds = [{"example_id": "X-0", "risk_severity": 1, "severity_confidence": 0.9},
             {"example_id": "X-1", "risk_severity": 1, "severity_confidence": 0.8},
             {"example_id": "X-2", "risk_severity": 4, "severity_confidence": 0.7},
             {"example_id": "X-3", "risk_severity": 1, "severity_confidence": 0.2},  # dangerous miss
             {"example_id": "X-4", "risk_severity": 3, "severity_confidence": 0.6}]
    s = summarise(gold, preds)
    assert s["n_danger"] == 1 and s["n_large"] == 1
    assert s["defer"][0.2]["danger_caught"] == 1 and s["defer"][0.2]["mae_kept"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
