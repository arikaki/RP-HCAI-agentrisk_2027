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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
