"""Write and check Codabench submissions.

Format (from the AgentRisk Codabench pages): a ZIP with `predictions.jsonl`
at its root, one JSON object per line, exactly one line per example, e.g.

    {"example_id": "AGENTRISK-004", "risk_severity": 2}
    {"example_id": "AGENTRISK-004", "risk_domains": ["No Material Harm"]}

Each subtask is a separate competition (1: 18508, 2a: 18509, 2b: 18510).
Only the field for that subtask is required; other target fields are ignored,
so one file with all three fields can be uploaded to each competition.
Always check with the official local evaluator from the Starting Kit:
    python3 evaluate.py submission.zip --task 1
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .labels import NO_HARM, NO_RISK, normalize_label, normalize_severity

FIELDS = {"1": "risk_severity", "2a": "risk_domains", "2b": "risk_drivers"}


def validate(pred_rows: list[dict], expected_ids: list[str], tasks=("1", "2a", "2b")) -> list[str]:
    """Return a list of problems (empty list = looks valid)."""
    problems = []
    ids = [r.get("example_id") for r in pred_rows]
    if len(ids) != len(set(ids)):
        problems.append("duplicate example_id")
    if set(ids) != set(expected_ids):
        problems.append(f"ids differ from expected: {len(set(expected_ids) - set(ids))} missing, "
                        f"{len(set(ids) - set(expected_ids))} unexpected")
    for r in pred_rows:
        eid = r.get("example_id")
        if "1" in tasks and normalize_severity(r.get("risk_severity")) is None:
            problems.append(f"{eid}: invalid risk_severity {r.get('risk_severity')!r}")
        for task, kind, none_label in (("2a", "domains", NO_HARM), ("2b", "drivers", NO_RISK)):
            if task not in tasks:
                continue
            labs = r.get(FIELDS[task])
            if not isinstance(labs, list) or not 1 <= len(labs) <= 2:
                problems.append(f"{eid}: {FIELDS[task]} must be a list of 1-2 labels")
                continue
            canon = [normalize_label(x, kind) for x in labs]
            if None in canon:
                problems.append(f"{eid}: unknown label in {labs}")
            elif len(set(canon)) != len(canon):
                problems.append(f"{eid}: duplicate labels {labs}")
            elif none_label in canon and len(canon) > 1:
                problems.append(f"{eid}: '{none_label}' must be used alone")
    return problems


def write_zip(pred_rows: list[dict], zip_path: str | Path, tasks=("1", "2a", "2b")) -> Path:
    """Keep only example_id + the requested target fields and zip them."""
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in pred_rows:
        out = {"example_id": r["example_id"]}
        for t in tasks:
            out[FIELDS[t]] = r[FIELDS[t]]
        lines.append(json.dumps(out, ensure_ascii=False))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("predictions.jsonl", "\n".join(lines) + "\n")
    return zip_path
