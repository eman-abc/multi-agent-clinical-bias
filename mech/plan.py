"""
mech/plan.py -- a frozen analysis plan you can verify.

The plan (analysis_plan.json) fixes every number the final analysis will use: which layers, how many folds and random
directions, the strengths, the significance rule, the minimum band length, the attention-vs-MLP decision margins.
`freeze_plan.py --freeze` writes ANALYSIS_PLAN.lock: a UTC timestamp plus the SHA-256 of the plan, the narrative
(ANALYSIS_PLAN.md), every script listed under "frozen_files", and every data file listed under "frozen_data".
Commit the lock to git. From then on:

  * the GPU scripts take their design parameters FROM the plan (--plan) and refuse a conflicting command-line value;
  * the analysis scripts print the plan status at the top of every report and store it in the output:
        FROZEN-OK        plan, narrative, code and data all match the lock
        MODIFIED         something differs from the lock (the report says what)   <- the analysis is no longer confirmatory
        NOT-FROZEN       no lock exists yet                                        <- exploratory
This does not stop anyone from editing files; it makes any edit visible, which is what a reader needs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
from pathlib import Path

LOCK_NAME = "ANALYSIS_PLAN.lock"
NARRATIVE = "ANALYSIS_PLAN.md"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_plan_hash(plan: dict) -> str:
    """Hash of the plan's CONTENT (key order and whitespace do not matter)."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_plan(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_lock(plan_path, root=".") -> dict:
    root = Path(root)
    plan = load_plan(plan_path)
    lock = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "plan_file": str(plan_path),
        "plan_sha256": canonical_plan_hash(plan),
        "narrative_sha256": sha256_file(root / NARRATIVE) if (root / NARRATIVE).exists() else None,
        "python": platform.python_version(),
        "files": {},
        "data": {},
    }
    for rel in plan.get("frozen_files", []):
        p = root / rel
        lock["files"][rel] = sha256_file(p) if p.exists() else None
    for rel in plan.get("frozen_data", []):
        p = root / rel
        lock["data"][rel] = sha256_file(p) if p.exists() else None
    return lock


def write_lock(plan_path, root=".", lock_path=None) -> dict:
    lock = make_lock(plan_path, root)
    path = Path(lock_path) if lock_path else Path(root) / LOCK_NAME
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def plan_status(plan_path, root=".", lock_path=None) -> dict:
    """{'status': 'FROZEN-OK' | 'MODIFIED' | 'NOT-FROZEN', 'frozen_at_utc': ..., 'differences': [...]}"""
    path = Path(lock_path) if lock_path else Path(root) / LOCK_NAME
    if not Path(plan_path).exists():
        return {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": [f"plan file {plan_path} not found"]}
    if not path.exists():
        return {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": ["no lock file: this run is exploratory"]}
    old = json.loads(path.read_text(encoding="utf-8"))
    new = make_lock(plan_path, root)
    diffs = []
    if old.get("plan_sha256") != new["plan_sha256"]:
        diffs.append("analysis_plan.json content changed")
    if old.get("narrative_sha256") != new["narrative_sha256"]:
        diffs.append(f"{NARRATIVE} changed")
    for group in ("files", "data"):
        for rel, h in old.get(group, {}).items():
            if new[group].get(rel) != h:
                diffs.append(f"{group[:-1] if group == 'files' else group}: {rel} {'missing' if new[group].get(rel) is None else 'changed'}")
        for rel in new[group]:
            if rel not in old.get(group, {}):
                diffs.append(f"{rel} was added to the plan after freezing")
    return {"status": "FROZEN-OK" if not diffs else "MODIFIED", "frozen_at_utc": old.get("frozen_at_utc"), "differences": diffs}


def banner(status: dict) -> str:
    s = status["status"]
    if s == "FROZEN-OK":
        return f"PLAN FROZEN-OK (frozen {status['frozen_at_utc']}): plan, narrative, code and data match the lock."
    if s == "NOT-FROZEN":
        return "PLAN NOT FROZEN: this analysis is EXPLORATORY."
    return ("!! PLAN MODIFIED AFTER FREEZING (frozen " + str(status["frozen_at_utc"]) + "): " + "; ".join(status["differences"]) +
            "\n!! This analysis is NO LONGER CONFIRMATORY. Say so in the paper, or restore the frozen files.")


def resolve(name: str, cli_value, plan_value, builtin=None, plan_given: bool = False):
    """
    One design parameter. No plan: the CLI value, else the built-in default. With a plan: the plan's value; a CLI value
    that disagrees with it is an ERROR (a frozen parameter must not be overridden silently).
    """
    if not plan_given:
        return cli_value if cli_value is not None else builtin
    if cli_value is not None and cli_value != plan_value:
        raise ValueError(f"--{name.replace('_', '-')} {cli_value!r} conflicts with the frozen plan value {plan_value!r}; edit and re-freeze the plan, or drop the flag")
    return plan_value
