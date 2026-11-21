#!/usr/bin/env python3
"""
freeze_plan.py -- freeze and verify the analysis plan. CPU only.

    python3 freeze_plan.py --freeze        # write ANALYSIS_PLAN.lock (timestamp + SHA-256 of plan, narrative, code, data)
    python3 freeze_plan.py --verify        # compare the files on disk with the lock
    python3 freeze_plan.py --show          # print the plan's key parameters

Freeze ONCE, BEFORE the final GPU runs, then commit the lock:
    git add analysis_plan.json ANALYSIS_PLAN.md ANALYSIS_PLAN.lock && git commit -m "Freeze analysis plan v1.0"
Optionally also deposit a timestamped copy somewhere you do not control (an OSF registration, or email the three files
to yourself and your advisor): a hash on your own disk proves nothing to a stranger, a third-party timestamp does.
"""
import argparse
import json
import sys

from mech import plan as P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="analysis_plan.json")
    ap.add_argument("--root", default=".")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--show", action="store_true")
    ap.add_argument("--force", action="store_true", help="freeze again even though a lock exists (this is itself a deviation: log it in ANALYSIS_PLAN.md)")
    args = ap.parse_args()

    if args.show:
        plan = P.load_plan(args.plan)
        print(json.dumps({k: plan[k] for k in ("title", "models", "channels", "pressure", "crossfit", "attn_mlp_split") if k in plan}, indent=2))
        return 0
    if args.verify:
        st = P.plan_status(args.plan, args.root)
        print(P.banner(st))
        return 0 if st["status"] == "FROZEN-OK" else 1

    import os
    lock_path = os.path.join(args.root, P.LOCK_NAME)
    if os.path.exists(lock_path) and not args.force:
        print(f"{lock_path} already exists. Freezing again changes what 'frozen' means; use --force only if you will log it as a deviation.")
        return 1
    lock = P.write_lock(args.plan, args.root)
    missing = [f for f, h in {**lock["files"], **lock["data"]}.items() if h is None]
    print(f"frozen at {lock['frozen_at_utc']}: plan sha256 {lock['plan_sha256'][:16]}..., {len(lock['files'])} code files, {len(lock['data'])} data files")
    if missing:
        print("WARNING: these files listed in the plan do not exist yet and are recorded as missing: " + ", ".join(missing))
        print("Create them and freeze again BEFORE running anything, or remove them from the plan.")
    print("Next: git add analysis_plan.json ANALYSIS_PLAN.md ANALYSIS_PLAN.lock && git commit -m 'Freeze analysis plan v1.0'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
