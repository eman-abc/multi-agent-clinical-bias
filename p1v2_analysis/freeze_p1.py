"""Freeze the behavioural analysis plan BEFORE running the analysis (the data are already collected, so this is a 'frozen after collection, before analysis' lock).
  python p1v2_analysis\\freeze_p1.py freeze --plan p1v2_analysis\\P1V2_ANALYSIS_PLAN.md --name p1v2
  python p1v2_analysis\\freeze_p1.py verify p1v2_analysis\\lock_p1v2.json
Hashes: the plan, the analysis and freeze scripts, the stimulus file, the cases file and EVERY answer file of the run. Freeze only after the run has finished."""
import argparse, glob, hashlib, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p, root):
    return os.path.relpath(os.path.abspath(p), os.path.abspath(root)).replace(os.sep, "/")


def tracked(root, plan, stim, cases, run):
    fs = [plan, os.path.join(HERE, "analyze_p1v2.py"), os.path.join(HERE, "freeze_p1.py"), os.path.join(root, stim), os.path.join(root, cases)]
    fs += sorted(glob.glob(os.path.join(root, "results", run, "steps", "*", "output.jsonl")))
    return fs


def check_ready(plan):
    text = open(plan, encoding="utf-8").read()
    bad = []
    if "____" in text:
        bad.append("plan still contains template blanks (____)")
    empty = [ln for ln in text.splitlines() if re.match(r"\|\s*\d+\s*\|", ln) and re.search(r"\|\s*\|\s*$", ln)]
    if empty:
        bad.append(f"{len(empty)} prediction row(s) have no probability; first: {empty[0][:70]}")
    return bad


def freeze(plan, name, root, stim, cases, run, force=False):
    bad = check_ready(plan)
    if bad and not force:
        raise SystemExit("refusing to freeze:\n  - " + "\n  - ".join(bad) + "\n(fix these, or pass --force for a deliberate exploratory freeze)")
    rec = dict(name=name, frozen_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), stage="after data collection, before analysis", run=run, stim=stim, cases=cases,
               plan=rel(plan, root), files={rel(f, root): sha(f) for f in tracked(root, plan, stim, cases, run)})
    path = os.path.join(HERE, f"lock_{name}.json")
    json.dump(rec, open(path, "w"), indent=1)
    return path, rec


def verify(lock, root):
    rec = json.load(open(lock))
    problems = []
    for r, h in rec["files"].items():
        p = os.path.join(root, *r.split("/"))
        if not os.path.exists(p):
            problems.append(f"missing: {r}")
        elif sha(p) != h:
            problems.append(f"changed: {r}")
    plan = os.path.join(root, *rec["plan"].split("/"))
    now = {rel(f, root) for f in tracked(root, plan, rec["stim"], rec["cases"], rec["run"])}
    for r in sorted(now - set(rec["files"])):
        problems.append(f"new file since freeze: {r}")
    return (not problems), problems, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--plan", required=True); f.add_argument("--name", default="p1v2"); f.add_argument("--root", default="."); f.add_argument("--force", action="store_true")
    f.add_argument("--stimuli", default=os.path.join("p1v2_data", "stimuli_luna_p0.jsonl")); f.add_argument("--cases", default=os.path.join("p1v2_data", "banks", "p1v2_bank_main", "cases_clean.jsonl")); f.add_argument("--run", default="p1v2_run1")
    v = sub.add_parser("verify"); v.add_argument("lock"); v.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.cmd == "freeze":
        p, rec = freeze(a.plan, a.name, a.root, a.stimuli, a.cases, a.run, a.force)
        print("frozen:", p, "| utc:", rec["frozen_utc"], "| files hashed:", len(rec["files"]))
    else:
        ok, probs, rec = verify(a.lock, a.root)
        print("FROZEN-OK" if ok else "MISMATCH", "| frozen", rec["frozen_utc"], "UTC")
        for q in probs:
            print("  -", q)
        sys.exit(0 if ok else 1)
