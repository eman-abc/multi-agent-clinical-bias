#!/usr/bin/env python3
"""
diagnose_unparseable.py -- CPU only. Why are answers missing, and is the missingness related to the design? Reads the LATEST record per (stimulus, model, draw) of a run.

    python diagnose_unparseable.py --run-dir results\\p1v2_run1 --stimuli p1v2_data\\stimuli_luna_p0.jsonl --out diagnose_run1.md

Per model: how many answers are missing, split into refusal / unparseable / transport failure / content filter; the error classes (numbers normalised) and finish reasons; what the replies actually say;
the missing rate by arm, task and every design factor (so a carrier, a source or a channel that triggers refusals is visible); and the cases that account for most of it.
A model whose failures are all 'failed after N attempts' has a rate-limit or routing problem (lower its workers); one whose failures are content filters has a content problem (report it, do not retry).
"""
import argparse
import collections
import glob
import json
import os
import re
import sys


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def klass(r):
    e = r.get("error") or ""
    if e.startswith("content_filter"):
        return "content_filter"
    if e:
        return "transport_failure"
    if r["answer"] is not None:
        return "answered"
    return r.get("confidence") or "unparseable"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stimuli", default=None)
    ap.add_argument("--out", default="diagnose_unparseable.md")
    a = ap.parse_args(argv)
    latest = {}
    for p in sorted(glob.glob(os.path.join(a.run_dir, "steps", "*agent_b_*", "output.jsonl"))):
        for r in jl(p):
            latest[(r["stimulus_id"], r["model"], r["draw"])] = r
    fac = {}
    if a.stimuli:
        for s in jl(a.stimuli):
            fac[s["stimulus_id"]] = {"arm": s["arm"], "task": s["task"], **{k: str(v) for k, v in s["factors"].items() if k in ("condition", "attribute", "form", "kind", "side", "source", "channel", "target_type")}}
    L = ["# Missing-answer diagnosis", ""]
    for model in sorted({k[1] for k in latest}):
        recs = [r for k, r in latest.items() if k[1] == model]
        cls = collections.Counter(klass(r) for r in recs)
        L += [f"## {model}  ({len(recs)} records)", "", f"- outcome: {dict(cls)}", f"- missing: {len(recs) - cls['answered']} ({100 * (len(recs) - cls['answered']) / len(recs):.1f}%)"]
        bad = [r for r in recs if r["answer"] is None]
        if not bad:
            L.append(""); continue
        L.append("- error classes: " + str(collections.Counter(re.sub(r"\d+", "N", (r.get("error") or "none")[:80]) for r in bad).most_common(6)))
        L.append("- finish reasons: " + str(collections.Counter(r.get("finish_reason") for r in bad).most_common(5)))
        L.append("- what the replies say: " + str(collections.Counter((r.get("raw_text") or "")[:50] for r in bad).most_common(5)))
        by = collections.defaultdict(lambda: [0, 0])
        for r in recs:
            for key, val in {"arm": r["arm"], "task": r["task"], **{k: v for k, v in fac.get(r["stimulus_id"], {}).items() if k not in ("arm", "task")}}.items():
                by[(key, val)][1] += 1
                by[(key, val)][0] += int(r["answer"] is None)
        rows = sorted(((100 * m / n, key, val, m, n) for (key, val), (m, n) in by.items() if n >= 20), reverse=True)
        L += ["", "| factor | value | missing | n | rate |", "|---|---|---|---|---|"] + [f"| {key} | {val} | {m} | {n} | {rate:.1f}% |" for rate, key, val, m, n in rows[:14]]
        cc = collections.Counter(r["case_id"] for r in bad)
        L += ["", f"- cases with a missing answer: {len(cc)}; the ten with most: {cc.most_common(10)}", ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
