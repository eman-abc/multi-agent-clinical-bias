#!/usr/bin/env python3
"""
inspect_bank.py -- CPU only. Read the base-note bank the way a reviewer would: how often did the writer need a second attempt and why, what did it change, and show real facts/note pairs side by side.

    python inspect_bank.py --bank-dir p1v2_data\\banks\\p1v2_smoke20b --run-dir results\\p1v2_smoke20b --k 6 --seed 1 --out bank_inspection.md

Reports: first-attempt pass rate and the reasons for rejected attempts; word ratio, content recall and negation-cue statistics; every note whose content recall is below 0.85 or that has fewer negation cues than its facts
(a dropped 'denies' / 'no' / 'without' reverses a finding: read those by eye); and k random pairs with a fixed seed, so the audit is reproducible. Pre-registration suggestion: read at least 40 pairs before the full bank is used.
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="bank_inspection.md")
    a = ap.parse_args(argv)
    cases = {c["case_id"]: c for c in jl(os.path.join(a.bank_dir, "cases_clean.jsonl"))}
    steps = glob.glob(os.path.join(a.run_dir, "steps", "*bank_luna", "output.jsonl"))
    recs = [r for p in steps for r in jl(p)]
    L = ["# Base-note bank inspection", "", f"{len(recs)} writer records from {len(steps)} step(s)."]
    first = sum(1 for r in recs if r["attempts"] and r["attempts"][0]["ok"])
    L.append(f"- passed on the first attempt: {first} of {len(recs)}; passed overall: {sum(r['passed'] for r in recs)}")
    why = collections.Counter(w.split(":")[0] for r in recs for at in r["attempts"] if not at["ok"] for w in at["why"])
    L.append(f"- reasons attempts were rejected: {dict(why)}")
    usage = [r.get("usage") or {} for r in recs]
    L.append(f"- tokens: prompt {sum(u.get('prompt_tokens', 0) for u in usage)}, completion {sum(u.get('completion_tokens', 0) for u in usage)}")
    soft = [(r, r["soft"]) for r in recs if r.get("soft")]
    if soft:
        wr = [s["word_ratio"] for _, s in soft]; cr = [s["content_recall"] for _, s in soft]
        L.append(f"- word ratio (note / facts): median {statistics.median(wr):.2f}, range {min(wr):.2f} to {max(wr):.2f}; content recall: median {statistics.median(cr):.2f}, minimum {min(cr):.2f}")
        L.append(f"- second-person words in {sum(s['second_person'] for _, s in soft)} notes")
        sus = [(r["case_id"], s) for r, s in soft if s["content_recall"] < 0.85 or s["negation_cues_note"] < s["negation_cues_facts"]]
        L += ["", f"## Read these by eye ({len(sus)})", ""] + [f"- case {cid}: recall {s['content_recall']}, negation cues facts {s['negation_cues_facts']} / note {s['negation_cues_note']}" for cid, s in sus]
    bank = {r["case_id"]: r for r in jl(os.path.join(a.bank_dir, "bank_luna.jsonl"))}
    if soft:
        from p1v2.basenotes import _NEG
        sents = lambda text: [s.strip() for s in text.replace("\n", " ").split(". ") if _NEG.search(s)]
        L += ["", "## The flagged cases: every sentence with a negation or absence cue, facts then note (is each finding still said the same way?)"]
        for cid, s in sus:
            L += ["", f"### case {cid}  (recall {s['content_recall']})", "facts:"] + [f"- {x}" for x in sents(cases[cid]["facts"])] + ["note:"] + [f"- {x}" for x in sents(bank[cid]["note_text"])]
    ids = sorted(bank)
    random.Random(a.seed).shuffle(ids)
    L += ["", f"## {a.k} random pairs (seed {a.seed})"]
    for cid in ids[: a.k]:
        L += ["", f"### case {cid}", "**facts**", "", cases[cid]["facts"], "", "**note**", "", bank[cid]["note_text"]]
    text = "\n".join(L) + "\n"
    open(a.out, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
