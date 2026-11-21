#!/usr/bin/env python3
"""
build_stimuli.py -- from a cleaned-cases file and a base-note bank, write every stimulus of the chosen arms to one JSONL (the file the Agent B runner reads).

    python build_stimuli.py --bank-dir p1v2_data\\banks\\p1v2_bank_main --bank luna --out p1v2_data\\stimuli_luna.jsonl
    python build_stimuli.py --bank-dir p1v2_data\\banks\\p1v2_bank_main --bank verbatim --arms bias --out p1v2_data\\stimuli_verbatim_bias.jsonl      # the writer-free control

Excluded cases are never built; a case without a note in the bank (writer failed) is skipped and listed; an attribute a case is ineligible for gets no carrier.
"""
import argparse
import collections
import json
import os
import sys

from p1v2 import stimuli as S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--bank", default="luna", choices=["luna", "verbatim"])
    ap.add_argument("--arms", nargs="+", default=["bias", "conformity", "interaction"], choices=["bias", "conformity", "interaction"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cases", type=int, default=None, help="use only the first N cases of the file (the sample is nested)")
    a = ap.parse_args(argv)
    cases = [json.loads(l) for l in open(os.path.join(a.bank_dir, "cases_clean.jsonl"), encoding="utf-8") if l.strip()]
    notes = {r["case_id"]: r["note_text"] for r in map(json.loads, open(os.path.join(a.bank_dir, f"bank_{a.bank}.jsonl"), encoding="utf-8"))}
    cases = cases[: a.n_cases] if a.n_cases else cases
    n, skipped, no_note = 0, [], []
    counts = collections.Counter()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for c in cases:
            if c["excluded"]:
                skipped.append(c["case_id"]); continue
            if c["case_id"] not in notes:
                no_note.append(c["case_id"]); continue
            case = {"case_id": c["case_id"], "question": c["question"], "options": c["options"], "correct": c["correct"]}
            for s in S.build_case_stimuli(case, notes[c["case_id"]], arms=tuple(a.arms), eligible=c["eligible"]):
                f.write(json.dumps({**s.to_json(), "bank": a.bank}, ensure_ascii=False) + "\n")
                counts[(s.arm, s.task)] += 1; n += 1
    print(json.dumps({"stimuli": n, "by_arm_task": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())}, "excluded_cases": len(skipped), "cases_without_a_note": no_note, "out": a.out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
