#!/usr/bin/env python3
"""
export_cases.py -- run from the project root (it imports your shared/ package). Writes the cases p1v2 reads, in the SAME order as your existing samples
(random.Random(42).sample is nested: the first 200 here are the 200 you already ran, the first 341 the 341).

    python export_cases.py --out p1v2_data/cases.jsonl            # all 801
    python export_cases.py --out p1v2_data/cases20.jsonl --n 20   # a small set for a smoke test

Each line: {case_id, facts, question, options{A..D}, correct}.  facts = the vignette with the trailing question stem removed (what a note is written from);
question = the full neutralised question (what Agent B's diagnostic prompt shows).
"""
import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p1v2_data/cases.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from shared.crew_pipeline import _strip_trailing_question_stem
    from shared.data_loader import load_amqa_dataset, sample_cases
    records = load_amqa_dataset(dry_run=False)
    chosen = sample_cases(records, a.n or len(records), seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps({"case_id": str(r.question_id), "facts": _strip_trailing_question_stem(r.neutralised_question), "question": r.neutralised_question,
                                "options": dict(r.options), "correct": r.correct_answer_idx}, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} cases to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
