#!/usr/bin/env python3
"""check_length_confound.py -- CPU only, read-only, safe to run while the GPU batch is running.

    python3 check_length_confound.py

A direction built as (mean of side A) - (mean of side B) can pick up "side A has more text" instead of the content that
is supposed to differ. If privileged-group (E+) notes are systematically longer or shorter than unprivileged (E-) ones,
every bias direction inherits a length axis -- and a length axis would also line up, with a consistent sign, with any
contrast where one side carries an extra sentence (the deference contrast: the pressured prompt adds a sentence).

For each attribute and carrier form it compares E+ against E- on the SAME case (characters and words), and as a
reference compares the neutral note N against its paraphrase P (same facts reworded, no cue). Two-sided exact sign test
on the sign of (E+ minus E-) per case; ties are dropped.
"""
import argparse
import json
import math
from collections import defaultdict


def sign_test_p(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_lengths(rows, attribute, cond_a, cond_b, signal_type=None):
    """{case_id: (a_text, b_text)} for two conditions of one attribute (signal_type=None: any)."""
    side = {cond_a: {}, cond_b: {}}
    for r in rows:
        if r.get("attribute") != attribute or r.get("rec_type") is not None or not r.get("note_text"):
            continue
        c = r.get("condition_label")
        if c in side and (signal_type is None or r.get("signal_type") in (signal_type, None)):
            side[c].setdefault(r["case_id"], r["note_text"])
    return {cid: (side[cond_a][cid], side[cond_b][cid]) for cid in side[cond_a] if cid in side[cond_b]}


def summarize(pairs):
    if not pairs:
        return None
    dc = [len(a) - len(b) for a, b in pairs.values()]
    dw = [len(a.split()) - len(b.split()) for a, b in pairs.values()]
    pos, neg = sum(x > 0 for x in dc), sum(x < 0 for x in dc)
    n = len(dc)
    mean = sum(dc) / n
    sd = (sum((x - mean) ** 2 for x in dc) / max(1, n - 1)) ** 0.5
    return {"n": n, "mean_chars": mean, "sd_chars": sd, "mean_words": sum(dw) / n,
            "pct_a_longer": 100 * pos / n, "pct_b_longer": 100 * neg / n, "p_sign": sign_test_p(pos, neg)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.source_jsonl, encoding="utf-8") if line.strip()]
    print(f"{len(rows)} notes read from {args.source_jsonl}\n")
    print(f"{'attribute':9s} {'carrier':8s} {'comparison':12s} {'n':>4s} {'mean len diff (chars)':>22s} {'words':>7s} {'% A longer':>11s} {'% B longer':>11s} {'sign test p':>12s}")
    flagged = []
    for attr in ("ses", "race", "sex"):
        for sig in ("PRX", "EXP", "IMP"):
            s = summarize(paired_lengths(rows, attr, "E+", "E-", sig))
            if s:
                print(f"{attr:9s} {sig:8s} {'E+ vs E-':12s} {s['n']:4d} {s['mean_chars']:+14.1f} (sd {s['sd_chars']:.1f}) {s['mean_words']:+7.1f} {s['pct_a_longer']:10.0f}% {s['pct_b_longer']:10.0f}% {s['p_sign']:12.2g}")
                if s["p_sign"] < 0.01 and abs(s["mean_chars"]) >= 5:
                    flagged.append((attr, sig, s["mean_chars"]))
        s = summarize(paired_lengths(rows, attr, "N", "P"))
        if s:
            print(f"{attr:9s} {'-':8s} {'N vs P':12s} {s['n']:4d} {s['mean_chars']:+14.1f} (sd {s['sd_chars']:.1f}) {s['mean_words']:+7.1f} {s['pct_a_longer']:10.0f}% {s['pct_b_longer']:10.0f}% {s['p_sign']:12.2g}")
        print()
    if flagged:
        print("SYSTEMATIC LENGTH DIFFERENCE (sign test p < 0.01 and at least 5 characters on average):")
        for attr, sig, m in flagged:
            print(f"  {attr} / {sig}: E+ is {'LONGER' if m > 0 else 'SHORTER'} than E- by {abs(m):.0f} characters on average")
        print("A bias direction built from these pairs can partly encode length. Compare with the Phase 4 pattern: if the attributes flagged here "
              "are the ones whose cosine with deference was consistently negative (SES, race), length is a candidate explanation.")
    else:
        print("No systematic length difference between E+ and E- (p < 0.01 and >= 5 characters) in any attribute / carrier form.")


if __name__ == "__main__":
    main()
