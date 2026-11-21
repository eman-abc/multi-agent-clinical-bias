#!/usr/bin/env python3
"""
audit_cases.py -- CPU only. Cleans every exported case, reports what the old exporter got wrong and which cases are excluded or ineligible, and writes the cleaned cases.

    python audit_cases.py --cases p1v2_data/cases.jsonl --out p1v2_data/cases_clean.jsonl --report p1v2_data/cases_audit.md
"""
import argparse
import collections
import json
import os
import sys

from p1v2 import facts as F


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="p1v2_data/cases_clean.jsonl")
    ap.add_argument("--report", default="p1v2_data/cases_audit.md")
    a = ap.parse_args(argv)
    raw = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    clean = [F.clean_case(r) for r in raw]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n = len(clean)
    legacy = collections.Counter(c["legacy_facts_status"] for c in clean)
    ids = lambda pred: [c["case_id"] for c in clean if pred(c)]
    L = ["# Case audit", "", f"{n} cases read.", "",
         f"Legacy `facts` (the exporter's field) against the cleaned facts: {dict(legacy)}. 'truncated' = the legacy text stops early (often inside a number, or before a data block); 'stem_retained' = the legacy text still contains the question stem.", "",
         f"- legacy truncated ({legacy.get('truncated', 0)}): {ids(lambda c: c['legacy_facts_status'] == 'truncated')}", f"- legacy stem retained ({legacy.get('stem_retained', 0)}): {ids(lambda c: c['legacy_facts_status'] == 'stem_retained')}", "",
         "Legacy cases written from wrong facts (truncated or stem retained) inside the nested samples you already ran: " + ", ".join(f"first {k}: {sum(c['legacy_facts_status'] in ('truncated', 'stem_retained') for c in clean[:k])} of {min(k, n)}" for k in (20, 200, 341, 801)), "",
         f"EXCLUDED from every arm: {sum(c['excluded'] for c in clean)} (no vignette: {ids(lambda c: c['flags']['empty_vignette'])}; image-dependent: {ids(lambda c: c['flags']['visual_ref'])})", "",
         f"Ineligible for SEX carriers: {sum(not c['eligible']['sex'] for c in clean)}; RACE: {sum(not c['eligible']['race'] for c in clean)}; SES: {sum(not c['eligible']['ses'] for c in clean)}.", "",
         f"For manual review (stem not final / not detected): {ids(lambda c: c['flags']['split'])}", "", "Most frequent reasons a case is ineligible:"]
    for key, label in (("sex_terms", "sex"), ("race_terms", "race"), ("ses_terms", "ses")):
        cnt = collections.Counter(t for c in clean for t in c["flags"][key])
        L.append(f"- {label}: {cnt.most_common(12)}")
    open(a.report, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
