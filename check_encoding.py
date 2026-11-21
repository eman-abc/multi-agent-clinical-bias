#!/usr/bin/env python3
"""
check_encoding.py -- CPU. Where do damaged characters ('ô' and 'à' where a curly quote or an ellipsis should be) sit in the notes? ('ö' is NOT flagged: it is correct in Sjögren and Schönlein.)

    python check_encoding.py <notes.jsonl> [<more.jsonl> ...] [--pairs bias_stage0/pairs.jsonl]

For every file: how many notes have proper typographic characters, how many have the damaged ones, WHICH CASES they belong to (with a snippet of the damaged text), and, with --pairs, whether those
cases are among the cases of the bias pilot and the interaction runs. Windows consoles can draw these letters themselves (Python writes cp1252 bytes, the console reads them as cp437), so switch the console
to UTF-8 first ([Console]::OutputEncoding = [Text.UTF8Encoding]::new(); $env:PYTHONUTF8 = "1") and trust only the counts of this script, which read the file directly.

Damaged text is a property of the SOURCE vignette: it feeds every note of its case in every condition, so it cannot create a difference between E+ and E- notes. It can still change how the model reads
that case. If a few cases carry it, list them in "exclude_cases" of stage3/natural_config.json and rerun a sensitivity analysis without them.
"""
import argparse
import collections
import json

BAD = "\u00f4\u00e0"                  # o-circumflex and a-grave only: o-umlaut is correct in Sjogren and Schonlein, so it is NOT a damage signature
GOOD = "\u201c\u201d\u2018\u2019\u2026\u2014\u2013"


def scan(path: str, case_field: str = "case_id", note_field: str = "note_text"):
    n = good = 0
    bad_by_case = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get(note_field) or ""
        n += 1
        good += any(c in t for c in GOOD)
        i = next((k for k, c in enumerate(t) if c in BAD), None)
        if i is not None:
            bad_by_case[str(r.get(case_field, "?"))].append(t[max(0, i - 45): i + 45])
    return n, good, bad_by_case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pairs", default=None, help="the bias pilot's pairs.jsonl: report which damaged cases are among its cases")
    args = ap.parse_args()
    pilot = set()
    if args.pairs:
        pilot = {json.loads(l)["case_id"] for l in open(args.pairs, encoding="utf-8") if l.strip()}
    for path in args.paths:
        n, good, bad = scan(path)
        nb = sum(len(v) for v in bad.values())
        print(f"\n{path}\n  {n} notes; with typographic quotes, ellipses or dashes: {good}; with the damaged characters: {nb} notes in {len(bad)} case(s)")
        for case, snips in sorted(bad.items(), key=lambda kv: -len(kv[1])):
            tag = ""
            if pilot:
                tag = "   <-- IN the bias pilot's cases" if case in pilot else "   (not among the bias pilot's cases)"
            print(f"  case {case}: {len(snips)} notes{tag}   e.g. {snips[0]!r}")
        if bad:
            print(f"  exclude_cases for the config: {sorted(bad)}")
        elif good:
            print("  no damaged characters: nothing to exclude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())