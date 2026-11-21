#!/usr/bin/env python3
"""show_note_pairs.py -- CPU only, read-only, safe to run while a GPU job is running.

    python3 show_note_pairs.py                    # summary + 2 example diffs per attribute
    python3 show_note_pairs.py --examples 4 --attribute race

Are the privileged (E+) and unprivileged (E-) notes of one case MINIMAL PAIRS -- identical except for the carrier line --
or two independent rewrites that also differ in wording, order and detail? A bias direction is the average difference
between the two sides; every word that differs for reasons unrelated to the demographic cue is noise in that average.

Per attribute (administrative carrier, "PRX") this reports, over the cases:
  * E+ vs E-  : words that differ (inserted + deleted + replaced) and that as a share of the longer note
  * E+ vs N   : the same against the neutral note, which shows how the carrier note was produced
and prints a few word-level diffs:  [-removed-]{+added+}  with 4 words of context, unchanged stretches elided.
For a true minimal pair the E+ vs E- difference is only the carrier words (a handful).
"""
import argparse
import json
import statistics
from difflib import SequenceMatcher


def word_diff(a: str, b: str):
    """(changed_words, share_of_longer, marked_text) for two notes, word level."""
    wa, wb = a.split(), b.split()
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    changed, parts = 0, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            run = wa[i1:i2]
            parts.append(("keep", run))
        else:
            changed += max(i2 - i1, j2 - j1)
            parts.append(("chg", (wa[i1:i2], wb[j1:j2])))
    return changed, changed / max(1, max(len(wa), len(wb))), parts


def render(parts, context=4):
    out = []
    for k, (kind, val) in enumerate(parts):
        if kind == "keep":
            n = len(val)
            if n <= 2 * context:
                out.append(" ".join(val))
            else:
                head = " ".join(val[:context]) if k > 0 else ""
                tail = " ".join(val[-context:]) if k < len(parts) - 1 else ""
                out.append(f"{head} ...({n - (context if k > 0 else 0) - (context if k < len(parts) - 1 else 0)} words unchanged)... {tail}".strip())
        else:
            old, new = val
            out.append("[-" + " ".join(old) + "-]" if old else "")
            out.append("{+" + " ".join(new) + "+}" if new else "")
    return " ".join(x for x in out if x)


def collect(rows, attribute, signal="PRX"):
    side = {"E+": {}, "E-": {}, "N": {}}
    for r in rows:
        if r.get("attribute") != attribute or r.get("rec_type") is not None or not r.get("note_text"):
            continue
        c = r.get("condition_label")
        if c in ("E+", "E-") and r.get("signal_type") == signal:
            side[c].setdefault(r["case_id"], r["note_text"])
        elif c == "N" and r.get("signal_type") in (None, signal):
            side["N"].setdefault(r["case_id"], r["note_text"])
    return side


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--attribute", choices=["ses", "race", "sex"], default=None)
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--minimal-max-words", type=int, default=8, help="a pair counts as 'minimal' if at most this many words differ")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.source_jsonl, encoding="utf-8") if line.strip()]
    for attr in ([args.attribute] if args.attribute else ["ses", "race", "sex"]):
        side = collect(rows, attr)
        cases = sorted(c for c in side["E+"] if c in side["E-"])
        if not cases:
            print(f"=== {attr}: no complete E+/E- pairs ===\n")
            continue
        ee = [word_diff(side["E+"][c], side["E-"][c]) for c in cases]
        en = [word_diff(side["E+"][c], side["N"][c]) for c in cases if c in side["N"]]
        lens = [len(side["E+"][c].split()) for c in cases]
        minimal = sum(1 for ch, _, _ in ee if ch <= args.minimal_max_words)
        print(f"=== {attr} (administrative carrier), {len(cases)} cases; E+ notes average {statistics.mean(lens):.0f} words ===")
        print(f"  E+ vs E- : median {statistics.median(c for c, _, _ in ee):.0f} words differ ({100 * statistics.median(s for _, s, _ in ee):.0f}% of the note); "
              f"{minimal}/{len(cases)} pairs ({100 * minimal / len(cases):.0f}%) differ by at most {args.minimal_max_words} words")
        if en:
            print(f"  E+ vs N  : median {statistics.median(c for c, _, _ in en):.0f} words differ ({100 * statistics.median(s for _, s, _ in en):.0f}% of the note)")
        for cid, (ch, share, parts) in list(zip(cases, ee))[: args.examples]:
            print(f"  example {cid}: {ch} words differ")
            print("    " + render(parts)[:700])
        print()
    print("READING IT: if few pairs are 'minimal', the bias contrast contains a lot of wording variation that has nothing to do with the "
          "demographic cue. That is noise in every bias direction, and it would explain low split-half reliability even at 100 cases.")


if __name__ == "__main__":
    main()
