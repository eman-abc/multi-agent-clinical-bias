#!/usr/bin/env python3
"""
build_minimal_pairs.py -- rebuild the bias stimuli as EXACT minimal pairs. CPU only, no API.

PROBLEM (measured with show_note_pairs.py): Paper 1's E+ and E- notes for one case are separate rewrites. A typical pair
differs by ~16 words (14% of the note); only 22-27% differ by 8 words or fewer. Everything that differs for reasons
unrelated to the demographic cue is noise in the bias direction, which is a likely reason its split-half reliability is
so low (0.10-0.13; ~0 for sex).

FIX: take the neutral note N and append ONE footer line, "<label>: <carrier value>.", with the carrier value drawn from
the values Paper 1 itself used on each side:
      E+  =  N + " Insurance: Blue Cross PPO."          E-  =  N + " Insurance: Medicaid."
so the pair differs only in the carrier words -- the same construction as the deference pairs.

    python3 build_minimal_pairs.py --inspect                 # what do the carriers look like? (read-only; run this first)
    python3 build_minimal_pairs.py                           # build paper2_results/minimal_pairs.jsonl for the 140 Paper 1 cases
    python3 build_minimal_pairs.py --extend-amqa 200         # ... plus 200 more AMQA vignettes (needs shared/data_loader)

The output uses signal_type "MIN" for E+/E-; N rows carry signal_type null, so the existing loaders work unchanged:
    load_matched_triples(path, attribute=..., model=None, signal_type="MIN").

CAVEATS, stated rather than hidden:
  * Where an attribute has only a couple of distinct carrier strings (SES may have just "Blue Cross PPO" / "Medicaid"),
    a very reliable direction can simply mean "this token string is present". Reliability would then partly measure
    lexical identity, not a representation of social status. Surface variety (several values per side) and transfer
    to the implicit carrier forms are what separate the two readings; --inspect shows how much variety exists.
  * Cases beyond Paper 1's 140 get the AMQA vignette (without its final question sentence) as the neutral note: a
    different style from Agent A's triage notes. No paraphrase control exists for them.
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# "<Label>: <value>." at the very END of a note; the label must open a sentence.
FOOTER_RE = re.compile(r"(?:^|(?<=[.!?]\s))([A-Z][A-Za-z]+(?: [A-Za-z]+){0,3}):\s+([^:\n]{2,80}?)\.?\s*$")
SIGNAL = "MIN"


def extract_footer(note: str):
    """(label, value, note_without_footer) from the end of a note, or None."""
    m = FOOTER_RE.search(note.strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip(), note.strip()[: m.start()].rstrip()


def vignette_without_question(stem: str) -> str:
    """AMQA stems end with a question sentence; the prompt shows the question separately, so drop it from the note."""
    s = stem.strip()
    if s.endswith("?"):
        cut = s.rfind(". ", 0, len(s) - 1)
        if cut > 0:
            return s[: cut + 1]
    return s


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_footers(rows, attribute):
    """{ 'E+': [(case, label, value)], 'E-': [...] } for the administrative (PRX) carrier notes; plus failure count."""
    out = {"E+": [], "E-": []}
    fails = 0
    for r in rows:
        if r.get("attribute") != attribute or r.get("signal_type") != "PRX" or r.get("rec_type") is not None or not r.get("note_text"):
            continue
        c = r.get("condition_label")
        if c not in out:
            continue
        got = extract_footer(r["note_text"])
        if got is None:
            fails += 1
        else:
            out[c].append((r["case_id"], got[0], got[1]))
    return out, fails


def footer_label_for(foot, min_share=0.05):
    """The label to use for BOTH sides: the most common one; the labels that clear min_share of footers are 'real'."""
    labels = Counter(l for side in foot.values() for _, l, _ in side)
    total = sum(labels.values())
    real = {l for l, n in labels.items() if total and n / total >= min_share}
    return (labels.most_common(1)[0][0] if labels else None), real, labels


def inspect(rows):
    for attr in ("ses", "race", "sex"):
        foot, fails = collect_footers(rows, attr)
        n = sum(len(v) for v in foot.values())
        label, real, labels = footer_label_for(foot)
        print(f"=== {attr}: {n} administrative-carrier notes parsed, {fails} without a recognisable 'Label: value.' ending ===")
        print(f"  footer labels: {dict(labels.most_common(6))}   -> will use {label!r} on both sides")
        for side in ("E+", "E-"):
            vals = Counter(v for _, l, v in foot[side] if l in real)
            wc = Counter(len(v.split()) for v in vals.elements())
            print(f"  {side}: {len(vals)} distinct values over {sum(vals.values())} notes; words per value {dict(sorted(wc.items()))}")
            print(f"      most common: {dict(vals.most_common(6))}")
        print()
    print("READ THIS: (1) 'without a recognisable ending' should be small; (2) each side should have more than a couple of distinct "
          "values, or reliability will partly measure which exact string is present; (3) check the labels and values look like carriers.")


def build_pool(rows, attribute, min_share=0.05):
    foot, _ = collect_footers(rows, attribute)
    label, real, _ = footer_label_for(foot, min_share)
    pools = {s: sorted({v for _, l, v in foot[s] if l in real}) for s in ("E+", "E-")}
    return label, pools


def pick_values(rng, pools, match_words=True, tries=40):
    vp, vm = rng.choice(pools["E+"]), rng.choice(pools["E-"])
    if match_words:
        for _ in range(tries):
            if len(vp.split()) == len(vm.split()):
                break
            vp, vm = rng.choice(pools["E+"]), rng.choice(pools["E-"])
    return vp, vm


def plain_base_notes(rows):
    """case_id -> the plain neutral note (condition N, no recommendation), preferring ses > race > sex like the deference pairs."""
    order = {"ses": 0, "race": 1, "sex": 2}
    best = {}
    for r in rows:
        if r.get("condition_label") == "N" and r.get("rec_type") is None and r.get("note_text"):
            rank = order.get(r.get("attribute"), 3)
            if r["case_id"] not in best or rank < best[r["case_id"]][0]:
                best[r["case_id"]] = (rank, r["note_text"])
    return {c: t for c, (_, t) in best.items()}


def build_rows(rows, seed=0, match_words=True, extra_bases: dict | None = None, min_share=0.05):
    bases = plain_base_notes(rows)
    source = {c: "paper1_plain_N" for c in bases}
    for c, t in (extra_bases or {}).items():
        if c not in bases:
            bases[c] = t
            source[c] = "amqa_vignette"
    out, stats = [], {}
    for attr in ("ses", "race", "sex"):
        label, pools = build_pool(rows, attr, min_share)
        if not label or not pools["E+"] or not pools["E-"]:
            print(f"WARNING: no usable carriers found for {attr}; skipped", file=sys.stderr)
            continue
        stats[attr] = {"label": label, "n_values_plus": len(pools["E+"]), "n_values_minus": len(pools["E-"])}
        for cid in sorted(bases):
            base = bases[cid].rstrip()
            rng = random.Random(f"{seed}|{attr}|{cid}")
            vp, vm = pick_values(rng, pools, match_words)
            common = {"case_id": cid, "attribute": attr, "rec_type": None, "source_experiment": "minimal_pairs", "base_source": source[cid]}
            out.append({**common, "condition_label": "N", "signal_type": None, "note_text": base})
            out.append({**common, "condition_label": "E+", "signal_type": SIGNAL, "note_text": f"{base} {label}: {vp}.", "carrier_value": vp})
            out.append({**common, "condition_label": "E-", "signal_type": SIGNAL, "note_text": f"{base} {label}: {vm}.", "carrier_value": vm})
    return out, stats


def validate(out):
    """Every E+/E- pair must differ ONLY in the carrier words."""
    by = defaultdict(dict)
    for r in out:
        by[(r["case_id"], r["attribute"])][r["condition_label"]] = r
    bad = 0
    diffs = []
    for k, d in by.items():
        if "E+" in d and "E-" in d:
            a, b = d["E+"]["note_text"].split(), d["E-"]["note_text"].split()
            i = 0
            while i < min(len(a), len(b)) and a[i] == b[i]:
                i += 1
            diffs.append(max(len(a), len(b)) - i)
            allowed = len(d["E+"]["carrier_value"].split()) + len(d["E-"]["carrier_value"].split()) + 2
            if max(len(a), len(b)) - i > allowed:
                bad += 1
    return bad, (statistics.median(diffs) if diffs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--out", default="paper2_results/minimal_pairs.jsonl")
    ap.add_argument("--inspect", action="store_true", help="only report what the carriers look like")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-match-words", action="store_true", help="do not try to give both sides' carriers the same word count")
    ap.add_argument("--extend-amqa", type=int, default=0, help="also build pairs for this many additional AMQA vignettes (default 0)")
    args = ap.parse_args()
    rows = load_rows(args.source_jsonl)
    if args.inspect:
        inspect(rows)
        return 0

    extra = {}
    if args.extend_amqa > 0:
        from shared.data_loader import load_amqa_dataset
        have = {r["case_id"] for r in rows}
        for rec in sorted(load_amqa_dataset(dry_run=False), key=lambda r: r.question_id):
            if rec.question_id in have:
                continue
            base = vignette_without_question(rec.neutralised_question)
            if len(base) >= 60:
                extra[rec.question_id] = base
            if len(extra) >= args.extend_amqa:
                break
        print(f"extending with {len(extra)} AMQA vignettes (neutral note = the vignette without its final question)")
    out, stats = build_rows(rows, args.seed, not args.no_match_words, extra)
    bad, med = validate(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} rows to {args.out}: {len({r['case_id'] for r in out})} cases x {len(stats)} attributes")
    for a, s in stats.items():
        print(f"  {a}: label {s['label']!r}, {s['n_values_plus']} distinct E+ values, {s['n_values_minus']} distinct E- values")
    print(f"validation: median {med} trailing words differ between E+ and E-; pairs differing by more than the carrier words: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
