#!/usr/bin/env python3
"""
build_bias_pairs.py -- CPU only, standard library only. Builds EXACT minimal pairs for the bias pilot from Paper 1's plain neutral notes.

    python build_bias_pairs.py --pool paper2_results/extracted_paper1_notes.jsonl --carriers bias_stage0/carriers_proposed.json --out bias_stage0/pairs.jsonl

For every case that has a plain neutral note (condition N, no recommendation, no carrier) it appends ONE footer line to that same note, so the two notes of
a pair are identical except for that line. For every attribute and every enabled form of the carrier file it writes:

    pair      the footer of a side-a string against the footer of a side-b string (the contrast whose direction is the bias direction)
    placebo_a two DIFFERENT side-a strings (no group meaning: how much of a direction comes from the words alone)
    placebo_b two DIFFERENT side-b strings

Controls built in:
  * strings of the two sides are length-matched by rank, so a direction cannot be tracking footer length
  * carriers and cases are split into two DISJOINT sets (A and B): a direction fitted on set A cases and carriers is tested on set B cases and carriers
  * every carrier string is used about equally often; assignment is deterministic
  * refusal notes (a model refusing to write a note) are dropped, and every pair is validated to differ only after the base note
Single-string forms (explicit labels such as "Race: Black") have no placebo; they are the positive control.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys

REFUSAL_PHRASES = ("i'm sorry, but i can", "i cannot assist", "i can't assist", "i cannot help with", "i can't help with", "unable to assist")


def looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return len(low.split()) < 60 and any(p in low for p in REFUSAL_PHRASES)


def norm_rec(v):
    return None if v in (None, "", "None") else str(v).upper()


def words(s: str) -> int:
    return len(s.split())


def load_plain_notes(pool_path: str):
    """One plain neutral note per case: condition N, no recommendation, no signal. Returns (notes, info)."""
    texts = collections.defaultdict(list)
    for line in open(pool_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("note_text")
        if t and r.get("condition_label") == "N" and norm_rec(r.get("rec_type")) is None and r.get("signal_type") in (None, "", "None"):
            if t not in texts[str(r.get("case_id"))]:
                texts[str(r.get("case_id"))].append(t)
    notes, refused, conflicting = {}, [], []
    for case in sorted(texts):
        if len(texts[case]) > 1:
            conflicting.append(case)
        t = texts[case][0]
        if looks_like_refusal(t):
            refused.append(case)
        else:
            notes[case] = t.strip()
    return notes, {"cases_with_plain_note": len(texts), "refusal_cases_dropped": refused, "cases_with_conflicting_plain_versions": conflicting}


def load_carriers(path: str) -> dict:
    cfg = json.load(open(path, encoding="utf-8"))
    minp = int(cfg.get("min_per_side", 6))
    for attr, acfg in cfg["attributes"].items():
        for form, f in acfg["forms"].items():
            if not f.get("enabled", True):
                continue
            a, b = f["a"], f["b"]
            if "{x}" not in f["template"]:
                raise ValueError(f"{attr}/{form}: template needs {{x}}")
            if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) & set(b):
                raise ValueError(f"{attr}/{form}: strings must be unique within a side and different between sides")
            if len(a) == 1 and len(b) == 1:
                continue                                                  # a single-string positive control: no placebo possible
            if len(a) < minp or len(b) < minp:
                raise ValueError(f"{attr}/{form}: needs at least {minp} strings per side (has {len(a)} and {len(b)}) so that two disjoint sets and placebos exist")
    return cfg


def footer(f: dict, x: str, surname: str) -> str:
    return f["template"].replace("{x}", x).replace("{surname}", surname)


def match_pairs(a: list, b: list):
    """Pair the strings of the two sides by length rank; returns [(a_i, b_i)] with similar word counts."""
    sa, sb = sorted(a, key=lambda s: (words(s), s)), sorted(b, key=lambda s: (words(s), s))
    n = min(len(sa), len(sb))
    return list(zip(sa[:n], sb[:n]))


def build(notes: dict, cfg: dict):
    surname = cfg.get("surname", "Baker")
    cases = sorted(notes)
    case_set = {c: ("A" if i % 2 == 0 else "B") for i, c in enumerate(cases)}          # balanced, deterministic, disjoint
    recs = [{"id": f"{c}|base", "kind": "base", "case_id": c, "text": notes[c]} for c in cases]
    use = collections.Counter()
    for attr, acfg in cfg["attributes"].items():
        for form, f in acfg["forms"].items():
            if not f.get("enabled", True):
                continue
            single = len(f["a"]) == 1 and len(f["b"]) == 1
            pairs = match_pairs(f["a"], f["b"])
            if single:
                by_set = {"A": pairs, "B": pairs}                       # one string per side: nothing to split, both sets use it (so no disjoint transfer test for this form)
            else:
                by_set = {"A": [p for k, p in enumerate(pairs) if k % 2 == 0], "B": [p for k, p in enumerate(pairs) if k % 2 == 1]}      # alternate by length rank: equal length spread
            rank = collections.Counter()
            for c in cases:
                s = case_set[c]
                plist = by_set[s]
                j = rank[s] % len(plist)
                rank[s] += 1
                (a_k, b_k) = plist[j]
                jobs = [("pair", (a_k, b_k))]
                if not single:
                    # the second string of a placebo: from the same side and the same carrier set, the closest in length (ties: the one used least so far)
                    def nearest(x, pool, tag):
                        cand = [s for s in pool if s != x]
                        return min(cand, key=lambda s: (abs(words(s) - words(x)), use[(attr, form, tag, s)], s))
                    a_next = nearest(a_k, [p[0] for p in plist], "placebo_a_neg")
                    b_next = nearest(b_k, [p[1] for p in plist], "placebo_b_neg")
                    use[(attr, form, "placebo_a_neg", a_next)] += 1
                    use[(attr, form, "placebo_b_neg", b_next)] += 1
                    jobs += [("placebo_a", (a_k, a_next)), ("placebo_b", (b_k, b_next))]
                for kind, (pos_s, neg_s) in jobs:
                    pid = f"{c}|{attr}|{form}|{kind}"
                    for role, x in (("pos", pos_s), ("neg", neg_s)):
                        ft = footer(f, x, surname)
                        recs.append({"id": f"{pid}|{role}", "pair_id": pid, "role": role, "kind": kind, "case_id": c, "attribute": attr, "form": form, "carrier_set": s,
                                     "carrier": x, "footer": ft, "single_string_form": single, "text": notes[c] + "\n" + ft, "footer_start": len(notes[c]) + 1})
                    use[(attr, form, kind, pos_s)] += 1
    return recs, case_set, use


def validate(recs: list, notes: dict) -> dict:
    """Every pair must be identical to the base note up to the footer, differ in the footer, and keep carrier sets disjoint."""
    by_pair = collections.defaultdict(dict)
    for r in recs:
        if r["kind"] == "base":
            continue
        assert r["text"][:r["footer_start"]] == notes[r["case_id"]] + "\n", r["id"]
        assert r["text"][r["footer_start"]:] == r["footer"], r["id"]
        by_pair[r["pair_id"]][r["role"]] = r
    stats = collections.defaultdict(list)
    sets = collections.defaultdict(lambda: {"A": set(), "B": set()})
    for pid, d in by_pair.items():
        p, n = d["pos"], d["neg"]
        assert p["text"][:p["footer_start"]] == n["text"][:n["footer_start"]], pid
        assert p["footer"] != n["footer"], pid
        key = (p["attribute"], p["form"], p["kind"])
        stats[key].append((abs(words(p["footer"]) - words(n["footer"])), max(words(p["footer"]), words(n["footer"])) / max(1, words(notes[p["case_id"]]))))
        if p["kind"] == "pair" and not p["single_string_form"]:
            sets[(p["attribute"], p["form"])][p["carrier_set"]] |= {p["carrier"], n["carrier"]}
    for key, d in sets.items():
        assert not (d["A"] & d["B"]), f"carrier sets overlap for {key}"
    return {"pairs": len(by_pair), "by_cell": {f"{k[0]}|{k[1]}|{k[2]}": {"n": len(v), "mean_abs_footer_word_diff": round(statistics.mean(x[0] for x in v), 2),
                                                                         "mean_share_of_words_in_footer_pct": round(100 * statistics.mean(x[1] for x in v), 1)} for k, v in sorted(stats.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--carriers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()
    notes, info = load_plain_notes(args.pool)
    if args.max_cases:
        notes = dict(sorted(notes.items())[:args.max_cases])
    cfg = load_carriers(args.carriers)
    recs, case_set, use = build(notes, cfg)
    report = validate(recs, notes)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    cfg_hash = hashlib.sha256(open(args.carriers, "rb").read()).hexdigest()
    manifest = {"pool": args.pool, "carriers_file": args.carriers, "carriers_sha256": cfg_hash, "n_cases": len(notes), "n_records": len(recs), **info, **report,
                "carrier_set_sizes": dict(collections.Counter(case_set.values())),
                "uses_per_string_min_max": [min(use.values()), max(use.values())] if use else None}
    json.dump(manifest, open(args.out.replace(".jsonl", "_manifest.json"), "w"), indent=1)
    print(f"cases with a plain note: {info['cases_with_plain_note']}; refusal notes dropped: {info['refusal_cases_dropped']}; cases with conflicting plain versions: {len(info['cases_with_conflicting_plain_versions'])}")
    print(f"{len(notes)} cases used; carrier sets A/B: {manifest['carrier_set_sizes']}; {report['pairs']} pairs; {len(recs)} records -> {args.out}")
    print("every pair validated: identical to the base note up to the footer, different footers, carrier sets disjoint\n")
    print(f"{'attribute|form|kind':26s} {'pairs':>6s} {'mean |footer word diff|':>24s} {'footer share of words':>22s}")
    for k, v in report["by_cell"].items():
        print(f"{k:26s} {v['n']:6d} {v['mean_abs_footer_word_diff']:24.2f} {v['mean_share_of_words_in_footer_pct']:21.1f}%")
    print("\nCompare with the OLD pairs (separate rewrites): a median of about 10% of the words differed and the worst tenth about 28%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())