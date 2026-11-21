"""test_build_bias_pairs.py -- CPU only.   python test_build_bias_pairs.py   (run from the folder that holds build_bias_pairs.py and bias_stage0/)"""
import collections
import contextlib
import copy
import io
import json
import os
import sys
import tempfile

import build_bias_pairs as B

CARRIERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bias_stage0", "carriers_proposed.json")


def make_pool(path, n=30):
    rows = []
    for i in range(n):
        text = f"A {20 + i % 7}-year-old patient presents with chest pain. Vitals are stable. " * 9          # about 100 words, like the real plain notes (median 98)
        for attr in ("race", "sex", "ses"):                                   # the plain note is identical across attributes, as in the real pool
            rows.append(dict(case_id=str(i), attribute=attr, condition_label="N", rec_type=None, signal_type=None, note_text=text))
        rows.append(dict(case_id=str(i), attribute="race", condition_label="E+", rec_type=None, signal_type="PRX", note_text=text + "\nInsurance: Medicaid"))     # must be ignored
        rows.append(dict(case_id=str(i), attribute="race", condition_label="N", rec_type="REC_WRONG", signal_type="N", note_text=text + " Recommendation: B."))  # must be ignored
    rows.append(dict(case_id="r1", attribute="ses", condition_label="N", rec_type=None, signal_type=None, note_text="I'm sorry, but I cannot assist with that request."))
    rows.append(dict(case_id="c1", attribute="ses", condition_label="N", rec_type=None, signal_type=None, note_text="Version one of the note. " * 10))
    rows.append(dict(case_id="c1", attribute="race", condition_label="N", rec_type=None, signal_type=None, note_text="Version two of the note. " * 10))
    open(path, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")


def test_plain_notes():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "pool.jsonl")
        make_pool(p)
        notes, info = B.load_plain_notes(p)
        assert len(notes) == 31 and "r1" not in notes and info["refusal_cases_dropped"] == ["r1"] and info["cases_with_conflicting_plain_versions"] == ["c1"]
        assert all("Recommendation" not in v and "Insurance:" not in v for v in notes.values())
    print("PASS: only plain neutral notes are used (recommendation and carrier notes ignored), refusals dropped, conflicting plain versions reported")


def test_exact_pairs_and_counts():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "pool.jsonl")
        make_pool(p)
        notes, _ = B.load_plain_notes(p)
        cfg = B.load_carriers(CARRIERS)
        recs, case_set, use = B.build(notes, cfg)
        rep = B.validate(recs, notes)                                          # raises if any pair differs outside the footer
        n = len(notes)
        enabled = [(a, f) for a, ac in cfg["attributes"].items() for f, fc in ac["forms"].items() if fc.get("enabled", True)]
        single = [(a, f) for a, f in enabled if len(cfg["attributes"][a]["forms"][f]["a"]) == 1]
        multi = [x for x in enabled if x not in single]
        assert rep["pairs"] == n * (len(single) + 3 * len(multi)), rep["pairs"]
        kinds = collections.Counter((r["attribute"], r["form"], r["kind"]) for r in recs if r.get("role") == "pos")
        for a, f in single:
            assert kinds[(a, f, "pair")] == n and kinds[(a, f, "placebo_a")] == 0           # explicit labels: no placebo
        for a, f in multi:
            assert all(kinds[(a, f, k)] == n for k in ("pair", "placebo_a", "placebo_b"))
        assert sum(1 for r in recs if r["kind"] == "base") == n
        # the old pairs differed in about 10% of the words; an exact pair differs only in its footer line
        assert all(v["mean_share_of_words_in_footer_pct"] < 6 for v in rep["by_cell"].values()), rep["by_cell"]
    print("PASS: every pair is the same base note plus one differing footer line; counts match (pair, two placebos for multi-string forms, none for explicit labels); footer is under 6% of the words")


def test_length_matching_and_placebo_and_sets():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "pool.jsonl")
        make_pool(p)
        notes, _ = B.load_plain_notes(p)
        cfg = B.load_carriers(CARRIERS)
        recs, case_set, use = B.build(notes, cfg)
        rep = B.validate(recs, notes)
        assert rep["by_cell"]["ses|PRX|pair"]["mean_abs_footer_word_diff"] <= 0.5, rep["by_cell"]["ses|PRX|pair"]        # strings are length-matched by rank
        # placebo pairs: both strings from the SAME side, and different
        for r in recs:
            if r.get("kind") in ("placebo_a", "placebo_b") and r["role"] == "pos":
                neg = next(x for x in recs if x["id"] == r["id"].replace("|pos", "|neg"))
                side = cfg["attributes"][r["attribute"]]["forms"][r["form"]]["a" if r["kind"] == "placebo_a" else "b"]
                assert r["carrier"] in side and neg["carrier"] in side and r["carrier"] != neg["carrier"]
        # carrier sets: cases split A/B, and the strings used by set-A cases never appear in set-B cases (per attribute and form)
        used = collections.defaultdict(lambda: {"A": set(), "B": set()})
        for r in recs:
            if r.get("kind") in ("pair", "placebo_a", "placebo_b"):
                used[(r["attribute"], r["form"])][r["carrier_set"]].add(r["carrier"])
        for key, d in used.items():
            f = cfg["attributes"][key[0]]["forms"][key[1]]
            if len(f["a"]) > 1:
                assert not (d["A"] & d["B"]), key
            else:
                assert d["A"] == d["B"], key                                          # a single-string form uses the same strings in both sets
        assert sorted(collections.Counter(case_set.values()).values()) == [15, 16]
        # every string is used about equally often within its set (pair kind only)
        cnt = collections.Counter((r["attribute"], r["form"], r["carrier"]) for r in recs if r.get("kind") == "pair")
        for (a, f), fc in ((k, cfg["attributes"][k[0]]["forms"][k[1]]) for k in used if len(cfg["attributes"][k[0]]["forms"][k[1]]["a"]) > 1):
            vals = [cnt[(a, f, s)] for s in fc["a"] + fc["b"] if cnt[(a, f, s)] > 0]
            assert max(vals) - min(vals) <= 2, (a, f, vals)
    print("PASS: strings are length-matched, placebo pairs use two different strings from the same side, cases and carriers split into disjoint sets, and strings are used evenly")


def test_deterministic_and_config_errors():
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "pool.jsonl")
        make_pool(p)
        notes, _ = B.load_plain_notes(p)
        cfg = B.load_carriers(CARRIERS)
        assert B.build(notes, cfg)[0] == B.build(notes, cfg)[0]
        for mutate, expect in ((lambda c: c["attributes"]["race"]["forms"]["PRX"].update(a=["Allison", "Anne"]), "at least"),
                               (lambda c: c["attributes"]["race"]["forms"]["PRX"].update(a=c["attributes"]["race"]["forms"]["PRX"]["a"][:-1] + ["Aisha"]), "different between sides"),
                               (lambda c: c["attributes"]["race"]["forms"]["PRX"].update(template="Patient name: Baker"), "{x}")):
            bad = copy.deepcopy(json.load(open(CARRIERS)))
            mutate(bad)
            path = os.path.join(t, "bad.json")
            json.dump(bad, open(path, "w"))
            try:
                B.load_carriers(path)
                raise AssertionError("bad config accepted")
            except ValueError as e:
                assert expect in str(e), str(e)
    print("PASS: building is deterministic; too few strings, strings shared between sides, and a template without {x} are all refused")


def test_cli():
    with tempfile.TemporaryDirectory() as t:
        p, out = os.path.join(t, "pool.jsonl"), os.path.join(t, "pairs.jsonl")
        make_pool(p)
        old, sys.argv = sys.argv, ["x", "--pool", p, "--carriers", CARRIERS, "--out", out]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = B.main()
        finally:
            sys.argv = old
        text = buf.getvalue()
        assert rc == 0 and "every pair validated" in text and "refusal notes dropped: ['r1']" in text and "ses|PRX|pair" in text
        rows = [json.loads(l) for l in open(out)]
        man = json.load(open(out.replace(".jsonl", "_manifest.json")))
        assert len(rows) == man["n_records"] and man["n_cases"] == 31 and len(man["carriers_sha256"]) == 64
    print("PASS: the command line writes the pairs and a manifest with the carrier file's hash, and prints the footer share of words for every cell")


if __name__ == "__main__":
    test_plain_notes()
    test_exact_pairs_and_counts()
    test_length_matching_and_placebo_and_sets()
    test_deterministic_and_config_errors()
    test_cli()
    print("\nALL BIAS-PAIR BUILDER TESTS PASSED")
