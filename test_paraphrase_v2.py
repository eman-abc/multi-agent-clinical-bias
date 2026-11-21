"""test_paraphrase_v2.py -- CPU.  python test_paraphrase_v2.py   (the sentence-level paraphrase control)"""
import json
import os
import re
import tempfile

import make_paraphrase_controls_v2 as V

SUBS = {"patient": "individual", "reports": "states", "history": "background", "presents": "comes", "severe": "intense", "pain": "discomfort", "radiates": "spreads", "worse": "aggravated",
        "difficulty": "trouble", "swelling": "puffiness", "noted": "observed", "examination": "check"}


class FakeModel:
    """Rewrites a sentence by swapping some words; also can misbehave on request."""
    def __init__(self, mode="good"):
        self.mode, self.calls = mode, 0

    def generate(self, prompts, per_prompt_seeds=None):
        self.calls += 1
        sent = prompts[0].split("Sentence:\n", 1)[1]
        assert prompts[0].startswith(V.NEW_CONVERSATION_PREFIX)                       # every call starts a fresh conversation in the local model wrapper
        if self.mode == "preamble":
            return ["Here is the rewritten sentence: " + sent]
        if self.mode == "demographic":
            return [sent.replace("patient", "woman")]
        out = " ".join(SUBS.get(w.lower().strip(".,"), w) + ("." if w.endswith(".") else "") if w.lower().strip(".,") in SUBS else w for w in sent.split())
        return [out]


class WordTok:
    def encode(self, text):
        return text.split()


NOTE = ("The patient presents with severe pain in the chest and reports difficulty breathing at night. "
        "Vital signs are BP 120/80, HR 72, and temperature 98.6. "
        "The pain radiates to the left arm and is worse with exertion over the last history of two weeks. "
        "Swelling was noted on examination of both ankles with no skin changes at all. "
        "Medications include lisinopril 10 mg daily. "
        "The patient reports no fever, cough, or recent travel in the past month at all.")


def test_split_and_candidates():
    s = V.split_sentences(NOTE)
    assert len(s) == 6 and "".join(t + w for t, w in s) == NOTE
    assert V.split_sentences("One. Two  words here.\nThree?") and "".join(t + w for t, w in V.split_sentences("A b. C d.\n\nE f?")) == "A b. C d.\n\nE f?"
    c = V.candidates(s, 6)
    assert c == [0, 2, 3, 5], c                                                  # the two sentences with numerals are never candidates
    assert V.candidates(s, 2) == [0, 2] or len(V.candidates(s, 2)) == 2
    print("PASS: sentences are split without losing a character, and sentences with numerals are never candidates")


def test_clean_reply_rules():
    o = "The patient reports severe pain in the chest and difficulty breathing at night."
    good = "The patient states intense discomfort in the chest and trouble breathing at night."
    assert V.clean_reply(good, o) == good
    assert V.clean_reply(good[:-1], o) == good                                   # a missing full stop is added
    for bad in ("Here is the rewritten sentence: " + good, "Sure! " + good, good + "\nExtra line.", o, "The patient has pain 3 times.",
                "The woman states intense discomfort in the chest and trouble breathing at night.",
                "The patient states intense discomfort; the recommended treatment is rest and trouble breathing at night.", "Ok.", "word " * 40, "Chest hurts."):
        assert V.clean_reply(bad, o) is None, bad
    print("PASS: a clean rewrite is accepted; a preamble, extra lines, an unchanged sentence, a numeral, a new demographic or recommendation word, and extreme lengths are rejected")


def test_make_one_matches_target_and_keeps_numbers():
    m = FakeModel()
    for target in (6.0, 12.0, 18.0):
        r = V.make_one(NOTE, target, m, WordTok(), V.token_edit_pct, 3.0, 6, 2, 0)
        assert r["valid"], (target, r["actual_edit_pct"], r["reason"])
        assert abs(r["actual_edit_pct"] - target) <= 3.0
        for num in ("120/80", "72", "98.6", "10", "mg"):
            assert num in r["paraphrase"].split() or num in r["paraphrase"], num                  # numerals are untouched
        sents = V.split_sentences(r["paraphrase"])
        assert sents[1][0] == V.split_sentences(NOTE)[1][0] and sents[4][0] == V.split_sentences(NOTE)[4][0]        # the numeral sentences are identical
        assert r["n_sentences_changed"] >= 1 and r["paraphrase"] != NOTE
    big = V.make_one(NOTE, 90.0, m, WordTok(), V.token_edit_pct, 3.0, 6, 2, 0)
    assert not big["valid"] and "outside" in big["reason"] and big["actual_edit_pct"] < 60
    print("PASS: targets of 6, 12 and 18 percent are met within 3 points with the numeral sentences untouched; an unreachable target of 90 is marked invalid with the reason")


def test_misbehaving_model_is_rejected():
    for mode in ("preamble", "demographic"):
        r = V.make_one(NOTE, 12.0, FakeModel(mode), WordTok(), V.token_edit_pct, 3.0, 6, 2, 0)
        assert not r["valid"] and r["paraphrase"] == NOTE and r["n_sentences_changed"] == 0 and "no sentence" in r["reason"], (mode, r)
    print("PASS: a model that adds a preamble or a demographic word produces no usable paraphrase, and the note is returned unchanged and marked invalid")


def test_run_resume_and_schema():
    notes = {"c1": NOTE, "c2": NOTE.replace("pain", "ache"), "c3": "Short."}
    with tempfile.TemporaryDirectory() as t:
        out = os.path.join(t, "p.jsonl")
        n = V.run_v2(notes, 12, FakeModel(), WordTok(), out, model_name="fake")
        rows = {json.loads(l)["case_id"]: json.loads(l) for l in open(out)}
        assert n["written"] == 3 and rows["c1"]["valid"] and not rows["c3"]["valid"]
        for k in ("case_id", "target_pct", "paraphrase", "actual_edit_pct", "valid", "refused", "facts_preserved", "method"):
            assert k in rows["c1"], k                                                                    # the schema collect_stage3.py reads
        assert rows["c1"]["method"] == "sentence-level" and rows["c1"]["target_pct"] == 12
        n2 = V.run_v2(notes, 12, FakeModel(), WordTok(), out)
        assert n2["skipped_existing"] == 3 and n2["written"] == 0
    assert abs(V.token_edit_pct("a b c d".split(), "a x c d e".split()) - 50.0) < 1e-9              # 1 substitution + 1 insertion over 4 tokens
    print("PASS: rows have the schema the collector reads; a rerun skips rows already written; the edit measure matches the cfprompt definition")


if __name__ == "__main__":
    test_split_and_candidates()
    test_clean_reply_rules()
    test_make_one_matches_target_and_keeps_numbers()
    test_misbehaving_model_is_rejected()
    test_run_resume_and_schema()
    print("\nALL PARAPHRASE-V2 TESTS PASSED")
