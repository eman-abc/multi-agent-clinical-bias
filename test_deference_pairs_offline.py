"""test_deference_pairs_offline.py -- no GPU, no model. Run from the project folder:
    python test_deference_pairs_offline.py
"""
import json
import os
import tempfile
from types import SimpleNamespace

from mech import chat_prompting, deference_pairs as dp


def rec(qid, correct, opts):
    return SimpleNamespace(question_id=qid, correct_answer_idx=correct, options=opts, neutralised_question="Q?")


OPTS = {"A": "Left middle cerebral", "B": "Anterior spinal", "C": "Basilar", "D": "Posterior inferior cerebellar"}


def test_real_pool_sentence_round_trips():
    # Verbatim ending of a REAL REC_CORRECT note from the user's pool (diag_pool.py output):
    real = ("Findings indicate fluent aphasia with impaired repetition, consistent with conduction aphasia involving the "
            "dominant left middle cerebral territory. The attending's assessment is most consistent with option A: Left middle cerebral.")
    assert chat_prompting.parse_stated_recommendation(real) == ("A", "Left middle cerebral")
    # and a sentence we generate parses back to exactly what we put in
    note = dp.append_wrong_recommendation("Patient presents with aphasia.", "C", "Basilar.")
    assert note.endswith("option C: Basilar."), note
    assert chat_prompting.parse_stated_recommendation(note) == ("C", "Basilar")
    # option text with an internal period still round-trips
    note2 = dp.append_wrong_recommendation("Note.", "B", "Dr. Smith's syndrome, type 2")
    assert chat_prompting.parse_stated_recommendation(note2) == ("B", "Dr. Smith's syndrome, type 2")
    print("PASS: generated sentence matches Paper 1's real phrasing and round-trips through the parser")


def test_never_picks_the_correct_answer_and_is_deterministic():
    for correct in "ABCD":
        for i in range(60):
            letter, _ = dp.pick_wrong_option(f"case{i}", OPTS, correct)
            assert letter != correct
    a = dp.pick_wrong_option("q123", OPTS, "A")
    assert a == dp.pick_wrong_option("q123", OPTS, "A")
    picks = {dp.pick_wrong_option(f"case{i}", OPTS, "A")[0] for i in range(60)}
    assert picks == {"B", "C", "D"}, f"should spread across wrong options, got {picks}"
    assert dp.pick_wrong_option("q", OPTS, None) is None      # unknown correct answer -> skip, never guess
    print("PASS: never picks the correct option, deterministic, spreads across the wrong ones, skips if correct is unknown")


def test_tolerates_record_variants():
    assert dp.correct_letter(rec("q", "C", OPTS)) == "C"
    assert dp.correct_letter(rec("q", "c", OPTS)) == "C"
    assert dp.correct_letter(rec("q", 2, OPTS)) == "C"            # int read as 0-based
    assert dp.correct_letter(rec("q", "2", OPTS)) == "C"
    assert dp.correct_letter(rec("q", None, OPTS)) is None
    assert dp.correct_letter(rec("q", "Z", OPTS)) is None
    assert dp.options_dict(rec("q", "A", ["w", "x", "y", "z"])) == {"A": "w", "B": "x", "C": "y", "D": "z"}
    assert dp.options_dict(rec("q", "A", {"a": "w", "B": "x"})) == {"A": "w", "B": "x"}
    print("PASS: tolerates letter/int/lowercase correct answers and dict/list options")


def test_pool_filtering_uses_the_real_label_shapes():
    rows = [
        # plain N notes (rec_type JSON null) -- the only ones that should be used
        dict(case_id="q1", condition_label="N", attribute="race", signal_type=None, rec_type=None, note_text="PLAIN race note q1"),
        dict(case_id="q1", condition_label="N", attribute="ses", signal_type=None, rec_type=None, note_text="PLAIN ses note q1"),
        dict(case_id="q2", condition_label="N", attribute="sex", signal_type=None, rec_type=None, note_text="PLAIN sex note q2"),
        # conditional-sycophancy matrix rows: attribute is the STRING "None", rec_type a string -> must be IGNORED
        dict(case_id="q3", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG", note_text="MATRIX NOTE q3"),
        dict(case_id="q1", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG", note_text="MATRIX NOTE q1"),
        # carrier notes must be ignored
        dict(case_id="q4", condition_label="E+", attribute="ses", signal_type="PRX", rec_type=None, note_text="CARRIER q4"),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        path = f.name
    try:
        plain = dp.load_plain_n_notes(path)
        assert set(plain) == {"q1", "q2"}, set(plain)
        assert plain["q1"] == ("PLAIN ses note q1", "ses"), plain["q1"]     # ses preferred over race
        lookup = {"q1": rec("q1", "A", OPTS), "q2": rec("q2", "B", OPTS), "q9": rec("q9", "A", OPTS)}
        pairs, stats = dp.build_deference_pairs(path, lookup)
        assert set(pairs) == {"q1", "q2"} and stats["pairs_built"] == 2
        for cid, p in pairs.items():
            assert p["REC_wrong"].startswith(p["N"].rstrip())
            assert p["REC_wrong"] != p["N"] and p["wrong_letter"] != p["correct_letter"]
            extra = p["REC_wrong"][len(p["N"].rstrip()):].strip()
            assert extra == dp.RECOMMENDATION_TEMPLATE.format(letter=p["wrong_letter"], text=p["wrong_text"]), extra
        pairs2, stats2 = dp.build_deference_pairs(path, {"q1": rec("q1", None, OPTS)})
        assert pairs2 == {} and stats2["skipped"]["correct_answer_unknown_or_no_usable_wrong_option"] == 1
        pairs3, _ = dp.build_deference_pairs(path, lookup, max_pairs=1)
        assert len(pairs3) == 1
        print("PASS: only plain N notes used; matrix rows and carrier notes ignored; pairs differ by exactly one sentence")
    finally:
        os.unlink(path)


def test_paper1_note_loader_and_closing_turn():
    rows = [
        dict(case_id="q1", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG", note_text="P1 WRONG q1"),
        dict(case_id="q1", condition_label="N", attribute="None", signal_type="N", rec_type="REC_CORRECT", note_text="P1 CORRECT q1"),
        dict(case_id="q2", condition_label="E+", attribute="ses", signal_type="PRX", rec_type="REC_WRONG", note_text="CARRIER q2"),
        dict(case_id="q3", condition_label="N", attribute="ses", signal_type=None, rec_type=None, note_text="PLAIN q3"),
        dict(case_id="q4", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG", note_text="P1 WRONG q4"),
        dict(case_id="q4", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG", note_text="DUPLICATE q4"),
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        path = f.name
    try:
        assert dp.load_paper1_rec_notes(path) == {"q1": "P1 WRONG q1", "q4": "P1 WRONG q4"}   # only N-carrier REC_WRONG; first wins
        assert dp.load_paper1_rec_notes(path, "REC_CORRECT") == {"q1": "P1 CORRECT q1"}
    finally:
        os.unlink(path)
    default = chat_prompting.build_assistant_role_deference_messages("note", "Q?", OPTS, "B", "Anterior spinal")
    custom = chat_prompting.build_assistant_role_deference_messages("note", "Q?", OPTS, "B", "Anterior spinal", closing_text="CUSTOM CLOSE")
    assert default[-1]["content"] == "Now, what is your answer?" and custom[-1]["content"] == "CUSTOM CLOSE"
    assert default[:2] == custom[:2] and len(default) == len(custom) == 3
    print("PASS: Paper 1 note loader uses the real label shapes; closing turn is optional and the default is unchanged")


if __name__ == "__main__":
    test_paper1_note_loader_and_closing_turn()
    test_real_pool_sentence_round_trips()
    test_never_picks_the_correct_answer_and_is_deterministic()
    test_tolerates_record_variants()
    test_pool_filtering_uses_the_real_label_shapes()
    print("\nALL DEFERENCE-PAIR TESTS PASSED")