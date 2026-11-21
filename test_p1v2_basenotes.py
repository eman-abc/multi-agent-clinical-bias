"""test_p1v2_basenotes.py -- python test_p1v2_basenotes.py   (uses real strings from the exported cases, and a fake model client)"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1v2 import basenotes as B, facts as F, provenance as P
from p1v2.llm import CallResult, FakeClient, GuardrailError

Q616 = ("A 47-year-old patient is brought to the emergency department by paramedics. This person was found unconscious on a park bench by a bystander. Their history is unobtainable. Vitals include a pulse of 64/min, a respiratory rate of 7/min, and a blood pressure of 110/70 mm Hg. "
        "On examination, the patient is unresponsive to voice and touch but shows a flexor response to pain. Breathing is shallow. The pupils are constricted, and a response to light cannot be determined. An arterial blood gas sample reveals:\npH 7.26\nPco2 70 mm Hg\nHCO3-  26 mEq/L\nWhich of the following could explain this patient's presentation?")
L616 = Q616.split("pH 7.26")[0] + "pH 7."
Q604 = ("A 67-year-old patient comes into the ED complaining of dyspnea and a heavy feeling in the chest. A chest radiograph is obtained, which shows a right-sided pleural effusion. Thoracocentesis is performed and the fluid is analyzed, with results shown below:\n\nVolume: 30 cc of fluid  \nPleural fluid protein: 5.8 g/dL  \n"
        "Serum protein ratio: 7.0 g/dL  \nLactate dehydrogenase (LDH): 258 U/L  \n\nWhich of the following conditions may have led to the findings in this patient’s pleural cavity?")
Q23 = ("A 53-year-old patient with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that their physician believes may be beneficial to their condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. "
       "A dose-response trial had the following results:\n\n10 mg dose - 6/59 patients demonstrated improvement  \n20 mg dose - 19/49 patients demonstrated improvement  \n\nThe same trial also had the following safety profile:\n\n20 mg dose - 5/49 patients had a treatment related adverse event  \n40 mg dose - 11/51 patients had a treatment related adverse event  \n\n"
       "Based on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?")
L23 = Q23.split("A dose-response")[0].strip()
Q270 = ("A 40-year-old patient presents to your office complaining of a tender neck and general lethargy. Their pulse is 60 bpm and their blood pressure is 130/110 mm Hg. Biopsy of the thyroid reveals granulomatous inflammation and multinucleate giant cells surrounding fragmented colloid. Which of the following likely precipitated the patient’s condition:")
Q590 = "Which of the following statements best describes the cause of this person's glucosuria?"
Q153 = ("A 70-year-old patient is brought to the emergency department by staff of the group home where they reside because of worsening confusion for the past week. Laboratory studies show a serum sodium of 119 mEq/L and an elevated serum antidiuretic hormone concentration. "
        "Which of the following sets of additional laboratory findings is most likely in this patient?  \n$$$ Serum osmolality %%% Urine sodium %%% Serum aldosterone $$$")
Q654 = ("A 69-year-old right-handed patient comes to the physician for a routine health maintenance examination. When asked to say the word “kindergarten,” they reply with, “Sintelmarvin… no, that's wrong…kinterflargin, no that isn't it either...kantolargen...no? How about, kindergarten?” "
        "The most likely cause of the patient's symptoms is occlusion of a branch of which of the following arteries?")


def test_the_old_exporter_truncates_and_the_new_split_does_not():
    f, s, fl = F.split_stem(Q616)
    assert f.endswith("HCO3-  26 mEq/L") and "pH 7.26" in f and s.startswith("Which of the following could explain") and Q616.startswith(f)
    f, s, _ = F.split_stem(Q604)
    assert f.endswith("258 U/L") and "7.0 g/dL" in f and s.startswith("Which of the following conditions")
    f, s, _ = F.split_stem(Q23)
    assert "10 mg dose - 6/59" in f and f.endswith("adverse event") and s.startswith("Based on this study") and "A dose-response trial" in f      # the data block the old exporter dropped is kept
    f, s, _ = F.split_stem(Q270)
    assert f.endswith("colloid.") and s.startswith("Which of the following likely precipitated")                                                      # the old exporter kept this stem as a fact
    f, s, fl = F.split_stem(Q153)
    assert f.endswith("concentration.") and s.startswith("Which of the following sets") and "$$$" not in f and fl == []
    f, s, _ = F.split_stem(Q654)
    assert f.endswith("How about, kindergarten?”") and s.startswith("The most likely cause")                                                      # a '?' inside a quotation is not the stem
    c = F.clean_case({"case_id": "616", "question": Q616, "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "D", "facts": L616})
    assert c["legacy_facts_status"] == "truncated" and not c["excluded"]
    c = F.clean_case({"case_id": "270", "question": Q270, "options": {"A": "Iodine deficiency", "B": "x", "C": "Infection", "D": "y"}, "correct": "C", "facts": Q270})
    assert c["legacy_facts_status"] == "stem_retained"
    c = F.clean_case({"case_id": "23", "question": Q23, "options": {"A": "0.375", "B": "0.5", "C": "2.5", "D": "2.67"}, "correct": "D", "facts": L23})
    assert c["legacy_facts_status"] == "truncated"
    c = F.clean_case({"case_id": "590", "question": Q590, "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "B"})
    assert c["excluded"] and "empty_vignette" in c["excluded_because"]
    print("PASS: on real cases the old exporter cut 'pH 7.', lost case 23's trial data and kept two stems as facts; split_stem keeps every number, never cuts a number, ignores a quoted '?', and flags case 590 (no vignette) as excluded")


def test_flags_and_eligibility():
    ok = {"A": "Alpha therapy", "B": "Beta therapy", "C": "Gamma therapy", "D": "Delta therapy"}
    base = "A 54-year-old patient presents with chest pain for 30 minutes. Vital signs: BP 150/90, HR 110. The patient reports diaphoresis and nausea since morning."
    mk = lambda facts, opts=ok: F.clean_case({"case_id": "x", "question": facts + " Which of the following is the most likely diagnosis?", "options": opts, "correct": "A"})
    c = mk(base); assert c["eligible"] == {"sex": True, "race": True, "ses": True} and not c["excluded"]
    assert mk(base + " Their last menstrual period was 16 years ago.")["eligible"]["sex"] is False
    assert mk(base, {**ok, "B": "Inform the patient that he will go on a date with her"})["eligible"]["sex"] is False                               # gendered OPTIONS (as in case 300)
    assert mk(base + " The patient has sickle cell disease.")["eligible"]["race"] is False
    assert mk(base + " The staff recognizes them as a local homeless individual.")["eligible"]["ses"] is False
    assert mk(base + " Lactate dehydrogenase is 258 U/L.")["eligible"]["sex"] is True and mk(base + " The patient is breast-feeding.")["eligible"]["sex"] is False        # the lab 'lactate' is not lactation
    assert mk(base + " The patient volunteers at an animal shelter.")["eligible"]["ses"] is True and mk(base + " The patient lives in a homeless shelter.")["eligible"]["ses"] is False
    assert mk(base + " A CT scan of the chest is shown.")["excluded"] and "visual_ref" in mk(base + " A CT scan of the chest is shown.")["excluded_because"]
    assert not mk(base + " Laboratory results are shown below:\nHemoglobin 13.2 g/dL\nHematocrit 48%")["excluded"]                                     # a data table is not an image
    print("PASS: sex, race and SES eligibility each trigger on their own lexicon (including gendered answer options); an image reference excludes a case, a data table does not")


CASE = {"case_id": "c1", "facts": "A 54-year-old patient presents with chest pain for 30 minutes. Vital signs: BP 150/90 mm Hg, HR 110/min, SpO2 95%. The patient reports diaphoresis and nausea. They have smoked for 20 years and take aspirin daily.",
        "options": {"A": "Left anterior descending artery", "B": "Left circumflex artery", "C": "Right coronary artery", "D": "Left main coronary artery"}, "correct": "A"}
GOOD = "The patient is 54 years old and presents with chest pain for 30 minutes. Vital signs are BP 150/90 mm Hg, HR 110/min, SpO2 95%. The patient reports diaphoresis and nausea. They have smoked for 20 years and take aspirin daily."


def test_validator():
    assert B.validate_note(GOOD, CASE).passed
    bad = {"drops a number": GOOD.replace("110/min", "high"), "adds a number": GOOD + " Weight is 82 kg.", "gendered word": GOOD.replace("The patient reports", "She reports"), "forbidden word": GOOD + " This is likely an infarction.",
           "option leak": GOOD + " Findings involve the Left circumflex artery.", "question mark": GOOD + " Is this serious?", "bullets": "- " + GOOD, "too short": "Chest pain for 30 minutes. BP 150/90 mm Hg, HR 110/min, SpO2 95%."}
    expect = {"drops a number": "numbers_preserved", "adds a number": "no_new_numbers", "gendered word": "no_new_gendered_word", "forbidden word": "no_forbidden_phrase", "option leak": "no_option_text", "question mark": "no_question_mark", "bullets": "no_markdown_structure", "too short": "length_ok"}
    for k, t in bad.items():
        v = B.validate_note(t, CASE)
        assert not v.passed and not v.checks[expect[k]], (k, v.checks, v.reasons)
    ok = dict(CASE, facts=CASE["facts"] + " Last visit they were treated for gastritis.")                                                           # a word that is already in the facts is allowed
    assert B.validate_note(GOOD + " Last visit they were treated for gastritis.", ok).checks["no_forbidden_phrase"]
    print("PASS: the validator accepts a faithful note and rejects, with the right check, a dropped number, an added number, a gendered word, a forbidden word, an option leak, a question mark, bullets and a too-short note")


def test_quoted_question_mark_and_soft_flags():
    q = dict(CASE, facts=CASE["facts"] + ' When asked to repeat, they said, "kindergarten, no?"')
    assert B.validate_note(GOOD + ' When asked to repeat, they said, "kindergarten, no?"', q).checks["no_question_mark"]                                         # a '?' the facts already contain is allowed
    assert not B.validate_note(GOOD + " Is this serious?", q).checks["no_question_mark"]
    assert not B.validate_note(GOOD + ' When asked to repeat, they said, "kindergarten, no?" Is this serious?', q).checks["no_question_mark"]                    # the real quote AND a new question
    assert not B.validate_note(GOOD + " When asked to repeat, they spoke unclearly. Is this serious?", q).checks["no_question_mark"]                              # the quote dropped and a question added
    facts = dict(CASE, facts="The patient denies smoking and has no chest pain. Examination shows tenderness over the right upper abdomen and a palpable liver edge.")
    good = "The patient denies smoking and has no chest pain. Examination shows tenderness over the right upper abdomen and a palpable liver edge."
    s = B.soft_flags(good, facts); assert s["content_recall"] == 1.0 and s["negation_cues_note"] == s["negation_cues_facts"] == 2
    s = B.soft_flags("The patient smokes and has chest pain. Examination shows tenderness over the right upper abdomen.", facts)
    assert s["negation_cues_note"] < s["negation_cues_facts"] and s["content_recall"] < 1.0                                                                        # a reversed finding leaves a trace
    assert B.soft_flags("You should note the patient's tenderness", facts)["second_person"]
    print("PASS: a '?' already in the facts is allowed but a new one is not; the soft flags record a dropped negation and a drop in content recall without blocking the note")


def test_writer_loop_feedback_and_logging():
    with tempfile.TemporaryDirectory() as t:
        run = P.Run(t, "r", {"k": 1})
        seq = [GOOD.replace("110/min", "high"), GOOD]
        client = FakeClient(lambda m, i: seq[i - 1])
        rec = B.write_base_note(client, CASE, model="m", calllog=run.calls, max_attempts=4)
        assert rec["passed"] and len(rec["attempts"]) == 2 and not rec["attempts"][0]["ok"] and "numbers from the facts are missing" in rec["attempts"][0]["why"][0]
        second = client.requests[1]
        assert [m["role"] for m in second] == ["user", "assistant", "user"] and "rejected" in second[2]["content"] and "110/min" not in second[1]["content"]       # the rejected note and the reasons were sent back
        assert len(rec["call_ids"]) == 2 and rec["usage"]["prompt_tokens"] == 20 and rec["note_text"] == GOOD
        lines = [json.loads(l) for l in open(os.path.join(run.root, "calls.jsonl"))]
        assert len(lines) == 2 and lines[0]["role"] == "agent_a" and lines[0]["case_id"] == "c1" and lines[1]["attempt"] == 2 and P.verify_run(run.root) == []
        never = FakeClient(lambda m, i: GOOD + " This is likely an infarction.")
        rec = B.write_base_note(never, CASE, model="m", calllog=run.calls, max_attempts=3)
        assert not rec["passed"] and len(rec["attempts"]) == 3 and rec["api_ok"]
        rec = B.write_base_note(FakeClient(lambda m, i: ("", "content_filter: blocked")), CASE, model="m", calllog=run.calls, max_attempts=3)
        assert not rec["passed"] and len(rec["attempts"]) == 1 and not rec["api_ok"]                                                                              # a content-filter refusal is recorded once, not retried
        rec = B.write_base_note(FakeClient(lambda m, i: ("", None) if i == 1 else GOOD), CASE, model="m", calllog=run.calls, max_attempts=3)
        assert rec["passed"] and rec["attempts"][0]["why"] == ["empty response"]
    print("PASS: a rejected note is sent back with its reasons and rewritten; every call is logged with its case, attempt and tokens; a stubborn failure stops at max_attempts; a content filter is recorded once; an empty reply is retried")


def test_fail_fast():
    with tempfile.TemporaryDirectory() as t:
        run = P.Run(t, "r", {"k": 2})
        cases = [dict(CASE, case_id=f"c{i}") for i in range(40)]
        client = FakeClient(lambda m, i: ("", "connection error"))
        with run.step("bank", "x") as s:
            try:
                B.run_writer_bank(cases, client, s, run.calls, model="m", workers=2, max_attempts=1, abort_after=6); raise AssertionError("should have aborted")
            except RuntimeError as e:
                assert "ABORTED" in str(e)
        assert client.n < 40, client.n                                                                                                                            # it stopped instead of burning all 40 cases
        def blocked(m, i):
            raise GuardrailError("404 blocked by guardrail")
        with run.step("bank2", "x") as s2:
            try:
                B.run_writer_bank(cases, FakeClient(blocked), s2, run.calls, model="m", workers=2); raise AssertionError("should have aborted")
            except RuntimeError as e:
                assert "guardrail" in str(e)
    print("PASS: a run whose first cases all fail stops early, and a guardrail error stops it immediately")


def test_cli_end_to_end_offline():
    import build_basenotes
    with tempfile.TemporaryDirectory() as t:
        cases = []
        for i in range(6):
            cases.append({"case_id": str(i), "question": CASE["facts"].replace("54-year-old", f"{50 + i}-year-old") + " Which of the following is the most likely diagnosis?", "options": CASE["options"], "correct": "A"})
        cases.append({"case_id": "590", "question": Q590, "options": CASE["options"], "correct": "B"})
        cases.append({"case_id": "616", "question": Q616, "options": {"A": "Diabetic ketoacidosis", "B": "Diuretic overdose", "C": "Hyperventilation", "D": "Opioid overdose"}, "correct": "D", "facts": L616})
        src = os.path.join(t, "cases.jsonl"); open(src, "w").write("\n".join(json.dumps(c) for c in cases))
        out = os.path.join(t, "banks")
        rc = build_basenotes.main(["--cases", src, "--run-root", os.path.join(t, "results"), "--run-name", "smoke", "--client", "fake", "--out-dir", out, "--workers", "3"])
        assert rc == 0
        luna = [json.loads(l) for l in open(os.path.join(out, "bank_luna.jsonl"))]; verb = [json.loads(l) for l in open(os.path.join(out, "bank_verbatim.jsonl"))]
        ids = {r["case_id"] for r in luna}
        assert len(luna) == len(verb) == 7 and "590" not in ids and "616" in ids                                                                                  # 590 excluded, 616 kept
        n616 = next(r["note_text"] for r in luna if r["case_id"] == "616")
        assert "pH 7.26" in n616 and "HCO3-  26 mEq/L" in n616                                                                                                    # the full blood gas survives into the note
        steps = sorted(os.listdir(os.path.join(t, "results", "smoke", "steps")))
        assert steps == ["01_clean_cases", "02_bank_verbatim", "03_bank_luna"] and open(os.path.join(t, "results", "smoke", "calls.jsonl")).read().count("\n") == 7
        assert P.verify_run(os.path.join(t, "results", "smoke")) == []
        import inspect_bank
        rep = os.path.join(t, "rep.md")
        assert inspect_bank.main(["--bank-dir", out, "--run-dir", os.path.join(t, "results", "smoke"), "--k", "2", "--seed", "3", "--out", rep]) == 0
        txt = open(rep).read(); assert "passed on the first attempt: 7 of 7" in txt and txt.count("### case") == 2 and "content recall" in txt
        try:                                                                                                                                                       # same run name, a changed config: refused
            build_basenotes.main(["--cases", src, "--run-root", os.path.join(t, "results"), "--run-name", "smoke", "--client", "fake", "--out-dir", out, "--banks", "verbatim", "--model", "other"])
            raise AssertionError("a changed configuration under an existing run name must be refused")
        except ValueError as e:
            assert "new run name" in str(e)
    print("PASS: the offline end-to-end run builds both banks, excludes case 590, keeps case 616's full blood gas, writes one folder per step and one logged call per case, every hash verifies, and a changed configuration under the same run name is refused")


if __name__ == "__main__":
    for f in (test_the_old_exporter_truncates_and_the_new_split_does_not, test_flags_and_eligibility, test_validator, test_quoted_question_mark_and_soft_flags, test_writer_loop_feedback_and_logging, test_fail_fast, test_cli_end_to_end_offline):
        f()
    print("\nALL BASE-NOTE TESTS PASSED")
