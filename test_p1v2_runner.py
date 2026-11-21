"""test_p1v2_runner.py -- python test_p1v2_runner.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1v2 import provenance as P, runner as R, stimuli as S
from p1v2.llm import CallResult, FakeClient, GuardrailError
import build_stimuli, run_agent_b

OPTS = {"A": "LAD", "B": "LCx", "C": "RCA", "D": "Left main"}


def make_bank(root, n=4, luna=True):
    os.makedirs(root, exist_ok=True)
    cases, notes = [], []
    for i in range(n):
        q = f"A {50 + i}-year-old patient presents with chest pain for 30 minutes. Vital signs: BP 150/90, HR 110. The patient reports diaphoresis. Which artery is most likely occluded?"
        cases.append({"case_id": str(i), "question": q, "options": OPTS, "correct": "A", "excluded": i == 3, "eligible": {"ses": True, "race": True, "sex": i != 1}})
        notes.append({"case_id": str(i), "bank": "luna", "source": "w", "note_text": f"The patient is {50 + i} years old and presents with chest pain for 30 minutes. Vital signs: BP 150/90, HR 110. The patient reports diaphoresis.", "note_sha": "x"})
    with open(os.path.join(root, "cases_clean.jsonl"), "w") as f: f.write("\n".join(json.dumps(c) for c in cases))
    with open(os.path.join(root, "bank_luna.jsonl"), "w") as f: f.write("\n".join(json.dumps(c) for c in notes[:3] + notes[3:]))
    return cases


def test_build_stimuli_respects_exclusion_and_eligibility():
    with tempfile.TemporaryDirectory() as t:
        make_bank(t); out = os.path.join(t, "s.jsonl")
        assert build_stimuli.main(["--bank-dir", t, "--bank", "luna", "--out", out]) == 0
        st = [json.loads(l) for l in open(out)]
        ids = {s["case_id"] for s in st}
        assert ids == {"0", "1", "2"}                                                                          # case 3 is excluded
        sex = {s["case_id"] for s in st if s["factors"].get("attribute") == "sex"}
        assert sex == {"0", "2"}                                                                               # case 1 is sex-ineligible
        assert len({s["stimulus_id"] for s in st}) == len(st) and all(s["bank"] == "luna" and s["messages"] for s in st)
    print("PASS: the stimulus file skips excluded cases and gives no sex carrier to a sex-ineligible case; ids are unique")


def stim_file(t, n=6):
    path = os.path.join(t, "s.jsonl"); stims = []
    for i in range(n):
        for task in ("esi", "usmle"):
            stims.append({"stimulus_id": f"s{i}{task}", "case_id": str(i), "task": task, "arm": "bias", "messages": [{"role": "user", "content": f"You are an emergency triage nurse. ESI {i}" if task == "esi" else f"clinical expert {i}"}], "factors": {}})
    open(path, "w").write("\n".join(json.dumps(s) for s in stims)); return path, stims


def test_run_parse_resume_and_retry():
    with tempfile.TemporaryDirectory() as t:
        path, stims = stim_file(t)
        run = P.Run(t, "r", {"k": 1}); spec = R.DEFAULT_MODELS["GPT-6-Luna"]
        script = lambda m, i: "3)" if "ESI" in m[0]["content"] else "Option B is wrong. So the answer is C"
        client = FakeClient(script)
        res = R.run_agent_b(path, stims, spec, client, run, draws=2, workers=3, progress_every=0)
        assert res["planned"] == 24 and res["done"] == 24 and client.n == 24
        recs = [json.loads(l) for p in sorted(__import__("glob").glob(os.path.join(run.root, "steps", "*agent_b_GPT_6_Luna", "output.jsonl"))) for l in open(p)]
        esi = [r for r in recs if r["task"] == "esi"]; us = [r for r in recs if r["task"] == "usmle"]
        assert all(r["answer"] == "3" and r["confidence"] == "exact" for r in esi) and all(r["answer"] == "C" and r["confidence"] == "explicit" for r in us)
        assert {(r["stimulus_id"], r["draw"]) for r in recs} == {(s["stimulus_id"], d) for s in stims for d in (0, 1)}
        client2 = FakeClient(script)                                                                           # the same command again: nothing left to do
        assert R.run_agent_b(path, stims, spec, client2, run, draws=2, workers=3, progress_every=0)["planned"] == 0 and client2.n == 0
        client3 = FakeClient(script)                                                                           # a third draw is only the missing third draw
        assert R.run_agent_b(path, stims, spec, client3, run, draws=3, workers=3, progress_every=0)["planned"] == 12 and client3.n == 12
        calls = [json.loads(l) for l in open(os.path.join(run.root, "calls.jsonl"))]
        assert len(calls) == 36 and calls[0]["role"] == "agent_b" and "request_sha" in calls[0] and "request" not in calls[0] and P.verify_run(run.root) == []
    with tempfile.TemporaryDirectory() as t:                                                                  # failed calls are retried on resume, a content filter is not
        path, stims = stim_file(t, 3); run = P.Run(t, "r", {"k": 2}); spec = R.DEFAULT_MODELS["DeepSeek-V4-Flash"]
        def flaky(m, i):
            if "ESI 0" in m[0]["content"]: return ("", "content_filter: blocked")
            if "ESI 1" in m[0]["content"]: return ("", "failed after 5 attempts")
            return "3" if "ESI" in m[0]["content"] else "A"
        R.run_agent_b(path, stims, spec, FakeClient(flaky), run, workers=2, progress_every=0)
        c = FakeClient(lambda m, i: "3" if "ESI" in m[0]["content"] else "A")
        r = R.run_agent_b(path, stims, spec, c, run, workers=2, progress_every=0)
        assert r["planned"] == 1 and c.n == 1                                                                  # only the transport failure ('ESI 1') is retried
    print("PASS: answers are read strictly ('3)' exact, 'Option B is wrong... answer is C' explicit); the same command makes no new calls; a third draw makes only the missing calls; the call log holds a request hash, not the request; a failed call is retried and a content filter is not")


def test_reasoning_is_never_the_answer_and_abort():
    with tempfile.TemporaryDirectory() as t:
        path, stims = stim_file(t, 3); run = P.Run(t, "r", {"k": 3}); spec = R.DEFAULT_MODELS["Gemini-3.8-Flash"]
        client = FakeClient(lambda m, i: CallResult(text="", reasoning="the answer is 3", finish_reason="length", usage={"completion_tokens": 1500}))
        try:
            R.run_agent_b(path, stims, spec, client, run, workers=2, abort_after=4, progress_every=0); raise AssertionError("empty content in every call must abort")
        except RuntimeError as e:
            assert "no content" in str(e)
        recs = [json.loads(l) for p in __import__("glob").glob(os.path.join(run.root, "steps", "*agent_b_Gemini_3_8_Flash", "output.jsonl")) for l in open(p)]
        assert recs and all(r["answer"] is None and r["confidence"] == "unparseable" and r["reasoning_chars"] > 0 for r in recs)    # the reasoning field is recorded, not read as the answer
        def blocked(m, i): raise GuardrailError("404 blocked")
        c = FakeClient(blocked)
        try:
            R.run_agent_b(path, stims, R.DEFAULT_MODELS["GPT-6-Luna"], c, run, workers=2, progress_every=0); raise AssertionError
        except RuntimeError as e:
            assert "guardrail" in str(e)
        assert c.n <= 4
    print("PASS: an empty-content reply with reasoning is recorded as unparseable (the reasoning is never read as the answer), repeated empty replies abort the run, and a guardrail error aborts immediately")


def test_content_filter_refusal_and_diagnosis():
    import diagnose_unparseable
    with tempfile.TemporaryDirectory() as t:
        path, stims = stim_file(t, 5); run = P.Run(t, "r", {"k": 5}); spec = R.DEFAULT_MODELS["GPT-6-Luna"]
        def script(m, i):
            if "ESI 2" in m[0]["content"]:
                return CallResult(text="I'm sorry, but I cannot assist with that request.", finish_reason="content_filter")       # the provider's filter, as case 104 produced
            if "ESI 3" in m[0]["content"]:
                return CallResult(error="failed after 9 attempts: Error code: 429 rate limit")
            return "3" if "ESI" in m[0]["content"] else "A"
        R.run_agent_b(path, stims, spec, FakeClient(script), run, workers=2, progress_every=0)
        recs = [json.loads(l) for p in __import__("glob").glob(os.path.join(run.root, "steps", "*agent_b_GPT_6_Luna", "output.jsonl")) for l in open(p)]
        cf = [r for r in recs if r["stimulus_id"] == "s2esi"][0]
        assert cf["error"].startswith("content_filter") and cf["answer"] is None and cf["confidence"] == "refusal"
        again = FakeClient(script); r = R.run_agent_b(path, stims, spec, again, run, workers=2, progress_every=0)
        assert r["planned"] == 1 and again.n == 1                                                                                 # only the rate-limited call is retried, not the content filter
        out = os.path.join(t, "d.md"); assert diagnose_unparseable.main(["--run-dir", str(run.root), "--stimuli", path, "--out", out]) == 0
        txt = open(out).read()
        assert "content_filter" in txt and "transport_failure" in txt and "failed after N attempts" in txt and "cases with a missing answer" in txt
    print("PASS: a provider content-filter reply is recorded as a refusal with its own error class and is not retried, a rate-limited call is retried, and the diagnosis separates the two and names the cases")


def test_models_in_parallel_share_one_verified_run():
    with tempfile.TemporaryDirectory() as t:
        bank = os.path.join(t, "bank"); make_bank(bank, n=6); sp = os.path.join(t, "stim.jsonl")
        build_stimuli.main(["--bank-dir", bank, "--bank", "luna", "--arms", "bias", "conformity", "--out", sp])
        n = sum(1 for _ in open(sp))
        rc = run_agent_b.main(["--stimuli", sp, "--models", "GPT-6-Luna", "DeepSeek-V4-Flash", "Gemini-3.8-Flash", "--run-root", os.path.join(t, "res"), "--run-name", "par", "--client", "fake", "--workers", "4", "--parallel-models"])
        assert rc == 0                                                                                          # includes verify_run: the call-log hash chain survives three models writing at once
        steps = sorted(os.listdir(os.path.join(t, "res", "par", "steps"))); assert len(steps) == 3 and len({s.split("_", 1)[1] for s in steps}) == 3
        assert sum(1 for _ in open(os.path.join(t, "res", "par", "calls.jsonl"))) == 3 * n
        out = os.path.join(t, "o.md")
        import diagnose_unparseable; assert diagnose_unparseable.main(["--run-dir", os.path.join(t, "res", "par"), "--stimuli", sp, "--out", out]) == 0
    print("PASS: three models run at the same time into one run folder: three distinct step folders, one hash-chained call log with every call, and every hash verifies")


def test_shuffle_estimate_and_cli():
    with tempfile.TemporaryDirectory() as t:
        path, stims = stim_file(t, 10); run = P.Run(t, "r", {"k": 4}); spec = R.DEFAULT_MODELS["GPT-6-Luna"]
        a = [s["stimulus_id"] for s, d in R.plan(stims, spec, run.root, 1)]; b = [s["stimulus_id"] for s, d in R.plan(stims, spec, run.root, 1)]
        assert a == b and a != [s["stimulus_id"] for s in stims]                                                # shuffled, deterministically
        e = R.estimate(stims, [spec], 2); assert e["calls"] == 40 and e["input_tokens_per_model"] > 0
        bank = os.path.join(t, "bank"); make_bank(bank); sp = os.path.join(t, "stim.jsonl")
        build_stimuli.main(["--bank-dir", bank, "--bank", "luna", "--arms", "bias", "--out", sp])
        assert run_agent_b.main(["--stimuli", sp, "--models", "GPT-6-Luna", "--run-root", os.path.join(t, "res"), "--run-name", "x", "--estimate"]) == 0
        rc = run_agent_b.main(["--stimuli", sp, "--models", "GPT-6-Luna", "--run-root", os.path.join(t, "res"), "--run-name", "x", "--client", "fake", "--workers", "3"])
        assert rc == 0
        rc = run_agent_b.main(["--stimuli", sp, "--models", "DeepSeek-V4-Flash", "--run-root", os.path.join(t, "res"), "--run-name", "x", "--client", "fake"])      # a second model joins the same run
        assert rc == 0 and len(os.listdir(os.path.join(t, "res", "x", "steps"))) == 2
    print("PASS: the call order is shuffled but reproducible, the estimate counts calls, and the CLI runs two models into one run folder with every hash verified")


if __name__ == "__main__":
    for f in (test_build_stimuli_respects_exclusion_and_eligibility, test_run_parse_resume_and_retry, test_content_filter_refusal_and_diagnosis, test_models_in_parallel_share_one_verified_run, test_reasoning_is_never_the_answer_and_abort, test_shuffle_estimate_and_cli):
        f()
    print("\nALL RUNNER TESTS PASSED")
