#!/usr/bin/env python3
"""
build_basenotes.py -- the base-note step. Cleans the cases, then builds the writer-free `verbatim` bank and the `luna` bank (GPT-6 Luna writes, code validates).

    python build_basenotes.py --cases p1v2_data/cases20.jsonl --run-name p1v2_smoke --client fake            # the whole flow offline, no API cost
    python build_basenotes.py --cases p1v2_data/cases20.jsonl --run-name p1v2_smoke                          # 20 real cases
    python build_basenotes.py --cases p1v2_data/cases.jsonl   --run-name p1v2_main --workers 8                # all 801

Everything is recorded under results/<run-name>/ (manifest, one folder per step with input, output, notes and hashes, calls.jsonl with every API call). The banks are written to p1v2_data/banks/<run-name>/.
`python -c "from p1v2.provenance import verify_run; print(verify_run('results/<run-name>'))"` re-checks every hash afterwards.
"""
import argparse
import collections
import glob
import json
import os
import sys

from p1v2 import basenotes as B
from p1v2 import facts as F
from p1v2 import provenance as P
from p1v2.llm import FakeClient, OpenRouterClient

DEFAULT_EXTRA = {"provider": {"order": ["azure"], "allow_fallbacks": False}}


def _fake_writer(messages, i):
    facts = messages[0]["content"].split("Case facts:\n", 1)[1]
    return "The patient presents as follows. " + facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--run-root", default="results")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--banks", nargs="+", default=["verbatim", "luna"], choices=["verbatim", "luna"])
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--client", default="openrouter", choices=["openrouter", "fake"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)

    raw = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    raw = raw[: a.n] if a.n else raw
    out_dir = a.out_dir or os.path.join("p1v2_data", "banks", a.run_name)
    os.makedirs(out_dir, exist_ok=True)
    code = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1v2", "*.py")))
    config = {"step": "basenotes", "cases_file": os.path.basename(a.cases), "n_cases": len(raw), "banks": a.banks, "model": a.model, "client": a.client, "max_attempts": a.max_attempts,
              "temperature": 0.0, "max_tokens": 2000, "extra_body": DEFAULT_EXTRA, "writer_prompt_sha": B.PROMPT_SHA}
    run = P.Run(a.run_root, a.run_name, config, code_files=code)

    clean = [F.clean_case(r) for r in raw]
    with run.step("clean_cases", purpose="split each question into vignette and stem without cutting numbers; flag image-dependent and no-vignette cases; decide per-attribute eligibility", inputs=a.cases) as s:
        for c in clean:
            s.append_output({k: c[k] for k in ("case_id", "flags", "eligible", "excluded", "excluded_because", "legacy_facts_status")})
        s.note(f"{len(clean)} cases; excluded {sum(c['excluded'] for c in clean)}; ineligible sex/race/ses = {sum(not c['eligible']['sex'] for c in clean)}/{sum(not c['eligible']['race'] for c in clean)}/{sum(not c['eligible']['ses'] for c in clean)}")
    with open(os.path.join(out_dir, "cases_clean.jsonl"), "w", encoding="utf-8") as f:
        for c in clean:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    included = [c for c in clean if not c["excluded"]]

    summary = {}
    for bank in a.banks:
        if bank == "verbatim":
            with run.step("bank_verbatim", purpose="writer-free control bank: the cleaned facts are the note", inputs=[{"case_id": c["case_id"]} for c in included]) as s:
                recs = [B.verbatim_record(c) for c in included]
                for r in recs:
                    s.append_output(r)
        else:
            client = FakeClient(_fake_writer) if a.client == "fake" else OpenRouterClient()
            with run.step("bank_luna", purpose=f"{a.model} writes one neutral note per case; deterministic validator accepts or rejects with reasons; up to {a.max_attempts} attempts with feedback", inputs=[{"case_id": c["case_id"]} for c in included]) as s:
                res = B.run_writer_bank(included, client, s, run.calls, model=a.model, extra_body=DEFAULT_EXTRA, workers=a.workers, max_attempts=a.max_attempts)
                recs = res["records"]
                fails = collections.Counter(w for r in recs if not r["passed"] for at in r["attempts"][-1:] for w in at["why"])
                s.note(f"{res['passed']} of {res['cases']} cases passed; last-attempt failure reasons: {dict(fails)}")
        good = [r for r in recs if r["passed"]]
        with open(os.path.join(out_dir, f"bank_{bank}.jsonl"), "w", encoding="utf-8") as f:
            for r in sorted(good, key=lambda r: r["case_id"]):
                f.write(json.dumps({k: r[k] for k in ("case_id", "bank", "source", "note_text", "note_sha")}, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"bank_{bank}_failed.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                if not r["passed"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[bank] = {"passed": len(good), "failed": len(recs) - len(good)}
    problems = P.verify_run(os.path.join(a.run_root, a.run_name))
    print(json.dumps({"summary": summary, "excluded_cases": sum(c["excluded"] for c in clean), "out_dir": out_dir, "verify_run_problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
