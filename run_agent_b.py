#!/usr/bin/env python3
"""
run_agent_b.py -- run Agent B models over a stimuli file. Resumable: run the same command again and only the missing or failed calls are made.

    python run_agent_b.py --stimuli p1v2_data\\stimuli_luna.jsonl --models GPT-6-Luna --run-name p1v2_run1 --estimate       # counts and a rough cost, no API
    python run_agent_b.py --stimuli p1v2_data\\stimuli_luna.jsonl --models GPT-6-Luna DeepSeek-V4-Flash Gemini-3.8-Flash --run-name p1v2_run1 --workers 12
    python run_agent_b.py --stimuli ... --n-cases 100 --draws 3 --run-name p1v2_retest        # a 3-draw test-retest on the first 100 cases (non-determinism at temperature 0)

Run the check_models_route.py first. The three models run one after another; each has its own step folder, so a model can be run, stopped and resumed independently.
"""
import argparse
import glob
import json
import os
import sys

from p1v2 import provenance as P
from p1v2 import runner as R
from p1v2.llm import FakeClient, OpenRouterClient


def _fake(messages, i):
    last = messages[-1]["content"]
    return "3" if "ESI" in messages[0]["content"] else "B"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--models", nargs="+", default=list(R.DEFAULT_MODELS))
    ap.add_argument("--run-root", default="results")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model-workers", nargs="*", default=[], help="per-model override, e.g. DeepSeek-V4-Flash=4 GPT-6-Luna=16")
    ap.add_argument("--n-cases", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--client", default="openrouter", choices=["openrouter", "fake"])
    ap.add_argument("--parallel-models", action="store_true", help="run the selected models at the same time in this process (each keeps its own worker count)")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--price-in", type=float, default=0.15, help="USD per million input tokens, for --estimate")
    ap.add_argument("--price-out", type=float, default=0.60)
    a = ap.parse_args(argv)
    stimuli = [json.loads(l) for l in open(a.stimuli, encoding="utf-8") if l.strip()]
    if a.n_cases:
        keep = []
        for s in stimuli:
            if s["case_id"] not in keep and len(keep) < a.n_cases:
                keep.append(s["case_id"])
        stimuli = [s for s in stimuli if s["case_id"] in set(keep)]
    if a.arms:
        stimuli = [s for s in stimuli if s["arm"] in a.arms]
    specs = [R.DEFAULT_MODELS[m] for m in a.models]
    if a.estimate:
        print(json.dumps(R.estimate(stimuli, specs, a.draws, a.price_in, a.price_out), indent=1))
        return 0
    code = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1v2", "*.py")))
    config = {"step": "agent_b", "stimuli_sha256": P.sha256_file(a.stimuli), "available_models": {n: [m.api_model_id, m.extra_body, m.max_tokens, m.temperature] for n, m in R.DEFAULT_MODELS.items()}, "client": a.client}      # models are chosen per call, so one can be added to an existing run
    run = P.Run(a.run_root, a.run_name, config, code_files=code)
    client = FakeClient(_fake) if a.client == "fake" else OpenRouterClient()
    mw = {k: int(v) for k, v in (x.split('=') for x in a.model_workers)}
    out, errors = {}, {}
    one = lambda spec: R.run_agent_b(a.stimuli, stimuli, spec, client, run, draws=a.draws, workers=mw.get(spec.name, a.workers))
    if a.parallel_models and len(specs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(specs)) as ex:
            futs = {spec.name: ex.submit(one, spec) for spec in specs}
            for name, f in futs.items():
                try:
                    out[name] = f.result()
                except Exception as e:  # noqa: BLE001      one model's failure must not discard the others' finished work
                    errors[name] = str(e)
    else:
        for spec in specs:
            out[spec.name] = one(spec)
    problems = P.verify_run(run.root)
    print(json.dumps({"results": out, "model_errors": errors, "verify_run_problems": problems}, indent=1))
    return 1 if (problems or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
