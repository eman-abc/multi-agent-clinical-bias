#!/usr/bin/env python3
"""
check_models_route.py -- ONE real call per model, before any large run. Run it from the project root (it imports your shared/ package):

    python check_models_route.py

It checks, with a tiny clinical note:
  1. Agent A / Validator: GENERATOR_MODEL_ID and VALIDATOR_MODEL_ID answer a trivial prompt through the routing in shared/crew_pipeline.py (a blocked or unrouted model fails here, in seconds);
  2. every Agent B model in AGENT_B_API_MODELS answers the ESI and the USMLE prompt, and the raw response is ONE answer token, optionally with a closing bracket or period (the parser in shared/inference_engine.py is only safe then);
  3. latency per call.
Exit code 0 only if every call worked and every Agent B answer is a single clean token. No file is written except check_models_route.txt.
"""
import re
import sys
import time

NOTE = ("A 54-year-old patient presents with crushing substernal chest pain radiating to the left arm for 30 minutes, with diaphoresis. Vital signs: BP 150/90 mmHg, HR 110 bpm, RR 22/min, SpO2 95%.")
Q = "Which artery is most likely occluded?"
OPTS = {"A": "Left anterior descending artery", "B": "Left circumflex artery", "C": "Right coronary artery", "D": "Left main coronary artery"}


def main() -> int:
    from shared.config_and_utils import AGENT_B_API_MODELS, GENERATOR_MODEL_ID, VALIDATOR_MODEL_ID
    from shared.crew_pipeline import _run_single_task_live
    from shared.inference_engine import run_esi_triage_chat_completions, run_usmle_diagnostic_chat_completions
    lines, bad = [], 0

    def rec(ok, name, detail, secs):
        nonlocal bad
        bad += 0 if ok else 1
        lines.append(f"{'OK  ' if ok else 'FAIL'}  {name:42s} {secs:5.1f}s  {detail}")
        print(lines[-1])

    for role, mid in (("Agent A (generator)", GENERATOR_MODEL_ID), ("Validator", VALIDATOR_MODEL_ID)):
        t0 = time.time()
        try:
            out = _run_single_task_live(None, "Reply with exactly one word: YES", expected_output="YES", model=mid)
            rec("YES" in out.upper(), f"{role}: {mid}", f"replied {out[:40]!r}", time.time() - t0)
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace("\n", " ")
            hint = " <- a guardrail or routing problem (check the model id in .env and the routing in _run_single_task_live)" if ("guardrail" in msg.lower() or "404" in msg) else ""
            rec(False, f"{role}: {mid}", f"{type(e).__name__}: {msg[:140]}{hint}", time.time() - t0)
    for spec in AGENT_B_API_MODELS:
        for task in ("esi", "usmle"):
            t0 = time.time()
            try:
                if task == "esi":
                    r = run_esi_triage_chat_completions(NOTE, model=spec.api_model_id, base_url=spec.base_url, api_key=spec.api_key, extra_body=spec.extra_body)
                else:
                    r = run_usmle_diagnostic_chat_completions(NOTE, Q, OPTS, model=spec.api_model_id, base_url=spec.base_url, api_key=spec.api_key, extra_body=spec.extra_body)
                raw = r.raw_text or ""
                m = re.fullmatch(r"\s*[\(\[]?\s*([A-D1-5])\s*[\)\]]?\s*[.:]?\s*", raw)         # one answer token; the prompt ends with '(' so '2)' is the model closing the bracket
                clean = bool(m) and m.group(1) == r.top_class
                rec(clean, f"Agent B: {spec.display_name} [{task}]", f"top_class {r.top_class!r}, raw {raw[:60]!r}, raw length {len(raw.strip())}" + ("" if clean else "  <- NOT a single answer token: the answer parser is unsafe for this model"), time.time() - t0)
            except Exception as e:  # noqa: BLE001
                rec(False, f"Agent B: {spec.display_name} [{task}]", f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " "), time.time() - t0)
    open("check_models_route.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nALL CHECKS PASSED" if not bad else f"\n{bad} CHECK(S) FAILED: fix them before any run")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
