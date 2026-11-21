"""
p1v2/runner.py -- the Agent B runner. One call = one (stimulus, model, draw). Resumable, shuffled, strictly parsed, and fail-fast.

* The answer is read from the model's CONTENT only, with parse.answer(). The legacy fallback that read the reasoning field when content was empty is gone: an empty content is recorded as unparseable, never guessed.
* Records are small (stimulus id, model, draw, answer, confidence, raw text, usage, provider, error, call id). The request is not repeated: the stimuli file is hashed in the step manifest and the call log carries `request_sha`,
  so any prompt can be reconstructed from the stimulus id without a gigabyte call log.
* Resume: records with an answer, or with a content-filter refusal, are done; a record whose call failed (error without text) is retried on the next invocation. The latest record per key wins in the analysis.
* Order is shuffled with a fixed seed, so an interrupted or limited run is a random sample of the design, not its first cases.
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import parse
from .llm import GuardrailError


@dataclass(frozen=True)
class ModelSpec:
    name: str
    api_model_id: str
    extra_body: dict | None = None
    max_tokens: int = 1500
    temperature: float = 0.0


AZURE = {"provider": {"order": ["azure"], "allow_fallbacks": False}}
DEFAULT_MODELS = {
    "DeepSeek-V4-Flash": ModelSpec("DeepSeek-V4-Flash", "deepseek/deepseek-v4-flash", AZURE),
    "GPT-6-Luna": ModelSpec("GPT-6-Luna", "openai/gpt-6-luna", AZURE),
    "Gemini-3.8-Flash": ModelSpec("Gemini-3.8-Flash", "google/gemini-3.8-flash", {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}}),
}


def request_sha(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def load_done(run_root, model_name: str) -> set[tuple]:
    """Keys (stimulus_id, model, draw) that need no further call, from every earlier step of this model in this run."""
    done: set[tuple] = set()
    for p in sorted(glob.glob(str(Path(run_root) / "steps" / f"*agent_b_{slug(model_name)}" / "output.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["stimulus_id"], r["model"], r["draw"])
            if r.get("error") and not r.get("raw_text") and not str(r["error"]).startswith("content_filter"):
                done.discard(key)                              # a failed call is retried; a later success re-adds it
            else:
                done.add(key)
    return done


def call_one(client, spec: ModelSpec, stim: dict, draw: int, calllog) -> dict:
    msgs = stim["messages"]
    res = client.chat(msgs, model=spec.api_model_id, temperature=spec.temperature, max_tokens=spec.max_tokens, extra_body=spec.extra_body)
    if res.finish_reason == "content_filter" and not res.error:
        res.error = "content_filter: finish_reason=content_filter"          # the provider's filter, not the model's answer: recorded as such, never retried
    text = (res.text or "").strip()
    answer, conf = parse.answer(text, stim["task"])
    cid = calllog.log(role="agent_b", model=spec.name, stimulus_id=stim["stimulus_id"], draw=draw, request_sha=request_sha(msgs), params={"temperature": spec.temperature, "max_tokens": spec.max_tokens, "extra_body": spec.extra_body},
                      response_raw=res.text, reasoning_chars=len(res.reasoning or ""), usage=res.usage, provider=res.provider, effective_model=res.model, finish_reason=res.finish_reason,
                      latency_s=round(res.latency_s, 3), error=res.error, retries=res.retries)
    return {"stimulus_id": stim["stimulus_id"], "model": spec.name, "draw": draw, "task": stim["task"], "arm": stim["arm"], "case_id": stim["case_id"], "answer": answer, "confidence": conf, "raw_text": text,
            "finish_reason": res.finish_reason, "reasoning_chars": len(res.reasoning or ""), "usage": res.usage, "provider": res.provider, "error": res.error, "call_id": cid}


def plan(stimuli: list[dict], spec: ModelSpec, run_root, draws: int, seed: int = 20261009) -> list[tuple]:
    done = load_done(run_root, spec.name)
    todo = [(s, d) for s in stimuli for d in range(draws) if (s["stimulus_id"], spec.name, d) not in done]
    random.Random(seed).shuffle(todo)
    return todo


def run_agent_b(stimuli_path, stimuli: list[dict], spec: ModelSpec, client, run, *, draws: int = 1, workers: int = 8, abort_after: int = 12, progress_every: int = 500, seed: int = 20261009) -> dict:
    todo = plan(stimuli, spec, run.root, draws, seed)
    stop, lock = threading.Event(), threading.Lock()
    st = {"finished": 0, "usable": 0, "why": None, "unparseable": 0, "failed": 0}

    def task(item):
        s, d = item
        if stop.is_set():
            return None
        try:
            rec = call_one(client, spec, s, d, run.calls)
        except GuardrailError as e:
            with lock:
                st["why"] = st["why"] or f"ABORTED by a guardrail/routing error: {e}"
            stop.set()
            return None
        with lock:
            st["finished"] += 1
            st["usable"] += int(bool(rec["raw_text"]))
            st["unparseable"] += int(rec["answer"] is None)
            st["failed"] += int(bool(rec["error"]) and not str(rec["error"]).startswith("content_filter"))
            if st["finished"] >= abort_after and st["usable"] == 0:
                st["why"] = st["why"] or f"ABORTED: the first {st['finished']} calls all returned no content (see calls.jsonl)"
                stop.set()
        return rec

    n_new, t0 = 0, time.time()
    with run.step(f"agent_b_{slug(spec.name)}", purpose=f"Agent B = {spec.api_model_id}: {len(todo)} calls ({draws} draw(s) per stimulus, temperature {spec.temperature}); strict answer parsing; resumable", inputs=str(stimuli_path)) as step:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            chunk = 4 * workers * 50
            for i in range(0, len(todo), chunk):
                if stop.is_set():
                    break
                for f in as_completed([ex.submit(task, it) for it in todo[i:i + chunk]]):
                    rec = f.result()
                    if rec is not None:
                        step.append_output(rec)
                        n_new += 1
                        if progress_every and n_new % progress_every == 0:
                            el = max(time.time() - t0, 1e-6); eta = (len(todo) - n_new) / (n_new / el) / 60
                            print(f"[{spec.name}] {n_new}/{len(todo)} calls, {n_new / el:.1f}/s, ETA {eta:.0f} min, {st['unparseable']} unparseable, {st['failed']} transport failures" + ("  <- many failures: lower --model-workers for this model" if n_new >= 200 and st["failed"] > 0.1 * n_new else ""), flush=True)
        step.note(f"{n_new} of {len(todo)} planned calls completed; unparseable answers: {st['unparseable']}; transport failures: {st['failed']}")
        if st["why"]:
            step.note(st["why"])
    if st["why"]:
        raise RuntimeError(st["why"])
    return {"planned": len(todo), "done": n_new, "unparseable": st["unparseable"], "transport_failures": st["failed"]}


def estimate(stimuli: list[dict], models: list[ModelSpec], draws: int, price_in_per_m: float = 0.15, price_out_per_m: float = 0.60, out_tokens: int = 4) -> dict:
    chars = sum(len(json.dumps(s["messages"])) for s in stimuli)
    tin = chars / 4 * draws
    calls = len(stimuli) * draws * len(models)
    return {"stimuli": len(stimuli), "models": [m.name for m in models], "draws": draws, "calls": calls, "input_tokens_per_model": int(tin), "usd_per_model_at_the_prices_given": round(tin / 1e6 * price_in_per_m + len(stimuli) * draws * out_tokens / 1e6 * price_out_per_m, 2),
            "note": "Gemini's mandatory reasoning adds output tokens that are not counted here"}
