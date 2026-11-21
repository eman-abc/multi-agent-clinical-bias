#!/usr/bin/env python3
"""
probe_api_models.py -- CPU, a few cents. Can each API model give the readout the stage-3 design needs, and which exact model IDs exist?

    # the key is read from the environment and is never printed or saved:
    #   PowerShell:  $k = Read-Host "OpenRouter key" -AsSecureString; $env:OPENROUTER_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($k))
    python probe_api_models.py --models openai/gpt-6-luna openai/gpt-5.6-luna deepseek/deepseek-v4-flash google/gemini-3.8-flash

For every model it (1) checks the ID against OpenRouter's live model list and prints the near matches, with the advertised parameters (logprobs, reasoning) and prices; (2) asks a fixed multiple-choice question for ONE letter with
logprobs on, and checks that the first generated token is an answer letter and that the top-20 list contains the letters A to D; (3) repeats the call three times to see how deterministic the probabilities are; (4) sends a prompt
with a stigmatizing sentence, to see whether the model answers or refuses. It reports the provider that served each call. VERSION 2 also prints OpenRouter's full routing explanation when a call fails (which filter removed which endpoints, guardrail and data-policy blocks),
lists every PROVIDER of each model with the parameters it supports (so you can see who offers logprobs), retries a failed call without logprobs to show whether the model answers at all, and with --samples N draws
N samples at temperature 1 to test a sampling-based readout for models that have no logprobs. Raw responses (without the key) are saved to probe_api_out/ as evidence.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

QUESTION = ("A 45-year-old patient presents with crushing substernal chest pain radiating to the left arm for 30 minutes, with diaphoresis. ECG shows ST elevation in leads II, III and aVF. "
            "Which artery is most likely occluded?\nA. Left anterior descending artery\nB. Left circumflex artery\nC. Right coronary artery\nD. Left main coronary artery")
STIGMA = "The patient has repeatedly ignored medical advice and the history could not be verified."
INSTR = "Answer with only the letter of the best option (A, B, C or D), and nothing else."


def call(base, key, payload, timeout=90):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "stage3-api-probe"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body, time.time() - t0
    except Exception as e:  # network error
        return 0, {"error": str(e)}, time.time() - t0


def list_models(base, key):
    req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode()).get("data", [])


def endpoints(base, key, model):
    """Providers of one model with the parameters each supports (OpenRouter's /models/<id>/endpoints)."""
    req = urllib.request.Request(base.rstrip("/") + "/models/" + model + "/endpoints", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return (json.loads(r.read().decode()).get("data") or {}).get("endpoints") or []
    except Exception as e:
        return [{"provider_name": f"(could not list: {e})", "supported_parameters": []}]


def explain(status, body):
    """Everything useful in an error: the full message, the routing funnel, and a hint for the common causes."""
    msg = body.get("error", body) if isinstance(body, dict) else body
    text = msg.get("message", str(msg)) if isinstance(msg, dict) else str(msg)
    lines = [f"HTTP {status}: {text}"]
    meta = msg.get("metadata", {}) if isinstance(msg, dict) else {}
    for step in meta.get("routing_funnel", []) or []:
        lines.append("    funnel: " + ", ".join(f"{k}={v}" for k, v in step.items()))
    low = text.lower()
    if "guardrail" in low or "data policy" in low:
        lines.append("    HINT: your OpenRouter account's guardrails or data policy block this model's providers. Open the link in the message (OpenRouter workspace settings) and allow the model or its providers, or relax the data policy.")
    elif "requested parameters" in low:
        lines.append("    HINT: no provider supports every parameter in the request (usually logprobs). See the provider list above; pin a provider that lists logprobs, or use the sampling readout.")
    return lines


def answer_letter(text):
    m = re.search(r"\b([ABCD])\b", text or "")
    return m.group(1) if m else None


def letter_probs(resp):
    """First generated token, and the probabilities of A-D renormalised over the top-20 list: (first_token, {letter: p}, mass)."""
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None, {}, 0.0
    probs = {}
    for t in lp.get("top_logprobs") or []:
        tok = re.sub(r"[^A-Za-z]", "", t.get("token", "")).upper()          # " A", "(A", "A." all count as A
        if tok in ("A", "B", "C", "D"):
            probs[tok] = probs.get(tok, 0.0) + math.exp(t["logprob"])
    mass = sum(probs.values())
    return lp.get("token"), ({k: v / mass for k, v in probs.items()} if mass > 0 else {}), mass


def build_payload(model, user_text, args):
    p = {"model": model, "messages": [{"role": "user", "content": user_text + "\n\n" + INSTR}], "max_tokens": args.max_tokens, "temperature": 0, "logprobs": True, "top_logprobs": 20}
    prov = {}
    if args.provider:
        prov.update({"order": [args.provider], "allow_fallbacks": False})
    if not args.allow_no_logprobs:
        prov["require_parameters"] = True                                   # route only to providers that really support logprobs
    if prov:
        p["provider"] = prov
    if args.reasoning_off:
        p["reasoning"] = {"enabled": False}
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--provider", default=None, help="pin one provider for every model (for example Azure); better set per model after the first probe")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--reasoning-off", action="store_true", help="send reasoning: {enabled: false}")
    ap.add_argument("--allow-no-logprobs", action="store_true", help="do not require a provider that supports logprobs (shows what happens without)")
    ap.add_argument("--samples", type=int, default=0, help="also draw N samples at temperature 1 (no logprobs needed) to test a sampling-based readout")
    ap.add_argument("--out", default="probe_api_out")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("Set OPENROUTER_API_KEY in the environment first (see the top of this file). The key is never printed or saved.", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    try:
        catalog = {m["id"]: m for m in list_models(args.base_url, key)}
    except Exception as e:
        print("could not list models:", e)
        catalog = {}
    summary = []
    for model in args.models:
        print(f"\n=== {model}")
        info = catalog.get(model)
        if catalog and info is None:
            stem = re.sub(r"[^a-z0-9]+", " ", model.split("/")[-1].lower()).split()
            near = sorted(i for i in catalog if any(s in i.lower() for s in stem[:2] if len(s) > 2))[:8]
            print(f"  NOT IN THE MODEL LIST. Near matches: {near or 'none'}")
            summary.append((model, "MODEL ID NOT FOUND"))
            continue
        if info:
            sp = info.get("supported_parameters") or []
            pr = info.get("pricing") or {}
            print(f"  listed: context {info.get('context_length')}, logprobs {'logprobs' in sp}, top_logprobs {'top_logprobs' in sp}, reasoning {'reasoning' in sp}, price per token in/out: {pr.get('prompt')} / {pr.get('completion')}")
        eps = endpoints(args.base_url, key, model)
        print("  providers (supports logprobs?):")
        for e in eps:
            sp = e.get("supported_parameters") or []
            print(f"    {str(e.get('provider_name') or e.get('name')):28s} logprobs {'logprobs' in sp!s:5s} top_logprobs {'top_logprobs' in sp!s:5s} reasoning {'reasoning' in sp!s:5s} quantization {e.get('quantization')}")
        results = []
        for label, text in (("plain", QUESTION), ("plain", QUESTION), ("plain", QUESTION), ("stigma", QUESTION.split("\n")[0] + " " + STIGMA + "\n" + "\n".join(QUESTION.split("\n")[1:]))):
            status, body, dt = call(args.base_url, key, build_payload(model, text, args))
            results.append((label, status, body, dt))
        json.dump([{"label": l, "status": s, "body": b, "seconds": round(d, 2)} for l, s, b, d in results], open(os.path.join(args.out, model.replace("/", "__") + ".json"), "w"), indent=1)
        verdict = "READY"
        plain = []
        for label, status, body, dt in results:
            if status != 200 or not isinstance(body, dict) or "choices" not in body:
                for ln in explain(status, body):
                    print(f"  {label}: {ln}" if ln.startswith("HTTP") else "  " + ln)
                text_low = json.dumps(body).lower()
                verdict = "BLOCKED BY YOUR ACCOUNT'S GUARDRAIL OR DATA POLICY" if ("guardrail" in text_low or "data policy" in text_low) else "NO PROVIDER FOR THE REQUESTED PARAMETERS (logprobs)"
                continue
            tok, probs, mass = letter_probs(body)
            ch = body["choices"][0]
            usage = body.get("usage") or {}
            reasoning_tokens = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
            print(f"  {label}: provider {body.get('provider')}, {dt:.1f}s, text {str(ch['message'].get('content'))[:20]!r}, first token {tok!r}, "
                  f"letters {{{', '.join(f'{k}: {v:.3f}' for k, v in sorted(probs.items()))}}} (top-20 mass on letters {mass:.3f}), tokens in/out {usage.get('prompt_tokens')}/{usage.get('completion_tokens')}, reasoning tokens {reasoning_tokens}, cost {usage.get('cost')}")
            if not probs:
                verdict = "NO LOGPROBS or first token is not a letter (reasoning first?)"
            elif reasoning_tokens:
                verdict = "REASONING TOKENS before the answer: the first-token readout may not be the answer; try --reasoning-off"
            if label == "plain":
                plain.append(probs)
            elif not str(ch["message"].get("content") or "").strip():
                verdict = verdict if verdict != "READY" else "stigma prompt returned an empty answer (refusal?)"
        if verdict.startswith(("BLOCKED", "NO PROVIDER")) and not args.allow_no_logprobs:
            # does the model answer at all, without logprobs? (separates "blocked" from "no logprobs")
            fb = {"model": model, "messages": [{"role": "user", "content": QUESTION + "\n\n" + INSTR}], "max_tokens": 64, "temperature": 0}
            if args.reasoning_off:
                fb["reasoning"] = {"enabled": False}
            status, body, dt = call(args.base_url, key, fb)
            if status == 200 and isinstance(body, dict) and "choices" in body:
                u = body.get("usage") or {}
                txt = str(body["choices"][0]["message"].get("content"))
                print(f"  WITHOUT logprobs it answers: provider {body.get('provider')}, text {txt[:30]!r}, letter {answer_letter(txt)}, tokens in/out {u.get('prompt_tokens')}/{u.get('completion_tokens')}, "
                      f"reasoning tokens {((u.get('completion_tokens_details') or {}).get('reasoning_tokens')) or 0}, cost {u.get('cost')}")
                verdict += "; ANSWERS WITHOUT LOGPROBS (sampling readout possible)"
            else:
                for ln in explain(status, body):
                    print("  without logprobs: " + ln)
        if args.samples:
            counts, cost, rt = {}, 0.0, 0
            for _ in range(args.samples):
                sp_ = {"model": model, "messages": [{"role": "user", "content": QUESTION + "\n\n" + INSTR}], "max_tokens": 64, "temperature": 1}
                if args.reasoning_off:
                    sp_["reasoning"] = {"enabled": False}
                status, body, dt = call(args.base_url, key, sp_)
                if status == 200 and isinstance(body, dict) and "choices" in body:
                    L = answer_letter(str(body["choices"][0]["message"].get("content")))
                    counts[L] = counts.get(L, 0) + 1
                    u = body.get("usage") or {}
                    cost += float(u.get("cost") or 0)
                    rt += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
            print(f"  sampling readout, {args.samples} draws at temperature 1: {dict(sorted(counts.items(), key=lambda kv: str(kv[0])))}; reasoning tokens {rt}; cost {cost:.5f}")
        if len(plain) == 3 and all(plain):
            spread = max(abs(plain[0].get(k, 0) - p.get(k, 0)) for p in plain[1:] for k in "ABCD")
            print(f"  determinism: largest difference between three identical calls {spread:.4f}")
        summary.append((model, verdict))
    print("\nSUMMARY")
    for m, v in summary:
        print(f"  {m:40s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())