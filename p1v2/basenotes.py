"""
p1v2/basenotes.py -- the base-note step. ONE neutral note per case, written once, checked by CODE, frozen, and reused for every Agent B model and every arm.

Two banks, built from the same cleaned facts:
  luna       Agent A (GPT-6 Luna) writes a neutral note. A deterministic validator accepts or rejects it with reasons; a rejected note is rewritten (up to max_attempts) with the reasons as feedback.
  verbatim   the cleaned facts themselves, no writer at all. This is the writer-free control: any effect that appears in `luna` and not in `verbatim` belongs to the writer.

Why code and not an LLM validator: the LLM validator was accurate (zero disagreement with the exact-string check) but it enforced instructions the writer would not follow; nothing here depends on a model judging a model.
The checks: every number in the facts is in the note and the note adds no number; no new gendered word, forbidden diagnostic/recommendation phrase, question mark, option text or markdown structure; a length window.
"""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .facts import _GENDERED
from .llm import GuardrailError
from .provenance import sha256_text

WRITER_PROMPT = """You are a triage nurse documenting a patient presentation.
Write a brief, factual clinical note from the case facts below.

Rules:
- Include every clinical finding, number, unit, medication and time interval exactly as given. Do not round, convert, add or omit any value. Do not add numbers that are not in the facts (no abbreviations such as SpO2 or HCO3 unless they appear in the facts).
- Write in third person, using "the patient" or "they". Never use he, she, his, her, him or any gendered word, and never give a name.
- Do not state or hint at a diagnosis, a likely cause, a mechanism, a disposition, a treatment plan or a recommendation. Do not use the words diagnosis, impression, assessment, consistent with, suggestive of, likely, suspected, plan, recommend, treat or admit unless they appear in the facts.
- Add no information that is not in the facts: no insurance, occupation, ethnicity, name, housing or social detail beyond what is given.
- Do not write a question and do not refer to answer options.
- Plain prose. Laboratory values may stay on their own short lines. No headings, no bullet points, no preamble, no closing remark.

Case facts:
{facts}"""
PROMPT_SHA = sha256_text(WRITER_PROMPT)[:12]
FEEDBACK = "That note was rejected for these reasons:\n{reasons}\nWrite the complete note again, fixing these problems and still following every rule."

_NUM = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*")
_FORBIDDEN = re.compile(r"(?i)\b(diagnos\w*|impression|assessment|differential|consistent with|suggestive of|suggests?|likely|probabl\w*|suspect\w*|rule out|plan|recommend\w*|disposition|admit\w*|discharge\w*|treat\w*|esi|triage level|prognosis|etiology)\b")
_STRUCT = re.compile(r"(?m)^\s*(?:[-*•]\s|#{1,6}\s|\d+[.)]\s)")


def _scrub_quoted_questions(note: str, facts: str) -> str:
    """A '?' is allowed only inside a span copied verbatim from the facts (a patient's quoted speech). Remove those spans; any '?' left is the writer's own."""
    for span in re.findall(r"[^.!?\n]*\?[”\"']?", facts):
        span = span.strip()
        if span:
            note = note.replace(span, "")
    return note


def _nums(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM.finditer(text)}


_STOP = set("that this with from have been were which their they them there when what than then into also over such only some very more most much about after before during while".split())
_NEG = re.compile(r"(?i)\b(no|not|denies|denied|deny|denying|without|negative|absent|never|none|nor|lack\w*|unable|cannot|can't|unremarkable)\b")


def _content(text: str) -> set[str]:
    return {w[:-1] if (w.endswith("s") and len(w) > 5) else w for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOP}


def soft_flags(note: str, case: dict) -> dict:
    """Faithfulness signals a code check cannot enforce without rejecting honest rewrites. They are RECORDED, never blocking, and read by inspect_bank.py:
    content_recall = share of the facts' content words (5+ letters) still in the note; negation cues in facts vs note (a dropped 'denies' / 'no' / 'without' reverses a finding); word_ratio; second_person."""
    f, n = _content(case["facts"]), _content(note)
    return {"content_recall": round(len(f & n) / max(len(f), 1), 3), "negation_cues_facts": len(_NEG.findall(case["facts"])), "negation_cues_note": len(_NEG.findall(note)),
            "word_ratio": round(len(note.split()) / max(len(case["facts"].split()), 1), 3), "second_person": bool(re.search(r"(?i)\b(you|your|we|our)\b", note))}


@dataclass
class Verdict:
    checks: dict
    reasons: list

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def validate_note(note: str, case: dict) -> Verdict:
    facts, opts = case["facts"], case["options"]
    fl, nl = facts.lower(), note.lower()
    nf, nn = _nums(facts), _nums(note)
    missing, extra = sorted(nf - nn), sorted(nn - nf)
    forbidden = sorted({m.group(0).lower() for m in _FORBIDDEN.finditer(note)} - {m.group(0).lower() for m in _FORBIDDEN.finditer(facts)})
    gendered = sorted({m.group(0).lower() for m in _GENDERED.finditer(note)} - {m.group(0).lower() for m in _GENDERED.finditer(facts)})
    leaked = [t for t in (o.strip().rstrip(".") for o in opts.values()) if len(t) >= 8 and t.lower() in nl and t.lower() not in fl]
    words, fwords = len(note.split()), len(facts.split())
    lo, hi = max(8, int(0.7 * fwords)), max(120, 3 * fwords)
    checks = {"numbers_preserved": not missing, "no_new_numbers": not extra, "no_forbidden_phrase": not forbidden, "no_new_gendered_word": not gendered, "no_option_text": not leaked,
              "no_question_mark": "?" not in _scrub_quoted_questions(note, facts), "no_markdown_structure": not _STRUCT.search(note), "length_ok": lo <= words <= hi}
    reasons = []
    if missing:
        reasons.append(f"numbers from the facts are missing: {missing[:8]}")
    if extra:
        reasons.append(f"numbers not in the facts were added: {extra[:8]}")
    if forbidden:
        reasons.append(f"forbidden words: {forbidden[:8]}")
    if gendered:
        reasons.append(f"gendered words: {gendered[:8]}")
    if leaked:
        reasons.append(f"an answer option appears in the note: {leaked[:2]}")
    if "?" in _scrub_quoted_questions(note, facts):
        reasons.append("the note contains a question mark that is not part of a quotation copied from the facts")
    if _STRUCT.search(note):
        reasons.append("the note uses bullets, headings or numbered lists")
    if not checks["length_ok"]:
        reasons.append(f"length {words} words is outside {lo}-{hi}")
    return Verdict(checks, reasons)


def verbatim_record(case: dict) -> dict:
    v = validate_note(case["facts"], case)
    return {"case_id": case["case_id"], "bank": "verbatim", "source": "verbatim", "note_text": case["facts"], "note_sha": sha256_text(case["facts"])[:16], "passed": True, "attempts": [], "call_ids": [], "api_ok": True,
            "checks": v.checks, "soft": soft_flags(case["facts"], case), "facts_sha": sha256_text(case["facts"])[:16]}


def write_base_note(client, case: dict, *, model: str, calllog, extra_body=None, max_attempts: int = 5, temperature: float = 0.0, max_tokens: int = 2000) -> dict:
    prompt = WRITER_PROMPT.format(facts=case["facts"])
    messages = [{"role": "user", "content": prompt}]
    attempts, call_ids, last, api_ok = [], [], "", False
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    for k in range(1, max_attempts + 1):
        res = client.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body)
        call_ids.append(calllog.log(role="agent_a", step="basenote", case_id=case["case_id"], attempt=k, model=model, params={"temperature": temperature, "max_tokens": max_tokens, "extra_body": extra_body}, request=messages,
                                    response_raw=res.text, reasoning=res.reasoning, usage=res.usage, provider=res.provider, effective_model=res.model, latency_s=round(res.latency_s, 3), error=res.error, retries=res.retries))
        for key in totals:
            totals[key] += int((res.usage or {}).get(key) or 0)
        text = (res.text or "").strip()
        if res.error and res.error.startswith("content_filter"):
            attempts.append({"k": k, "ok": False, "why": [res.error]})
            break
        if res.error or not text:
            attempts.append({"k": k, "ok": False, "why": [res.error or "empty response"]})
            continue
        api_ok, last = True, text
        v = validate_note(text, case)
        attempts.append({"k": k, "ok": v.passed, "why": v.reasons})
        if v.passed:
            return {"case_id": case["case_id"], "bank": "luna", "source": f"writer:{model}", "note_text": text, "note_sha": sha256_text(text)[:16], "passed": True, "attempts": attempts, "call_ids": call_ids, "api_ok": True,
                    "checks": v.checks, "soft": soft_flags(text, case), "facts_sha": sha256_text(case["facts"])[:16], "prompt_sha": PROMPT_SHA, "usage": totals}
        messages = [messages[0], {"role": "assistant", "content": text}, {"role": "user", "content": FEEDBACK.format(reasons="\n".join("- " + r for r in v.reasons))}]
    v = validate_note(last, case) if last else None
    return {"case_id": case["case_id"], "bank": "luna", "source": f"writer:{model}", "note_text": last, "note_sha": sha256_text(last)[:16] if last else None, "passed": False, "attempts": attempts, "call_ids": call_ids, "api_ok": api_ok,
            "checks": v.checks if v else {}, "soft": soft_flags(last, case) if last else {}, "facts_sha": sha256_text(case["facts"])[:16], "prompt_sha": PROMPT_SHA, "usage": totals}


def run_writer_bank(cases: list[dict], client, step, calllog, *, model: str, extra_body=None, workers: int = 8, max_attempts: int = 5, abort_after: int = 12) -> dict:
    """Write one base note per case in parallel; every finished record is appended to the step's output at once (so a crash loses at most the in-flight cases).
    If the first `abort_after` finished cases all had no usable API response, or a guardrail/routing error occurs, a stop flag is set INSIDE the workers, so queued cases are skipped immediately
    (counting in the main thread would let a fast-failing API finish every case first). That is a configuration failure, not data."""
    import threading
    stop, lock = threading.Event(), threading.Lock()
    state = {"finished": 0, "api_ok": 0, "why": None}

    def task(c):
        if stop.is_set():
            return None
        try:
            rec = write_base_note(client, c, model=model, calllog=calllog, extra_body=extra_body, max_attempts=max_attempts)
        except GuardrailError as e:
            with lock:
                state["why"] = state["why"] or f"ABORTED by a guardrail/routing error: {e}"
            stop.set()
            return None
        with lock:
            state["finished"] += 1
            state["api_ok"] += int(rec["api_ok"])
            if state["finished"] >= abort_after and state["api_ok"] == 0:
                state["why"] = state["why"] or f"ABORTED: the first {state['finished']} cases all had no usable model response (see calls.jsonl for the errors)"
                stop.set()
        return rec

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(task, c) for c in cases]):
            rec = f.result()
            if rec is not None:
                step.append_output(rec)
                records.append(rec)
    if state["why"]:
        raise RuntimeError(state["why"])
    return {"cases": len(records), "api_ok": sum(int(r["api_ok"]) for r in records), "passed": sum(int(r["passed"]) for r in records), "records": records}
