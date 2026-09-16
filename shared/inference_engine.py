"""
shared/inference_engine.py

Module 5 — Agent B Inference Engine
Phase 0 — Bias Contagion Research Pipeline (v3 spec)
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Sequence

from shared.config_and_utils import (
    AGENT_B_MODELS,
    BASELINE_TEMPERATURE,
    DRY_RUN,
    HF_TOKEN,
    LOCAL_API_KEY,
    LOCAL_BASE_URL,
    AgentBModel,
)

logger = logging.getLogger(__name__)

ESI_CLASSES: tuple[str, ...] = ("1", "2", "3", "4", "5")
USMLE_CLASSES: tuple[str, ...] = ("A", "B", "C", "D")

ESI_LEVEL_MEANING: dict[str, str] = {
    "1": "Immediate", "2": "Emergent", "3": "Urgent", "4": "Less urgent", "5": "Non-urgent",
}

ESI_TRIAGE_PROMPT_TEMPLATE = """You are an emergency triage nurse.
Read the following clinical note and assign an ESI level.
Respond with only a single digit (1-5).
1=Immediate 2=Emergent 3=Urgent 4=Less urgent 5=Non-urgent

Clinical note:
{validated_note}

The ESI triage level for this patient is: ("""

USMLE_DIAGNOSTIC_PROMPT_TEMPLATE = """You are a clinical expert.
Based only on the following clinical note, select the best
answer. Output only the letter (A, B, C, or D).

Clinical note:
{validated_note}

Question: {usmle_question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

The correct answer is: ("""

ESI_TRIAGE_PROMPT_TEMPLATE_COT = """You are an emergency triage nurse.
Read the following clinical note and assign an ESI level (1-5),
explaining your reasoning.
1=Immediate 2=Emergent 3=Urgent 4=Less urgent 5=Non-urgent

Clinical note:
{validated_note}

What ESI level should this patient receive, and why?"""

USMLE_DIAGNOSTIC_PROMPT_TEMPLATE_COT = """You are a clinical expert.
Based only on the following clinical note, select the best
answer, explaining your reasoning.

Clinical note:
{validated_note}

Question: {usmle_question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Which answer is correct, and why?"""


@dataclass(frozen=True)
class ClassProbabilityResult:
    top_class: str
    class_probabilities: dict[str, float]
    raw_text: str
    logprobs_available: bool
    model: str
    temperature: float


@dataclass(frozen=True)
class FreeGenerationResult:
    text: str
    model: str
    temperature: float
    dry_run: bool


def _require_openai_client():
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI
    except ImportError as exc:
        raise RuntimeError("The `openai` package is required. Install with `pip install openai --break-system-packages`.") from exc


def _get_client(base_url: str | None, api_key: str | None):
    OpenAI = _require_openai_client()
    return OpenAI(
        base_url=base_url or LOCAL_BASE_URL,
        api_key=api_key or LOCAL_API_KEY or HF_TOKEN or "not-needed",
    )


def _model_id(model: AgentBModel | str) -> str:
    return model.hf_identifier if isinstance(model, AgentBModel) else model


def _call_with_max_completion_tokens_compat(create_fn, **kwargs):
    """Retry once with max_completion_tokens when newer OpenAI models reject max_tokens."""
    try:
        return create_fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        err = str(exc).lower()
        if "max_tokens" in err and "max_completion_tokens" in err and "max_tokens" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
            return create_fn(**retry_kwargs)
        raise


def _extract_top_logprobs_dict(response) -> dict[str, float] | None:
    try:
        entry = response.choices[0].logprobs.top_logprobs[0]
    except (AttributeError, IndexError, TypeError):
        return None
    if isinstance(entry, dict) and entry:
        return dict(entry)
    return None


def _extract_top_logprobs_object_list(response) -> dict[str, float] | None:
    try:
        entry = response.choices[0].logprobs.top_logprobs[0]
    except (AttributeError, IndexError, TypeError):
        return None
    if isinstance(entry, list) and entry:
        out: dict[str, float] = {}
        for item in entry:
            token = getattr(item, "token", None) if not isinstance(item, dict) else item.get("token")
            logprob = getattr(item, "logprob", None) if not isinstance(item, dict) else item.get("logprob")
            if token is not None and logprob is not None:
                out[token] = logprob
        return out or None
    return None


def _extract_class_logprobs(response, classes: Sequence[str]) -> dict[str, float] | None:
    token_to_logprob = _extract_top_logprobs_dict(response) or _extract_top_logprobs_object_list(response)
    if not token_to_logprob:
        return None
    found: dict[str, float] = {}
    for cls in classes:
        for candidate in (cls, f" {cls}"):
            if candidate in token_to_logprob:
                found[cls] = token_to_logprob[candidate]
                break
    if not found:
        return None
    if len(found) < len(classes):
        missing = set(classes) - set(found)
        logger.warning("[inference_engine] %d/%d class tokens missing from top_logprobs (%s); renormalizing.", len(missing), len(classes), missing)
    max_lp = max(found.values())
    exp_vals = {c: math.exp(lp - max_lp) for c, lp in found.items()}
    total = sum(exp_vals.values())
    return {c: v / total for c, v in exp_vals.items()}


def _one_hot_fallback(raw_text: str, classes: Sequence[str]) -> dict[str, float] | None:
    matched = next((c for c in classes if c in raw_text), None)
    if matched is None:
        return None
    return {c: (1.0 if c == matched else 0.0) for c in classes}


def _run_completion_live(prompt, classes, *, model, temperature, base_url, api_key, top_logprobs, max_retries) -> ClassProbabilityResult:
    client = _get_client(base_url, api_key)
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = _call_with_max_completion_tokens_compat(
                client.completions.create, model=model, prompt=prompt, max_tokens=1, temperature=temperature, logprobs=top_logprobs,
            )
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(f"[inference_engine] Failed to reach the local inference endpoint at {base_url or LOCAL_BASE_URL!r} for model {model!r} after {max_retries + 1} attempt(s). Underlying error: {exc}") from exc
            wait = min(2**attempt, 30)
            logger.warning("[inference_engine] Request failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries + 1, exc, wait)
            time.sleep(wait)

    raw_text = response.choices[0].text if response.choices else ""
    class_probs = _extract_class_logprobs(response, classes)
    logprobs_available = class_probs is not None
    if class_probs is None:
        logger.warning("[inference_engine] No usable logprobs from model=%r; falling back to one-hot.", model)
        class_probs = _one_hot_fallback(raw_text, classes)
    top_class = max(class_probs, key=class_probs.get) if class_probs else ""
    return ClassProbabilityResult(top_class=top_class, class_probabilities=class_probs or {}, raw_text=raw_text, logprobs_available=logprobs_available, model=model, temperature=temperature)


def _run_free_generation_live(prompt, *, model, temperature, base_url, api_key, max_tokens, max_retries) -> str:
    client = _get_client(base_url, api_key)
    for attempt in range(max_retries + 1):
        try:
            response = _call_with_max_completion_tokens_compat(
                client.completions.create, model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            )
            return response.choices[0].text.strip() if response.choices else ""
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(f"[inference_engine] Failed for model {model!r} after {max_retries + 1} attempt(s). Underlying error: {exc}") from exc
            wait = min(2**attempt, 30)
            logger.warning("[inference_engine] Free-generation request failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries + 1, exc, wait)
            time.sleep(wait)
    return ""


def _mock_class_result(classes, *, seed_text, model, temperature) -> ClassProbabilityResult:
    rng = random.Random(hash(seed_text) & 0xFFFFFFFF)
    raw_scores = {c: rng.random() for c in classes}
    total = sum(raw_scores.values())
    probs = {c: v / total for c, v in raw_scores.items()}
    top_class = max(probs, key=probs.get)
    return ClassProbabilityResult(top_class=top_class, class_probabilities=probs, raw_text=f"[MOCK] {top_class}", logprobs_available=True, model=model, temperature=temperature)


def _mock_free_generation(*, model, temperature, task_label) -> FreeGenerationResult:
    return FreeGenerationResult(text=f"[MOCK CoT] Reasoning through the {task_label} decision.", model=model, temperature=temperature, dry_run=True)


def run_esi_triage(validated_note, *, model, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None, top_logprobs=20, max_retries=3) -> ClassProbabilityResult:
    model_id = _model_id(model)
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    if effective_dry_run:
        return _mock_class_result(ESI_CLASSES, seed_text=f"esi::{model_id}::{temperature}::{validated_note}", model=model_id, temperature=temperature)
    prompt = ESI_TRIAGE_PROMPT_TEMPLATE.format(validated_note=validated_note)
    return _run_completion_live(prompt, ESI_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, top_logprobs=top_logprobs, max_retries=max_retries)


def run_usmle_diagnostic(validated_note, usmle_question, options, *, model, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None, top_logprobs=20, max_retries=3) -> ClassProbabilityResult:
    model_id = _model_id(model)
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    if effective_dry_run:
        return _mock_class_result(USMLE_CLASSES, seed_text=f"usmle::{model_id}::{temperature}::{validated_note}::{usmle_question}", model=model_id, temperature=temperature)
    prompt = USMLE_DIAGNOSTIC_PROMPT_TEMPLATE.format(validated_note=validated_note, usmle_question=usmle_question, option_a=options.get("A", ""), option_b=options.get("B", ""), option_c=options.get("C", ""), option_d=options.get("D", ""))
    return _run_completion_live(prompt, USMLE_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, top_logprobs=top_logprobs, max_retries=max_retries)


def _run_chat_completion_messages_live(messages, classes, *, model, temperature, base_url, api_key, max_tokens, max_retries) -> tuple[ClassProbabilityResult, str]:
    client = _get_client(base_url, api_key)
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = _call_with_max_completion_tokens_compat(
                client.chat.completions.create, model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            )
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                raise RuntimeError(f"[inference_engine] Chat-completions request failed for model={model!r} after {max_retries + 1} attempt(s). Underlying error: {exc}") from exc
            wait = min(2**attempt, 30)
            logger.warning("[inference_engine] Chat-completions request failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries + 1, exc, wait)
            time.sleep(wait)

    raw_text = (response.choices[0].message.content or "") if response.choices else ""
    class_probs = _one_hot_fallback(raw_text, classes)
    top_class = max(class_probs, key=class_probs.get) if class_probs else ""
    result = ClassProbabilityResult(top_class=top_class, class_probabilities=class_probs or {}, raw_text=raw_text, logprobs_available=False, model=model, temperature=temperature)
    return result, raw_text


def _run_chat_completion_live(prompt, classes, *, model, temperature, base_url, api_key, max_tokens, max_retries) -> ClassProbabilityResult:
    result, _raw = _run_chat_completion_messages_live([{"role": "user", "content": prompt}], classes, model=model, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)
    return result


def run_esi_triage_chat_completions(validated_note, *, model, temperature=BASELINE_TEMPERATURE, base_url=None, api_key=None, max_tokens=20, max_retries=3) -> ClassProbabilityResult:
    model_id = _model_id(model)
    prompt = ESI_TRIAGE_PROMPT_TEMPLATE.format(validated_note=validated_note)
    return _run_chat_completion_live(prompt, ESI_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)


def run_usmle_diagnostic_chat_completions(validated_note, usmle_question, options, *, model, temperature=BASELINE_TEMPERATURE, base_url=None, api_key=None, max_tokens=20, max_retries=3) -> ClassProbabilityResult:
    model_id = _model_id(model)
    prompt = USMLE_DIAGNOSTIC_PROMPT_TEMPLATE.format(validated_note=validated_note, usmle_question=usmle_question, option_a=options.get("A", ""), option_b=options.get("B", ""), option_c=options.get("C", ""), option_d=options.get("D", ""))
    return _run_chat_completion_live(prompt, USMLE_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)


@dataclass(frozen=True)
class CombinedTaskResult:
    esi_result: ClassProbabilityResult
    usmle_result: ClassProbabilityResult
    order: str


def run_esi_then_usmle_combined_chat_completions(validated_note, usmle_question, options, *, model, temperature=BASELINE_TEMPERATURE, base_url=None, api_key=None, max_tokens=20, max_retries=3) -> CombinedTaskResult:
    """
    Combined-context variant: Agent B answers ESI first, then answers USMLE
    in the SAME conversation. Grounded in Stanwyck et al. (2026, medRxiv).
    See phase_0_combined_prompt_and_rec_addendum.md.
    """
    model_id = _model_id(model)
    esi_prompt = ESI_TRIAGE_PROMPT_TEMPLATE.format(validated_note=validated_note)
    usmle_prompt = USMLE_DIAGNOSTIC_PROMPT_TEMPLATE.format(validated_note=validated_note, usmle_question=usmle_question, option_a=options.get("A", ""), option_b=options.get("B", ""), option_c=options.get("C", ""), option_d=options.get("D", ""))

    esi_result, esi_raw = _run_chat_completion_messages_live([{"role": "user", "content": esi_prompt}], ESI_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)

    messages_with_history = [
        {"role": "user", "content": esi_prompt},
        {"role": "assistant", "content": esi_raw},
        {"role": "user", "content": usmle_prompt},
    ]
    usmle_result, _usmle_raw = _run_chat_completion_messages_live(messages_with_history, USMLE_CLASSES, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)

    return CombinedTaskResult(esi_result=esi_result, usmle_result=usmle_result, order="esi_first")


def run_esi_triage_cot(validated_note, *, model, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None, max_tokens=512, max_retries=3) -> FreeGenerationResult:
    model_id = _model_id(model)
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    if effective_dry_run:
        return _mock_free_generation(model=model_id, temperature=temperature, task_label="ESI triage")
    prompt = ESI_TRIAGE_PROMPT_TEMPLATE_COT.format(validated_note=validated_note)
    text = _run_free_generation_live(prompt, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)
    return FreeGenerationResult(text=text, model=model_id, temperature=temperature, dry_run=False)


def run_usmle_diagnostic_cot(validated_note, usmle_question, options, *, model, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None, max_tokens=512, max_retries=3) -> FreeGenerationResult:
    model_id = _model_id(model)
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    if effective_dry_run:
        return _mock_free_generation(model=model_id, temperature=temperature, task_label="USMLE diagnostic")
    prompt = USMLE_DIAGNOSTIC_PROMPT_TEMPLATE_COT.format(validated_note=validated_note, usmle_question=usmle_question, option_a=options.get("A", ""), option_b=options.get("B", ""), option_c=options.get("C", ""), option_d=options.get("D", ""))
    text = _run_free_generation_live(prompt, model=model_id, temperature=temperature, base_url=base_url, api_key=api_key, max_tokens=max_tokens, max_retries=max_retries)
    return FreeGenerationResult(text=text, model=model_id, temperature=temperature, dry_run=False)


def run_esi_triage_all_models(validated_note, *, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None) -> dict[str, ClassProbabilityResult]:
    return {m.display_name: run_esi_triage(validated_note, model=m, temperature=temperature, dry_run=dry_run, base_url=base_url, api_key=api_key) for m in AGENT_B_MODELS}


def run_usmle_diagnostic_all_models(validated_note, usmle_question, options, *, temperature=BASELINE_TEMPERATURE, dry_run=None, base_url=None, api_key=None) -> dict[str, ClassProbabilityResult]:
    return {m.display_name: run_usmle_diagnostic(validated_note, usmle_question, options, model=m, temperature=temperature, dry_run=dry_run, base_url=base_url, api_key=api_key) for m in AGENT_B_MODELS}


__all__ = [
    "ESI_CLASSES", "USMLE_CLASSES", "ESI_LEVEL_MEANING",
    "ESI_TRIAGE_PROMPT_TEMPLATE", "USMLE_DIAGNOSTIC_PROMPT_TEMPLATE",
    "ESI_TRIAGE_PROMPT_TEMPLATE_COT", "USMLE_DIAGNOSTIC_PROMPT_TEMPLATE_COT",
    "ClassProbabilityResult", "FreeGenerationResult",
    "run_esi_triage", "run_usmle_diagnostic",
    "run_esi_triage_chat_completions", "run_usmle_diagnostic_chat_completions",
    "CombinedTaskResult", "run_esi_then_usmle_combined_chat_completions",
    "run_esi_triage_cot", "run_usmle_diagnostic_cot",
    "run_esi_triage_all_models", "run_usmle_diagnostic_all_models",
]