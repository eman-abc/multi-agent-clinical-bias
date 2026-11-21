"""
mech/local_inference.py

Option 2 ("use the records from Paper 1"): run ONLY the local Agent B
inference step on already-generated notes -- zero Agent A calls, zero
Validator calls, zero OPENROUTER_API_KEY usage. Reuses
shared/inference_engine.py's chat-completions functions unchanged (they
already accept any base_url/api_key, which is what makes the
local-vLLM-as-APIModelSpec trick from run_phase1_bridge_replication.py
work here too).

This module has NO dependency on shared/crew_pipeline.py (Agent A/
Validator) and NO dependency on shared/conditions.py's condition-matrix
generation -- it only needs a note_text and, for USMLE, the original
case's question+options, which come from Paper 1's own AMQA loader
(shared/data_loader.py) keyed by case_id -- the dataset itself, not any
Paper 1 RESULT file.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.config_and_utils import APIModelSpec
from shared.inference_engine import run_esi_triage_chat_completions, run_usmle_diagnostic_chat_completions


@dataclass(frozen=True)
class LocalPrediction:
    task: str  # "esi" | "usmle"
    top_class: str | None
    class_probabilities: dict[str, float] | None
    raw_text: str | None


def run_esi_on_note(note_text: str, model_spec: APIModelSpec, *, temperature: float = 0.0) -> LocalPrediction:
    result = run_esi_triage_chat_completions(
        note_text, model=model_spec.api_model_id, temperature=temperature,
        base_url=model_spec.base_url, api_key=model_spec.api_key, extra_body=model_spec.extra_body,
    )
    return LocalPrediction(task="esi", top_class=result.top_class, class_probabilities=result.class_probabilities, raw_text=result.raw_text)


def run_usmle_on_note(note_text: str, usmle_question: str, options: dict[str, str], model_spec: APIModelSpec, *, temperature: float = 0.0) -> LocalPrediction:
    result = run_usmle_diagnostic_chat_completions(
        note_text, usmle_question, options, model=model_spec.api_model_id, temperature=temperature,
        base_url=model_spec.base_url, api_key=model_spec.api_key, extra_body=model_spec.extra_body,
    )
    return LocalPrediction(task="usmle", top_class=result.top_class, class_probabilities=result.class_probabilities, raw_text=result.raw_text)


__all__ = ["LocalPrediction", "run_esi_on_note", "run_usmle_on_note"]