"""
shared/config_and_utils.py

Module 1 — Environment & Utils
Phase 0 — Bias Contagion Research Pipeline (v3 spec)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Final

from dotenv import load_dotenv

load_dotenv(override=False)


def _get_bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    logging.warning("Environment variable %s=%r is not a recognized boolean; defaulting to %r", var_name, raw, default)
    return default


def _get_str_env(var_name: str, default: str | None = None) -> str | None:
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        return default
    return value


OPENAI_API_KEY: Final[str | None] = _get_str_env("OPENAI_API_KEY")
OPENAI_BASE_URL: Final[str | None] = _get_str_env("OPENAI_BASE_URL")
HF_TOKEN: Final[str | None] = _get_str_env("HF_TOKEN")
LOCAL_BASE_URL: Final[str] = _get_str_env("LOCAL_BASE_URL", default="http://localhost:8000/v1")  # type: ignore[assignment]
LOCAL_API_KEY: Final[str | None] = _get_str_env("LOCAL_API_KEY")

DRY_RUN: Final[bool] = _get_bool_env("DRY_RUN", default=True)

if DRY_RUN:
    logging.info("[config_and_utils] DRY_RUN is active -- all API-touching modules must return mock data.")

GENERATOR_MODEL_ID: Final[str] = _get_str_env("GENERATOR_MODEL_ID", default="gpt-4o-mini-2024-07-18")  # type: ignore[assignment]
VALIDATOR_MODEL_ID: Final[str] = _get_str_env("VALIDATOR_MODEL_ID", default=GENERATOR_MODEL_ID)  # type: ignore[assignment]

GENERATOR_MODEL_PRICE_PER_1M_INPUT_USD: Final[float] = 0.15
GENERATOR_MODEL_PRICE_PER_1M_OUTPUT_USD: Final[float] = 0.60


@dataclass(frozen=True)
class AgentBModel:
    display_name: str
    hf_identifier: str
    primary_paper: str
    role: str


AGENT_B_MODELS: Final[tuple[AgentBModel, ...]] = (
    AgentBModel(display_name="Llama-3.1-8B-Instruct", hf_identifier="meta-llama/Llama-3.1-8B-Instruct", primary_paper="Kumarappan & Mujoo 2026 (primary); Pandey 2026; Wang et al. 2026", role="Full verified mechanistic pipeline; deference direction"),
    AgentBModel(display_name="OLMo-7B-Instruct", hf_identifier="allenai/OLMo-7B-Instruct", primary_paper="Ahsan et al. 2025 (primary)", role="Clinical-native gender sensor; domain-matched"),
    AgentBModel(display_name="Gemma-2-9B-IT", hf_identifier="google/gemma-2-9b-it", primary_paper="Kumarappan & Mujoo 2026 (cross-family); Pandey 2026", role="Independent family replication"),
    AgentBModel(display_name="Gemma-2-27B-IT", hf_identifier="google/gemma-2-27b-it", primary_paper="Karvonen & Marks 2025 (primary)", role="ACE bias direction extraction validated at this scale"),
    AgentBModel(display_name="Mistral-7B-Instruct-v0.3", hf_identifier="mistralai/Mistral-7B-Instruct-v0.3", primary_paper="Kumarappan & Mujoo 2026 (full replication); Pandey 2026", role="Layer-for-layer L14-L18 mechanistic replication"),
)

AGENT_B_MODEL_IDS: Final[dict[str, str]] = {model.display_name: model.hf_identifier for model in AGENT_B_MODELS}

BASELINE_TEMPERATURE: Final[float] = 0.8
EXPERIMENT_5_LOW_TEMPERATURE: Final[float] = 0.2

MAX_VALIDATION_REGENERATION_ATTEMPTS: Final[int] = 2

N_CASES_GO_NO_GO: Final[int] = 40
N_CASES_PUBLISHABILITY: Final[int] = 200


@dataclass(frozen=True)
class PipelineConfig:
    dry_run: bool = DRY_RUN
    openai_api_key: str | None = OPENAI_API_KEY
    openai_base_url: str | None = OPENAI_BASE_URL
    hf_token: str | None = HF_TOKEN
    local_base_url: str = LOCAL_BASE_URL
    local_api_key: str | None = LOCAL_API_KEY
    generator_model_id: str = GENERATOR_MODEL_ID
    agent_b_models: tuple[AgentBModel, ...] = field(default_factory=lambda: AGENT_B_MODELS)
    baseline_temperature: float = BASELINE_TEMPERATURE
    experiment_5_low_temperature: float = EXPERIMENT_5_LOW_TEMPERATURE
    max_validation_regeneration_attempts: int = MAX_VALIDATION_REGENERATION_ATTEMPTS
    n_cases_go_no_go: int = N_CASES_GO_NO_GO
    n_cases_publishability: int = N_CASES_PUBLISHABILITY

    def require_openai_key(self) -> str:
        if self.dry_run:
            return self.openai_api_key or "dry-run-placeholder-key"
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but DRY_RUN is False.")
        return self.openai_api_key

    def require_hf_token(self) -> str:
        if self.dry_run:
            return self.hf_token or "dry-run-placeholder-token"
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN is not set but DRY_RUN is False.")
        return self.hf_token


def get_config() -> PipelineConfig:
    return PipelineConfig()


def _is_content_filter_error(exc: Exception) -> bool:
    """
    True if this exception is Azure OpenAI's content-management policy
    filter blocking a request -- confirmed live during this project on
    BOTH sides of the pipeline: Agent B's inference calls (already
    handled in shared/experiments.py) AND Agent A's Note-Writer/Note-
    Validator calls in shared/crew_pipeline.py (crashed an entire
    run_full_experiment.py batch before this was added there too --
    the Validator sending a genuinely disturbing/violent clinical note
    back to Azure for CHECK 1/2/3 can trigger the same filter as sending
    it to Agent B). Clinical vignettes involving trauma, violence, or
    substance use can trigger this per-request, even though the
    underlying content is entirely legitimate clinical text -- a real,
    structural limitation of running this research against a commercial
    content-moderated API, not a bug in this project's prompts.

    Deliberately narrow: only matches this specific failure mode. Any
    OTHER exception (auth, connection, rate limit, a genuine bug) should
    still raise and stop the run -- silently absorbing those alongside
    this one, different, expected failure mode would hide real problems.

    Single shared definition (not duplicated in crew_pipeline.py and
    experiments.py separately) so both layers of the pipeline recognize
    the exact same failure mode consistently.
    """
    msg = str(exc).lower()
    return "content_filter" in msg or "responsibleaipolicyviolation" in msg


__all__ = [
    "DRY_RUN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "HF_TOKEN", "LOCAL_BASE_URL", "LOCAL_API_KEY",
    "GENERATOR_MODEL_ID", "VALIDATOR_MODEL_ID",
    "GENERATOR_MODEL_PRICE_PER_1M_INPUT_USD", "GENERATOR_MODEL_PRICE_PER_1M_OUTPUT_USD",
    "AgentBModel", "AGENT_B_MODELS", "AGENT_B_MODEL_IDS",
    "BASELINE_TEMPERATURE", "EXPERIMENT_5_LOW_TEMPERATURE", "MAX_VALIDATION_REGENERATION_ATTEMPTS",
    "N_CASES_GO_NO_GO", "N_CASES_PUBLISHABILITY", "PipelineConfig", "get_config",
    "_is_content_filter_error",
]