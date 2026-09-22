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
OPENROUTER_API_KEY: Final[str | None] = _get_str_env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: Final[str] = _get_str_env("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")  # type: ignore[assignment]

DRY_RUN: Final[bool] = _get_bool_env("DRY_RUN", default=True)

if DRY_RUN:
    logging.info("[config_and_utils] DRY_RUN is active -- all API-touching modules must return mock data.")

GENERATOR_MODEL_ID: Final[str] = _get_str_env("GENERATOR_MODEL_ID", default="openai/gpt-5.6-luna")  # type: ignore[assignment]
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


@dataclass(frozen=True)
class APIModelSpec:
    """
    One API-routed Agent B model (Azure OpenAI directly, or a model behind
    OpenRouter) -- the counterpart to AgentBModel above, which is for the
    FUTURE open-weight/vLLM path. Each spec carries its own connection
    details so multiple providers can be mixed in one experiment run
    without one config silently overriding another.

    `extra_body`: OpenRouter-specific fields (e.g. {"provider": {"order":
    [...], "allow_fallbacks": False}}) that aren't part of the standard
    OpenAI Chat Completions schema -- passed through via the OpenAI SDK's
    own `extra_body` kwarg, its documented mechanism for exactly this case
    (a non-OpenAI-standard field on an OpenAI-compatible endpoint). None
    for a plain Azure/OpenAI deployment, which needs no such routing hint.
    """
    display_name: str
    api_model_id: str
    base_url: str | None
    api_key: str | None
    extra_body: dict | None = None
    primary_paper: str = ""
    role: str = ""


AGENT_B_API_MODELS: Final[tuple[APIModelSpec, ...]] = (
    # APIModelSpec(
    #     display_name="gpt-5.4-mini", api_model_id="gpt-5.4-mini",
    #     base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY,
    #     role="Primary Azure deployment, this project's original Agent B model",
    # ),
    APIModelSpec(
        display_name="DeepSeek-V4-Flash", api_model_id="deepseek/deepseek-v4-flash",
        base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
        extra_body={"provider": {"order": ["azure"], "allow_fallbacks": False}},
        role="Cross-model comparison via OpenRouter",
    ),
    APIModelSpec(
        display_name="GPT-5.6-Luna", api_model_id="openai/gpt-5.6-luna",
        base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
        extra_body={"provider": {"order": ["azure"], "allow_fallbacks": False}},
        role="Cross-model comparison via OpenRouter",
    ),
    APIModelSpec(
        display_name="Gemini-3.8-Flash", api_model_id="google/gemini-3.8-flash",
        base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY,
        extra_body={"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}},
        role="Cross-model comparison via OpenRouter, different upstream provider",
    ),
)
# IMPORTANT: logprobs reliability through OpenRouter is NOT guaranteed
# uniformly across models/providers -- confirmed via a real, documented
# OpenRouter issue (their own GitHub tracker: logprobs requested but
# returned empty for at least one model, while the identical request
# against that model's native endpoint worked correctly). Verify each of
# these three with a standalone test call (see test_openrouter_logprobs.py)
# BEFORE trusting them in a real experiment run -- exactly the same
# discipline already applied to the Azure gpt-5.4-mini deployment, which
# turned up its own two real, undocumented constraints (top_logprobs<=5,
# max_completion_tokens>=150).

BASELINE_TEMPERATURE: Final[float] = 0.0

# Client-side request timeout (seconds) for every Agent A/Validator/Agent B
# API call. Confirmed live: with no timeout set, the OpenAI SDK's own
# default (several minutes) meant a genuinely hung request -- e.g. the
# first-ever call through a brand-new, unverified path like OpenRouter --
# produced total silence with no error, no case-completion print, and no
# checkpoint file, for a very long time before anything happened. A short,
# explicit timeout turns a silent hang into a fast, visible error instead,
# which then correctly triggers this project's own existing retry/backoff
# logic (see inference_engine.py) rather than leaving the person staring
# at a blank terminal wondering if anything is happening at all.
REQUEST_TIMEOUT_SECONDS: Final[float] = _get_str_env("REQUEST_TIMEOUT_SECONDS", default="45")  # type: ignore[assignment]
REQUEST_TIMEOUT_SECONDS: Final[float] = float(REQUEST_TIMEOUT_SECONDS)
# CHANGED from 0.8. Verified live during this project: FairMedQA/AMQA --
# the paper this project's entire metric framework (CFR, AD, McNemar) is
# built on and compared against -- explicitly states "we set the
# temperature to 0.0 for all model queries... to ensure stable and
# deterministic outputs," for both their generation and evaluation steps.
# At temperature>0, the same input can sample a different output on
# repeated identical calls, meaning some of any observed E+ vs. E-
# difference is guaranteed sampling noise, not signal -- this was directly
# implicated in this project's own reversion-fraction sparsity problem
# (35 of 37 cases showed zero shift on either side; the aggregate was
# driven by 1-2 non-zero cases). temperature=0.0 removes that noise
# source and makes CFR/AD numbers genuinely comparable to FairMedQA's own
# published values, rather than an apples-to-oranges comparison.
EXPERIMENT_5_LOW_TEMPERATURE: Final[float] = 0.2
EXPLORATORY_HIGH_TEMPERATURE: Final[float] = 0.8
# NEW: the project's OLD baseline (0.8), kept as a documented, deliberate
# SECONDARY condition -- not discarded. Okawa (2026, already in this
# project's own reading list) treats temperature as a genuine independent
# variable, not just noise to eliminate: her finding is that temperature
# itself controls how readily bias amplifies in multi-agent settings.
# This constant exists so that question stays testable (alongside
# EXPERIMENT_5_LOW_TEMPERATURE=0.2) without it being confused for the
# primary measurement temperature.

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
    exploratory_high_temperature: float = EXPLORATORY_HIGH_TEMPERATURE
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
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL",
    "APIModelSpec", "AGENT_B_API_MODELS",
    "REQUEST_TIMEOUT_SECONDS",
    "GENERATOR_MODEL_ID", "VALIDATOR_MODEL_ID",
    "GENERATOR_MODEL_PRICE_PER_1M_INPUT_USD", "GENERATOR_MODEL_PRICE_PER_1M_OUTPUT_USD",
    "AgentBModel", "AGENT_B_MODELS", "AGENT_B_MODEL_IDS",
    "BASELINE_TEMPERATURE", "EXPERIMENT_5_LOW_TEMPERATURE", "EXPLORATORY_HIGH_TEMPERATURE", "MAX_VALIDATION_REGENERATION_ATTEMPTS",
    "N_CASES_GO_NO_GO", "N_CASES_PUBLISHABILITY", "PipelineConfig", "get_config",
    "_is_content_filter_error",
]