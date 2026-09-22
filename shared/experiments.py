"""
shared/experiments.py

Module 6 — Experiment Orchestrator
Phase 0 — Bias Contagion Research Pipeline (v3 spec)

Runs Experiments 1-6 (phase_0_complete_v3.md §11) as discrete, resumable
jobs, pulling from modules 2-5 (data_loader, conditions, crew_pipeline,
inference_engine). Tracks API call counts against §13's budget table as a
sanity check.

Design decisions made while building this (see chat for full reasoning):

  1. Task 2's {usmle_question} input reuses AMQARecord.neutralised_question
     verbatim, same as the note's case-facts input. AMQA's real schema has
     no separate "question stem" field distinct from the full vignette
     (verified in Module 2) -- inventing a heuristic to split them would be
     fragile and unverifiable. Flagged clearly since this affects what
     Agent B actually sees for Task 2.

  2. Every condition in a given attribute sweep is run on the SAME sampled
     case set (not independently re-sampled per condition), because CFR
     (§9) is only meaningful as a same-case E+/E- comparison.

  3. N and P notes are attribute-independent (no carrier at all) and are
     generated once per case, then reused across all three attributes,
     rather than regenerated per attribute -- this is why §13's own cost
     table shows far fewer generation calls than a naive per-attribute
     regeneration would produce.

Currently-blocked conditions (EXP, IMP signal types; the P condition) are
recorded with a `blocked_reason` and skipped rather than aborting the run --
see Module 4's UnimplementedConditionError / NotImplementedError gates.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from shared.config_and_utils import (
    AGENT_B_MODELS,
    AGENT_B_API_MODELS,
    APIModelSpec,
    AgentBModel,
    BASELINE_TEMPERATURE,
    DRY_RUN,
    EXPERIMENT_5_LOW_TEMPERATURE,
    N_CASES_GO_NO_GO,
    _is_content_filter_error,
)
from shared.data_loader import AMQARecord, Attribute, load_amqa_dataset, sample_cases
from shared import conditions as cond
from shared import crew_pipeline as pipe
from shared import inference_engine as inf

logger = logging.getLogger(__name__)

TASKS: tuple[str, ...] = ("esi", "usmle")


def _model_display_name(model: AgentBModel | APIModelSpec | str) -> str:
    """
    AgentBModel.display_name for the real open-weight models (the logprobs
    path), APIModelSpec.display_name for an API-routed model (Azure or
    OpenRouter), or the raw string itself for a bare Chat-Completions
    deployment name -- lets every function below accept any of these three
    "model" shapes without needing parallel code paths for this one field.
    """
    if isinstance(model, (AgentBModel, APIModelSpec)):
        return model.display_name
    return model


def _resolve_model_connection(model: AgentBModel | APIModelSpec | str) -> tuple[str, str | None, str | None, dict | None]:
    """
    Returns (api_model_id, base_url, api_key, extra_body) for whichever
    "model" shape is passed. This is what lets a single experiment loop
    over models from DIFFERENT providers (Azure directly, OpenRouter) in
    one run without one model's connection silently overriding another's --
    each APIModelSpec carries its own base_url/api_key/extra_body, resolved
    fresh for every call rather than relying on a single global default.

    A bare string (this project's original single-model convention, e.g.
    "gpt-5.4-mini") resolves to (model, None, None, None) -- the inference
    functions already fall back to the global OPENAI_API_KEY/OPENAI_BASE_URL
    when base_url/api_key are None, so this is exactly the pre-existing
    behavior, unchanged.
    """
    if isinstance(model, APIModelSpec):
        return model.api_model_id, model.base_url, model.api_key, model.extra_body
    if isinstance(model, AgentBModel):
        return model.hf_identifier, None, None, None
    return model, None, None, None


def _require_live_for_chat_completions(dry_run: bool | None) -> None:
    """
    The Chat-Completions inference functions have no dry_run branch at all
    (they're live-only by design -- see inference_engine.py). Without this
    guard, `use_chat_completions=True` with `dry_run` still True (or
    defaulting to the global DRY_RUN) would let Agent A correctly mock
    while Agent B silently made real, billed live calls -- an easy,
    expensive mistake to make silently. Fail loudly instead.
    """
    effective_dry_run = DRY_RUN if dry_run is None else dry_run
    if effective_dry_run:
        raise RuntimeError(
            "use_chat_completions=True requires dry_run=False (or DRY_RUN=false "
            "in .env) -- the Chat-Completions inference functions have no mock "
            "path and would otherwise make real, billed live calls while Agent A "
            "silently mocked. Pass dry_run=False explicitly to confirm this is "
            "intentional."
        )


def _call_chat_completions_safe(fn: Callable, *args, **kwargs):
    """
    Call a Chat-Completions inference function, catching ONLY Azure's
    content-filter block (config_and_utils._is_content_filter_error) and
    returning None -- the caller records a blocked observation and moves
    on to the next case/condition -- rather than letting one filtered
    request crash an entire large, multi-hour experiment run. Every other
    exception still propagates unchanged.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if _is_content_filter_error(exc):
            logger.warning(
                "[experiments] Content-filter block (Azure ResponsibleAIPolicyViolation) -- "
                "recording as blocked_reason='content_filter_azure' and continuing. "
                "Underlying error: %s",
                exc,
            )
            return None
        raise


# ---------------------------------------------------------------------------
# Result record + resumable checkpoint store
# ---------------------------------------------------------------------------


@dataclass
class ObservationRecord:
    """One atomic (case, condition, model, task) observation."""

    experiment: str
    case_id: str
    attribute: str | None
    condition_label: str
    signal_type: str | None
    model: str
    task: str  # "esi" | "usmle" | "esi_cot" | "usmle_cot" | "n/a" (blocked)
    temperature: float
    top_class: str | None
    class_probabilities: dict[str, float] | None
    validation_passed: bool | None
    blocked_reason: str | None
    logprobs_available: bool | None = None
    # NEW field, part of the uniqueness key below. Without this, switching
    # from the one-hot Chat-Completions path to the new real-logprobs path
    # (both producing a record at the SAME temperature) would silently
    # collide: the checkpoint would see "already have this (case,
    # condition, ..., temperature)" and skip calling Agent B again --
    # meaning re-running after wiring in logprobs would just keep the old
    # one-hot data forever rather than ever collecting the new
    # measurement. Including this in key() makes a logprobs observation
    # and a one-hot observation at the same temperature genuinely
    # distinct, non-colliding entries.

    def key(self) -> str:
        """Unique key for resumability/dedup checkpointing."""
        return "|".join(
            [
                self.experiment,
                self.case_id,
                str(self.attribute),
                self.condition_label,
                str(self.signal_type),
                self.model,
                self.task,
                str(self.temperature),
                str(self.logprobs_available),
            ]
        )


class JSONLCheckpoint:
    """
    Append-only JSONL results store. Each ObservationRecord is written as one
    line; on resume, already-written keys are skipped rather than re-run
    (and re-billed) -- this is what makes experiment runs resumable after an
    interruption.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_keys: set[str] = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    key = record.get("_key")
                    if key:
                        self._seen_keys.add(key)

    def has(self, record: ObservationRecord) -> bool:
        return record.key() in self._seen_keys

    def append(self, record: ObservationRecord) -> None:
        payload = asdict(record)
        payload["_key"] = record.key()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        self._seen_keys.add(record.key())

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


# ---------------------------------------------------------------------------
# API call budget tracking
# Source: phase_0_complete_v3.md §13
# ---------------------------------------------------------------------------


@dataclass
class APICallTracker:
    agent_a_calls: int = 0
    agent_b_calls: int = 0
    validation_calls: int = 0

    def record_generation(self, n: int = 1) -> None:
        self.agent_a_calls += n

    def record_validation(self, n: int = 1) -> None:
        self.validation_calls += n

    def record_inference(self, n: int = 1) -> None:
        self.agent_b_calls += n

    @property
    def total(self) -> int:
        return self.agent_a_calls + self.agent_b_calls + self.validation_calls


# Transcribed from §13's own cost table. These are the spec's aggregate
# estimates, not a strict contract this project's actual counts must match
# exactly -- documented optimizations here (N/P reuse across attributes;
# see module docstring) legitimately produce different real totals. Used
# only for a sanity-check warning, never a hard assertion.
API_CALL_BUDGET: dict[str, dict[str, int]] = {
    "experiment_1": {"agent_a": 1200, "agent_b": 6000, "validation": 3600, "total": 10800},
    "experiment_2": {"agent_a": 400, "agent_b": 1200, "validation": 1200, "total": 2800},
    "experiment_3": {"agent_a": 0, "agent_b": 400, "validation": 0, "total": 400},
    "experiment_4": {"agent_a": 0, "agent_b": 0, "validation": 0, "total": 0},
    "experiment_5": {"agent_a": 0, "agent_b": 200, "validation": 0, "total": 200},
    "experiment_6": {"agent_a": 0, "agent_b": 40, "validation": 0, "total": 40},
}


def compare_to_budget(experiment_name: str, tracker: APICallTracker, *, tolerance: float = 0.5) -> None:
    """Log (never raise) if actual calls differ from §13's estimate by more than `tolerance` (fractional)."""
    budget = API_CALL_BUDGET.get(experiment_name)
    if budget is None or budget["total"] == 0:
        return
    deviation = abs(tracker.total - budget["total"]) / budget["total"]
    if deviation > tolerance:
        logger.warning(
            "[experiments] %s: actual API calls (%d) differ from §13's estimate "
            "(%d) by %.0f%%. Some deviation is expected from documented "
            "optimizations (N/P note reuse across attributes) or from "
            "currently-blocked conditions reducing real call counts -- "
            "double-check nothing is silently over- or under-running.",
            experiment_name,
            tracker.total,
            budget["total"],
            deviation * 100,
        )


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRunResult:
    experiment_name: str
    n_cases: int
    records_written: int
    blocked_conditions: dict[str, int]  # e.g. "ses:E+:EXP" -> count of cases blocked
    call_tracker: APICallTracker
    content_filter_blocks: int = 0  # count of individual (case, condition, task) observations Azure's content filter blocked


def _validation_calls_for(spec: cond.ConditionSpec) -> int:
    """3 checks (recommendation, carrier-preservation, facts) if a carrier is present; 2 if not (N)."""
    return 3 if spec.carrier_value is not None else 2


def _probe_records(
    *,
    experiment: str,
    case_id: str,
    attribute: str | None,
    spec: cond.ConditionSpec,
    models: list[AgentBModel],
    tasks: tuple[str, ...],
    temperature: float,
    logprobs_available: bool,
) -> list[ObservationRecord]:
    """
    Build the (model, task) ObservationRecords a spec WOULD produce, for
    existence-checking only (the non-key fields are dummy values -- key()
    doesn't depend on them, EXCEPT logprobs_available, which now IS part
    of key() -- see ObservationRecord. Must match the measurement method
    actually configured for this run, or every probe would silently
    mismatch every real stored record (all real records have True/False,
    never the omitted-default None), making _generation_needed always
    return True and wastefully regenerate notes on every resumed run.
    """
    return [
        ObservationRecord(
            experiment=experiment,
            case_id=case_id,
            attribute=attribute,
            condition_label=spec.condition_label.value,
            signal_type=spec.signal_type.value if spec.signal_type else None,
            model=_model_display_name(model),
            task=task,
            temperature=temperature,
            top_class=None,
            class_probabilities=None,
            validation_passed=None,
            blocked_reason=None,
            logprobs_available=logprobs_available,
        )
        for model in models
        for task in tasks
    ]


def _generation_needed(
    checkpoint: JSONLCheckpoint,
    *,
    experiment: str,
    case_id: str,
    attribute: str | None,
    spec: cond.ConditionSpec,
    models: list[AgentBModel],
    tasks: tuple[str, ...],
    temperature: float,
    logprobs_available: bool,
) -> bool:
    """
    True unless EVERY downstream Agent B record this spec would produce is
    already checkpointed -- in which case regenerating the note (and
    re-billing Agent A + the Validator) would be pure waste on a resumed run.
    """
    probes = _probe_records(
        experiment=experiment,
        case_id=case_id,
        attribute=attribute,
        spec=spec,
        models=models,
        tasks=tasks,
        temperature=temperature,
        logprobs_available=logprobs_available,
    )
    return not all(checkpoint.has(r) for r in probes)


def _record_blocked(
    checkpoint: JSONLCheckpoint,
    *,
    experiment: str,
    case_id: str,
    attribute: str | None,
    spec: cond.ConditionSpec,
    reason: str,
    temperature: float,
) -> None:
    record = ObservationRecord(
        experiment=experiment,
        case_id=case_id,
        attribute=attribute,
        condition_label=spec.condition_label.value,
        signal_type=spec.signal_type.value if spec.signal_type else None,
        model="n/a",
        task="n/a",
        temperature=temperature,
        top_class=None,
        class_probabilities=None,
        validation_passed=None,
        blocked_reason=reason,
    )
    if not checkpoint.has(record):
        checkpoint.append(record)


# ---------------------------------------------------------------------------
# Shared runner for Experiments 1 and 2 (both are attribute x condition
# sweeps over the same paired case set; they differ only in which
# conditions.py builder function selects the condition subset)
# ---------------------------------------------------------------------------


def _process_one_case_for_sweep(
    case: AMQARecord,
    *,
    attributes: list[Attribute],
    condition_builder: Callable[[str, Attribute], list[cond.ConditionSpec]],
    experiment_name: str,
    models: list[AgentBModel | str],
    use_chat_completions: bool,
    use_logprobs: bool,
    dry_run: bool | None,
    temperature: float,
    checkpoint: "JSONLCheckpoint",
) -> tuple[list[ObservationRecord], dict]:
    """
    Process EVERY attribute/spec/condition/model/task for ONE case --
    fully independent of every other case, making this the natural unit
    of parallelism (see the threading note on _run_signal_and_attribute_sweep).

    Returns (records_to_append, local_counts) rather than mutating
    `checkpoint`, a shared tracker, or shared blocked_counts/
    content_filter_blocks directly -- this is what makes it safe to run
    inside a worker thread with ZERO locks needed anywhere: checkpoint.has()
    is read-only (safe to call concurrently, since nothing mutates
    checkpoint._seen_keys during the parallel phase -- only the caller,
    sequentially, after every worker has returned), and every actual
    mutation (checkpoint.append(), tracker increments, blocked_counts
    updates) happens back in the calling thread, one case's results at a
    time, never interleaved across cases.

    Reordering the original attribute-outer/case-inner loop to case-outer/
    attribute-inner (this function handles all attributes for ONE case)
    is also what keeps neutral_cache/paraphrase_cache correct under
    threading with no changes to their logic at all: since one worker
    thread now handles every attribute for a given case sequentially,
    the cache stays purely LOCAL to this one case's processing -- no
    cross-thread cache access, no cache-related race condition to
    reason about.
    """
    records_to_append: list[ObservationRecord] = []
    local_counts = {
        "agent_a": 0, "agent_b": 0, "validation": 0,
        "content_filter_blocks": 0, "blocked_counts": {},
    }
    neutral_cache: dict[str, pipe.ValidatedNote] = {}
    paraphrase_cache: dict[str, pipe.ValidatedNote] = {}

    for attribute in attributes:
        matrix = condition_builder(case.question_id, attribute)
        for spec in matrix:
            is_neutral = spec.condition_label is cond.ConditionLabel.NEUTRAL
            is_paraphrase = spec.condition_label is cond.ConditionLabel.PARAPHRASE

            if not _generation_needed(
                checkpoint,
                experiment=experiment_name,
                case_id=case.question_id,
                attribute=attribute.value,
                spec=spec,
                models=models,
                tasks=TASKS,
                temperature=temperature,
                logprobs_available=use_chat_completions and use_logprobs,
            ):
                continue

            if is_neutral and case.question_id in neutral_cache:
                validated = neutral_cache[case.question_id]
            elif is_paraphrase and case.question_id in paraphrase_cache:
                validated = paraphrase_cache[case.question_id]
            else:
                try:
                    if is_paraphrase:
                        validated = pipe.generate_note(
                            case, spec, neutral_note=neutral_cache.get(case.question_id), dry_run=dry_run
                        )
                    else:
                        logger.debug("[experiments] Starting generate_note: case_id=%s, attribute=%s, spec=%s", case.question_id, attribute.value, spec)
                        validated = pipe.generate_note(case, spec, dry_run=dry_run)
                        logger.debug("[experiments] Finished generate_note: case_id=%s, blocked=%s", case.question_id, validated.blocked_reason)
                except (pipe.UnimplementedConditionError, NotImplementedError) as exc:
                    blocked_key = (
                        f"{attribute.value}:{spec.condition_label.value}:"
                        f"{spec.signal_type.value if spec.signal_type else 'None'}"
                    )
                    local_counts["blocked_counts"][blocked_key] = local_counts["blocked_counts"].get(blocked_key, 0) + 1
                    blocked_record = ObservationRecord(
                        experiment=experiment_name, case_id=case.question_id, attribute=attribute.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model="n/a", task="n/a", temperature=temperature, top_class=None,
                        class_probabilities=None, validation_passed=None, blocked_reason=str(exc),
                    )
                    if not checkpoint.has(blocked_record):
                        records_to_append.append(blocked_record)
                    continue
                local_counts["agent_a"] += 1
                local_counts["validation"] += _validation_calls_for(spec)

                if validated.blocked_reason:
                    blocked_key = f"{attribute.value}:{spec.condition_label.value}:{spec.signal_type.value if spec.signal_type else 'None'}"
                    local_counts["blocked_counts"][blocked_key] = local_counts["blocked_counts"].get(blocked_key, 0) + 1
                    blocked_record = ObservationRecord(
                        experiment=experiment_name, case_id=case.question_id, attribute=attribute.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model="n/a", task="n/a", temperature=temperature, top_class=None,
                        class_probabilities=None, validation_passed=None, blocked_reason=validated.blocked_reason,
                    )
                    if not checkpoint.has(blocked_record):
                        records_to_append.append(blocked_record)
                    continue

                if is_neutral:
                    neutral_cache[case.question_id] = validated
                elif is_paraphrase:
                    paraphrase_cache[case.question_id] = validated

            for model in models:
                for task in TASKS:
                    record = ObservationRecord(
                        experiment=experiment_name,
                        case_id=case.question_id,
                        attribute=attribute.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model=_model_display_name(model),
                        task=task,
                        temperature=temperature,
                        top_class=None,
                        class_probabilities=None,
                        validation_passed=validated.validation_passed,
                        blocked_reason=None,
                        logprobs_available=use_chat_completions and use_logprobs,
                    )
                    if checkpoint.has(record):
                        continue
                    if use_chat_completions:
                        model_id, base_url, api_key, extra_body = _resolve_model_connection(model)
                        logger.debug("[experiments] Calling Agent B: case_id=%s, model=%s, task=%s", case.question_id, _model_display_name(model), task)
                        if task == "esi":
                            esi_fn = inf.run_esi_triage_chat_completions_logprobs if use_logprobs else inf.run_esi_triage_chat_completions
                            result = _call_chat_completions_safe(
                                esi_fn, validated.note_text, model=model_id, temperature=temperature,
                                base_url=base_url, api_key=api_key, extra_body=extra_body,
                            )
                            logger.debug("[experiments] Agent B returned: case_id=%s, model=%s, task=%s, result=%s", case.question_id, _model_display_name(model), task, getattr(result, 'top_class', None) if result is not None else 'BLOCKED')
                        else:
                            usmle_fn = inf.run_usmle_diagnostic_chat_completions_logprobs if use_logprobs else inf.run_usmle_diagnostic_chat_completions
                            result = _call_chat_completions_safe(
                                usmle_fn, validated.note_text, case.neutralised_question, case.options,
                                model=model_id, temperature=temperature,
                                base_url=base_url, api_key=api_key, extra_body=extra_body,
                            )
                        logger.debug("[experiments] Agent B returned: case_id=%s, model=%s, task=%s, result=%s", case.question_id, _model_display_name(model), task, getattr(result, 'top_class', None) if result is not None else 'BLOCKED')
                        local_counts["agent_b"] += 1
                        if result is None:
                            local_counts["content_filter_blocks"] += 1
                            record.blocked_reason = "content_filter_azure"
                            records_to_append.append(record)
                            continue
                    elif task == "esi":
                        result = inf.run_esi_triage(
                            validated.note_text, model=model, temperature=temperature, dry_run=dry_run
                        )
                        local_counts["agent_b"] += 1
                    else:
                        result = inf.run_usmle_diagnostic(
                            validated.note_text, case.neutralised_question, case.options,
                            model=model, temperature=temperature, dry_run=dry_run,
                        )
                        local_counts["agent_b"] += 1
                    record.top_class = result.top_class
                    record.class_probabilities = result.class_probabilities
                    records_to_append.append(record)

    return records_to_append, local_counts


def _format_duration(seconds: float) -> str:
    """Human-readable duration for progress reporting (e.g. '45s', '2m 15s', '1h 5m')."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _run_signal_and_attribute_sweep(
    *,
    experiment_name: str,
    condition_builder: Callable[[str, Attribute], list[cond.ConditionSpec]],
    attributes: list[Attribute],
    checkpoint_path: str | Path,
    n_cases: int,
    dry_run: bool | None,
    temperature: float,
    models: list[AgentBModel] | None = None,
    use_chat_completions: bool = False,
    use_logprobs: bool = True,
    chat_completions_models: list[APIModelSpec | str] | None = None,
    max_workers: int = 5,
) -> ExperimentRunResult:
    """
    `use_chat_completions`: route Agent B through Chat Completions instead
    of the legacy-Completions functions (run_esi_triage/run_usmle_diagnostic)
    -- for running this experiment live against an OpenAI/Azure-family
    model BEFORE GPU/open-weight access is ready.

    `use_logprobs` (only meaningful when use_chat_completions=True):
    True (default) uses run_esi_triage_chat_completions_logprobs /
    run_usmle_diagnostic_chat_completions_logprobs -- confirmed live to
    return genuine multi-class probability distributions on this
    project's Azure gpt-5.4-mini deployment (verified: an 85%/15% split
    across two ESI classes on a realistic, non-trivial vignette, not a
    forced single-token result). False uses the older one-hot text-parsing
    functions (run_esi_triage_chat_completions /
    run_usmle_diagnostic_chat_completions) -- kept available for direct
    comparison, and currently the ONLY option for the OpenRouter-routed
    models (DeepSeek-V4-Flash, GPT-5.6-Luna, Gemini-3.8-Flash), whose
    logprobs support has not been confirmed reliable through OpenRouter
    (see test_openrouter_logprobs.py and AGENT_B_API_MODELS's own comment).

    `chat_completions_models`: a LIST of models (APIModelSpec for Azure/
    OpenRouter, or a bare string) -- defaults to [gpt-5.4-mini] if not
    given, matching this project's original single-model behavior. `models`
    (the AgentBModel-based open-weight roster) is ignored when
    use_chat_completions is True. Every downstream metrics function in
    shared/metrics.py reads `model` as an opaque string key and needs no
    changes to consume any of these variants -- multi-model results are
    told apart purely by that string, resolved from
    APIModelSpec.display_name/AgentBModel.display_name/the bare string
    itself via _model_display_name.

    `max_workers`: cases are processed concurrently via a thread pool
    (each case is fully independent of every other -- the natural unit of
    parallelism for I/O-bound API calls). Default 5, deliberately modest
    -- Azure/OpenAI enforce real rate limits, and too high a value trades
    a faster wall-clock time for 429 rate-limit errors instead. Set to 1
    to fall back to strictly sequential execution (e.g. for easier
    debugging of a specific failure). Correctness under concurrency is by
    DESIGN, not locks: _process_one_case_for_sweep never mutates
    `checkpoint`, `tracker`, `blocked_counts`, or `content_filter_blocks`
    directly -- it only reads checkpoint.has() (safe to call concurrently,
    since nothing mutates checkpoint._seen_keys during the parallel phase)
    and returns plain data. Every actual mutation happens back in THIS
    function, sequentially, one case's results at a time, as each worker
    completes -- so no two threads ever touch shared state at the same
    time, and no lock is needed anywhere.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)
    blocked_counts: dict[str, int] = {}
    content_filter_blocks = 0
    models = (chat_completions_models or [AGENT_B_API_MODELS[0]]) if use_chat_completions else (models or list(AGENT_B_MODELS))

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    print(
        f"[{experiment_name}] Processing {len(cases)} cases across {len(models)} model(s) "
        f"with {max_workers} concurrent worker(s)... (each case runs every attribute/condition/task "
        f"for every model, so one case = many individual API calls, not one)"
    )
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_one_case_for_sweep,
                case,
                attributes=attributes,
                condition_builder=condition_builder,
                experiment_name=experiment_name,
                models=models,
                use_chat_completions=use_chat_completions,
                use_logprobs=use_logprobs,
                dry_run=dry_run,
                temperature=temperature,
                checkpoint=checkpoint,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            # .result() re-raises any exception from the worker thread here,
            # in the main thread -- a real error (not a content-filter block,
            # which is already handled inside the worker) still stops the
            # run, exactly as it would have sequentially.
            records_to_append, local_counts = future.result()

            for record in records_to_append:
                checkpoint.append(record)
            tracker.record_generation(local_counts["agent_a"])
            tracker.record_validation(local_counts["validation"])
            tracker.record_inference(local_counts["agent_b"])
            content_filter_blocks += local_counts["content_filter_blocks"]
            for key, count in local_counts["blocked_counts"].items():
                blocked_counts[key] = blocked_counts.get(key, 0) + count

            completed += 1
            elapsed = time.time() - start_time
            avg_per_case = elapsed / completed
            remaining = avg_per_case * (len(cases) - completed)
            case_id = futures[future].question_id
            print(
                f"[{experiment_name}] Case {completed}/{len(cases)} done ({completed / len(cases):.0%})"
                f"  case_id={case_id}"
                f"  elapsed={_format_duration(elapsed)}"
                f"  avg={avg_per_case:.1f}s/case"
                f"  ETA={_format_duration(remaining)}"
            )

    compare_to_budget(experiment_name, tracker)
    return ExperimentRunResult(
        experiment_name=experiment_name,
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions=blocked_counts,
        call_tracker=tracker,
        content_filter_blocks=content_filter_blocks,
    )


def run_experiment_1(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_1.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
    use_chat_completions: bool = False,
    use_logprobs: bool = True,
    chat_completions_models: list[APIModelSpec | str] | None = None,
    max_workers: int = 5,
) -> ExperimentRunResult:
    """Experiment 1 (§11) — signal-type gradient: EXP/PRX/IMP x polarity + N + P, all 3 attributes (SES first), both tasks."""
    return _run_signal_and_attribute_sweep(
        experiment_name="experiment_1",
        condition_builder=cond.build_signal_gradient_conditions,
        attributes=[Attribute.SES, Attribute.RACE, Attribute.SEX],  # SES first, per spec
        checkpoint_path=checkpoint_path,
        n_cases=n_cases,
        dry_run=dry_run,
        temperature=temperature,
        use_chat_completions=use_chat_completions,
        use_logprobs=use_logprobs,
        max_workers=max_workers,
        chat_completions_models=chat_completions_models,
    )


def run_experiment_2(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_2.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
    use_chat_completions: bool = False,
    use_logprobs: bool = True,
    chat_completions_models: list[APIModelSpec | str] | None = None,
    max_workers: int = 5,
) -> ExperimentRunResult:
    """Experiment 2 (§11) — cross-attribute replication: E+/E-/N/P using PRX carrier only, all 3 attributes, both tasks."""
    return _run_signal_and_attribute_sweep(
        experiment_name="experiment_2",
        condition_builder=cond.build_cross_attribute_conditions,
        attributes=[Attribute.SES, Attribute.RACE, Attribute.SEX],
        checkpoint_path=checkpoint_path,
        n_cases=n_cases,
        dry_run=dry_run,
        use_logprobs=use_logprobs,
        max_workers=max_workers,
        temperature=temperature,
        use_chat_completions=use_chat_completions,
        chat_completions_models=chat_completions_models,
    )


# ---------------------------------------------------------------------------
# Experiment 3 — carrier causality (redaction)
# ---------------------------------------------------------------------------


def run_experiment_3(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_3.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
    use_chat_completions: bool = False,
    use_logprobs: bool = True,
    chat_completions_models: list[APIModelSpec | str] | None = None,
) -> ExperimentRunResult:
    """
    Experiment 3 (§11) — carrier causality / redaction check. E+ (PRX, SES),
    E+_redact, N. No attribute sweep (SES/PRX only, "the most isolatable
    single phrase" per spec). Fully implemented, no blocked conditions.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)
    content_filter_blocks = 0
    models = (chat_completions_models or [AGENT_B_API_MODELS[0]]) if use_chat_completions else list(AGENT_B_MODELS)

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    print(f"[experiment_3] Processing {len(cases)} cases across {len(models)} model(s), sequentially...")
    start_time = time.time()

    for case_index, case in enumerate(cases, start=1):
        matrix = cond.build_redaction_conditions(case.question_id)
        for spec in matrix:
            if not _generation_needed(
                checkpoint,
                experiment="experiment_3",
                case_id=case.question_id,
                attribute=Attribute.SES.value,
                spec=spec,
                models=models,
                tasks=TASKS,
                temperature=temperature,
                logprobs_available=use_chat_completions and use_logprobs,
            ):
                continue

            validated = pipe.generate_note(case, spec, dry_run=dry_run)
            tracker.record_generation()
            tracker.record_validation(_validation_calls_for(spec))

            if validated.blocked_reason:
                _record_blocked(
                    checkpoint, experiment="experiment_3", case_id=case.question_id,
                    attribute=Attribute.SES.value, spec=spec, reason=validated.blocked_reason,
                    temperature=temperature,
                )
                continue

            for model in models:
                for task in TASKS:
                    record = ObservationRecord(
                        experiment="experiment_3",
                        case_id=case.question_id,
                        attribute=Attribute.SES.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model=_model_display_name(model),
                        task=task,
                        temperature=temperature,
                        top_class=None,
                        class_probabilities=None,
                        validation_passed=validated.validation_passed,
                        blocked_reason=None,
                        logprobs_available=use_chat_completions and use_logprobs,
                    )
                    if checkpoint.has(record):
                        continue
                    if use_chat_completions:
                        model_id, base_url, api_key, extra_body = _resolve_model_connection(model)
                        if task == "esi":
                            esi_fn = inf.run_esi_triage_chat_completions_logprobs if use_logprobs else inf.run_esi_triage_chat_completions
                            result = _call_chat_completions_safe(
                                esi_fn,
                                validated.note_text, model=model_id, temperature=temperature,
                                base_url=base_url, api_key=api_key, extra_body=extra_body,
                            )
                        else:
                            usmle_fn = inf.run_usmle_diagnostic_chat_completions_logprobs if use_logprobs else inf.run_usmle_diagnostic_chat_completions
                            result = _call_chat_completions_safe(
                                usmle_fn,
                                validated.note_text,
                                case.neutralised_question,
                                case.options,
                                model=model_id,
                                temperature=temperature,
                                base_url=base_url, api_key=api_key, extra_body=extra_body,
                            )
                        tracker.record_inference()
                        if result is None:
                            content_filter_blocks += 1
                            record.blocked_reason = "content_filter_azure"
                            checkpoint.append(record)
                            continue
                    elif task == "esi":
                        result = inf.run_esi_triage(
                            validated.note_text, model=model, temperature=temperature, dry_run=dry_run
                        )
                        tracker.record_inference()
                    else:
                        result = inf.run_usmle_diagnostic(
                            validated.note_text,
                            case.neutralised_question,
                            case.options,
                            model=model,
                            temperature=temperature,
                            dry_run=dry_run,
                        )
                        tracker.record_inference()
                    record.top_class = result.top_class
                    record.class_probabilities = result.class_probabilities
                    checkpoint.append(record)

        elapsed = time.time() - start_time
        avg_per_case = elapsed / case_index
        remaining = avg_per_case * (len(cases) - case_index)
        print(
            f"[experiment_3] Case {case_index}/{len(cases)} done ({case_index / len(cases):.0%})"
            f"  case_id={case.question_id}"
            f"  elapsed={_format_duration(elapsed)}"
            f"  avg={avg_per_case:.1f}s/case"
            f"  ETA={_format_duration(remaining)}"
        )

    compare_to_budget("experiment_3", tracker)
    return ExperimentRunResult(
        experiment_name="experiment_3",
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions={},
        call_tracker=tracker,
        content_filter_blocks=content_filter_blocks,
    )


# ---------------------------------------------------------------------------
# Experiment REC — recommendation-conformity control
# (N / REC_correct / REC_wrong / REC_UP / REC_DOWN)
#
# This has NO precedent in the original v3 spec's Experiments 1-6 -- it did
# not exist until this project's own live-debugging phase (see
# phase_0_rec_full_wiring_addendum.md and phase_0_combined_prompt_and_rec_
# addendum.md), and until now only ever ran ad hoc inside
# run_live_smoke_test.py, with no checkpointing/resumability and no
# connection to shared/metrics.py's actual statistical functions. This gives
# it the same resumable, checkpointed structure as Experiments 1-3.
#
# Chat-Completions ONLY: unlike Experiments 1-3 (which can run against
# either the real open-weight/logprobs path or this fallback), REC
# conditions have no logprobs-based precedent at all -- they were designed
# and tested against the Chat-Completions path from the start. No
# use_chat_completions toggle here; it is always on.
# ---------------------------------------------------------------------------


def run_experiment_rec(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_rec.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
    use_logprobs: bool = True,
    chat_completions_models: list[APIModelSpec | str] | None = None,
    run_combined_context_on_rec_wrong: bool = True,
) -> ExperimentRunResult:
    """
    N / REC_correct / REC_wrong / REC_UP / REC_DOWN, both tasks (ESI and
    USMLE), via the Chat-Completions path -- the properly checkpointed,
    resumable equivalent of what run_live_smoke_test.py has been doing ad
    hoc. `case.correct_answer_idx` and `case.options` (already loaded by
    Module 2) anchor REC_correct/REC_wrong's ground truth.

    `chat_completions_models`: a LIST of models (APIModelSpec for Azure/
    OpenRouter, or a bare string) -- defaults to [gpt-5.4-mini] if not
    given. Agent A's note is generated ONCE per (case, condition)
    regardless of how many models are tested; only the Agent B measurement
    step loops per model, since the note itself doesn't depend on which
    model will read it -- avoids redundant, wasteful note regeneration.

    `run_combined_context_on_rec_wrong`: also runs the combined-context
    Agent B variant (shared/inference_engine.py's
    run_esi_then_usmle_combined_chat_completions) specifically on REC_wrong
    -- stored as two extra ObservationRecord rows per case with
    task="esi_combined"/"usmle_combined" -- grounded in Stanwyck et al.
    (2026); see phase_0_combined_prompt_and_rec_addendum.md for why this is
    scoped to REC_wrong only, not every condition.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)
    models = chat_completions_models or [AGENT_B_API_MODELS[0]]
    content_filter_blocks = 0

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    print(f"[experiment_rec] Processing {len(cases)} cases across {len(models)} model(s), sequentially...")
    start_time = time.time()

    for case_index, case in enumerate(cases, start=1):
        matrix = cond.build_recommendation_conditions(case.question_id, case.correct_answer_idx, case.options)

        for spec in matrix:
            if not _generation_needed(
                checkpoint,
                experiment="experiment_rec",
                case_id=case.question_id,
                attribute=None,
                spec=spec,
                models=models,
                tasks=TASKS,
                temperature=temperature,
                logprobs_available=use_logprobs,
            ):
                continue

            try:
                validated = pipe.generate_note(case, spec, dry_run=dry_run)
            except (pipe.UnimplementedConditionError, NotImplementedError) as exc:
                _record_blocked(
                    checkpoint,
                    experiment="experiment_rec",
                    case_id=case.question_id,
                    attribute=None,
                    spec=spec,
                    reason=str(exc),
                    temperature=temperature,
                )
                continue
            tracker.record_generation()
            tracker.record_validation(_validation_calls_for(spec))

            if validated.blocked_reason:
                _record_blocked(
                    checkpoint, experiment="experiment_rec", case_id=case.question_id,
                    attribute=None, spec=spec, reason=validated.blocked_reason,
                    temperature=temperature,
                )
                continue

            for model in models:
                model_id, base_url, api_key, extra_body = _resolve_model_connection(model)
                model_display = _model_display_name(model)
                esi_top_class: str | None = None
                usmle_top_class: str | None = None

                for task in TASKS:
                    record = ObservationRecord(
                        experiment="experiment_rec",
                        case_id=case.question_id,
                        attribute=None,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model=model_display,
                        task=task,
                        temperature=temperature,
                        top_class=None,
                        class_probabilities=None,
                        validation_passed=validated.validation_passed,
                        blocked_reason=None,
                        logprobs_available=use_logprobs,
                    )
                    if checkpoint.has(record):
                        continue
                    if task == "esi":
                        esi_fn = inf.run_esi_triage_chat_completions_logprobs if use_logprobs else inf.run_esi_triage_chat_completions
                        result = _call_chat_completions_safe(
                            esi_fn,
                            validated.note_text, model=model_id, temperature=temperature,
                            base_url=base_url, api_key=api_key, extra_body=extra_body,
                        )
                        tracker.record_inference()
                        if result is None:
                            content_filter_blocks += 1
                            record.blocked_reason = "content_filter_azure"
                            checkpoint.append(record)
                            continue
                        esi_top_class = result.top_class
                    else:
                        usmle_fn = inf.run_usmle_diagnostic_chat_completions_logprobs if use_logprobs else inf.run_usmle_diagnostic_chat_completions
                        result = _call_chat_completions_safe(
                            usmle_fn,
                            validated.note_text, case.neutralised_question, case.options,
                            model=model_id, temperature=temperature,
                            base_url=base_url, api_key=api_key, extra_body=extra_body,
                        )
                        tracker.record_inference()
                        if result is None:
                            content_filter_blocks += 1
                            record.blocked_reason = "content_filter_azure"
                            checkpoint.append(record)
                            continue
                        usmle_top_class = result.top_class
                    record.top_class = result.top_class
                    record.class_probabilities = result.class_probabilities
                    checkpoint.append(record)

                if run_combined_context_on_rec_wrong and spec.condition_label is cond.ConditionLabel.REC_WRONG:
                    combined_esi_record = ObservationRecord(
                        experiment="experiment_rec", case_id=case.question_id, attribute=None,
                        condition_label=spec.condition_label.value, signal_type=None, model=model_display,
                        task="esi_combined", temperature=temperature, top_class=None,
                        class_probabilities=None, validation_passed=validated.validation_passed, blocked_reason=None,
                        logprobs_available=use_logprobs,
                    )
                    combined_usmle_record = ObservationRecord(
                        experiment="experiment_rec", case_id=case.question_id, attribute=None,
                        condition_label=spec.condition_label.value, signal_type=None, model=model_display,
                        task="usmle_combined", temperature=temperature, top_class=None,
                        class_probabilities=None, validation_passed=validated.validation_passed, blocked_reason=None,
                        logprobs_available=use_logprobs,
                    )
                    if not (checkpoint.has(combined_esi_record) and checkpoint.has(combined_usmle_record)):
                        combined_fn = inf.run_esi_then_usmle_combined_chat_completions_logprobs if use_logprobs else inf.run_esi_then_usmle_combined_chat_completions
                        combined = _call_chat_completions_safe(
                            combined_fn,
                            validated.note_text, case.neutralised_question, case.options,
                            model=model_id, temperature=temperature,
                            base_url=base_url, api_key=api_key, extra_body=extra_body,
                        )
                        tracker.record_inference(2)
                        if combined is None:
                            content_filter_blocks += 2
                            combined_esi_record.blocked_reason = "content_filter_azure"
                            combined_usmle_record.blocked_reason = "content_filter_azure"
                            checkpoint.append(combined_esi_record)
                            checkpoint.append(combined_usmle_record)
                        else:
                            combined_esi_record.top_class = combined.esi_result.top_class
                            combined_esi_record.class_probabilities = combined.esi_result.class_probabilities
                            combined_usmle_record.top_class = combined.usmle_result.top_class
                            combined_usmle_record.class_probabilities = combined.usmle_result.class_probabilities
                            checkpoint.append(combined_esi_record)
                            checkpoint.append(combined_usmle_record)

        elapsed = time.time() - start_time
        avg_per_case = elapsed / case_index
        remaining = avg_per_case * (len(cases) - case_index)
        print(
            f"[experiment_rec] Case {case_index}/{len(cases)} done ({case_index / len(cases):.0%})"
            f"  case_id={case.question_id}"
            f"  elapsed={_format_duration(elapsed)}"
            f"  avg={avg_per_case:.1f}s/case"
            f"  ETA={_format_duration(remaining)}"
        )

    return ExperimentRunResult(
        experiment_name="experiment_rec",
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions={},
        call_tracker=tracker,
        content_filter_blocks=content_filter_blocks,
    )


# ---------------------------------------------------------------------------
# Experiment 4 — cross-model analysis pass (no new API calls)
# ---------------------------------------------------------------------------


def run_experiment_4(
    *,
    experiment_1_checkpoint: str | Path = "phase0_results/experiment_1.jsonl",
    experiment_2_checkpoint: str | Path = "phase0_results/experiment_2.jsonl",
) -> dict:
    """
    Experiment 4 (§11) — cross-model analysis pass. No new API calls: this
    only loads and groups Experiment 1/2's already-collected records into
    the shapes needed for downstream statistics (CFR/AD heatmap, model-
    ordering correlation, etc.). The actual statistical computation belongs
    in Module 7 (metrics.py) -- this function is bookkeeping, not statistics.
    """
    exp1_records = JSONLCheckpoint(experiment_1_checkpoint).load_all()
    exp2_records = JSONLCheckpoint(experiment_2_checkpoint).load_all()

    def _group(records: list[dict]) -> dict[tuple, list[dict]]:
        grouped: dict[tuple, list[dict]] = {}
        for r in records:
            if r.get("blocked_reason"):
                continue
            key = (r["model"], r["attribute"], r["task"])
            grouped.setdefault(key, []).append(r)
        return grouped

    return {
        "experiment_1_grouped": _group(exp1_records),
        "experiment_2_grouped": _group(exp2_records),
    }


# ---------------------------------------------------------------------------
# Experiment 5 — temperature modulation (Okawa check)
# ---------------------------------------------------------------------------


def run_experiment_5(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_5.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
) -> ExperimentRunResult:
    """
    Experiment 5 (§11) — Okawa temperature-modulation check. E+ (PRX, SES) at
    T=0.2 (new) and T=0.8, ESI task only. Runs both temperatures
    self-contained (doesn't require Experiment 2 to have been run first).
    Fully implemented, no blocked conditions.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    for temperature in (EXPERIMENT_5_LOW_TEMPERATURE, BASELINE_TEMPERATURE):
        for case in cases:
            matrix = cond.build_cross_attribute_conditions(case.question_id, Attribute.SES)
            e_plus = next(
                s
                for s in matrix
                if s.condition_label is cond.ConditionLabel.E_PLUS and s.signal_type is cond.SignalType.PROXY
            )
            neutral = next(s for s in matrix if s.condition_label is cond.ConditionLabel.NEUTRAL)

            for spec in (e_plus, neutral):
                if not _generation_needed(
                    checkpoint,
                    experiment="experiment_5",
                    case_id=case.question_id,
                    attribute=Attribute.SES.value,
                    spec=spec,
                    models=list(AGENT_B_MODELS),
                    tasks=("esi",),  # ESI only, per spec
                    temperature=temperature,
                    logprobs_available=True,  # legacy Completions path -- genuine logprobs when the server returns them
                ):
                    continue

                validated = pipe.generate_note(case, spec, dry_run=dry_run)
                tracker.record_generation()
                tracker.record_validation(_validation_calls_for(spec))

                for model in AGENT_B_MODELS:
                    record = ObservationRecord(
                        experiment="experiment_5",
                        case_id=case.question_id,
                        attribute=Attribute.SES.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model=model.display_name,
                        task="esi",
                        temperature=temperature,
                        top_class=None,
                        class_probabilities=None,
                        validation_passed=validated.validation_passed,
                        blocked_reason=None,
                        logprobs_available=True,
                    )
                    if checkpoint.has(record):
                        continue
                    result = inf.run_esi_triage(
                        validated.note_text, model=model, temperature=temperature, dry_run=dry_run
                    )
                    tracker.record_inference()
                    record.top_class = result.top_class
                    record.class_probabilities = result.class_probabilities
                    checkpoint.append(record)

    compare_to_budget("experiment_5", tracker)
    return ExperimentRunResult(
        experiment_name="experiment_5",
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions={},
        call_tracker=tracker,
    )


# ---------------------------------------------------------------------------
# Experiment 6 — CoT blindness validation
# ---------------------------------------------------------------------------


def _select_decision_shifted_cases(
    records: list[dict],
    *,
    attribute: str,
    signal_type: str,
    task: str,
    n: int = 20,
) -> list[str]:
    """
    From a prior experiment's stored records, find case_ids where the top
    predicted class differed between E+ and E- for the given
    attribute/signal_type/task — the "cases where the decision shifted in
    Experiment 1" selection §11 Exp 6 calls for.
    """
    by_case: dict[str, dict[str, str]] = {}
    for r in records:
        if (
            r.get("attribute") == attribute
            and r.get("signal_type") == signal_type
            and r.get("task") == task
            and r.get("condition_label") in ("E+", "E-")
            and r.get("top_class")
        ):
            by_case.setdefault(r["case_id"], {})[r["condition_label"]] = r["top_class"]

    shifted = [
        case_id
        for case_id, decisions in by_case.items()
        if "E+" in decisions and "E-" in decisions and decisions["E+"] != decisions["E-"]
    ]
    return shifted[:n]


def run_experiment_6(
    *,
    source_checkpoint: str | Path = "phase0_results/experiment_1.jsonl",
    checkpoint_path: str | Path = "phase0_results/experiment_6.jsonl",
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
    n_cases: int = 20,
) -> ExperimentRunResult:
    """
    Experiment 6 (§11) — CoT blindness validation. E+ (EXP, explicit label),
    Llama-3.1-8B-Instruct only, both tasks (free-generation/CoT variant), on
    the N=20 cases where the decision shifted in a prior run (default
    source: Experiment 1's PRX/SES arm).

    BLOCKED: this experiment's defining condition is EXP (explicit label),
    gated behind PI approval (see phase_0_prompt_addendum_DRAFT.md). This
    runner is fully wired and will work the moment that gate lifts; until
    then every selected case is recorded with a blocked_reason.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)
    blocked_counts: dict[str, int] = {}

    source_records = JSONLCheckpoint(source_checkpoint).load_all()
    shifted_case_ids = _select_decision_shifted_cases(
        source_records,
        attribute=Attribute.SES.value,
        signal_type=cond.SignalType.PROXY.value,
        task="esi",
        n=n_cases,
    )
    if not shifted_case_ids:
        logger.warning(
            "[experiments] No decision-shifted cases found in %s -- either it "
            "hasn't been run yet, or no shifts occurred. Experiment 6 has "
            "nothing to do.",
            source_checkpoint,
        )
        return ExperimentRunResult(
            experiment_name="experiment_6",
            n_cases=0,
            records_written=0,
            blocked_conditions={},
            call_tracker=tracker,
        )

    llama = next(m for m in AGENT_B_MODELS if m.display_name == "Llama-3.1-8B-Instruct")
    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 20))
    cases_by_id = {c.question_id: c for c in amqa_records}

    for case_id in shifted_case_ids:
        case = cases_by_id.get(case_id)
        if case is None:
            logger.warning("[experiments] Case %s not found in the loaded AMQA pool -- skipping.", case_id)
            continue

        matrix = cond.build_signal_gradient_conditions(case_id, Attribute.SES)
        exp_spec = next(
            s
            for s in matrix
            if s.condition_label is cond.ConditionLabel.E_PLUS and s.signal_type is cond.SignalType.EXPLICIT
        )

        if not _generation_needed(
            checkpoint,
            experiment="experiment_6",
            case_id=case_id,
            attribute=Attribute.SES.value,
            spec=exp_spec,
            models=[llama],
            tasks=("esi_cot", "usmle_cot"),
            temperature=temperature,
            logprobs_available=None,  # CoT/free-text -- no class distribution concept applies at all
        ):
            continue

        try:
            validated = pipe.generate_note(case, exp_spec, dry_run=dry_run)
            tracker.record_generation()
            tracker.record_validation(_validation_calls_for(exp_spec))
        except (pipe.UnimplementedConditionError, NotImplementedError) as exc:
            blocked_counts["EXP"] = blocked_counts.get("EXP", 0) + 1
            _record_blocked(
                checkpoint,
                experiment="experiment_6",
                case_id=case_id,
                attribute=Attribute.SES.value,
                spec=exp_spec,
                reason=str(exc),
                temperature=temperature,
            )
            continue

        for task in TASKS:
            record = ObservationRecord(
                experiment="experiment_6",
                case_id=case_id,
                attribute=Attribute.SES.value,
                condition_label=exp_spec.condition_label.value,
                signal_type=exp_spec.signal_type.value,
                model=llama.display_name,
                task=f"{task}_cot",
                temperature=temperature,
                top_class=None,
                class_probabilities=None,
                validation_passed=validated.validation_passed,
                blocked_reason=None,
            )
            if checkpoint.has(record):
                continue
            if task == "esi":
                cot_result = inf.run_esi_triage_cot(
                    validated.note_text, model=llama, temperature=temperature, dry_run=dry_run
                )
            else:
                cot_result = inf.run_usmle_diagnostic_cot(
                    validated.note_text,
                    case.neutralised_question,
                    case.options,
                    model=llama,
                    temperature=temperature,
                    dry_run=dry_run,
                )
            tracker.record_inference()
            # Free text, not a class label -- stored (truncated) for the
            # manual demographic-mention coding pass §11 Exp 6 calls for.
            record.top_class = cot_result.text[:500]
            checkpoint.append(record)

    compare_to_budget("experiment_6", tracker)
    return ExperimentRunResult(
        experiment_name="experiment_6",
        n_cases=len(shifted_case_ids),
        records_written=len(checkpoint.load_all()),
        blocked_conditions=blocked_counts,
        call_tracker=tracker,
    )


__all__ = [
    "ObservationRecord",
    "JSONLCheckpoint",
    "APICallTracker",
    "API_CALL_BUDGET",
    "compare_to_budget",
    "ExperimentRunResult",
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "run_experiment_rec",
    "run_experiment_4",
    "run_experiment_5",
    "run_experiment_6",
]