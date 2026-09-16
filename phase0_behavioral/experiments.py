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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from shared.config_and_utils import (
    AGENT_B_MODELS,
    AgentBModel,
    BASELINE_TEMPERATURE,
    EXPERIMENT_5_LOW_TEMPERATURE,
    N_CASES_GO_NO_GO,
)
from shared.data_loader import AMQARecord, Attribute, load_amqa_dataset, sample_cases
from shared import conditions as cond
from shared import crew_pipeline as pipe
from shared import inference_engine as inf

logger = logging.getLogger(__name__)

TASKS: tuple[str, ...] = ("esi", "usmle")


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
) -> list[ObservationRecord]:
    """
    Build the (model, task) ObservationRecords a spec WOULD produce, for
    existence-checking only (the non-key fields are dummy values -- key()
    doesn't depend on them).
    """
    return [
        ObservationRecord(
            experiment=experiment,
            case_id=case_id,
            attribute=attribute,
            condition_label=spec.condition_label.value,
            signal_type=spec.signal_type.value if spec.signal_type else None,
            model=model.display_name,
            task=task,
            temperature=temperature,
            top_class=None,
            class_probabilities=None,
            validation_passed=None,
            blocked_reason=None,
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
) -> ExperimentRunResult:
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)
    blocked_counts: dict[str, int] = {}
    models = models or list(AGENT_B_MODELS)

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    # Attribute-independent cache: N (no carrier at all) and P (paraphrase,
    # matched against a fixed SES/PRX reference regardless of which attribute
    # is being studied -- see crew_pipeline.generate_paraphrase_note's
    # docstring) don't depend on the attribute -- generate each once per
    # case, reuse across all three attribute sweeps.
    neutral_cache: dict[str, pipe.ValidatedNote] = {}
    paraphrase_cache: dict[str, pipe.ValidatedNote] = {}

    for attribute in attributes:
        for case in cases:
            matrix = condition_builder(case.question_id, attribute)
            for spec in matrix:
                is_neutral = spec.condition_label is cond.ConditionLabel.NEUTRAL
                is_paraphrase = spec.condition_label is cond.ConditionLabel.PARAPHRASE

                # Resumability: skip generation entirely if every downstream
                # Agent B record for this spec is already checkpointed --
                # otherwise a resumed run would needlessly re-bill Agent A +
                # the Validator for a note whose decisions are all already on
                # disk (agent_b_calls was already being skipped per-record
                # below, but agent_a/validation calls were not -- fixed here).
                if not _generation_needed(
                    checkpoint,
                    experiment=experiment_name,
                    case_id=case.question_id,
                    attribute=attribute.value,
                    spec=spec,
                    models=models,
                    tasks=TASKS,
                    temperature=temperature,
                ):
                    continue

                if is_neutral and case.question_id in neutral_cache:
                    validated = neutral_cache[case.question_id]
                elif is_paraphrase and case.question_id in paraphrase_cache:
                    validated = paraphrase_cache[case.question_id]
                else:
                    try:
                        if is_paraphrase:
                            # Pass the cached N through if we have it, so P
                            # doesn't trigger its own redundant N generation
                            # (crew_pipeline.generate_paraphrase_note falls
                            # back to generating N itself only if omitted).
                            validated = pipe.generate_note(
                                case, spec, neutral_note=neutral_cache.get(case.question_id), dry_run=dry_run
                            )
                        else:
                            validated = pipe.generate_note(case, spec, dry_run=dry_run)
                    except (pipe.UnimplementedConditionError, NotImplementedError) as exc:
                        blocked_key = (
                            f"{attribute.value}:{spec.condition_label.value}:"
                            f"{spec.signal_type.value if spec.signal_type else 'None'}"
                        )
                        blocked_counts[blocked_key] = blocked_counts.get(blocked_key, 0) + 1
                        _record_blocked(
                            checkpoint,
                            experiment=experiment_name,
                            case_id=case.question_id,
                            attribute=attribute.value,
                            spec=spec,
                            reason=str(exc),
                            temperature=temperature,
                        )
                        continue
                    tracker.record_generation()
                    tracker.record_validation(_validation_calls_for(spec))
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
                            model=model.display_name,
                            task=task,
                            temperature=temperature,
                            top_class=None,
                            class_probabilities=None,
                            validation_passed=validated.validation_passed,
                            blocked_reason=None,
                        )
                        if checkpoint.has(record):
                            continue
                        if task == "esi":
                            result = inf.run_esi_triage(
                                validated.note_text, model=model, temperature=temperature, dry_run=dry_run
                            )
                        else:
                            result = inf.run_usmle_diagnostic(
                                validated.note_text,
                                case.neutralised_question,  # see module docstring, decision #1
                                case.options,
                                model=model,
                                temperature=temperature,
                                dry_run=dry_run,
                            )
                        tracker.record_inference()
                        record.top_class = result.top_class
                        record.class_probabilities = result.class_probabilities
                        checkpoint.append(record)

    compare_to_budget(experiment_name, tracker)
    return ExperimentRunResult(
        experiment_name=experiment_name,
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions=blocked_counts,
        call_tracker=tracker,
    )


def run_experiment_1(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_1.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
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
    )


def run_experiment_2(
    *,
    checkpoint_path: str | Path = "phase0_results/experiment_2.jsonl",
    n_cases: int = N_CASES_GO_NO_GO,
    dry_run: bool | None = None,
    temperature: float = BASELINE_TEMPERATURE,
) -> ExperimentRunResult:
    """Experiment 2 (§11) — cross-attribute replication: E+/E-/N/P using PRX carrier only, all 3 attributes, both tasks."""
    return _run_signal_and_attribute_sweep(
        experiment_name="experiment_2",
        condition_builder=cond.build_cross_attribute_conditions,
        attributes=[Attribute.SES, Attribute.RACE, Attribute.SEX],
        checkpoint_path=checkpoint_path,
        n_cases=n_cases,
        dry_run=dry_run,
        temperature=temperature,
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
) -> ExperimentRunResult:
    """
    Experiment 3 (§11) — carrier causality / redaction check. E+ (PRX, SES),
    E+_redact, N. No attribute sweep (SES/PRX only, "the most isolatable
    single phrase" per spec). Fully implemented, no blocked conditions.
    """
    tracker = APICallTracker()
    checkpoint = JSONLCheckpoint(checkpoint_path)

    amqa_records = load_amqa_dataset(dry_run=dry_run, mock_n=max(n_cases, 1))
    cases = sample_cases(amqa_records, n_cases, seed=42)

    for case in cases:
        matrix = cond.build_redaction_conditions(case.question_id)
        for spec in matrix:
            if not _generation_needed(
                checkpoint,
                experiment="experiment_3",
                case_id=case.question_id,
                attribute=Attribute.SES.value,
                spec=spec,
                models=list(AGENT_B_MODELS),
                tasks=TASKS,
                temperature=temperature,
            ):
                continue

            validated = pipe.generate_note(case, spec, dry_run=dry_run)
            tracker.record_generation()
            tracker.record_validation(_validation_calls_for(spec))

            for model in AGENT_B_MODELS:
                for task in TASKS:
                    record = ObservationRecord(
                        experiment="experiment_3",
                        case_id=case.question_id,
                        attribute=Attribute.SES.value,
                        condition_label=spec.condition_label.value,
                        signal_type=spec.signal_type.value if spec.signal_type else None,
                        model=model.display_name,
                        task=task,
                        temperature=temperature,
                        top_class=None,
                        class_probabilities=None,
                        validation_passed=validated.validation_passed,
                        blocked_reason=None,
                    )
                    if checkpoint.has(record):
                        continue
                    if task == "esi":
                        result = inf.run_esi_triage(
                            validated.note_text, model=model, temperature=temperature, dry_run=dry_run
                        )
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

    compare_to_budget("experiment_3", tracker)
    return ExperimentRunResult(
        experiment_name="experiment_3",
        n_cases=len(cases),
        records_written=len(checkpoint.load_all()),
        blocked_conditions={},
        call_tracker=tracker,
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
    "run_experiment_4",
    "run_experiment_5",
    "run_experiment_6",
]
