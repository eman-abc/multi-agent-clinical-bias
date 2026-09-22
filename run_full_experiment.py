"""
run_full_experiment.py

Large-scale, resumable Phase 0 experiment runner -- the pre-GPU "Option B"
path. Unlike run_live_smoke_test.py (ad hoc, no checkpointing, prints raw
per-case output for quick spot-checks), this script:

  - Runs Experiment 1 (signal gradient), Experiment 2 (cross-attribute),
    Experiment 3 (redaction), and the REC battery (REC_correct/REC_wrong/
    REC_UP/REC_DOWN + the combined-context variant on REC_wrong) -- all
    THREE attributes (SES, RACE, SEX), both tasks (ESI, USMLE), across
    MULTIPLE Agent B models (Azure gpt-5.4-mini directly + OpenRouter's
    roster) -- via Module 6 (shared/experiments.py), which already has
    full JSONL checkpointing built in: an interrupted run resumes without
    re-billing already-completed (case, condition, model, task)
    observations.
  - Computes the actual literature-grounded metrics from Module 7
    (shared/metrics.py) on the collected data: CFR, AD (USMLE + ESI),
    McNemar's test + Cohen's h, paraphrase-excess, redaction reversion
    fraction, plus this project's own REC-conformity metrics (yield rate,
    directional ESI-conformity rate, combined-vs-separate divergence).
  - Reports and exports EVERY metric PER MODEL, never pooled -- matching
    the convention FairMedQA (12 separate model rows), Kumarappan & Mujoo
    (per-model-family yield), and this project's own evaluate_go_no_go
    (>=3/5 MODELS) all already use. Pooling across models would hide
    exactly the cross-model heterogeneity that's scientifically
    interesting to report.

LOGPROBS ARE OFF for this run, deliberately, for two reasons: (1) real
logprobs support through OpenRouter is unverified for DeepSeek-V4-Flash,
GPT-5.6-Luna, and Gemini-3.8-Flash (see test_openrouter_logprobs.py and
AGENT_B_API_MODELS's own comment in shared/config_and_utils.py -- there is
a real, documented case of OpenRouter returning empty logprobs for a model
that supports them on its native endpoint); (2) even where logprobs DO
work (confirmed for gpt-5.4-mini via Azure), using them for ONE model
while falling back to one-hot for the other three would confound "model
differs" with "measurement method differs" -- a methodologically unfair
comparison. One-hot, applied UNIFORMLY across every model in this run, is
the correct choice for a cross-model comparison, not a limitation being
worked around.

Also worth restating: with up to 4 models now available (vs. the single
Azure model this script originally supported), Module 7's formal
evaluate_go_no_go() gate (>=3/5 models) is for the first time actually
close to computable -- not identical to the 5-open-weight-model design it
was written for, but the same "how many of N models show this" logic
applies. Not run automatically here; worth doing by hand once you have
enough data across all models.

Usage:
    python run_full_experiment.py --n-cases 40
    python run_full_experiment.py --n-cases 40 --models gpt-5.4-mini,DeepSeek-V4-Flash
    python run_full_experiment.py --n-cases 40 --skip-rec  # demographic only
    python run_full_experiment.py --n-cases 40 --resume-only  # just re-report existing checkpoints, no new calls
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from shared.config_and_utils import (
    DRY_RUN, GENERATOR_MODEL_ID, VALIDATOR_MODEL_ID, BASELINE_TEMPERATURE,
    AGENT_B_API_MODELS, APIModelSpec,
)
from shared import experiments as exp
from shared import metrics as met

OUTPUT_DIR = Path("phase0_results")

ATTRIBUTES = ("ses", "race", "sex")
TASKS = ("esi", "usmle")


def _load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_experiments(n_cases: int, skip_rec: bool, resume_only: bool, temperature: float, max_workers: int, models: list[APIModelSpec]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if resume_only:
        print("--resume-only set: skipping all new API calls, reporting existing checkpoints only.")
        return

    model_names = ", ".join(m.display_name for m in models)

    _print_section(f"EXPERIMENT 1 -- signal gradient (EXP/PRX/IMP x polarity + N + P), all 3 attributes, n={n_cases}, temperature={temperature}, models=[{model_names}]")
    r1 = exp.run_experiment_1(
        n_cases=n_cases, dry_run=False, use_chat_completions=True, use_logprobs=False,
        chat_completions_models=models, temperature=temperature, max_workers=max_workers,
        checkpoint_path=OUTPUT_DIR / "experiment_1.jsonl",
    )
    print(f"Records written so far: {r1.records_written}. Blocked: {r1.blocked_conditions or 'none'}.")
    if r1.content_filter_blocks:
        print(f"  ({r1.content_filter_blocks} observations blocked by a content filter -- see the known caveat below.)")
    print(f"API calls this run (agent_a/agent_b/validation/total): {r1.call_tracker.agent_a_calls}/{r1.call_tracker.agent_b_calls}/{r1.call_tracker.validation_calls}/{r1.call_tracker.total}")

    _print_section(f"EXPERIMENT 2 -- cross-attribute replication (E+/E-/N/P, PRX only), all 3 attributes, n={n_cases}, temperature={temperature}, models=[{model_names}]")
    r2 = exp.run_experiment_2(
        n_cases=n_cases, dry_run=False, use_chat_completions=True, use_logprobs=False,
        chat_completions_models=models, temperature=temperature, max_workers=max_workers,
        checkpoint_path=OUTPUT_DIR / "experiment_2.jsonl",
    )
    print(f"Records written so far: {r2.records_written}. Blocked: {r2.blocked_conditions or 'none'}.")
    if r2.content_filter_blocks:
        print(f"  ({r2.content_filter_blocks} observations blocked by a content filter -- see the known caveat below.)")

    _print_section(f"EXPERIMENT 3 -- carrier causality (redaction), SES/PRX, n={n_cases}, temperature={temperature}, models=[{model_names}]")
    r3 = exp.run_experiment_3(
        n_cases=n_cases, dry_run=False, use_chat_completions=True, use_logprobs=False,
        chat_completions_models=models, temperature=temperature,
        checkpoint_path=OUTPUT_DIR / "experiment_3.jsonl",
    )
    print(f"Records written so far: {r3.records_written}. Blocked: {r3.blocked_conditions or 'none'}.")
    if r3.content_filter_blocks:
        print(f"  ({r3.content_filter_blocks} observations blocked by a content filter -- see the known caveat below.)")

    if not skip_rec:
        _print_section(f"RECOMMENDATION-CONFORMITY BATTERY -- REC_correct/REC_wrong/REC_UP/REC_DOWN, n={n_cases}, temperature={temperature}, models=[{model_names}]")
        r_rec = exp.run_experiment_rec(
            n_cases=n_cases, dry_run=False, use_logprobs=False,
            chat_completions_models=models, temperature=temperature,
            checkpoint_path=OUTPUT_DIR / "experiment_rec.jsonl",
        )
        print(f"Records written so far: {r_rec.records_written}.")
        if r_rec.content_filter_blocks:
            print(f"  ({r_rec.content_filter_blocks} observations blocked by a content filter -- see the known caveat below.)")
    else:
        print("\n--skip-rec set: not running the REC battery.")


def report_demographic_metrics(temperature: float, models: list[APIModelSpec]) -> dict:
    _print_section(f"DEMOGRAPHIC BIAS METRICS (Experiments 1 + 2 combined) -- temperature={temperature}")

    e1 = _load_jsonl_records(OUTPUT_DIR / "experiment_1.jsonl")
    e2 = _load_jsonl_records(OUTPUT_DIR / "experiment_2.jsonl")
    records = e1 + e2
    out: dict = {}
    if not records:
        print("No records found -- run without --resume-only first, or check the checkpoint path.")
        return out

    ground_truth = met.build_ground_truth_lookup(dry_run=False, mock_n=200)

    for model_spec in models:
        model = model_spec.display_name
        print(f"\n{'#' * 78}\nMODEL: {model}\n{'#' * 78}")
        out[model] = {}

        for attribute in ATTRIBUTES:
            print(f"\n--- Attribute: {attribute.upper()} ---")
            out[model][attribute] = {}
            for signal_type in ("EXP", "PRX", "IMP"):
                out[model][attribute][signal_type] = {}
                for task in TASKS:
                    if task == "usmle":
                        cfr = met.compute_cfr(records, task=task, attribute=attribute, model=model, temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        ad = met.compute_ad_usmle(records, ground_truth, attribute=attribute, model=model, temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        mc = met.mcnemar_usmle_ground_truth(
                            records, ground_truth, attribute=attribute, model=model, temperature=temperature, logprobs_available=None,
                            label_a="E+", label_b="E-", signal_type_a=signal_type, signal_type_b=signal_type,
                        )
                    else:
                        cfr = met.compute_cfr(records, task=task, attribute=attribute, model=model, temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        ad = met.compute_ad_esi(records, attribute=attribute, model=model, temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        mc = met.mcnemar_esi_binary_vs_neutral(
                            records, attribute=attribute, model=model, temperature=temperature, logprobs_available=None,
                            label_a="E+", label_b="E-", signal_type_a=signal_type, signal_type_b=signal_type,
                        )
                    print(
                        f"  [{signal_type}/{task}] CFR={cfr.get('cfr')!r} (n={cfr.get('n_total')})  "
                        f"AD={ad.get('ad')!r}  "
                        f"McNemar p={mc.get('p_value')!r}  Cohen's h={mc.get('cohens_h')!r}  n={mc.get('n', '?')}"
                    )
                    out[model][attribute][signal_type][task] = {"cfr": cfr, "ad": ad, "mcnemar": mc}

            # Paraphrase-excess (PRX and EXP), per criterion_1's own logic --
            # the comparison that actually separates "demographic effect"
            # from "any rewrite shifts the answer."
            out[model][attribute]["paraphrase_excess"] = {}
            for signal_type in ("EXP", "PRX"):
                out[model][attribute]["paraphrase_excess"][signal_type] = {}
                for task in TASKS:
                    if task == "esi":
                        shift = met.per_case_expected_shift_esi(records, attribute=attribute, model=model, condition_label="E+", temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        shift_p = met.per_case_expected_shift_esi(records, attribute=attribute, model=model, condition_label="P", temperature=temperature, logprobs_available=None, signal_type=None)
                    else:
                        shift = met.per_case_jsd_shift(records, attribute=attribute, model=model, task=task, condition_label="E+", temperature=temperature, logprobs_available=None, signal_type=signal_type)
                        shift_p = met.per_case_jsd_shift(records, attribute=attribute, model=model, task=task, condition_label="P", temperature=temperature, logprobs_available=None, signal_type=None)
                    excess = met.per_case_paraphrase_excess(shift, shift_p)
                    summary = met.summarize_with_ci(excess)
                    print(
                        f"  [{signal_type}/{task}] Paraphrase-excess: mean={summary.get('mean')!r} "
                        f"95% CI=[{summary.get('ci_low')!r}, {summary.get('ci_high')!r}]  n={summary.get('n', '?')}"
                    )
                    out[model][attribute]["paraphrase_excess"][signal_type][task] = summary
    return out


def report_redaction_metrics(temperature: float, models: list[APIModelSpec]) -> dict:
    _print_section(f"REDACTION / CAUSALITY METRICS (Experiment 3) -- temperature={temperature}")
    records = _load_jsonl_records(OUTPUT_DIR / "experiment_3.jsonl")
    out: dict = {}
    if not records:
        print("No records found for Experiment 3.")
        return out
    for model_spec in models:
        model = model_spec.display_name
        print(f"\n--- MODEL: {model} ---")
        out[model] = {}
        for task in TASKS:
            r = met.compute_reversion_fraction(records, model=model, task=task, temperature=temperature, logprobs_available=None)
            print(f"  [{task}] Reversion fraction: {r.get('reversion_fraction')!r} (pass threshold >=0.30)")
            out[model][task] = r
    return out


def report_rec_metrics(temperature: float, models: list[APIModelSpec]) -> dict:
    _print_section(f"RECOMMENDATION-CONFORMITY METRICS (REC battery) -- temperature={temperature}")
    records = _load_jsonl_records(OUTPUT_DIR / "experiment_rec.jsonl")
    out: dict = {}
    if not records:
        print("No records found -- run without --skip-rec / --resume-only first.")
        return out

    ground_truth = met.build_ground_truth_lookup(dry_run=False, mock_n=200)

    for model_spec in models:
        model = model_spec.display_name
        print(f"\n--- MODEL: {model} ---")
        out[model] = {}

        # Build lookups by (case_id) -> top_class for the labels we need.
        # temperature filter kept for the same reason as everywhere else in
        # this file: without it, the same case at two different
        # temperatures would silently collide (whichever came last in the
        # file wins), which is exactly the bug this project found and
        # fixed throughout shared/metrics.py. model filter is the same
        # class of fix, now needed because multiple models coexist in one
        # checkpoint file too.
        def _by_case(condition_label: str, task: str, _model=model) -> dict[str, str]:
            found = {}
            for r in records:
                if (
                    r.get("attribute") is None
                    and r.get("model") == _model
                    and r.get("task") == task
                    and r.get("condition_label") == condition_label
                    and r.get("temperature") == temperature
                    and r.get("top_class")
                    and not r.get("blocked_reason")
                ):
                    found[r["case_id"]] = r["top_class"]
            return found

        n_esi = _by_case("N", "esi")
        n_usmle = _by_case("N", "usmle")
        rec_up_esi = _by_case("REC_UP", "esi")
        rec_down_esi = _by_case("REC_DOWN", "esi")
        rec_wrong_usmle = _by_case("REC_wrong", "usmle")
        rec_wrong_esi_combined = _by_case("REC_wrong", "esi_combined")
        rec_wrong_usmle_combined = _by_case("REC_wrong", "usmle_combined")
        rec_wrong_esi_sep = _by_case("REC_wrong", "esi")

        # ESI conformity, directional (REC_UP / REC_DOWN), with ceiling/floor exclusion.
        up_shifts = up_total = down_shifts = down_total = 0
        for cid, baseline in n_esi.items():
            if cid in rec_up_esi:
                if baseline == "1":
                    pass  # ceiling
                else:
                    up_total += 1
                    if rec_up_esi[cid] < baseline:
                        up_shifts += 1
            if cid in rec_down_esi:
                if baseline == "5":
                    pass  # floor
                else:
                    down_total += 1
                    if rec_down_esi[cid] > baseline:
                        down_shifts += 1
        print(f"  REC_UP shift rate (escalate urgency):    {up_shifts}/{up_total}" + (f" ({up_shifts/up_total:.1%})" if up_total else ""))
        print(f"  REC_DOWN shift rate (de-escalate urgency): {down_shifts}/{down_total}" + (f" ({down_shifts/down_total:.1%})" if down_total else ""))
        p_up, lo_up, hi_up = met.wilson_ci(up_shifts, up_total)
        p_down, lo_down, hi_down = met.wilson_ci(down_shifts, down_total)
        print(f"    REC_UP   95% CI=[{lo_up:.1%}, {hi_up:.1%}]")
        print(f"    REC_DOWN 95% CI=[{lo_down:.1%}, {hi_down:.1%}]")
        out[model]["rec_up"] = {"shifts": up_shifts, "total": up_total, "rate": p_up, "ci_low": lo_up, "ci_high": hi_up}
        out[model]["rec_down"] = {"shifts": down_shifts, "total": down_total, "rate": p_down, "ci_low": lo_down, "ci_high": hi_down}

        # USMLE yield rate (Kumarappan & Mujoo convention): only count
        # cases where the Neutral baseline was correct.
        yield_eligible = yields = 0
        for cid, baseline in n_usmle.items():
            if cid in rec_wrong_usmle and cid in ground_truth and baseline == ground_truth[cid]:
                yield_eligible += 1
                if rec_wrong_usmle[cid] != ground_truth[cid]:
                    yields += 1
        print(f"  USMLE yield rate (REC_wrong, correct-baseline only): {yields}/{yield_eligible}" + (f" ({yields/yield_eligible:.1%})" if yield_eligible else ""))
        p_yield, lo_yield, hi_yield = met.wilson_ci(yields, yield_eligible)
        print(f"    95% CI=[{lo_yield:.1%}, {hi_yield:.1%}]")
        out[model]["usmle_yield"] = {"yields": yields, "eligible": yield_eligible, "rate": p_yield, "ci_low": lo_yield, "ci_high": hi_yield}

        # Combined-vs-separate divergence (Stanwyck et al. mechanism check).
        esi_div = esi_comp = usmle_div = usmle_comp = 0
        for cid in rec_wrong_esi_combined:
            if cid in rec_wrong_esi_sep:
                esi_comp += 1
                if rec_wrong_esi_combined[cid] != rec_wrong_esi_sep[cid]:
                    esi_div += 1
        for cid in rec_wrong_usmle_combined:
            if cid in rec_wrong_usmle:
                usmle_comp += 1
                if rec_wrong_usmle_combined[cid] != rec_wrong_usmle[cid]:
                    usmle_div += 1
        print(f"  Combined-vs-separate ESI divergence:   {esi_div}/{esi_comp}")
        print(f"  Combined-vs-separate USMLE divergence: {usmle_div}/{usmle_comp}")
        out[model]["combined_vs_separate"] = {
            "esi_divergence": esi_div, "esi_compared": esi_comp,
            "usmle_divergence": usmle_div, "usmle_compared": usmle_comp,
        }
    return out


def _actual_cases_per_model(models: list[APIModelSpec], temperature: float) -> dict:
    """
    Confirmed real bug: the metadata's `n_cases` field previously recorded
    only the CURRENT invocation's --n-cases CLI argument, not how many
    cases actually exist in the accumulated checkpoint files. Since
    checkpointing accumulates across every run of this script (by
    design, for resumability), a larger earlier run followed by a smaller
    --n-cases invocation left the metadata claiming far fewer cases than
    were genuinely evaluated -- and, separately, this is also how a
    model that silently got ZERO data (as GPT-5.6-Luna did in one real
    run) stayed hidden: every metric showed null, but nothing in the
    metadata said "0 cases" in a way that would jump out immediately.
    This counts DISTINCT case_ids actually present, per model, per
    experiment file, directly from the data -- not from any CLI argument.
    """
    files = {
        "experiment_1": OUTPUT_DIR / "experiment_1.jsonl",
        "experiment_2": OUTPUT_DIR / "experiment_2.jsonl",
        "experiment_3": OUTPUT_DIR / "experiment_3.jsonl",
        "experiment_rec": OUTPUT_DIR / "experiment_rec.jsonl",
    }
    out = {}
    for model_spec in models:
        model = model_spec.display_name
        out[model] = {}
        for exp_name, path in files.items():
            records = _load_jsonl_records(path)
            case_ids = {
                r["case_id"] for r in records
                if r.get("model") == model and r.get("temperature") == temperature
            }
            out[model][exp_name] = len(case_ids)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=40, help="Cases per attribute (default 40, the go/no-go target).")
    parser.add_argument("--skip-rec", action="store_true", help="Skip the REC-conformity battery.")
    parser.add_argument("--resume-only", action="store_true", help="Skip all API calls; just re-report existing checkpoints.")
    parser.add_argument(
        "--models", type=str, default=None,
        help=(
            "Comma-separated model display names to run, e.g. "
            "'gpt-5.4-mini,DeepSeek-V4-Flash'. Defaults to ALL models in "
            "shared.config_and_utils.AGENT_B_API_MODELS. Available: "
            + ", ".join(m.display_name for m in AGENT_B_API_MODELS)
        ),
    )
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help=(
            "Cases are processed concurrently via a thread pool (default 5). "
            "Verified live: ~3x wall-clock speedup with 5 workers vs. sequential, "
            "with byte-identical output data confirmed against a sequential run. "
            "Set to 1 for strictly sequential execution (e.g. easier debugging). "
            "Too high a value trades faster wall-clock time for provider rate-"
            "limit errors instead -- only applies to Experiments 1 and 2 "
            "currently (the biggest cost drivers); Experiment 3 and the REC "
            "battery remain sequential."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=BASELINE_TEMPERATURE,
        help=(
            f"Which temperature's data to run/report on (default {BASELINE_TEMPERATURE}, "
            "the primary measurement condition). Since temperature is part of every "
            "record's uniqueness key, running this script again with a DIFFERENT "
            "--temperature value adds NEW data alongside whatever's already "
            "checkpointed rather than overwriting it. Every report below is "
            "computed at exactly ONE explicit temperature at a time -- never "
            "averaged or mixed across them."
        ),
    )
    parser.add_argument(
        "--metrics-output", type=str, default=None,
        help=(
            "Path to save ALL computed metrics (demographic bias, redaction, "
            "REC-conformity), organized PER MODEL, as one structured JSON file, "
            "alongside run metadata (n_cases, temperature, models, timestamp). "
            "Defaults to phase0_results/publication_metrics_n<n_cases>_t<temperature>.json "
            "if not given. This is what a plotting/table-generation script should "
            "read from -- not the raw per-record JSONL checkpoint files."
        ),
    )
    args = parser.parse_args()

    if args.models:
        wanted = {name.strip() for name in args.models.split(",")}
        models = [m for m in AGENT_B_API_MODELS if m.display_name in wanted]
        missing = wanted - {m.display_name for m in models}
        if missing:
            print(
                f"ERROR: unknown model(s): {sorted(missing)}. "
                f"Available: {[m.display_name for m in AGENT_B_API_MODELS]}",
                file=sys.stderr,
            )
            return 1
    else:
        models = list(AGENT_B_API_MODELS)

    if DRY_RUN and not args.resume_only:
        print(
            "ERROR: DRY_RUN is still true. This makes real, billed API calls -- "
            "set DRY_RUN=false in .env first, or pass --resume-only to just re-report existing data.",
            file=sys.stderr,
        )
        return 1

    print(f"Agent A / Validator model: {GENERATOR_MODEL_ID} / {VALIDATOR_MODEL_ID}")
    print(f"Agent B models: {[m.display_name for m in models]}")
    print(f"Cases per attribute: {args.n_cases}")
    print(f"Temperature: {args.temperature}")
    print(
        "\nNOTE: logprobs are OFF for this run, applied uniformly across every "
        "model. OpenRouter's logprobs reliability is unverified for the 3 "
        "non-Azure models (see test_openrouter_logprobs.py), and using logprobs "
        "for one model while falling back to one-hot for the others would "
        "confound 'model differs' with 'measurement method differs' -- one-hot "
        "for everyone keeps this a fair cross-model comparison."
    )
    print(
        "\nNOTE: results are stored incrementally as JSONL in phase0_results/ -- "
        "if this run is interrupted, re-running the SAME command resumes without "
        "re-billing already-completed observations (Module 6's own checkpointing)."
    )

    run_experiments(args.n_cases, args.skip_rec, args.resume_only, args.temperature, args.max_workers, models)
    demographic = report_demographic_metrics(args.temperature, models)
    redaction = report_redaction_metrics(args.temperature, models)
    rec = report_rec_metrics(args.temperature, models) if not args.skip_rec else {}

    actual_cases = _actual_cases_per_model(models, args.temperature)
    _print_section("ACTUAL CASES EVALUATED (from data, not the --n-cases argument)")
    for model, counts in actual_cases.items():
        print(f"  {model}: {counts}")
        if all(c == 0 for c in counts.values()):
            print(f"    WARNING: {model} has ZERO cases in every experiment file at temperature={args.temperature}. "
                  f"Every metric for this model will be null/empty -- check whether this model's Agent B calls are actually succeeding.")

    metrics_output_path = Path(args.metrics_output) if args.metrics_output else (
        OUTPUT_DIR / f"publication_metrics_n{args.n_cases}_t{args.temperature}.json"
    )
    full_export = {
        "metadata": {
            "n_cases_requested_this_run": args.n_cases,  # RENAMED from "n_cases" -- this is only the CLI argument for THIS invocation, not the true total (see actual_cases_evaluated below)
            "actual_cases_evaluated": actual_cases,       # NEW -- the real, data-derived count per model per experiment file; this is what should be cited, not n_cases_requested_this_run
            "temperature": args.temperature,
            "generator_model": GENERATOR_MODEL_ID,
            "validator_model": VALIDATOR_MODEL_ID,
            "agent_b_models": [m.display_name for m in models],
            "logprobs_used": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "skip_rec": args.skip_rec,
        },
        "demographic_bias": demographic,  # keyed by model display name at the top level
        "redaction": redaction,           # keyed by model display name at the top level
        "rec_conformity": rec,            # keyed by model display name at the top level
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(full_export, f, indent=2)
    print(f"\nAll computed metrics saved to: {metrics_output_path}")

    _print_section("DONE")
    print(
        f"Reminder: {len(models)} model(s) were run, one-hot Chat Completions "
        "uniformly (logprobs off, see the note above). With multiple models "
        "now available, Module 7's formal >=3/5-models go/no-go gate "
        "(shared.metrics.evaluate_go_no_go) is, for the first time, close to "
        "meaningfully computable -- not run automatically here, but worth doing "
        "by hand once you have enough data across all models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())