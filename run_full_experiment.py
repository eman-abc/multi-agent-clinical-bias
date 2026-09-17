"""
run_full_experiment.py

Large-scale, resumable Phase 0 experiment runner -- the pre-GPU "Option B"
path. Unlike run_live_smoke_test.py (ad hoc, no checkpointing, prints raw
per-case output for quick spot-checks), this script:

  - Runs Experiment 1 (signal gradient), Experiment 2 (cross-attribute),
    Experiment 3 (redaction), and the REC battery (REC_correct/REC_wrong/
    REC_UP/REC_DOWN + the combined-context variant on REC_wrong) -- all
    THREE attributes (SES, RACE, SEX), both tasks (ESI, USMLE) -- via
    Module 6 (shared/experiments.py), which already has full JSONL
    checkpointing built in: an interrupted run resumes without re-billing
    already-completed (case, condition, model, task) observations.
  - Computes the actual literature-grounded metrics from Module 7
    (shared/metrics.py) on the collected data: CFR, AD (USMLE + ESI),
    McNemar's test + Cohen's h, paraphrase-excess, redaction reversion
    fraction, plus this project's own REC-conformity metrics (yield rate,
    directional ESI-conformity rate, combined-vs-separate divergence).

IMPORTANT CAVEAT, stated plainly rather than glossed over: Module 7's
evaluate_go_no_go() requires >=3 of 5 Agent B models to pass each
criterion (phase_0_complete_v3.md's own §14 design, written for the
5-model open-weight sweep). Chat-Completions only gives you ONE model
(gpt-5.4-mini by default) -- 3/5 is mathematically impossible to satisfy
with n=1 model, regardless of how strong or significant the real effect
is. This script does NOT run the literal go/no-go gate for that reason --
running it would produce a misleading permanent "NULL" no matter what the
data shows. Instead, it reports every underlying metric directly (CFR, AD,
p-values, effect sizes), which ARE meaningful with one model, and leaves
the formal >=3/5 go/no-go evaluation for after GPU/open-weight access lets
Module 6's real logprobs-based path run across all five models.

This also does NOT use logprobs-based measurement (run_esi_triage /
run_usmle_diagnostic) -- those need the legacy Completions API, which
GPT-4o/4.1/5-family models don't support (confirmed earlier in this
project). Every result here comes from the Chat-Completions fallback
(one-hot text parsing) -- a real, usable measurement, just without the
full probability distribution logprobs would give.

Usage:
    python run_full_experiment.py --n-cases 40
    python run_full_experiment.py --n-cases 40 --skip-rec  # demographic only
    python run_full_experiment.py --n-cases 40 --resume-only  # just re-report existing checkpoints, no new calls
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from shared.config_and_utils import DRY_RUN, GENERATOR_MODEL_ID, VALIDATOR_MODEL_ID, BASELINE_TEMPERATURE
from shared import experiments as exp
from shared import metrics as met

CHAT_COMPLETIONS_MODEL = "gpt-5.4-mini"
OUTPUT_DIR = Path("phase0_results")

ATTRIBUTES = ("ses", "race", "sex")
TASKS = ("esi", "usmle")


def _load_jsonl_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    import json

    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_experiments(n_cases: int, skip_rec: bool, resume_only: bool, temperature: float, max_workers: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if resume_only:
        print("--resume-only set: skipping all new API calls, reporting existing checkpoints only.")
        return

    _print_section(f"EXPERIMENT 1 -- signal gradient (EXP/PRX/IMP x polarity + N + P), all 3 attributes, n={n_cases}, temperature={temperature}, max_workers={max_workers}")
    r1 = exp.run_experiment_1(
        n_cases=n_cases, dry_run=False, use_chat_completions=True,
        chat_completions_model=CHAT_COMPLETIONS_MODEL, temperature=temperature, max_workers=max_workers,
        checkpoint_path=OUTPUT_DIR / "experiment_1.jsonl",
    )
    print(f"Records written so far: {r1.records_written}. Blocked: {r1.blocked_conditions or 'none'}.")
    if r1.content_filter_blocks:
        print(f"  ({r1.content_filter_blocks} observations blocked by Azure's content filter -- see the known caveat below.)")
    print(f"API calls this run (agent_a/agent_b/validation/total): {r1.call_tracker.agent_a_calls}/{r1.call_tracker.agent_b_calls}/{r1.call_tracker.validation_calls}/{r1.call_tracker.total}")

    _print_section(f"EXPERIMENT 2 -- cross-attribute replication (E+/E-/N/P, PRX only), all 3 attributes, n={n_cases}, temperature={temperature}, max_workers={max_workers}")
    r2 = exp.run_experiment_2(
        n_cases=n_cases, dry_run=False, use_chat_completions=True,
        chat_completions_model=CHAT_COMPLETIONS_MODEL, temperature=temperature, max_workers=max_workers,
        checkpoint_path=OUTPUT_DIR / "experiment_2.jsonl",
    )
    print(f"Records written so far: {r2.records_written}. Blocked: {r2.blocked_conditions or 'none'}.")
    if r2.content_filter_blocks:
        print(f"  ({r2.content_filter_blocks} observations blocked by Azure's content filter -- see the known caveat below.)")

    _print_section(f"EXPERIMENT 3 -- carrier causality (redaction), SES/PRX, n={n_cases}, temperature={temperature}")
    r3 = exp.run_experiment_3(
        n_cases=n_cases, dry_run=False, use_chat_completions=True,
        chat_completions_model=CHAT_COMPLETIONS_MODEL, temperature=temperature,
        checkpoint_path=OUTPUT_DIR / "experiment_3.jsonl",
    )
    print(f"Records written so far: {r3.records_written}. Blocked: {r3.blocked_conditions or 'none'}.")
    if r3.content_filter_blocks:
        print(f"  ({r3.content_filter_blocks} observations blocked by Azure's content filter -- see the known caveat below.)")

    if not skip_rec:
        _print_section(f"RECOMMENDATION-CONFORMITY BATTERY -- REC_correct/REC_wrong/REC_UP/REC_DOWN, n={n_cases}, temperature={temperature}")
        r_rec = exp.run_experiment_rec(
            n_cases=n_cases, dry_run=False,
            chat_completions_model=CHAT_COMPLETIONS_MODEL, temperature=temperature,
            checkpoint_path=OUTPUT_DIR / "experiment_rec.jsonl",
        )
        print(f"Records written so far: {r_rec.records_written}.")
        if r_rec.content_filter_blocks:
            print(f"  ({r_rec.content_filter_blocks} observations blocked by Azure's content filter -- see the known caveat below.)")
    else:
        print("\n--skip-rec set: not running the REC battery.")


def report_demographic_metrics(temperature: float) -> dict:
    _print_section(f"DEMOGRAPHIC BIAS METRICS (Experiments 1 + 2 combined) -- temperature={temperature}")

    e1 = _load_jsonl_records(OUTPUT_DIR / "experiment_1.jsonl")
    e2 = _load_jsonl_records(OUTPUT_DIR / "experiment_2.jsonl")
    records = e1 + e2
    out: dict = {}
    if not records:
        print("No records found -- run without --resume-only first, or check the checkpoint path.")
        return out

    ground_truth = met.build_ground_truth_lookup(dry_run=False, mock_n=200)
    model = CHAT_COMPLETIONS_MODEL

    for attribute in ATTRIBUTES:
        print(f"\n--- Attribute: {attribute.upper()} ---")
        out[attribute] = {}
        for signal_type in ("EXP", "PRX", "IMP"):
            out[attribute][signal_type] = {}
            for task in TASKS:
                if task == "usmle":
                    cfr = met.compute_cfr(records, task=task, attribute=attribute, model=model, temperature=temperature, signal_type=signal_type)
                    ad = met.compute_ad_usmle(records, ground_truth, attribute=attribute, model=model, temperature=temperature, signal_type=signal_type)
                    mc = met.mcnemar_usmle_ground_truth(
                        records, ground_truth, attribute=attribute, model=model, temperature=temperature,
                        label_a="E+", label_b="E-", signal_type_a=signal_type, signal_type_b=signal_type,
                    )
                else:
                    cfr = met.compute_cfr(records, task=task, attribute=attribute, model=model, temperature=temperature, signal_type=signal_type)
                    ad = met.compute_ad_esi(records, attribute=attribute, model=model, temperature=temperature, signal_type=signal_type)
                    mc = met.mcnemar_esi_binary_vs_neutral(
                        records, attribute=attribute, model=model, temperature=temperature,
                        label_a="E+", label_b="E-", signal_type_a=signal_type, signal_type_b=signal_type,
                    )
                print(
                    f"  [{signal_type}/{task}] CFR={cfr.get('cfr')!r} (n={cfr.get('n_total')})  "
                    f"AD={ad.get('ad')!r}  "
                    f"McNemar p={mc.get('p_value')!r}  Cohen's h={mc.get('cohens_h')!r}  n={mc.get('n', '?')}"
                )
                out[attribute][signal_type][task] = {"cfr": cfr, "ad": ad, "mcnemar": mc}

        # Paraphrase-excess (PRX and EXP), per criterion_1's own logic -- the
        # comparison that actually separates "demographic effect" from
        # "any rewrite shifts the answer."
        out[attribute]["paraphrase_excess"] = {}
        for signal_type in ("EXP", "PRX"):
            out[attribute]["paraphrase_excess"][signal_type] = {}
            for task in TASKS:
                if task == "esi":
                    shift = met.per_case_expected_shift_esi(records, attribute=attribute, model=model, condition_label="E+", temperature=temperature, signal_type=signal_type)
                    shift_p = met.per_case_expected_shift_esi(records, attribute=attribute, model=model, condition_label="P", temperature=temperature, signal_type=None)
                else:
                    shift = met.per_case_jsd_shift(records, attribute=attribute, model=model, task=task, condition_label="E+", temperature=temperature, signal_type=signal_type)
                    shift_p = met.per_case_jsd_shift(records, attribute=attribute, model=model, task=task, condition_label="P", temperature=temperature, signal_type=None)
                excess = met.per_case_paraphrase_excess(shift, shift_p)
                summary = met.summarize_with_ci(excess)
                print(
                    f"  [{signal_type}/{task}] Paraphrase-excess: mean={summary.get('mean')!r} "
                    f"95% CI=[{summary.get('ci_low')!r}, {summary.get('ci_high')!r}]  n={summary.get('n', '?')}"
                )
                out[attribute]["paraphrase_excess"][signal_type][task] = summary
    return out


def report_redaction_metrics(temperature: float) -> dict:
    _print_section(f"REDACTION / CAUSALITY METRICS (Experiment 3) -- temperature={temperature}")
    records = _load_jsonl_records(OUTPUT_DIR / "experiment_3.jsonl")
    out: dict = {}
    if not records:
        print("No records found for Experiment 3.")
        return out
    for task in TASKS:
        r = met.compute_reversion_fraction(records, model=CHAT_COMPLETIONS_MODEL, task=task, temperature=temperature)
        print(f"  [{task}] Reversion fraction: {r.get('reversion_fraction')!r} (pass threshold >=0.30)")
        out[task] = r
    return out


def report_rec_metrics(temperature: float) -> dict:
    _print_section(f"RECOMMENDATION-CONFORMITY METRICS (REC battery) -- temperature={temperature}")
    records = _load_jsonl_records(OUTPUT_DIR / "experiment_rec.jsonl")
    out: dict = {}
    if not records:
        print("No records found -- run without --skip-rec / --resume-only first.")
        return out

    ground_truth = met.build_ground_truth_lookup(dry_run=False, mock_n=200)
    model = CHAT_COMPLETIONS_MODEL

    # Build lookups by (case_id) -> top_class for the labels we need.
    # temperature filter added here for the same reason as everywhere else
    # in this file: without it, the same case at two different
    # temperatures would silently collide (whichever came last in the
    # file wins), which is exactly the bug this project found and fixed
    # throughout shared/metrics.py.
    def _by_case(condition_label: str, task: str) -> dict[str, str]:
        out = {}
        for r in records:
            if (
                r.get("attribute") is None
                and r.get("model") == model
                and r.get("task") == task
                and r.get("condition_label") == condition_label
                and r.get("temperature") == temperature
                and r.get("top_class")
                and not r.get("blocked_reason")
            ):
                out[r["case_id"]] = r["top_class"]
        return out

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
    out["rec_up"] = {"shifts": up_shifts, "total": up_total, "rate": p_up, "ci_low": lo_up, "ci_high": hi_up}
    out["rec_down"] = {"shifts": down_shifts, "total": down_total, "rate": p_down, "ci_low": lo_down, "ci_high": hi_down}

    # USMLE yield rate (Kumarappan & Mujoo convention): only count cases
    # where the Neutral baseline was correct.
    yield_eligible = yields = 0
    for cid, baseline in n_usmle.items():
        if cid in rec_wrong_usmle and cid in ground_truth and baseline == ground_truth[cid]:
            yield_eligible += 1
            if rec_wrong_usmle[cid] != ground_truth[cid]:
                yields += 1
    print(f"  USMLE yield rate (REC_wrong, correct-baseline only): {yields}/{yield_eligible}" + (f" ({yields/yield_eligible:.1%})" if yield_eligible else ""))
    p_yield, lo_yield, hi_yield = met.wilson_ci(yields, yield_eligible)
    print(f"    95% CI=[{lo_yield:.1%}, {hi_yield:.1%}]")
    out["usmle_yield"] = {"yields": yields, "eligible": yield_eligible, "rate": p_yield, "ci_low": lo_yield, "ci_high": hi_yield}

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
    out["combined_vs_separate"] = {
        "esi_divergence": esi_div, "esi_compared": esi_comp,
        "usmle_divergence": usmle_div, "usmle_compared": usmle_comp,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, default=40, help="Cases per attribute (default 40, the go/no-go target).")
    parser.add_argument("--skip-rec", action="store_true", help="Skip the REC-conformity battery.")
    parser.add_argument("--resume-only", action="store_true", help="Skip all API calls; just re-report existing checkpoints.")
    parser.add_argument(
        "--max-workers", type=int, default=5,
        help=(
            "Cases are processed concurrently via a thread pool (default 5). "
            "Verified live: ~3x wall-clock speedup with 5 workers vs. sequential, "
            "with byte-identical output data confirmed against a sequential run. "
            "Set to 1 for strictly sequential execution (e.g. easier debugging). "
            "Too high a value trades faster wall-clock time for Azure/OpenAI "
            "rate-limit (429) errors instead -- only applies to Experiments 1 and "
            "2 currently (the biggest cost drivers); Experiment 3 and the REC "
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
            "checkpointed rather than overwriting it -- so 0.0 (primary), 0.2 "
            "(Experiment 5's point), and 0.8 (this project's old baseline, now a "
            "secondary Okawa-style comparison) can all coexist in the same files. "
            "Every report below is computed at exactly ONE explicit temperature at "
            "a time -- never averaged or mixed across them."
        ),
    )
    parser.add_argument(
        "--metrics-output", type=str, default=None,
        help=(
            "Path to save ALL computed metrics (demographic bias, redaction, "
            "REC-conformity) as one structured JSON file, alongside run metadata "
            "(n_cases, temperature, model, timestamp). Defaults to "
            "phase0_results/publication_metrics_n<n_cases>_t<temperature>.json "
            "if not given. This is what a plotting/table-generation script should "
            "read from -- not the raw per-record JSONL checkpoint files."
        ),
    )
    args = parser.parse_args()

    if DRY_RUN and not args.resume_only:
        print(
            "ERROR: DRY_RUN is still true. This makes real, billed API calls -- "
            "set DRY_RUN=false in .env first, or pass --resume-only to just re-report existing data.",
            file=sys.stderr,
        )
        return 1

    print(f"Agent A / Validator model: {GENERATOR_MODEL_ID} / {VALIDATOR_MODEL_ID}")
    print(f"Agent B model (Chat-Completions): {CHAT_COMPLETIONS_MODEL}")
    print(f"Cases per attribute: {args.n_cases}")
    print(f"Temperature: {args.temperature}")
    print(
        "\nNOTE: results are stored incrementally as JSONL in phase0_results/ -- "
        "if this run is interrupted, re-running the SAME command resumes without "
        "re-billing already-completed observations (Module 6's own checkpointing). "
        "Running again with a DIFFERENT --temperature adds new data alongside "
        "what's already there rather than overwriting it -- every metric below is "
        "computed at exactly ONE explicit temperature, never mixed."
    )

    run_experiments(args.n_cases, args.skip_rec, args.resume_only, args.temperature, args.max_workers)
    demographic = report_demographic_metrics(args.temperature)
    redaction = report_redaction_metrics(args.temperature)
    rec = report_rec_metrics(args.temperature) if not args.skip_rec else {}

    metrics_output_path = Path(args.metrics_output) if args.metrics_output else (
        OUTPUT_DIR / f"publication_metrics_n{args.n_cases}_t{args.temperature}.json"
    )
    full_export = {
        "metadata": {
            "n_cases": args.n_cases,
            "temperature": args.temperature,
            "generator_model": GENERATOR_MODEL_ID,
            "validator_model": VALIDATOR_MODEL_ID,
            "agent_b_model": CHAT_COMPLETIONS_MODEL,
            "logprobs_used": True,  # experiments.py's use_logprobs defaults to True; this script never overrides it
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "skip_rec": args.skip_rec,
        },
        "demographic_bias": demographic,
        "redaction": redaction,
        "rec_conformity": rec,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(full_export, f, indent=2)
    print(f"\nAll computed metrics saved to: {metrics_output_path}")

    _print_section("DONE")
    print(
        "Reminder: Agent B measurement uses real logprobs by default (verified "
        "live: genuine multi-class distributions on realistic clinical "
        "vignettes, not one-hot text parsing) via the Chat-Completions API -- "
        "this is a real measurement, but still a single model, not the 5-model "
        "open-weight sweep Module 6/7 were originally spec'd for. The formal "
        ">=3/5-model go/no-go gate isn't run here -- it can't be meaningfully "
        "satisfied with a single model. Re-run the real go/no-go evaluation "
        "(shared.metrics.evaluate_go_no_go) once GPU/open-weight access lets "
        "the five real Agent B models run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())