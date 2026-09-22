"""
analyze_multi_model_sycophancy.py

Comprehensive metrics + analysis across MULTIPLE conditional-sycophancy
data files (e.g. Gemini and DeepSeek, run separately, each interrupted
partway through before reaching the full intended n). Built specifically
for the case where you have partial, uneven data and need to know exactly
how much of it is actually usable -- not just re-run the single-model
analysis twice and eyeball the difference.

Does three things existing tooling doesn't:

1. COVERAGE REPORT (per model) -- this matters more than it looks. Jobs
   are submitted as ONE FLAT POOL across all cases and conditions (not
   case-by-case sequentially), so an interruption partway through a run
   typically leaves MOST cases PARTIALLY complete, not a clean "first N
   cases done, the rest untouched." A case missing even one condition
   can't be used in a paired comparison (McNemar needs both E+ and E- for
   the SAME case; the binary-yield metric needs the N baseline present).
   This report tells you, per model: how many rows exist, how many UNIQUE
   cases have any data at all, how many have a usable N baseline, and how
   many have the FULL condition sweep -- so "7000 rows" turns into an
   honest "here's how many are actually pairable."

2. VALIDATION QUALITY REPORT (per model) -- validation_passed rate, which
   specific check (recommendation / carrier / facts) failed most often
   when it did fail, and the distribution of how many regeneration
   attempts it took. Only meaningful for data collected with the
   validation-enabled version of run_full_conditional_sycophancy.py;
   reported as "not available" otherwise rather than silently skipped.

3. CROSS-MODEL COMPARISON -- the one thing genuinely missing from
   analyze_conditional_sycophancy.py, which only ever looks at one model
   at a time. Puts Gemini's and DeepSeek's signal-type findings side by
   side, both as a table and a grouped figure, and flags any signal-type/
   rec-type cell where the two models' confidence intervals don't overlap
   at all -- the closest a non-overlapping-CI heuristic can get to "these
   two models behave differently here" without a formal two-sample test.

Everything from analyze_conditional_sycophancy.py's own per-model
analysis (binary yield, paired McNemar, signal-type-pooled shifts, all
existing figures) is reused directly by importing that module -- nothing
here duplicates that computation.

Usage:
    python analyze_multi_model_sycophancy.py \
        --file Gemini-3.8-Flash=phase0_results/experiment_conditional_sycophancy_full_Gemini-3.8-Flash.jsonl \
        --file DeepSeek-V4-Flash=phase0_results/experiment_conditional_sycophancy_full_DeepSeek-V4-Flash.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_conditional_sycophancy as single


# ---------------------------------------------------------------------------
# Expected job shape -- must match run_full_conditional_sycophancy.py's own
# job-building loop exactly, or the coverage report would silently lie.
# 3 attributes x 2 polarities x 3 signal types x 4 (task,rec_type) pairs
# = 72 demographic-sweep jobs, plus 4 baseline jobs (REC_UP/DOWN/CORRECT/
# WRONG at N) = 76 jobs per case, per model.
# ---------------------------------------------------------------------------
ATTRIBUTES = ("ses", "race", "sex")
POLARITIES = ("E+", "E-")
SIGNAL_TYPES = ("EXP", "PRX", "IMP")
DEMOGRAPHIC_REC_TASK_PAIRS = (("REC_UP", "esi"), ("REC_DOWN", "esi"), ("REC_CORRECT", "usmle"), ("REC_WRONG", "usmle"))
BASELINE_REC_TASK_PAIRS = (("REC_UP", "esi"), ("REC_DOWN", "esi"), ("REC_CORRECT", "usmle"), ("REC_WRONG", "usmle"))
JOBS_PER_CASE = len(BASELINE_REC_TASK_PAIRS) + len(ATTRIBUTES) * len(POLARITIES) * len(SIGNAL_TYPES) * len(DEMOGRAPHIC_REC_TASK_PAIRS)  # 4 + 72 = 76


def compute_coverage_report(records: list[dict]) -> dict:
    by_case = defaultdict(set)  # case_id -> set of (task, attribute, condition_label, signal_type, rec_type) tuples present
    for r in records:
        by_case[r["case_id"]].add((r["task"], r["attribute"], r["condition_label"], r["signal_type"], r["rec_type"]))

    n_cases_any_data = len(by_case)
    n_cases_full = sum(1 for combos in by_case.values() if len(combos) == JOBS_PER_CASE)
    n_cases_has_n_baseline = sum(
        1 for combos in by_case.values()
        if any(c[2] == "N" for c in combos)
    )
    completeness = sorted(len(combos) / JOBS_PER_CASE for combos in by_case.values())

    return {
        "total_rows": len(records),
        "unique_cases": n_cases_any_data,
        "cases_with_n_baseline": n_cases_has_n_baseline,
        "cases_fully_complete": n_cases_full,
        "cases_partial": n_cases_any_data - n_cases_full,
        "completeness_fraction_min": completeness[0] if completeness else None,
        "completeness_fraction_median": completeness[len(completeness) // 2] if completeness else None,
        "completeness_fraction_max": completeness[-1] if completeness else None,
        "jobs_per_case_expected": JOBS_PER_CASE,
    }


def compute_validation_quality_report(records: list[dict]) -> dict:
    have_validation_field = any("validation_passed" in r for r in records)
    if not have_validation_field:
        return {"available": False}

    n = len(records)
    n_passed = sum(1 for r in records if r.get("validation_passed") is True)
    n_failed = sum(1 for r in records if r.get("validation_passed") is False)
    n_unknown = n - n_passed - n_failed

    # NEW: track whether per-check detail and attempt counts were actually
    # COLLECTED in this file, not just whether they happen to show zero --
    # confirmed real gap: run_full_conditional_sycophancy.py changed
    # schema partway through this project (validation_details/
    # validation_attempts/note_text/agent_b_raw_text were added later), so
    # an older file (validation_passed only) and a newer file (full
    # detail) both silently report "failure breakdown: {}" and "attempts:
    # {}" for entirely different reasons -- one because nothing failed,
    # the other because the field was never recorded. Comparing two
    # models' "check_failure_counts" side by side is misleading unless
    # both files even HAVE that data to compare.
    has_check_detail = any(r.get("validation_details") is not None for r in records)
    has_attempts_data = any(r.get("validation_attempts") is not None for r in records)
    has_note_text = any(r.get("note_text") is not None for r in records)
    has_raw_text = any(r.get("agent_b_raw_text") is not None for r in records)

    check_failures = Counter()
    for r in records:
        details = r.get("validation_details")
        if not details:
            continue
        for check in ("rec_ok", "carrier_ok", "facts_ok"):
            if details.get(check) is False:
                check_failures[check] += 1

    attempts = [r["validation_attempts"] for r in records if r.get("validation_attempts") is not None]
    attempts_hist = Counter(attempts)

    return {
        "available": True,
        "n_records": n,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_unknown_no_validation_data": n_unknown,
        "pass_rate": n_passed / n if n else None,
        "has_check_level_detail": has_check_detail,   # NEW -- False means check_failure_counts below is not meaningful, not "zero failures"
        "check_failure_counts": dict(check_failures) if has_check_detail else None,
        "has_attempts_data": has_attempts_data,        # NEW -- same distinction for attempts_histogram
        "attempts_histogram": dict(sorted(attempts_hist.items())) if has_attempts_data else None,
        "has_note_text": has_note_text,                # NEW -- whether the actual generated note is recoverable from this file at all
        "has_agent_b_raw_text": has_raw_text,
    }


def analyze_one_model(label: str, path: Path, output_dir: Path) -> dict:
    records = single.load_records(path)
    detected_model = single.detect_model(records)
    print(f"\n{'='*88}\n{label}  ({path})\n{'='*88}")
    print(f"Detected model field in data: {detected_model!r}")
    if detected_model != label and detected_model != "multiple_models":
        print(f"NOTE: you labeled this {label!r} but the data itself says {detected_model!r} -- using your label for output naming, the data's own value for filtering.")

    coverage = compute_coverage_report(records)
    print(f"\nCoverage: {coverage['total_rows']} rows, {coverage['unique_cases']} unique cases "
          f"({coverage['cases_fully_complete']} fully complete, {coverage['cases_partial']} partial), "
          f"completeness range [{coverage['completeness_fraction_min']:.0%}, {coverage['completeness_fraction_max']:.0%}], "
          f"median {coverage['completeness_fraction_median']:.0%}")

    validation = compute_validation_quality_report(records)
    if validation["available"]:
        detail_note = "" if validation["has_check_level_detail"] else "  (no per-check detail recorded in this file)"
        print(f"Validation: {validation['pass_rate']:.1%} passed first-class ({validation['n_passed']}/{validation['n_records']}), "
              f"failure breakdown: {validation['check_failure_counts']}{detail_note}")
        if not validation["has_note_text"]:
            print("  NOTE: this file has no note_text field -- the actual generated notes are not recoverable from it (collected before that field was added).")
    else:
        print("Validation: no validation_passed field in this data (collected before validation was added).")

    baselines, baseline_probs = single.build_baselines(records)
    binary, per_case_yield = single.compute_binary_yield(records, baselines)
    mcnemar_results = single.compute_paired_mcnemar(per_case_yield)
    signal_pooled = single.compute_signal_type_pooled(records, baseline_probs)
    by_attr_raw = single.collect_signal_type_by_attribute_raw(records, baseline_probs)
    pooled_raw = single.collect_signal_type_pooled_raw(records, baseline_probs)

    n_significant = sum(1 for r in mcnemar_results if r["p_value"] < 0.05)
    print(f"Paired McNemar (demographic polarity): {n_significant}/{len(mcnemar_results)} significant at p<0.05")

    model_dir = output_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    single.plot_signal_type_finding(signal_pooled, model_dir)
    single.plot_usmle_signal_type_finding(signal_pooled, model_dir)
    single.plot_signal_type_by_attribute(by_attr_raw, model_dir)
    single.plot_shift_heatmap(by_attr_raw, model_dir)
    single.plot_raw_distribution_strip(pooled_raw, model_dir)
    single.plot_mcnemar_overview(mcnemar_results, model_dir)
    single.plot_binary_yield_by_signal(binary, model_dir)

    return {
        "coverage": coverage,
        "validation": validation,
        "binary_yield": binary,
        "paired_mcnemar_polarity": mcnemar_results,
        "signal_type_pooled": signal_pooled,
    }


def plot_cross_model_esi_comparison(pooled_by_model: dict, output_dir: Path) -> None:
    """Gemini vs DeepSeek, same axes, ESI signal-type shift."""
    labels = list(pooled_by_model.keys())
    signal_types = ["EXP", "PRX", "IMP"]
    rec_types = ["REC_UP", "REC_DOWN"]
    fig, axes = plt.subplots(1, len(rec_types), figsize=(6 * len(rec_types), 5), sharey=True)
    palette = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]

    for ax, rec_type in zip(axes, rec_types):
        x = np.arange(len(signal_types))
        width = 0.8 / max(len(labels), 1)
        for i, label in enumerate(labels):
            means, err_lo, err_hi = [], [], []
            for sig in signal_types:
                entry = pooled_by_model[label].get("esi", {}).get(sig, {}).get(rec_type)
                if entry and entry.get("mean") is not None:
                    means.append(entry["mean"]); err_lo.append(entry["mean"] - entry["ci_low"]); err_hi.append(entry["ci_high"] - entry["mean"])
                else:
                    means.append(0); err_lo.append(0); err_hi.append(0)
            offset = (i - (len(labels) - 1) / 2) * width
            ax.bar(x + offset, means, width=width * 0.9, yerr=[err_lo, err_hi], capsize=3, label=label, color=palette[i % len(palette)], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(signal_types)
        ax.set_title(rec_type)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Mean shift in expected ESI value vs. N baseline")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-model comparison (ESI): does the signal-type pattern replicate across models?")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"cross_model_esi_comparison.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved cross_model_esi_comparison.png / .pdf ({len(labels)} models)")


def plot_cross_model_usmle_comparison(pooled_by_model: dict, output_dir: Path) -> None:
    """
    Gemini vs DeepSeek, same axes, USMLE belief-in-recommended-answer.
    NEW -- this is the one nothing in this script surfaced before: the
    biggest divergence found in this project's real data (Gemini ~85%
    belief in a stated WRONG diagnosis vs. DeepSeek ~18%, for EXP) lives
    entirely in USMLE/REC_WRONG, which the ESI-only comparison above
    structurally cannot show. REC_CORRECT is plotted alongside it as the
    ceiling/sanity check (both models should sit near 1.0 there), exactly
    matching the single-model USMLE figure's own convention.
    """
    labels = list(pooled_by_model.keys())
    signal_types = ["EXP", "PRX", "IMP"]
    rec_types = ["REC_CORRECT", "REC_WRONG"]
    fig, axes = plt.subplots(1, len(rec_types), figsize=(6 * len(rec_types), 5), sharey=True)
    palette = ["#2980b9", "#c0392b", "#27ae60", "#8e44ad"]

    for ax, rec_type in zip(axes, rec_types):
        x = np.arange(len(signal_types))
        width = 0.8 / max(len(labels), 1)
        for i, label in enumerate(labels):
            means, err_lo, err_hi = [], [], []
            for sig in signal_types:
                entry = pooled_by_model[label].get("usmle", {}).get(sig, {}).get(rec_type)
                if entry and entry.get("mean") is not None:
                    means.append(entry["mean"]); err_lo.append(entry["mean"] - entry["ci_low"]); err_hi.append(entry["ci_high"] - entry["mean"])
                else:
                    means.append(0); err_lo.append(0); err_hi.append(0)
            offset = (i - (len(labels) - 1) / 2) * width
            ax.bar(x + offset, means, width=width * 0.9, yerr=[err_lo, err_hi], capsize=3, label=label, color=palette[i % len(palette)], alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(signal_types)
        ax.set_title(rec_type)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Mean P(target letter) -- belief in the recommended answer")
    axes[0].legend(fontsize=8)
    fig.suptitle("Cross-model comparison (USMLE): does belief in a stated recommendation replicate across models?")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"cross_model_usmle_comparison.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved cross_model_usmle_comparison.png / .pdf ({len(labels)} models)")


def compare_models_table(pooled_by_model: dict) -> list[dict]:
    """
    Flags signal-type/rec-type cells where two models' CIs don't overlap
    at all -- the closest a CI heuristic gets to 'these differ' without a
    formal test. NOW COVERS BOTH TASKS -- previously ESI only, which
    meant the single largest divergence in this project's real data
    (USMLE/REC_WRONG, Gemini vs DeepSeek) never appeared in this table at
    all, only in the raw per-model JSON a person would have to
    cross-reference by hand.
    """
    labels = list(pooled_by_model.keys())
    rows = []
    for task, rec_types in (("esi", ("REC_UP", "REC_DOWN")), ("usmle", ("REC_CORRECT", "REC_WRONG"))):
        for sig in ("EXP", "PRX", "IMP"):
            for rec_type in rec_types:
                cell = {"task": task, "signal_type": sig, "rec_type": rec_type}
                entries = {}
                for label in labels:
                    e = pooled_by_model[label].get(task, {}).get(sig, {}).get(rec_type)
                    entries[label] = e
                    cell[label] = e
                if len(labels) == 2 and all(entries.values()):
                    a, b = entries[labels[0]], entries[labels[1]]
                    non_overlapping = a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]
                    cell["cis_non_overlapping"] = non_overlapping
                rows.append(cell)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action="append", required=True, metavar="LABEL=PATH",
                         help="A model's data file as LABEL=PATH, e.g. --file DeepSeek-V4-Flash=phase0_results/experiment_conditional_sycophancy_full_DeepSeek-V4-Flash.jsonl. Repeat for each model.")
    parser.add_argument("--output-dir", type=str, default="figures/cross_model")
    parser.add_argument("--metrics-output", type=str, default="phase0_results/multi_model_sycophancy_metrics.json")
    args = parser.parse_args()

    files = {}
    for spec in args.file:
        if "=" not in spec:
            raise SystemExit(f"--file must be LABEL=PATH, got: {spec!r}")
        label, path = spec.split("=", 1)
        files[label] = Path(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_model_results = {}
    pooled_by_model = {}
    for label, path in files.items():
        if not path.exists():
            print(f"WARNING: {path} does not exist -- skipping {label}.")
            continue
        result = analyze_one_model(label, path, output_dir)
        per_model_results[label] = result
        pooled_by_model[label] = result["signal_type_pooled"]

    if len(pooled_by_model) >= 2:
        print(f"\n{'='*88}\nCROSS-MODEL COMPARISON\n{'='*88}")
        plot_cross_model_esi_comparison(pooled_by_model, output_dir)
        plot_cross_model_usmle_comparison(pooled_by_model, output_dir)
        comparison_table = compare_models_table(pooled_by_model)
        n_non_overlapping = sum(1 for row in comparison_table if row.get("cis_non_overlapping"))
        print(f"\n{n_non_overlapping}/{len(comparison_table)} cells show non-overlapping CIs between models:")
        for row in comparison_table:
            flag = row.get("cis_non_overlapping")
            if flag is not None:
                marker = " <-- CIs do NOT overlap" if flag else ""
                print(f"  [{row['task']}/{row['signal_type']}/{row['rec_type']}]{marker}")
    else:
        comparison_table = []
        print("\nOnly one model's data provided -- skipping cross-model comparison (need at least 2 --file arguments).")

    export = {
        "metadata": {
            "files": {label: str(path) for label, path in files.items()},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "per_model": per_model_results,
        "cross_model_comparison": comparison_table,
    }
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nAll computed metrics saved to: {metrics_path}")
    print(f"All figures written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())