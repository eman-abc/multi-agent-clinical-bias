"""
generate_publication_figures.py

Reads a publication_metrics_*.json file (produced by run_full_experiment.py)
and generates publication-quality static figures (PNG + PDF) summarizing:

1. Paraphrase-excess forest plot -- ONE FILE PER MODEL (mean + 95% CI for
   every attribute x signal-type x task combination). This is the key
   demographic-bias result: a CI that excludes zero is the only thing that
   counts as a real, noise-controlled signal in this project's own design
   (see "Compared to What?", arXiv:2605.01048). Separate files per model,
   not one combined chart, because with up to 4 models x 3 attributes x 3
   signal types x 2 tasks, a single chart would have 70+ rows and be
   unreadable.
2. REC-conformity GROUPED bar chart -- ONE combined chart, models side by
   side within each of REC_UP/REC_DOWN/USMLE-yield. Unlike the forest
   plot, direct cross-model comparison IS the point here, and there are
   only 3 metrics, so grouping fits comfortably in one readable chart.
3. CFR heatmap -- ONE figure, one subplot per model, so the same
   at-a-glance summary is directly comparable side by side rather than
   requiring the reader to flip between separate files.

Usage:
    python generate_publication_figures.py phase0_results/publication_metrics_n200_t0.0.json
    python generate_publication_figures.py phase0_results/publication_metrics_n200_t0.0.json --output-dir figures/
"""

import json
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed -- just write files
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return _normalize_to_multi_model(data)


_KNOWN_ATTRIBUTES = {"ses", "race", "sex"}


def _normalize_to_multi_model(data: dict) -> dict:
    """
    Older runs of run_full_experiment.py (before multi-model support was
    added) produced demographic_bias/redaction/rec_conformity keyed
    directly by attribute/task, with a single `agent_b_model` string in
    metadata -- not the current per-model-keyed structure. Detect that
    shape here and wrap it so every plotting function below can assume
    the current (model-keyed) structure unconditionally, rather than
    branching in every single plot function.
    """
    demo = data.get("demographic_bias", {})
    is_old_shape = bool(demo) and set(demo.keys()) <= _KNOWN_ATTRIBUTES
    if not is_old_shape:
        return data  # already model-keyed, or empty -- nothing to do

    model = data.get("metadata", {}).get("agent_b_model", "model")
    print(f"Detected single-model (pre-multi-model) JSON shape -- wrapping under model={model!r}.")
    data = dict(data)  # shallow copy, don't mutate the caller's dict
    data["demographic_bias"] = {model: data["demographic_bias"]}
    data["redaction"] = {model: data["redaction"]} if data.get("redaction") else {}
    data["rec_conformity"] = {model: data["rec_conformity"]} if data.get("rec_conformity") else {}
    return data


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def plot_paraphrase_excess_forest(data: dict, output_dir: Path) -> None:
    demo = data.get("demographic_bias", {})
    if not demo:
        print("No demographic_bias data found -- skipping paraphrase-excess forest plots.")
        return

    for model, attrs in demo.items():
        rows = []  # (label, mean, ci_low, ci_high, n)
        for attribute, attr_data in attrs.items():
            pe = attr_data.get("paraphrase_excess", {})
            for signal_type, signal_data in pe.items():
                for task, summary in signal_data.items():
                    if summary.get("mean") is None:
                        continue
                    label = f"{attribute}/{signal_type}/{task}"
                    rows.append((label, summary["mean"], summary["ci_low"], summary["ci_high"], summary.get("n", "?")))

        if not rows:
            print(f"[{model}] No paraphrase-excess entries with data -- skipping forest plot.")
            continue

        rows.sort(key=lambda r: r[0])
        labels = [r[0] for r in rows]
        means = [r[1] for r in rows]
        ci_los = [r[2] for r in rows]
        ci_his = [r[3] for r in rows]
        ns = [r[4] for r in rows]

        y_pos = np.arange(len(labels))
        fig_height = max(4, 0.35 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        err_low = [m - lo for m, lo in zip(means, ci_los)]
        err_high = [hi - m for m, hi in zip(means, ci_his)]

        # Color: red if CI excludes zero (a real, noise-controlled signal by
        # this project's own design), gray otherwise.
        colors = ["#c0392b" if (lo > 0 or hi < 0) else "#7f8c8d" for lo, hi in zip(ci_los, ci_his)]

        ax.errorbar(means, y_pos, xerr=[err_low, err_high], fmt="o", markersize=5,
                    ecolor="black", elinewidth=1, capsize=3, zorder=2)
        ax.scatter(means, y_pos, c=colors, s=40, zorder=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{lbl}  (n={n})" for lbl, n in zip(labels, ns)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Paraphrase-excess (95% bootstrap CI)")
        ax.set_title(f"Demographic bias: paraphrase-excess -- {model}")
        ax.grid(axis="x", linestyle=":", alpha=0.5)

        fig.tight_layout()
        fname = f"paraphrase_excess_forest_{_safe_filename(model)}"
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"{fname}.{ext}", dpi=200)
        plt.close(fig)
        print(f"Saved {fname}.png / .pdf ({len(rows)} conditions)")


def plot_rec_conformity_bars(data: dict, output_dir: Path) -> None:
    rec = data.get("rec_conformity", {})
    if not rec:
        print("No rec_conformity data found -- skipping REC-conformity bar chart.")
        return

    models = list(rec.keys())
    metric_defs = [
        ("rec_up", "REC_UP\n(escalate urgency)", "total"),
        ("rec_down", "REC_DOWN\n(de-escalate urgency)", "total"),
        ("usmle_yield", "USMLE yield\n(stated wrong diagnosis)", "eligible"),
    ]

    # Collect (model, metric) -> (rate%, err_low, err_high, n)
    values = {}
    for model in models:
        for key, _, n_field in metric_defs:
            entry = rec[model].get(key)
            if not entry:
                continue
            values[(model, key)] = (
                entry["rate"] * 100,
                (entry["rate"] - entry["ci_low"]) * 100,
                (entry["ci_high"] - entry["rate"]) * 100,
                entry[n_field],
            )

    if not values:
        print("No REC-conformity entries with data -- skipping bar chart.")
        return

    n_models = len(models)
    n_metrics = len(metric_defs)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(7 + n_models, 5))

    palette = ["#2980b9", "#8e44ad", "#27ae60", "#d35400", "#c0392b", "#16a085"]
    for i, model in enumerate(models):
        rates, err_los, err_his, ns = [], [], [], []
        for key, _, _ in metric_defs:
            v = values.get((model, key))
            rates.append(v[0] if v else 0)
            err_los.append(v[1] if v else 0)
            err_his.append(v[2] if v else 0)
            ns.append(v[3] if v else 0)
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width=width * 0.9, yerr=[err_los, err_his],
                      capsize=3, label=model, color=palette[i % len(palette)], alpha=0.85)
        for xi, (rate, n) in enumerate(zip(rates, ns)):
            ax.text(x[xi] + offset, rate + max(err_his, default=0) * 0.1 + 1, f"n={n}", ha="center", fontsize=6, rotation=90)

    ax.set_ylabel("Rate (%)")
    ax.set_title("Recommendation-conformity by model (Wilson 95% CI)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in metric_defs], fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"rec_conformity_bars.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved rec_conformity_bars.png / .pdf ({n_models} model(s), {n_metrics} metrics)")


def plot_cfr_heatmap(data: dict, output_dir: Path) -> None:
    demo = data.get("demographic_bias", {})
    if not demo:
        print("No demographic_bias data found -- skipping CFR heatmap.")
        return

    models = list(demo.keys())
    signal_types = ["EXP", "PRX", "IMP"]
    tasks = ["esi", "usmle"]
    col_labels = [f"{s}/{t}" for s in signal_types for t in tasks]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 2.5 + 0.4 * 3), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        attributes = sorted(demo[model].keys())
        matrix = np.full((len(attributes), len(col_labels)), np.nan)
        for i, attribute in enumerate(attributes):
            for j, (s, t) in enumerate((s, t) for s in signal_types for t in tasks):
                cell = demo[model].get(attribute, {}).get(s, {}).get(t, {})
                cfr = cell.get("cfr", {}).get("cfr")
                if cfr is not None:
                    matrix[i, j] = cfr

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(attributes)))
        ax.set_yticklabels([a.upper() for a in attributes], fontsize=8)
        ax.set_title(model, fontsize=10)

        for i in range(len(attributes)):
            for j in range(len(col_labels)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                            color="black" if val > 0.7 else "white")

    fig.colorbar(im, ax=axes, label="CFR (higher = more counterfactually fair)", shrink=0.8)
    fig.suptitle("Counterfactual Fairness Rate by model / attribute / signal type / task")

    fname = "cfr_heatmap"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{fname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}.png / .pdf ({len(models)} model(s))")


def plot_redaction_reversion(data: dict, output_dir: Path) -> None:
    """
    Experiment 3's redaction/causality check: does deleting the carrier
    phrase revert Agent B's decision back toward baseline? Never
    visualized before in this script. Shown alongside the raw
    mean_delta_e_plus/mean_delta_redact values, since the reversion
    fraction ratio can look extreme (even negative) when both deltas are
    small -- the raw deltas make it clear whether that's a real reversal
    or just noise in a near-zero signal.
    """
    redaction = data.get("redaction", {})
    if not redaction:
        print("No redaction data found -- skipping the redaction/reversion figure.")
        return

    models = list(redaction.keys())
    tasks = ["esi", "usmle"]
    fig, axes = plt.subplots(1, 2, figsize=(5 + 2.5 * len(models), 4.5))

    # Left panel: the reversion fraction itself, with the pre-registered pass threshold marked.
    ax = axes[0]
    x = np.arange(len(tasks))
    width = 0.8 / max(len(models), 1)
    palette = ["#2980b9", "#8e44ad", "#27ae60", "#d35400"]
    for i, model in enumerate(models):
        vals = [redaction[model].get(t, {}).get("reversion_fraction") for t in tasks]
        vals = [v if v is not None else 0 for v in vals]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=model, color=palette[i % len(palette)], alpha=0.85)
    ax.axhline(0.30, color="black", linewidth=1, linestyle="--", label="pre-registered pass threshold (0.30)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("Reversion fraction")
    ax.set_title("Does redacting the carrier revert the decision?", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Right panel: the raw deltas underlying that ratio, so a small or
    # negative reversion fraction can be read in context rather than
    # taken as a dramatic reversal.
    ax = axes[1]
    for i, model in enumerate(models):
        e_plus = [redaction[model].get(t, {}).get("mean_delta_e_plus", 0) for t in tasks]
        redact = [redaction[model].get(t, {}).get("mean_delta_redact", 0) for t in tasks]
        offset_e = (i - (len(models) - 1) / 2) * width - width * 0.22
        offset_r = (i - (len(models) - 1) / 2) * width + width * 0.22
        ax.bar(x + offset_e, e_plus, width=width * 0.4, color=palette[i % len(palette)], alpha=0.9, label=f"{model}: Δ(E+,N)" if i == 0 else None)
        ax.bar(x + offset_r, redact, width=width * 0.4, color=palette[i % len(palette)], alpha=0.4, hatch="//", label=f"{model}: Δ(redact,N)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("Mean shift from N baseline")
    ax.set_title("Underlying deltas (solid=E+, hatched=redact)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Experiment 3: carrier-phrase causal ablation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"redaction_reversion.{ext}", dpi=200)
    plt.close(fig)
    print(f"Saved redaction_reversion.png / .pdf ({len(models)} model(s))")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to a publication_metrics_*.json file from run_full_experiment.py")
    parser.add_argument("--output-dir", default="figures", help="Directory to write figures into (created if missing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(args.metrics_json)
    meta = data.get("metadata", {})
    print(f"Loaded metrics: n_cases={meta.get('n_cases')}, temperature={meta.get('temperature')}, "
          f"models={meta.get('agent_b_models')}, logprobs_used={meta.get('logprobs_used')}")

    plot_paraphrase_excess_forest(data, output_dir)
    plot_rec_conformity_bars(data, output_dir)
    plot_cfr_heatmap(data, output_dir)
    plot_redaction_reversion(data, output_dir)

    print(f"\nAll figures written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
