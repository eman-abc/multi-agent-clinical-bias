#!/usr/bin/env python3
"""
run_attn_mlp_split.py -- does the pressure effect act through ATTENTION or through MLPs? GPU; saves raw P(correct).

Patches the clean run's last-token attention output, MLP output, or both, into the pressured run for every layer of a
WINDOW at once (primary window L14-L18; control windows of the same width elsewhere), plus a descriptive single-layer
sweep. Restoration of P(correct) is computed afterwards by analyze_attn_mlp.py under the frozen decision rules.

    python run_attn_mlp_split.py --model Llama-3.1-8B-Instruct --channel assistant --plan analysis_plan.json
    python run_attn_mlp_split.py --model Mistral-7B-Instruct-v0.1 --channel user --windows 14-18,8-12 --sweep 10-24

Built-in validity check, run on every case: patching attention AND MLP, and attention ALONE, at EVERY layer must reproduce
the clean P(correct) (exact identities for rotary-position models, see mech/patching_window.py). If it does not hold to 1e-3 the
run is flagged INVALID. About 40 forward passes per case, a few minutes per model and channel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mech import ace_sensor, chat_prompting, deference_pairs, patching
from mech import patching_window as pw
from mech import plan as P
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset
from shared.inference_engine import USMLE_CLASSES
import run_phase3_deference_sensor as p3

OUT_DIR = "paper2_results/attn_mlp"
TOL = 1e-3
BUILTIN = {"primary_window": [14, 18], "control_windows": [[8, 12], [26, 30]], "single_layer_sweep": [10, 24], "n_pairs": 100,
           "min_clean_p_correct": 0.8, "pressure": "consensus"}


def parse_range(s):
    a, b = str(s).split("-")
    return [int(a), int(b)]


def parse_ranges(s):
    return None if s is None else [parse_range(x) for x in str(s).split(",")]


def out_path(model, pressure, channel, out_dir=OUT_DIR):
    return Path(out_dir) / f"attn_mlp_{model}_{pressure}_{channel}.npz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    ap.add_argument("--channel", required=True, choices=["user", "assistant"])
    ap.add_argument("--plan", default=None)
    ap.add_argument("--allow-modified-plan", action="store_true")
    ap.add_argument("--windows", default=None, help="primary window first, then controls, e.g. 14-18,8-12,26-30")
    ap.add_argument("--sweep", default=None, help="single-layer sweep range, e.g. 10-24")
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--min-clean-p-correct", type=float, default=None)
    ap.add_argument("--pressure", default=None, choices=list(deference_pairs.PRESSURES))
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    plan = P.load_plan(args.plan) if args.plan else None
    status = P.plan_status(args.plan) if args.plan else {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": ["run without --plan"]}
    print(P.banner(status))
    if status["status"] == "MODIFIED" and not args.allow_modified_plan:
        print("Refusing to run a confirmatory analysis on a modified plan (or pass --allow-modified-plan).", file=sys.stderr)
        return 2
    given = plan is not None
    sp = plan["attn_mlp_split"] if given else {}
    cases = plan["cases"] if given else {}
    cli_windows = parse_ranges(args.windows)
    try:
        cli_primary = cli_windows[0] if cli_windows else None
        cli_controls = cli_windows[1:] if cli_windows else None
        primary = P.resolve("windows", cli_primary, sp.get("primary_window"), BUILTIN["primary_window"], given)
        controls = P.resolve("windows", cli_controls, sp.get("control_windows"), BUILTIN["control_windows"], given)
        sweep = P.resolve("sweep", parse_range(args.sweep) if args.sweep else None, sp.get("single_layer_sweep"), BUILTIN["single_layer_sweep"], given)
        n_pairs = P.resolve("n_pairs", args.n_pairs, cases.get("n_pairs_cap"), BUILTIN["n_pairs"], given)
        thr = P.resolve("min_clean_p_correct", args.min_clean_p_correct, (cases.get("min_clean_p_correct") or {}).get(args.model), BUILTIN["min_clean_p_correct"], given)
        pressure = P.resolve("pressure", args.pressure, plan.get("pressure") if given else None, BUILTIN["pressure"], given)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    windows = [primary] + [list(w) for w in controls]
    roles = ["primary"] + [f"control{k + 1}" for k in range(len(controls))]
    L = spec.n_layers
    for a, b in windows + [sweep]:
        if not (0 <= a <= b < L):
            print(f"ERROR: range {a}-{b} is outside this model's {L} layers", file=sys.stderr)
            return 2

    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    pairs, _ = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    kept, clean_tokens, clean_probs, sel = p3.select_cases_model_knows(model, pairs, case_lookup, ids, min_p_correct=thr, n_pairs=n_pairs)
    print(f"know-the-answer filter (>= {thr}): kept {len(kept)} of {sel['candidates_tested']} tested")
    if len(kept) < 10:
        print("ERROR: fewer than 10 cases pass the filter", file=sys.stderr)
        return 1
    case_ids, pressured = [], {}
    for cid in kept:
        p = pairs[cid]
        pt = p3.build_pressured_tokens(model, args.channel, case_lookup[cid], p["N"], p["REC_wrong"], wrong_letter=p["wrong_letter"], wrong_text=p["wrong_text"], pressure=pressure)
        if pt is not None:
            pressured[cid] = pt
            case_ids.append(cid)
    n = len(case_ids)
    corr = [pairs[c]["correct_letter"] for c in case_ids]
    clean_pc = np.array([clean_probs[c][corr[i]] for i, c in enumerate(case_ids)])
    press_pc = np.array([ace_sensor.plain_class_probs(model, pressured[c], ids)[corr[i]] for i, c in enumerate(case_ids)])
    print(f"{n} cases; mean P(correct) clean {clean_pc.mean():.3f}, pressured {press_pc.mean():.3f}, gap {clean_pc.mean() - press_pc.mean():.3f}")

    types = list(pw.TYPES)
    sweep_layers = list(range(sweep[0], sweep[1] + 1))
    patched_window = np.zeros((len(windows), len(types), n))
    patched_sweep = np.zeros((len(sweep_layers), 2, n))
    sanity = np.zeros(n)
    sanity_attn = np.zeros(n)
    t0 = time.time()
    for i, cid in enumerate(case_ids):
        acts = patching.capture_last_token_activations(model, clean_tokens[cid], L, components=("attn_out", "mlp_out"))
        for wi, (a, b) in enumerate(windows):
            for ti, t in enumerate(types):
                patched_window[wi, ti, i] = pw.patch_window_and_predict(model, pressured[cid], acts, layers=range(a, b + 1), components=pw.TYPES[t], class_token_ids=ids)[corr[i]]
        for si, l in enumerate(sweep_layers):
            for ci, comp in enumerate(("attn_out", "mlp_out")):
                patched_sweep[si, ci, i] = pw.patch_window_and_predict(model, pressured[cid], acts, layers=[l], components=(comp,), class_token_ids=ids)[corr[i]]
        sanity[i] = pw.patch_window_and_predict(model, pressured[cid], acts, layers=range(L), components=pw.TYPES["both"], class_token_ids=ids)[corr[i]]
        sanity_attn[i] = pw.patch_window_and_predict(model, pressured[cid], acts, layers=range(L), components=pw.TYPES["attn_out"], class_token_ids=ids)[corr[i]]
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} cases, {time.time() - t0:.0f}s")

    worst = float(max(np.abs(sanity - clean_pc).max(), np.abs(sanity_attn - clean_pc).max()))
    valid = worst <= TOL
    print(f"EXACTNESS CHECK: patching attention+MLP, and attention alone, at all {L} layers reproduces the clean P(correct) to within {worst:.2e} "
          + ("(valid)" if valid else f"-- INVALID: exceeds {TOL}; do not use this run"))
    path = out_path(spec.display_name, pressure, args.channel, args.out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": spec.display_name, "channel": args.channel, "pressure": pressure, "windows": windows, "roles": roles, "types": types,
            "sweep_layers": sweep_layers, "min_clean_p_correct": thr, "n_pairs_cap": n_pairs, "exactness_max_abs_diff": worst, "valid": bool(valid),
            "plan_status": status["status"], "plan_frozen_at": status["frozen_at_utc"], "n_layers": L}
    np.savez(path, meta=np.array(json.dumps(meta)), case_ids=np.array(case_ids), clean_pc=clean_pc, pressured_pc=press_pc,
             windows=np.array(windows), patched_window=patched_window, sweep_layers=np.array(sweep_layers), patched_sweep=patched_sweep, sanity=sanity, sanity_attn=sanity_attn)
    print(f"saved {path}\nnext: python analyze_attn_mlp.py")
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
