#!/usr/bin/env python3
"""
run_crossfit.py -- held-out validation of the deference direction. GPU; saves RAW probabilities, the CPU script
analyze_crossfit.py applies the frozen rules.

For one model and channel (consensus pressure): keep the same cases Phase 3 kept; split them into K folds; for each fold
build the deference direction at every layer from the OTHER folds only; steer each held-out case's pressured prompt
along ITS fold's direction (strength in gap units measured in the training folds) and record P(stated wrong option).
Do the same with R random unit directions (same effective size). A case is never steered by a direction that saw it.

    python run_crossfit.py --model Mistral-7B-Instruct-v0.1 --channel user --plan analysis_plan.json     # the final run
    python run_crossfit.py --model Llama-3.1-8B-Instruct --channel assistant --layers 14-16 --n-random 5   # a quick look

With --plan, every design parameter is READ FROM THE PLAN and a conflicting flag is an error. Without it the built-in
defaults equal the plan's, and the output says the run is not frozen. Resumable: a rerun skips finished layers.
Cost: n_cases x layers x (real strengths + random strengths x draws) forward passes (about 8 per second on a 4090).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from mech import ace_sensor, chat_prompting, crossfit, deference_pairs, reliability
from mech import plan as P
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset
from shared.inference_engine import USMLE_CLASSES
import run_phase3_deference_sensor as p3

OUT_DIR = "paper2_results/crossfit"
BUILTIN = {"layers": list(range(12, 25)), "folds": 10, "seed": 0, "real_coefficients": [1.0, 2.0], "random_coefficients": [2.0],
           "n_random": 20, "n_pairs": 100, "min_clean_p_correct": 0.8, "pressure": "consensus"}


def parse_layers(s):
    if s is None:
        return None
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_floats(s):
    return None if s is None else [float(x) for x in str(s).split(",")]


def out_paths(model, pressure, channel, out_dir=OUT_DIR):
    stem = Path(out_dir) / f"crossfit_{model}_{pressure}_{channel}"
    return Path(str(stem) + ".npz"), Path(str(stem) + "_acts.npz")


def resolve_params(args, plan, model):
    """Design parameters: from the plan when one is given (conflicts are errors), else flags, else built-ins."""
    given = plan is not None
    cf = plan["crossfit"] if given else {}
    cases = plan["cases"] if given else {}
    cli = {"layers": parse_layers(args.layers), "folds": args.folds, "seed": args.seed, "real_coefficients": parse_floats(args.coefficients),
           "random_coefficients": parse_floats(args.random_coefficients), "n_random": args.n_random, "n_pairs": args.n_pairs,
           "min_clean_p_correct": args.min_clean_p_correct, "pressure": args.pressure}
    planv = {"layers": cf.get("layers"), "folds": cf.get("folds"), "seed": cf.get("seed"), "real_coefficients": cf.get("real_coefficients"),
             "random_coefficients": cf.get("random_coefficients"), "n_random": cf.get("n_random"), "n_pairs": cases.get("n_pairs_cap"),
             "min_clean_p_correct": (cases.get("min_clean_p_correct") or {}).get(model), "pressure": plan.get("pressure") if given else None}
    return {k: P.resolve(k, cli[k], planv[k], BUILTIN[k], given) for k in BUILTIN}


def meta_for(args, params, spec, case_ids):
    return {"model": spec.display_name, "channel": args.channel, "pressure": params["pressure"], "layers": params["layers"],
            "real_coefficients": params["real_coefficients"], "random_coefficients": params["random_coefficients"], "n_random": params["n_random"],
            "folds": params["folds"], "seed": params["seed"], "min_clean_p_correct": params["min_clean_p_correct"], "n_pairs_cap": params["n_pairs"],
            "case_ids": list(case_ids)}


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    ap.add_argument("--channel", required=True, choices=["user", "assistant"])
    ap.add_argument("--plan", default=None, help="analysis_plan.json: take every design parameter from it")
    ap.add_argument("--allow-modified-plan", action="store_true", help="run even though the frozen plan was modified (the output is then labelled non-confirmatory)")
    ap.add_argument("--layers", default=None, help="e.g. 12-24 or 14,16,18")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--coefficients", default=None, help="real-direction strengths in gap units, e.g. 1,2")
    ap.add_argument("--random-coefficients", default=None, help="strengths at which random directions are run, e.g. 2")
    ap.add_argument("--n-random", type=int, default=None)
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--min-clean-p-correct", type=float, default=None)
    ap.add_argument("--pressure", default=None, choices=list(deference_pairs.PRESSURES))
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--stop-after-layers", type=int, default=None, help="stop after this many NEW layers (testing / splitting a long run)")
    ap.add_argument("--restart", action="store_true", help="discard a partial result instead of resuming it")
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    plan = P.load_plan(args.plan) if args.plan else None
    status = P.plan_status(args.plan) if args.plan else {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": ["run without --plan"]}
    print(P.banner(status))
    if status["status"] == "MODIFIED" and not args.allow_modified_plan:
        print("Refusing to run a confirmatory analysis on a modified plan. Restore the frozen files, or pass --allow-modified-plan to run non-confirmatory.", file=sys.stderr)
        return 2
    try:
        params = resolve_params(args, plan, args.model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for need in ("real_coefficients", "random_coefficients"):
        if not params[need]:
            print(f"ERROR: {need} is empty", file=sys.stderr)
            return 2
    layers, K, R, seed = params["layers"], params["folds"], params["n_random"], params["seed"]
    rc, rrc = params["real_coefficients"], params["random_coefficients"]

    # ---- cases (identical selection to Phase 3)
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    pairs, pair_stats = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
    print(f"candidate pairs: {pair_stats}")
    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    kept, clean_tokens, clean_probs, sel = p3.select_cases_model_knows(model, pairs, case_lookup, ids, min_p_correct=params["min_clean_p_correct"], n_pairs=params["n_pairs"])
    print(f"know-the-answer filter (>= {params['min_clean_p_correct']}): kept {len(kept)} of {sel['candidates_tested']} tested")
    if len(kept) < max(10, K):
        print(f"ERROR: {len(kept)} cases is too few for {K} folds", file=sys.stderr)
        return 1
    pressured_tokens, pressured_probs, case_ids = {}, {}, []
    for cid in kept:
        p = pairs[cid]
        pt = p3.build_pressured_tokens(model, args.channel, case_lookup[cid], p["N"], p["REC_wrong"], wrong_letter=p["wrong_letter"], wrong_text=p["wrong_text"], pressure=params["pressure"])
        if pt is not None:
            pressured_tokens[cid] = pt
            pressured_probs[cid] = ace_sensor.plain_class_probs(model, pt, ids)
            case_ids.append(cid)
    n = len(case_ids)
    stated = np.array([pairs[c]["wrong_letter"] for c in case_ids])
    correct = np.array([pairs[c]["correct_letter"] for c in case_ids])
    ps_clean = np.array([clean_probs[c][pairs[c]["wrong_letter"]] for c in case_ids])
    ps_press = np.array([pressured_probs[c][pairs[c]["wrong_letter"]] for c in case_ids])
    pc_clean = np.array([clean_probs[c][pairs[c]["correct_letter"]] for c in case_ids])
    pc_press = np.array([pressured_probs[c][pairs[c]["correct_letter"]] for c in case_ids])
    print(f"{n} cases; mean P(stated) clean {ps_clean.mean():.3f} -> pressured {ps_press.mean():.3f}; mean P(correct) {pc_clean.mean():.3f} -> {pc_press.mean():.3f}")

    # ---- activations once, directions per fold (CPU)
    print("Collecting clean and pressured last-token activations...")
    acts_pos = reliability.collect_activations(model, [clean_tokens[c] for c in case_ids], spec.n_layers)
    acts_neg = reliability.collect_activations(model, [pressured_tokens[c] for c in case_ids], spec.n_layers)
    folds = crossfit.assign_folds(n, K, seed)
    dirs, gaps = crossfit.fold_directions(acts_pos, acts_neg, folds, layers)
    d_model = acts_pos.shape[2]
    rdirs = {l: crossfit.random_directions(l, R, d_model, seed) for l in layers}

    mpath, apath = out_paths(spec.display_name, params["pressure"], args.channel, args.out_dir)
    meta = meta_for(args, params, spec, case_ids)
    nL, nC, nRC = len(layers), len(rc), len(rrc)
    real = np.full((nL, nC, n), np.nan, dtype=np.float32)
    rand = np.full((nL, nRC, R, n), np.nan, dtype=np.float32)
    done = np.zeros(nL, dtype=bool)
    if mpath.exists() and not args.restart:
        z = np.load(mpath, allow_pickle=False)
        old = {k: v for k, v in json.loads(str(z["meta"])).items() if k not in ("plan_status", "plan_frozen_at")}
        if old == meta:
            real, rand, done = z["real"].copy(), z["rand"].copy(), z["done"].copy()
            print(f"RESUMING: {int(done.sum())} of {nL} layers already done in {mpath}")
        else:
            print(f"ERROR: {mpath} exists with different settings or cases; use --restart to discard it", file=sys.stderr)
            return 1
    atomic_savez(apath, acts_pos=acts_pos.astype(np.float16), acts_neg=acts_neg.astype(np.float16), case_ids=np.array(case_ids))

    def save():
        atomic_savez(mpath, meta=np.array(json.dumps({**meta, "plan_status": status["status"], "plan_frozen_at": status["frozen_at_utc"]})),
                     case_ids=np.array(case_ids), stated=stated, correct=correct, p_stated_clean=ps_clean, p_stated_pressured=ps_press,
                     p_correct_clean=pc_clean, p_correct_pressured=pc_press, folds=folds, gaps=gaps, layers=np.array(layers),
                     real_coefficients=np.array(rc), random_coefficients=np.array(rrc), real=real, rand=rand, done=done)

    per_layer = n * (nC + nRC * R)
    print(f"{nL} layers x {n} cases x ({nC} real + {nRC}x{R} random) = {nL * per_layer} forward passes; about {nL * per_layer / 8 / 60:.0f} min at 8 passes/s")
    new_layers = 0
    t_start = time.time()
    for li, layer in enumerate(layers):
        if done[li]:
            continue
        t0 = time.time()
        for i, cid in enumerate(case_ids):
            f = folds[i]
            for ci, c in enumerate(rc):
                _, probs = ace_sensor.steered_class_prediction(model, pressured_tokens[cid], ids, layer=layer, direction=dirs[li, f], coefficient=c * gaps[li, f])
                real[li, ci, i] = probs[stated[i]]
            for ri, c in enumerate(rrc):
                for r in range(R):
                    _, probs = ace_sensor.steered_class_prediction(model, pressured_tokens[cid], ids, layer=layer, direction=rdirs[layer][r], coefficient=c * gaps[li, f])
                    rand[li, ri, r, i] = probs[stated[i]]
        done[li] = True
        save()
        new_layers += 1
        prim = rc.index(rrc[0]) if rrc[0] in rc else 0
        dr = real[li, prim].mean() - ps_press.mean()
        dn = rand[li, 0].mean(axis=1) - ps_press.mean()          # per random draw, mean over cases
        left = int((~done).sum())
        print(f"  layer {layer:2d} done in {time.time() - t0:5.0f}s | held-out change in P(stated) at +{rrc[0]:g}: real {dr:+.3f}, random draws mean {dn.mean():+.3f} (most negative {dn.min():+.3f}) | "
              f"{left} layers left, ~{left * (time.time() - t_start) / new_layers / 60:.0f} min")
        if args.stop_after_layers and new_layers >= args.stop_after_layers:
            print(f"--stop-after-layers {args.stop_after_layers} reached; rerun the same command to resume.")
            return 0
    print(f"\nfinished: {mpath}\nnext: python analyze_crossfit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
