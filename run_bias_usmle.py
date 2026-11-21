#!/usr/bin/env python3
"""
run_bias_usmle.py -- Stage C1: the BIAS direction, measured on USMLE prompts.

Phase 2 built the bias direction on ESI-triage prompts (digit answers); the deference direction (Phase 3) lives on USMLE
prompts (letter answers). Comparing them across formats leaves a low cosine uninterpretable. This script builds the
bias direction on the SAME prompt format as the deference direction, from the same Paper 1 notes:

  contrast : E+ (privileged-group carrier) minus E- (unprivileged-group carrier), same case, same facts -- Phase 2's
             convention exactly (positive = E+), at the last token of the USMLE prompt, every layer.
  behavior : does the carrier change the USMLE answer at all, and MORE than mere rewording does? The paraphrase notes
             ("P": same facts reworded, no demographic cue) set the bar -- Compared-to-What's central warning is that
             a counterfactual flip rate can be matched by a paraphrase.
  validity : steer NEUTRAL prompts along the direction (strength in gap units, as in Phase 3) and see whether P(correct)
             moves more than a RANDOM direction of the same size moves it. There is no sign prediction here (we do not
             know which way "privileged" should push P(correct)), so the test is on the SIZE of the swing.
  ceiling  : split-half reliability of the direction and sign-flip null directions are saved for the comparison grid.

GATE G1 (roadmap): is there any causal effect beyond random? If not, the bias direction is readable on USMLE prompts but
does not move USMLE answers, and the format-matched comparison has to rest on the ESI side (stage C2).

Usage (Llama; use --min-clean-p-correct 0.5 for Mistral):
    python run_bias_usmle.py --model Llama-3.1-8B-Instruct --attribute ses --skip-validation      # ~3 min: behavior + direction + reliability
    python run_bias_usmle.py --model Llama-3.1-8B-Instruct --attribute ses --layers 10,12,14,16,18,20,22,24,26,28
    python run_bias_usmle.py --model Llama-3.1-8B-Instruct --attribute ses                          # full sweep, ~15 min
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mech import ace_sensor, chat_prompting, deference_pairs, reliability
from mech.checkpoint import DirectionStore, SteeringCheckpoint, SteeringObservation, export_metrics_json
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset
from shared.inference_engine import USMLE_CLASSES
from run_phase2_bias_sensor import load_matched_triples
import run_phase3_deference_sensor as p3

OUT_ROOT = "paper2_results/phase2usmle"
DIRECTIONS_DIR = f"{OUT_ROOT}/directions"
STEERING_LOG = f"{OUT_ROOT}/steering_validation.jsonl"
METRICS_DIR = f"{OUT_ROOT}/metrics"


# ------------------------------------------------------------------ pure helpers (unit-tested)

def load_paraphrase_notes(jsonl_path, *, attribute: str) -> dict[str, str]:
    """case_id -> the paraphrase control note ("P": same facts, reworded, no demographic cue) for one attribute."""
    out: dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("condition_label") == "P" and r.get("attribute") == attribute and r.get("rec_type") is None
                    and r.get("note_text") and not r.get("blocked_reason")):
                out.setdefault(r["case_id"], r["note_text"])
    return out


def boot_ci(x, boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def behavior_stats(rows: list[dict], boot: int = 2000) -> dict:
    """rows: [{'correct': 'A', 'probs': {'E+': {...}, 'E-': {...}, 'N': {...}, 'P': {...}}}] (P may be missing)."""
    pc = lambda r, c: r["probs"][c][r["correct"]]
    top = lambda r, c: max(r["probs"][c], key=r["probs"][c].get)
    out = {"n_cases": len(rows)}
    out["mean_p_correct"] = {c: float(np.mean([pc(r, c) for r in rows if c in r["probs"]])) for c in ("E+", "E-", "N", "P") if any(c in r["probs"] for r in rows)}
    out["signed_gap_eplus_minus_eminus"] = boot_ci([pc(r, "E+") - pc(r, "E-") for r in rows], boot)
    out["abs_gap_counterfactual"] = boot_ci([abs(pc(r, "E+") - pc(r, "E-")) for r in rows], boot)
    out["top_answer_agreement_counterfactual"] = float(np.mean([top(r, "E+") == top(r, "E-") for r in rows]))
    withp = [r for r in rows if "P" in r["probs"]]
    if withp:
        out["n_with_paraphrase"] = len(withp)
        out["abs_gap_paraphrase"] = boot_ci([abs(pc(r, "N") - pc(r, "P")) for r in withp], boot)
        out["top_answer_agreement_paraphrase"] = float(np.mean([top(r, "N") == top(r, "P") for r in withp]))
        # the number that matters: does the demographic swap move P(correct) MORE than rewording does? (paired by case)
        out["abs_gap_counterfactual_minus_paraphrase"] = boot_ci([abs(pc(r, "E+") - pc(r, "E-")) - abs(pc(r, "N") - pc(r, "P")) for r in withp], boot)
    return out


def swing_flags(ms: dict, mr: dict | None, pos_c: float, neg_c: float, min_effect: float) -> dict:
    """Bias has no predicted SIGN on USMLE, so the test is the SIZE of the swing between +c and -c, against a random direction."""
    if ms.get(pos_c) is None or ms.get(neg_c) is None:
        return {"swing": None, "random_swing": None, "moves_answers": False, "beats_random": None}
    swing = ms[pos_c] - ms[neg_c]
    rswing = (mr[pos_c] - mr[neg_c]) if (mr is not None and mr.get(pos_c) is not None and mr.get(neg_c) is not None) else None
    moves = abs(swing) >= min_effect
    beats = None if mr is None else bool(moves and rswing is not None and abs(swing) > abs(rswing))
    return {"swing": float(swing), "random_swing": rswing, "moves_answers": bool(moves), "beats_random": beats}


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    ap.add_argument("--attribute", required=True, choices=["ses", "race", "sex"])
    ap.add_argument("--signal-type", default="PRX", choices=["PRX", "EXP", "IMP"])
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--n-pairs", type=int, default=140, help="Cases used to EXTRACT the direction (more data = a steadier direction).")
    ap.add_argument("--validate-n", type=int, default=100, help="Max cases kept for the steering check.")
    ap.add_argument("--min-clean-p-correct", type=float, default=0.8, help="Steering-check cases must have P(correct) >= this on the NEUTRAL prompt (0.5 for Mistral). 0 disables.")
    ap.add_argument("--coef-unit", default="gap", choices=["gap", "absolute"])
    ap.add_argument("--coefficients", default=None, help="Default -2,-1,1,2 (gap) or -8,-4,4,8 (absolute).")
    ap.add_argument("--predict-at", type=float, default=None, help="Strength at which the swing is judged (default 1 for gap units).")
    ap.add_argument("--min-effect", type=float, default=0.01, help="Smallest swing in P(correct) that counts as an effect.")
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--layers", default=None, help="Comma-separated layers for the steering check (default: all). Extraction always covers every layer.")
    ap.add_argument("--skip-validation", action="store_true", help="Behavior check + direction + reliability only (a few minutes).")
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    if args.coefficients is None:
        args.coefficients = "-2,-1,1,2" if args.coef_unit == "gap" else "-8,-4,4,8"
    coefficients = [float(x) for x in args.coefficients.split(",")]
    sym = sorted(c for c in coefficients if c > 0 and -c in coefficients)
    pos_c = float(args.predict_at) if args.predict_at is not None else (1.0 if (args.coef_unit == "gap" and 1.0 in sym) else (sym[-1] if sym else max(coefficients)))
    neg_c = -pos_c
    if pos_c not in coefficients or neg_c not in coefficients:
        print(f"ERROR: +-{pos_c:g} must both be among the coefficients {coefficients}", file=sys.stderr)
        return 1

    # ---- cheap, fail-fast work first
    print("Loading AMQA dataset...")
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    triples = load_matched_triples(args.source_jsonl, attribute=args.attribute, model=None, signal_type=args.signal_type)
    paraphrase = load_paraphrase_notes(args.source_jsonl, attribute=args.attribute)
    case_ids = [c for c in sorted(triples) if c in case_lookup and deference_pairs.correct_letter(case_lookup[c]) is not None][: args.n_pairs]
    print(f"{len(triples)} complete (E+, E-, N) triples for attribute={args.attribute!r}, signal_type={args.signal_type!r}; "
          f"{len(case_ids)} usable with an AMQA answer key; paraphrase notes for {sum(c in paraphrase for c in case_ids)} of them")
    if len(case_ids) < 10:
        print("ERROR: fewer than 10 usable cases.", file=sys.stderr)
        return 1

    print(f"Loading model {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    print(f"Resolved USMLE class tokens in context: {ids}")

    # ---- prompts: every note goes through the SAME USMLE prompt builder Phase 3 uses
    tok = {}
    for cid in case_ids:
        rec = case_lookup[cid]
        for cond in ("E+", "E-", "N"):
            tok[(cid, cond)] = p3.build_clean_tokens(model, rec, triples[cid][cond])
        if cid in paraphrase:
            tok[(cid, "P")] = p3.build_clean_tokens(model, rec, paraphrase[cid])

    # ---- 1. behavior: does the carrier change the answer, and more than rewording?
    rows = []
    for cid in case_ids:
        probs = {c: ace_sensor.plain_class_probs(model, tok[(cid, c)], ids) for c in ("E+", "E-", "N", "P") if (cid, c) in tok}
        rows.append({"cid": cid, "correct": deference_pairs.correct_letter(case_lookup[cid]), "probs": probs})
    beh = behavior_stats(rows)
    print(f"\nBEHAVIOR (USMLE prompts, attribute={args.attribute}, {beh['n_cases']} cases):")
    print("  mean P(correct): " + "  ".join(f"{c} {v:.3f}" for c, v in beh["mean_p_correct"].items()))
    sg = beh["signed_gap_eplus_minus_eminus"]
    ag = beh["abs_gap_counterfactual"]
    print(f"  signed gap P(correct|E+) - P(correct|E-): {sg[0]:+.3f} [{sg[1]:+.3f}, {sg[2]:+.3f}]")
    print(f"  mean |gap| E+ vs E-: {ag[0]:.3f} [{ag[1]:.3f}, {ag[2]:.3f}]   top answer agrees: {100 * beh['top_answer_agreement_counterfactual']:.0f}%")
    if "abs_gap_paraphrase" in beh:
        pg = beh["abs_gap_paraphrase"]
        dd = beh["abs_gap_counterfactual_minus_paraphrase"]
        print(f"  mean |gap| N vs reworded N (paraphrase): {pg[0]:.3f} [{pg[1]:.3f}, {pg[2]:.3f}]   top answer agrees: {100 * beh['top_answer_agreement_paraphrase']:.0f}%")
        print(f"  demographic swap minus rewording, |gap|: {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]  "
              f"{'<- demographic swap moves P(correct) MORE than rewording' if dd[1] > 0 else '<- not distinguishable from rewording'}")
    else:
        print("  (no paraphrase notes found for this attribute: cannot separate demographic effects from rewording)")

    # ---- 2. direction + reliability (all layers, every case)
    print(f"\nExtracting the bias direction at all {spec.n_layers} layers from {len(case_ids)} E+/E- pairs...")
    pos_all = reliability.collect_activations(model, [tok[(c, "E+")] for c in case_ids], spec.n_layers)
    neg_all = reliability.collect_activations(model, [tok[(c, "E-")] for c in case_ids], spec.n_layers)
    results = [ace_sensor._direction_from_vectors(pos_all[:, l, :], neg_all[:, l, :], len(case_ids), "last_token") for l in range(spec.n_layers)]
    store = DirectionStore(DIRECTIONS_DIR)
    for l, res in enumerate(results):
        store.save(model=spec.display_name, attribute=args.attribute, layer=l, direction=res.direction, mean_pos=res.mean_pos, mean_neg=res.mean_neg,
                   std=res.std, bias_scalar=res.bias_scalar, n_pairs=res.n_pairs, extraction_point=res.extraction_point)
    print(f"Saved {len(results)} layer directions to {DIRECTIONS_DIR}/ (attribute={args.attribute!r})")
    split = reliability.split_half_cosines(pos_all, neg_all, n_splits=args.n_splits)
    null_dirs = reliability.signflip_null_directions(pos_all, neg_all, n_null=args.n_null)
    reliability.save_reliability(reliability.reliability_path("bias_usmle", spec.display_name, args.attribute), split_half=split, null_dirs=null_dirs, n_pairs=len(case_ids))
    sh = split.mean(axis=0)
    mid = slice(14, 25) if spec.n_layers > 25 else slice(0, spec.n_layers)
    print(f"Split-half reliability of the direction: mean cosine {sh.mean():.3f} over all layers, {sh[mid].mean():.3f} over layers "
          f"{'14-24' if spec.n_layers > 25 else 'all (small model)'} (full-sample estimate {float(reliability.spearman_brown(sh[mid].mean())):.3f})")

    metrics = {"model": spec.display_name, "attribute": args.attribute, "signal_type": args.signal_type, "prompt_format": "USMLE",
               "n_extraction_pairs": len(case_ids), "behavior": beh, "split_half_mean_by_layer": [float(x) for x in sh]}
    if args.skip_validation:
        export_metrics_json(metrics, f"{METRICS_DIR}/phase2usmle_metrics_{spec.display_name}_{args.attribute}.json")
        print("\n--skip-validation: stopping after behavior + direction + reliability.")
        return 0

    # ---- 3. steering check on NEUTRAL prompts, against a random direction
    kept = []
    for r in rows:
        if r["probs"]["N"][r["correct"]] >= args.min_clean_p_correct:
            kept.append(r["cid"])
            if len(kept) >= args.validate_n:
                break
    print(f"\nSteering check: {len(kept)} cases with P(correct) >= {args.min_clean_p_correct} on the neutral prompt.")
    if len(kept) < 5:
        print("ERROR: fewer than 5 cases pass the filter; lower --min-clean-p-correct.", file=sys.stderr)
        return 1
    base = {r["cid"]: r["probs"]["N"] for r in rows}
    corr = {r["cid"]: r["correct"] for r in rows}
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(spec.n_layers))
    gap = {l: float(np.dot(results[l].mean_pos - results[l].mean_neg, results[l].direction)) for l in range(spec.n_layers)}
    unit = {l: (gap[l] if args.coef_unit == "gap" else 1.0) for l in gap}
    do_random = not args.no_random_control
    attr_key = f"{args.attribute}_usmle"
    print("Clean-to-pressured style distance along the direction (E+ minus E-, activation units): " + ", ".join(f"L{l}={gap[l]:.1f}" for l in (4, 10, 14, 16, 18, 22, 28) if l in gap))
    print(f"Strengths in {'MULTIPLES OF THAT DISTANCE' if args.coef_unit == 'gap' else 'raw units'}; judged at +-{pos_c:g}; shifts under {args.min_effect:g} count as no effect.")
    ckpt = SteeringCheckpoint(STEERING_LOG)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    per_layer, n_moves, n_beats = {}, 0, 0
    for layer in layers:
        direction = results[layer].direction
        rng = np.random.default_rng(2000 + layer)
        rdir = rng.normal(size=direction.shape[0])
        rdir /= np.linalg.norm(rdir)
        d_dim = {c: [] for c in coefficients}
        d_rnd = {c: [] for c in coefficients}
        for cid in kept:
            for coef in coefficients:
                for kind, dvec, store_d in (("dim", direction, d_dim), ("rnd", rdir, d_rnd)):
                    if kind == "rnd" and not do_random:
                        continue
                    top_class, probs = ace_sensor.steered_class_prediction(model, tok[(cid, "N")], ids, layer=layer, direction=dvec, coefficient=coef * unit[layer])
                    store_d[coef].append(probs[corr[cid]] - base[cid][corr[cid]])
                    obs = SteeringObservation(model=spec.display_name, attribute=attr_key + ("" if kind == "dim" else "__random"), layer=layer, coefficient=coef,
                                              case_id=cid, task="usmle", condition_source="N_prompt_steered", top_class=None, class_probabilities=None, raw_text=None)
                    if not ckpt.has(obs):
                        obs.top_class, obs.class_probabilities = top_class, probs
                        ckpt.append(obs)
        ms = {c: mean(d_dim[c]) for c in coefficients}
        mr = {c: mean(d_rnd[c]) for c in coefficients} if do_random else None
        fl = swing_flags(ms, mr, pos_c, neg_c, args.min_effect)
        n_moves += fl["moves_answers"]
        n_beats += bool(fl["beats_random"])
        per_layer[layer] = {"d_p_correct": ms, "d_p_correct_random": mr, **fl}
        shown = "  ".join(f"{c:+.0f}:{ms[c]:+.3f}" for c in coefficients if ms[c] is not None)
        sw = f"{fl['swing']:+.3f}" if fl["swing"] is not None else "n/a"
        rs = f"{fl['random_swing']:+.3f}" if fl["random_swing"] is not None else "n/a"
        print(f"  layer={layer:2d}  swing {sw} (random {rs})  moves answers: {str(fl['moves_answers']):5s}  beats random: {str(fl['beats_random']):5s}  | dP(correct) by coef  {shown}")

    metrics.update({"n_validation_cases": len(kept), "validation_cases": kept, "min_clean_p_correct": args.min_clean_p_correct, "coefficients": coefficients,
                    "coef_unit": args.coef_unit, "gap_by_layer": gap, "predict_at": pos_c, "min_effect": args.min_effect, "random_control": do_random,
                    "per_layer": per_layer, "layers_moving_answers": n_moves, "layers_beating_random": n_beats if do_random else None,
                    "per_case": {c: {"correct": corr[c], "n_probs": base[c]} for c in kept}})
    export_metrics_json(metrics, f"{METRICS_DIR}/phase2usmle_metrics_{spec.display_name}_{args.attribute}.json")
    print(f"\nSummary (G1 readout): {n_moves}/{len(layers)} layers move P(correct) by at least {args.min_effect:g} at +-{pos_c:g}"
          + (f"; {n_beats}/{len(layers)} also beat the random direction." if do_random else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
