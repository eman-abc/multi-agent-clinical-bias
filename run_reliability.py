#!/usr/bin/env python3
"""
run_reliability.py -- Stage C4: reliability ceilings for the directions that ALREADY exist.

For one contrast it rebuilds the exact prompts the original extraction used, keeps the PER-CASE last-token activations
(the original scripts threw them away), and from them computes
  * a split-half reliability curve (how similar the direction is to itself on a fresh half of the cases),
  * sign-flip null directions (what noise alone looks like),
and, as an integrity check, recomputes the full-sample direction and compares it with the one saved earlier. The check
must read ~1.000; if it does not, the prompts or cases differ from the original run and the reliability numbers do NOT
describe the saved direction.

Contrasts (--kind):
  bias_esi    : E+ minus E- on ESI prompts, as Phase 2 built it. Use the SAME --n-pairs as that run (30 for ses, 100 for race/sex).
  def_usmle   : clean minus pressured on USMLE prompts, as Phase 3 built it. Use the SAME flags as the queue
                (--n-pairs 100 and --min-clean-p-correct 0.8 for Llama, 0.5 for Mistral).

Output: paper2_results/reliability/<kind>__<model>__<key>.npz  (read by run_phase4_grid.py)

    python run_reliability.py --model Mistral-7B-Instruct-v0.1 --kind bias_esi --attribute race --n-pairs 100
    python run_reliability.py --model Llama-3.1-8B-Instruct --kind def_usmle --channel assistant --min-clean-p-correct 0.8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mech import ace_sensor, chat_prompting, deference_pairs, reliability
from mech.checkpoint import DirectionStore
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset
from shared.inference_engine import USMLE_CLASSES
from run_phase2_bias_sensor import load_matched_triples
import run_phase3_deference_sensor as p3

PHASE2_DIRS = "paper2_results/phase2/directions"
PHASE3_DIRS = "paper2_results/phase3/directions"


def compare_with_saved(store: DirectionStore, model_name: str, attr_key: str, recomputed: np.ndarray):
    """Per-layer cosine between the recomputed directions and the saved ones; None if nothing is saved."""
    saved_layers = store.list_layers(model_name, attr_key)
    if not saved_layers:
        return None
    cos = []
    for l in saved_layers:
        if l < recomputed.shape[0]:
            s = store.load(model_name, attr_key, l)["direction"]
            cos.append(float(np.dot(s, recomputed[l]) / (np.linalg.norm(s) * np.linalg.norm(recomputed[l]))))
    return np.array(cos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    ap.add_argument("--kind", required=True, choices=["bias_esi", "def_usmle"])
    ap.add_argument("--attribute", choices=["ses", "race", "sex"], help="bias_esi only")
    ap.add_argument("--pressure", default="consensus", choices=list(deference_pairs.PRESSURES), help="def_usmle only")
    ap.add_argument("--channel", choices=["user", "assistant"], help="def_usmle only")
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--min-clean-p-correct", type=float, default=0.8, help="def_usmle only: same value the Phase 3 run used")
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()
    if args.kind == "bias_esi" and not args.attribute:
        ap.error("--kind bias_esi needs --attribute")
    if args.kind == "def_usmle" and not args.channel:
        ap.error("--kind def_usmle needs --channel")

    spec = MECH_MODELS_BY_NAME[args.model]
    if args.kind == "bias_esi":
        triples = load_matched_triples(args.source_jsonl, attribute=args.attribute, model=None, signal_type="PRX")
        case_ids = sorted(triples)[: args.n_pairs]
        if len(case_ids) < 10:
            print(f"ERROR: only {len(case_ids)} triples", file=sys.stderr)
            return 1
        print(f"bias_esi / {args.attribute}: {len(case_ids)} E+/E- pairs. Loading {spec.display_name}...")
        model = ace_sensor.load_model(spec, device=args.device)

        def esi_tokens(note):
            body = chat_prompting.ESI_BODY_TEMPLATE.format(validated_note=note)
            return chat_prompting.tokenize_chat_prompt(model, chat_prompting.build_user_role_messages(body), chat_prompting.ESI_SUFFIX)

        pos_tokens = [esi_tokens(triples[c]["E+"]) for c in case_ids]
        neg_tokens = [esi_tokens(triples[c]["E-"]) for c in case_ids]
        store, saved_key, key = DirectionStore(PHASE2_DIRS), args.attribute, args.attribute
    else:
        case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
        pairs, _ = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
        print(f"def_usmle / {args.pressure}/{args.channel}: {len(pairs)} candidate pairs. Loading {spec.display_name}...")
        model = ace_sensor.load_model(spec, device=args.device)
        ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
        kept, clean_tokens, _, sel = p3.select_cases_model_knows(model, pairs, case_lookup, ids, min_p_correct=args.min_clean_p_correct, n_pairs=args.n_pairs)
        print(f"know-the-answer filter (>= {args.min_clean_p_correct}): kept {len(kept)} of {sel['candidates_tested']} tested")
        if len(kept) < 10:
            print("ERROR: fewer than 10 cases pass the filter.", file=sys.stderr)
            return 1
        case_ids = []
        pos_tokens, neg_tokens = [], []
        for cid in kept:
            p = pairs[cid]
            pt = p3.build_pressured_tokens(model, args.channel, case_lookup[cid], p["N"], p["REC_wrong"], wrong_letter=p["wrong_letter"], wrong_text=p["wrong_text"], pressure=args.pressure)
            if pt is not None:
                case_ids.append(cid)
                pos_tokens.append(clean_tokens[cid])
                neg_tokens.append(pt)
        store, saved_key, key = DirectionStore(PHASE3_DIRS), p3.deference_attribute(args.pressure, args.channel), f"{args.pressure}_{args.channel}"

    print(f"Collecting per-case activations ({len(case_ids)} pairs x 2 prompts, all {spec.n_layers} layers)...")
    pos_all = reliability.collect_activations(model, pos_tokens, spec.n_layers)
    neg_all = reliability.collect_activations(model, neg_tokens, spec.n_layers)

    recomputed = reliability.directions_from_arrays(pos_all, neg_all)
    cos_saved = compare_with_saved(store, spec.display_name, saved_key, recomputed)
    if cos_saved is None:
        print(f"NOTE: no saved direction under {store.root if hasattr(store, 'root') else ''} / {saved_key!r}: integrity check skipped.")
        match = None
    else:
        match = float(cos_saved.mean())
        print(f"INTEGRITY: recomputed vs saved direction, mean cosine {match:.5f} (worst layer {cos_saved.min():.5f}) "
              + ("-- matches" if cos_saved.min() > 0.999 else "-- DOES NOT MATCH: different cases or prompts than the original run; do not use these reliabilities for the saved direction"))

    split = reliability.split_half_cosines(pos_all, neg_all, n_splits=args.n_splits)
    null = reliability.signflip_null_directions(pos_all, neg_all, n_null=args.n_null)
    path = reliability.reliability_path(args.kind, spec.display_name, key)
    reliability.save_reliability(path, split_half=split, null_dirs=null, n_pairs=len(case_ids), matches_saved_mean_cos=match)
    sh = split.mean(axis=0)
    mid = slice(14, 25) if spec.n_layers > 25 else slice(0, spec.n_layers)
    print(f"Split-half reliability: mean cosine {sh.mean():.3f} over all layers, {sh[mid].mean():.3f} over layers "
          f"{'14-24' if spec.n_layers > 25 else 'all (small model)'} (full-sample estimate {float(reliability.spearman_brown(sh[mid].mean())):.3f}); n = {len(case_ids)} pairs")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
