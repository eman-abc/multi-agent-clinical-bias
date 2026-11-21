"""
run_phase3_localization_patching.py

The activation-patching localization study -- Kumarappan & Mujoo's OWN
method (patch clean activations into a corrupted run, measure restoration),
not the mean-difference extraction run_phase3_deference_sensor.py does.
Separate script, separate question: that script asks "what direction is
the deference signal," this one asks "where in the network does patching
it back in matter most" -- see mech/patching.py's module docstring for why
these are kept as two different techniques instead of one.

CLEAN = N (neutral, the model's own unperturbed judgment).
CORRUPTED = REC_wrong (external pressure toward a stated wrong answer),
built per --channel -- see run_phase3_deference_sensor.py's
build_clean_tokens/build_pressured_tokens, reused here directly so both
scripts build channel-framed prompts identically rather than maintaining
two separate implementations of the same construction.

COARSE-THEN-FINE STAGING, so this doesn't become an unbounded GPU bill:
  Stage 1 (default): every layer, resid_post only, small n (15-20) --
    cheap, gives a localization curve fast.
  Stage 2 (--fine-layers): attn_out + mlp_out split, ONLY for the layers
    you specify (typically the top few from stage 1) -- this is what
    actually tests Kumarappan & Mujoo's attention-vs-MLP claim, but it's
    2x the forward passes of stage 1 per layer, so only run it where
    stage 1 already points somewhere.

Usage:
    python run_phase3_localization_patching.py --model Mistral-7B-Instruct-v0.3 --channel user
    python run_phase3_localization_patching.py --model Llama-3.1-8B-Instruct --channel assistant --fine-layers 14,15,16,17,18
"""

from __future__ import annotations

import argparse
import sys

from mech import ace_sensor, chat_prompting, deference_pairs, patching
from mech.checkpoint import export_metrics_json
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset, AMQARecord
from shared.inference_engine import USMLE_CLASSES
from run_phase3_deference_sensor import (build_clean_tokens, build_pressured_tokens, select_cases_model_knows,
                                         behavioral_summary, print_behavioral)



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    parser.add_argument("--channel", required=True, choices=["user", "assistant"], help="Same channel framing as run_phase3_deference_sensor.py -- which role carries the stated wrong recommendation.")
    parser.add_argument("--pressure", default="consensus", choices=list(deference_pairs.PRESSURES), help="Same pressure type as run_phase3_deference_sensor.py (default consensus).")
    parser.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    parser.add_argument("--n-pairs", type=int, default=40, help="Maximum number of cases to KEEP after the know-the-answer filter. Localization is a SCAN (one pass per layer per case), so it is cheap even at 40-100.")
    parser.add_argument("--min-clean-p-correct", type=float, default=0.8, help="Same know-the-answer filter as run_phase3_deference_sensor.py (Kumarappan & Mujoo Section 4.1). 0 disables it.")
    parser.add_argument("--fine-layers", default=None, help="Comma-separated layers to also run the attn_out/mlp_out split on (stage 2). Default: skip stage 2.")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = parser.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]

    # Cheap, fail-fast work first (no GPU): AMQA lookup + the SAME candidate pairs run_phase3_deference_sensor.py uses
    # (same builder, same deterministic ordering and wrong-answer choice -> the two scripts see identical stimuli).
    print("Loading AMQA dataset for USMLE question/options lookup...")
    amqa_records = load_amqa_dataset(dry_run=False)
    case_lookup: dict[str, AMQARecord] = {r.question_id: r for r in amqa_records}
    pairs, pair_stats = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
    print(f"Candidate pairs: {pair_stats}")
    if len(pairs) < 5:
        print(f"ERROR: only {len(pairs)} usable pairs -- see the stats line above.", file=sys.stderr)
        return 1

    print(f"Loading model {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)

    class_token_ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    print(f"Resolved USMLE class tokens in context: {class_token_ids}")

    kept, clean_tokens, clean_probs, sel_stats = select_cases_model_knows(
        model, pairs, case_lookup, class_token_ids, min_p_correct=args.min_clean_p_correct, n_pairs=args.n_pairs,
    )
    print(f"Know-the-answer filter (clean P(correct) >= {args.min_clean_p_correct}): kept {len(kept)} of {sel_stats['candidates_tested']} candidates tested.")
    if len(kept) < 5:
        print("ERROR: fewer than 5 cases pass the filter. Lower --min-clean-p-correct (e.g. 0.5).", file=sys.stderr)
        return 1

    print(f"Building chat-templated ({args.channel}-channel) pressured prompts...")
    corrupted_tokens: dict = {}
    corrupted_probs: dict = {}
    n_failed = 0
    for cid in kept:
        cr, pr = case_lookup[cid], pairs[cid]
        ct = build_pressured_tokens(model, args.channel, cr, pr["N"], pr["REC_wrong"], wrong_letter=pr["wrong_letter"], wrong_text=pr["wrong_text"], pressure=args.pressure)
        if ct is None:
            n_failed += 1
            continue
        corrupted_tokens[cid] = ct
        corrupted_probs[cid] = ace_sensor.plain_class_probs(model, ct, class_token_ids)
    valid = [c for c in kept if c in corrupted_tokens]
    clean_tokens = {c: clean_tokens[c] for c in valid}
    if n_failed:
        print(f"NOTE: {n_failed} cases skipped -- could not build the pressured prompt.")
    behavioral = behavioral_summary(valid, pairs, clean_probs, corrupted_probs)
    print_behavioral(behavioral, f"{args.pressure}/{args.channel}")

    # K&M's metric: P(correct) -- the clean-to-pressured gap in P(correct) is what patching tries to close.
    def p_correct(probs, cid):
        return probs.get(pairs[cid]["correct_letter"])

    print(f"\nSTAGE 1 -- coarse sweep, all {spec.n_layers} layers, resid_post only, n={len(valid)}, metric = P(correct)...")
    stage1 = patching.run_localization_sweep(
        model, clean_tokens=clean_tokens, corrupted_tokens=corrupted_tokens,
        class_token_ids=class_token_ids, ev_fn=p_correct, n_layers=spec.n_layers, components=("resid_post",),
    )
    gap = None
    if stage1 and stage1[0].clean_ev is not None and stage1[0].corrupted_ev is not None:
        gap = stage1[0].clean_ev - stage1[0].corrupted_ev
        print(f"  clean P(correct) {stage1[0].clean_ev:.3f}   pressured P(correct) {stage1[0].corrupted_ev:.3f}   gap {gap:.3f}")
        if gap < 0.10:
            print("  WARNING: the gap is under 0.10 -- the pressure barely lowers P(correct) here, so restoration fractions below "
                  "are dividing by a tiny number and are not meaningful. Treat this channel's localization as uninformative.")
    for r in stage1:
        flag = ""
        if r.restoration_fraction is not None and r.restoration_fraction >= 0.5:
            flag = "  <-- candidate: restores >=50% of the clean/pressured gap"
        print(f"  layer={r.layer:2d} [{r.component}]  restoration={r.restoration_fraction!r}  patched P(correct)={r.patched_ev!r}{flag}")

    results_export = {
        "metric": "P(correct)", "pressure": args.pressure, "channel": args.channel, "n_cases": len(valid), "cases": valid,
        "selection": sel_stats, "behavioral": behavioral, "clean_minus_pressured_gap": gap,
        "stimuli": "plain Experiment-1 N notes + ONE appended stated-wrong-recommendation sentence (mech/deference_pairs.py)",
        "stage1_resid_post": [r.__dict__ for r in stage1],
    }

    if args.fine_layers:
        fine_layers = [int(x) for x in args.fine_layers.split(",")]
        print(f"\nSTAGE 2 -- attn_out/mlp_out split at layers {fine_layers}...")
        stage2 = patching.run_localization_sweep(
            model, clean_tokens=clean_tokens, corrupted_tokens=corrupted_tokens,
            class_token_ids=class_token_ids, ev_fn=p_correct, n_layers=spec.n_layers,
            components=("attn_out", "mlp_out"), layers=fine_layers,
        )
        for r in stage2:
            print(f"  layer={r.layer:2d} [{r.component}]  restoration={r.restoration_fraction!r}")
        results_export["stage2_attn_mlp_split"] = [r.__dict__ for r in stage2]

    out_path = f"paper2_results/phase3/metrics/localization_{spec.display_name}_{args.pressure}_{args.channel}.json"
    export_metrics_json(results_export, out_path)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())