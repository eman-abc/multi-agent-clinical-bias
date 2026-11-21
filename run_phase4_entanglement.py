"""
run_phase4_entanglement.py

Phase 4 -- the actual test this whole project is structured around: are the
bias direction (Phase 2, demographic E+/E- contrast) and the deference
direction (Phase 3, N/REC_wrong contrast) the same geometric object, or
genuinely separate ones? Computed as cosine similarity between the two
directions, layer by layer, for one model/attribute pair.

REQUIRES: both Phase 2 (run_phase2_bias_sensor.py, for the given
--attribute) and Phase 3 (run_phase3_deference_sensor.py) to have already
been run for the same --model -- this script only reads
paper2_results/phase2/directions/ and paper2_results/phase3/directions/,
it never re-extracts anything.

SAME RECIPE FOR BOTH DIRECTIONS IS WHAT MAKES THIS VALID: both were
extracted via the identical whitened-mean-difference method
(mech/ace_sensor.py), not two different techniques -- see
run_phase3_deference_sensor.py's module docstring for why that matters.
A high or low cosine similarity here is evidence about the underlying
geometry, not an artifact of comparing apples to oranges.

READING THE RESULT (Dual-Stance Evaluation of Sycophancy, arXiv:2606.11205,
is the closest precedent for what a "confirmed entangled" finding would
look like -- a steering direction meant to separate two concepts turned
out not to):
  |cosine| close to 1 at some layer -> the two directions point the same
    way there -- entangled at that layer, the live alternative hypothesis.
  |cosine| close to 0 at some layer -> genuinely separate signals at that
    layer -- the outcome the project's research question is actually
    testing for.
  No threshold is specified by any paper this project draws on -- what
  counts as "close to 1" vs "close to 0" is a judgment call to make
  explicit when you report this, not a number this script decides for you.

Usage:
    python run_phase4_entanglement.py --model Llama-3.1-8B-Instruct --attribute ses
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from mech.checkpoint import DirectionStore, export_metrics_json
from mech.model_registry import MECH_MODELS_BY_NAME
from run_phase3_deference_sensor import deference_attribute  # the one function that builds the Phase 3 direction key from (pressure, channel)

PHASE2_DIRECTIONS_DIR = "paper2_results/phase2/directions"
PHASE3_DIRECTIONS_DIR = "paper2_results/phase3/directions"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    parser.add_argument("--attribute", required=True, choices=["ses", "race", "sex"], help="Which Phase 2 bias direction to compare against Phase 3's deference direction.")
    parser.add_argument("--pressure", default="consensus", choices=["single", "consensus"], help="Which Phase 3 pressure type's deference direction to use (must match what run_phase3_deference_sensor.py was run with; default consensus).")
    parser.add_argument("--channel", required=True, choices=["user", "assistant"], help="Which Phase 3 channel's deference direction to use -- Phase 3 now saves one direction set per channel (paper2_results/phase3/directions/, attribute='deference_user' or 'deference_assistant'), since user-role and assistant-role pressure are a priori different signals (Kumarappan & Mujoo's own channel-framing finding). Run Phase 4 once per channel if you want both; don't average across them.")
    args = parser.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    bias_store = DirectionStore(PHASE2_DIRECTIONS_DIR)
    deference_store = DirectionStore(PHASE3_DIRECTIONS_DIR)
    deference_key = deference_attribute(args.pressure, args.channel)

    bias_layers = set(bias_store.list_layers(spec.display_name, args.attribute))
    deference_layers = set(deference_store.list_layers(spec.display_name, deference_key))
    common_layers = sorted(bias_layers & deference_layers)

    if not common_layers:
        print(
            f"ERROR: no layers found with BOTH a bias direction (attribute={args.attribute!r}) "
            f"and a deference direction (attribute={deference_key!r}) for model={args.model!r}. "
            f"Run run_phase2_bias_sensor.py and run_phase3_deference_sensor.py --pressure {args.pressure} --channel {args.channel} "
            f"for this model first.",
            file=sys.stderr,
        )
        return 1

    missing_bias = sorted(deference_layers - bias_layers)
    missing_deference = sorted(bias_layers - deference_layers)
    if missing_bias:
        print(f"NOTE: {len(missing_bias)} layers have a deference direction but no bias direction (layers {missing_bias}) -- skipped.")
    if missing_deference:
        print(f"NOTE: {len(missing_deference)} layers have a bias direction but no deference direction (layers {missing_deference}) -- skipped.")

    print(f"Comparing bias ({args.attribute}) vs. deference ({deference_key}) directions at {len(common_layers)} common layers for {spec.display_name}...\n")

    per_layer = []
    for layer in common_layers:
        bias_d = bias_store.load(spec.display_name, args.attribute, layer)
        deference_d = deference_store.load(spec.display_name, deference_key, layer)
        sim = cosine_similarity(bias_d["direction"], deference_d["direction"])
        per_layer.append({"layer": layer, "cosine_similarity": sim})
        print(f"  layer={layer:2d}  cosine(bias, deference) = {sim:+.3f}")

    sims = [r["cosine_similarity"] for r in per_layer if not np.isnan(r["cosine_similarity"])]
    summary = {
        "model": spec.display_name,
        "attribute": args.attribute,
        "deference_pressure": args.pressure,
        "deference_channel": args.channel,
        "n_common_layers": len(common_layers),
        "per_layer": per_layer,
        "mean_abs_cosine_similarity": float(np.mean(np.abs(sims))) if sims else None,
        "max_abs_cosine_similarity": float(np.max(np.abs(sims))) if sims else None,
        "layer_of_max_abs_similarity": common_layers[int(np.argmax(np.abs(sims)))] if sims else None,
        "note": (
            "No entanglement/orthogonality threshold is specified by any paper this "
            "project draws on. Report the per-layer curve and state your own judgment "
            "call explicitly rather than citing a pass/fail number this script invented."
        ),
    }
    out_path = f"paper2_results/phase4/entanglement_{spec.display_name}_{args.attribute}_{args.pressure}_{args.channel}.json"
    export_metrics_json(summary, out_path)
    print(f"\nMean |cosine similarity| across common layers: {summary['mean_abs_cosine_similarity']}")
    print(f"Max |cosine similarity|: {summary['max_abs_cosine_similarity']} at layer {summary['layer_of_max_abs_similarity']}")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())