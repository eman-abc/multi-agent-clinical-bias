"""
run_phase2_bias_sensor.py

Phase 2 -- build and validate the bias-direction sensor (Karvonen & Marks'
ACE recipe, arXiv:2506.10922) for one model, one demographic attribute.

DATA SOURCE -- reads notes only, never touches phase0_results/ directly:
by default this reads paper2_results/extracted_paper1_notes.jsonl, the
output of extract_paper1_notes.py (a read-only snapshot of Paper 1's
note_text -- see that script's docstring for the read-only guarantee).
That file has NO `model` field (the notes are model-agnostic -- the same
Agent-A-generated text was read by whichever Agent B models Paper 1
tested), so load_matched_triples below does not filter by model when
reading it.

You can also point --source-jsonl at either of
run_phase1_bridge_replication.py's own outputs
(paper2_results/phase1/experiment_1.jsonl in --mode generate, or
experiment_1_reused.jsonl in --mode reuse) if you want triples specific
to a fresh generation run for this exact model -- both of those DO have a
`model` field (ObservationRecord schema), so load_matched_triples filters
by it when present. Either source works; this script auto-detects which
schema it's reading.

WORKFLOW:
  1. Load N matched (E+, E-, N) note triples for one attribute from an
     existing checkpoint file.
  2. Load the model (TransformerLens).
  3. Extract the ACE direction at every layer (one forward pass per
     example, all layers at once -- see mech/ace_sensor.py).
  4. "Validate before trusting" (Karvonen & Marks' own methodology, see
     mech/ace_sensor.py's module docstring for the exact quote and the
     honest caveat about how this project operationalizes it): for a
     subset of layers and a sweep of steering coefficients, apply
     additive steering to NEUTRAL notes and check whether the ESI
     decision shifts in OPPOSITE directions for +coefficient vs.
     -coefficient -- the causal signature this project is treating as
     "the direction is doing something, not just correlating with
     something."
  5. Save: directions (mech/checkpoint.py's DirectionStore), every
     steering-validation observation (SteeringCheckpoint, JSONL,
     resumable), and one summary metrics JSON per (model, attribute) run.

Usage:
    python run_phase2_bias_sensor.py --model Mistral-7B-Instruct-v0.3 --attribute ses
    python run_phase2_bias_sensor.py --model Llama-3.1-8B-Instruct --attribute ses --layers 8,12,16,20,24 --coefficients=-8,-4,4,8

NOTE on --coefficients: always use the `--coefficients=VALUE` form (with
the `=`), not `--coefficients VALUE`, whenever the first value is
negative. argparse can't distinguish a bare "-8,-4,4,8" token from an
unrecognized flag unless the value is a pure negative number on its own
("-8" alone would be fine; "-8,-4,4,8" is not, because of the comma) --
without the `=`, argparse refuses to consume it as this option's value
and raises "expected one argument" instead of a parse error that
explains why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from mech import ace_sensor, chat_prompting
from mech.checkpoint import DirectionStore, SteeringCheckpoint, SteeringObservation, export_metrics_json
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.inference_engine import ESI_CLASSES
from shared.metrics import expected_ordinal_value

DIRECTIONS_DIR = "paper2_results/phase2/directions"
STEERING_CHECKPOINT = "paper2_results/phase2/steering_validation.jsonl"


def load_matched_triples(jsonl_path: str | Path, *, attribute: str, model: str | None, signal_type: str = "PRX") -> dict[str, dict[str, str]]:
    """
    Returns {case_id: {"E+": note_text, "E-": note_text, "N": note_text}},
    keeping only case_ids where all three are present.

    `signal_type` defaults to "PRX" (the administrative-carrier condition
    Phase 0's own design calls "the most isolatable single phrase" --
    cleanest, least-confounded carrier form, good for a first extraction).
    Pass "EXP" or "IMP" to test whether a direction extracted/validated on
    one carrier form generalizes to another -- Karvonen & Marks' own
    central finding is that bias survives realistic, non-explicit framing,
    so a direction that only works for PRX and silently fails for IMP
    would be a real, worth-knowing gap, not just a parameter sweep.

    Schema-flexible (see module docstring): if a record has a `model`
    field, it's filtered against the `model` argument (Phase 1's own
    ObservationRecord-schema outputs); if not (extract_paper1_notes.py's
    output, which is deliberately model-agnostic), no model filtering is
    applied -- pass model=None explicitly when reading that file.
    """
    by_case: dict[str, dict[str, str]] = {}
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run extract_paper1_notes.py first (for the "
            f"default source), or run_phase1_bridge_replication.py for "
            f"model={model!r} (for a model-specific source)."
        )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            has_model_field = "model" in r
            model_ok = (r.get("model") == model) if (has_model_field and model is not None) else True
            if (
                r.get("attribute") == attribute
                and model_ok
                and r.get("signal_type") in (signal_type, None)
                and r.get("condition_label") in ("E+", "E-", "N")
                and r.get("note_text")
                and not r.get("blocked_reason")
            ):
                by_case.setdefault(r["case_id"], {})[r["condition_label"]] = r["note_text"]

    complete = {cid: notes for cid, notes in by_case.items() if all(k in notes for k in ("E+", "E-", "N"))}
    return complete


def run_clamp_effectiveness_test(model, *, results, case_ids: list[str], positive_tokens: list[torch.Tensor], negative_tokens: list[torch.Tensor], layers: list[int], class_token_ids: dict[str, int]) -> dict:
    """
    THE FIX for a real gap: ace_sensor.ace_edit_transform (Karvonen & Marks'
    actual clamp equation) was implemented from the start but never called
    anywhere -- only additive_steering_transform was used, which validates
    causality but doesn't test the paper's own actual claim (does the clamp
    reduce measured bias). This runs that missing experiment: for each
    tested layer, apply the ACE clamp to E+ and E- notes, read the ESI
    decision, and compute a CFR-style agreement rate the same way
    shared/metrics.py's compute_cfr does for the API models -- "clamped
    CFR" should move toward 1.0 (E+ and E- decisions agree) if the clamp
    is doing what Karvonen & Marks report, compared to the UNCLAMPED CFR
    at the same layer as the baseline to beat.

    `positive_tokens`/`negative_tokens`: already chat-templated (see
    mech/chat_prompting.py) -- this function only reads logits, it never
    tokenizes anything itself.
    """
    from mech.ace_sensor import make_hook, ace_edit_transform, steered_class_prediction

    out = {}
    for layer in layers:
        direction = results[layer].direction
        bias_scalar = results[layer].bias_scalar

        def _clamped_prediction(tokens):
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = make_hook(direction, ace_edit_transform, coefficient=1.0, bias_scalar=bias_scalar)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
            next_token_logits = logits[0, -1, :]
            # THE FIX, same bug as ace_sensor.steered_class_prediction had: use
            # precomputed in-context token ids, not a per-call isolated re-tokenization.
            class_logits = {cls: float(next_token_logits[tid].item()) for cls, tid in class_token_ids.items()}
            max_logit = max(class_logits.values())
            exp_vals = {c: float(np.exp(v - max_logit)) for c, v in class_logits.items()}
            total = sum(exp_vals.values())
            return max(exp_vals, key=lambda c: exp_vals[c] / total)

        def _unclamped_prediction(tokens):
            top_class, _ = steered_class_prediction(model, tokens, class_token_ids, layer=layer, direction=direction, coefficient=0.0)
            return top_class

        n_agree_clamped = n_agree_unclamped = 0
        for cid, pos_tok, neg_tok in zip(case_ids, positive_tokens, negative_tokens):
            if _clamped_prediction(pos_tok) == _clamped_prediction(neg_tok):
                n_agree_clamped += 1
            if _unclamped_prediction(pos_tok) == _unclamped_prediction(neg_tok):
                n_agree_unclamped += 1

        out[layer] = {
            "clamped_cfr": n_agree_clamped / len(case_ids),
            "unclamped_cfr": n_agree_unclamped / len(case_ids),
            "n_cases": len(case_ids),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    parser.add_argument("--attribute", required=True, choices=["ses", "race", "sex"])
    parser.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl", help="Default: extract_paper1_notes.py's output (model-agnostic). Or point at Phase 1's own output for a model-specific source.")
    parser.add_argument("--signal-type", default="PRX", choices=["PRX", "EXP", "IMP"], help="Which carrier form to extract/validate on. Default PRX (cleanest). Try EXP/IMP to test whether the direction generalizes across framing -- Karvonen & Marks' own central finding is that bias survives realistic, non-explicit framing.")
    parser.add_argument("--n-pairs", type=int, default=30, help="Max matched triples to use for extraction (more = better estimate, more GPU time).")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices to run steering validation on. Default: EVERY layer (was every-4th when compute was the constraint; with real GPU access, a full per-layer sweep is cheap relative to extraction and gives an actual localization curve instead of a sparse sample).")
    parser.add_argument("--test-clamp", action="store_true", help="THE FIX: also run the actual ACE clamp-effectiveness test (Karvonen & Marks' own success metric -- does clamping raise CFR toward 1.0) at the layers passing the opposite-sign check. This was previously never wired up even though ace_edit_transform existed.")
    parser.add_argument("--coefficients", default="-8,-4,4,8", help="Comma-separated additive-steering coefficients to sweep (both signs needed for the opposite-direction validation test).")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"], help="Default: auto-detect (cuda if available, else cpu). Force 'cpu' for a local dry run (e.g. --model gpt2) even on a machine that also has a GPU.")
    args = parser.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    if not spec.primary_papers or spec.primary_papers == ("None -- Used exclusively for local dry runs",):
        print(f"NOTE: {args.model!r} is a dry-run/smoke-test model, not one of Paper 2's actual targets. "
              f"Extracted directions and steering results are for validating the PIPELINE MECHANICS only "
              f"(shapes, file I/O, no crashes) -- they carry no scientific claim about bias/deference for "
              f"this model and shouldn't be reported as a Paper 2 finding.")

    print(f"Loading matched (E+,E-,N) triples for attribute={args.attribute!r}, signal_type={args.signal_type!r}, model={args.model!r} from {args.source_jsonl}...")
    triples = load_matched_triples(args.source_jsonl, attribute=args.attribute, model=args.model, signal_type=args.signal_type)
    if len(triples) < 5:
        print(f"ERROR: only {len(triples)} complete triples found -- need Phase 1 run for this model/attribute first (or more cases).", file=sys.stderr)
        return 1
    case_ids = sorted(triples.keys())[: args.n_pairs]
    print(f"Using {len(case_ids)} of {len(triples)} available triples.")

    positive_texts = [triples[cid]["E+"] for cid in case_ids]
    negative_texts = [triples[cid]["E-"] for cid in case_ids]
    neutral_texts = {cid: triples[cid]["N"] for cid in case_ids}

    print(f"Loading model {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)

    # THE FIX: build real chat-templated tokens (see mech/chat_prompting.py)
    # instead of tokenizing the flat ESI prompt string directly -- every
    # extraction/validation prompt below goes through this, not model.to_tokens
    # on a raw string. Single user-turn (no channel variable here -- see
    # run_phase3_deference_sensor.py for where channel framing actually applies).
    # THE FIX: resolve class tokens ONCE, in the real context after the suffix --
    # not per-call isolated re-tokenization (see chat_prompting.resolve_class_token_ids
    # and ace_sensor.steered_class_prediction's docstrings; confirmed broken on Mistral this session).
    class_token_ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.ESI_SUFFIX, ESI_CLASSES)
    print(f"Resolved ESI class tokens in context: {class_token_ids}")

    def _esi_tokens(note_text: str) -> torch.Tensor:
        body = chat_prompting.ESI_BODY_TEMPLATE.format(validated_note=note_text)
        messages = chat_prompting.build_user_role_messages(body)
        return chat_prompting.tokenize_chat_prompt(model, messages, chat_prompting.ESI_SUFFIX)

    positive_tokens = [_esi_tokens(t) for t in positive_texts]
    negative_tokens = [_esi_tokens(t) for t in negative_texts]
    neutral_tokens = {cid: _esi_tokens(t) for cid, t in neutral_texts.items()}

    # --- Extraction (Karvonen & Marks' ACE recipe, every layer, one pass per example) ---
    print(f"Extracting ACE directions at all {spec.n_layers} layers ({len(case_ids)} examples/group, one forward pass each)...")
    results = ace_sensor.extract_directions_all_layers(
        model, positive_tokens=positive_tokens, negative_tokens=negative_tokens, n_layers=spec.n_layers,
    )
    direction_store = DirectionStore(DIRECTIONS_DIR)
    for layer, res in enumerate(results):
        direction_store.save(
            model=spec.display_name, attribute=args.attribute, layer=layer,
            direction=res.direction, mean_pos=res.mean_pos, mean_neg=res.mean_neg, std=res.std,
            bias_scalar=res.bias_scalar, n_pairs=res.n_pairs, extraction_point=res.extraction_point,
        )
    print(f"Saved {len(results)} layer directions to {DIRECTIONS_DIR}/")

    # --- Causal validation ("steer before trusting") ---
    validation_layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(spec.n_layers))
    coefficients = [float(x) for x in args.coefficients.split(",")]
    print(f"Validating via additive steering at layers={validation_layers}, coefficients={coefficients}...")

    steering_ckpt = SteeringCheckpoint(STEERING_CHECKPOINT)
    layer_pass_count = 0

    # Unsteered baseline (coefficient=0) is a no-op regardless of which
    # layer's hook is nominally attached, so it's computed ONCE per case
    # here, not once per (layer, case) inside the loop below -- an earlier
    # draft of this script recomputed it per layer, which is correct but
    # wastes exactly as many forward passes as there are validation layers.
    print("Computing unsteered baseline (once per case, reused across every validation layer)...")
    baseline_by_case: dict[str, float] = {}
    for cid in case_ids:
        _, base_probs = ace_sensor.steered_class_prediction(model, neutral_tokens[cid], class_token_ids, layer=validation_layers[0], direction=results[validation_layers[0]].direction, coefficient=0.0)
        baseline_by_case[cid] = expected_ordinal_value(base_probs)

    for layer in validation_layers:
        direction = results[layer].direction
        shifts_by_coef: dict[float, list[float]] = {c: [] for c in coefficients}

        for cid in case_ids:
            tokens = neutral_tokens[cid]
            base_ev = baseline_by_case[cid]

            for coef in coefficients:
                obs_key_task = "esi"
                obs = SteeringObservation(
                    model=spec.display_name, attribute=args.attribute, layer=layer, coefficient=coef,
                    case_id=cid, task=obs_key_task, condition_source="N",
                    top_class=None, class_probabilities=None, raw_text=None,
                )
                if steering_ckpt.has(obs):
                    continue
                top_class, probs = ace_sensor.steered_class_prediction(model, tokens, class_token_ids, layer=layer, direction=direction, coefficient=coef)
                ev = expected_ordinal_value(probs)
                if ev is not None and base_ev is not None:
                    shifts_by_coef[coef].append(ev - base_ev)
                obs.top_class = top_class
                obs.class_probabilities = probs
                steering_ckpt.append(obs)

        mean_shift = {c: (sum(v) / len(v) if v else None) for c, v in shifts_by_coef.items()}
        pos_coefs = [c for c in coefficients if c > 0]
        neg_coefs = [c for c in coefficients if c < 0]
        pos_shift = next((mean_shift[c] for c in pos_coefs if mean_shift[c] is not None), None)
        neg_shift = next((mean_shift[c] for c in neg_coefs if mean_shift[c] is not None), None)
        opposite_signs = pos_shift is not None and neg_shift is not None and (pos_shift > 0) != (neg_shift > 0) and pos_shift != 0 and neg_shift != 0
        print(f"  layer={layer:2d}  mean ESI shift by coefficient: {mean_shift}  -> opposite-sign pass: {opposite_signs}")
        if opposite_signs:
            layer_pass_count += 1

    clamp_results = None
    if args.test_clamp:
        # Tests clamp effectiveness at every validated layer (not just the
        # ones that passed the opposite-sign check) -- deliberately: a
        # layer failing the steering test could still show a clamp effect,
        # since the clamp (project-to-midpoint) and additive steering
        # (push-by-a-fixed-amount) are different operations on the same
        # direction, and it's worth knowing if they disagree.
        print(f"\n--test-clamp set: running the actual ACE clamp-effectiveness test at all {len(validation_layers)} validated layers...")
        clamp_results = run_clamp_effectiveness_test(
            model, results=results, case_ids=case_ids, positive_tokens=positive_tokens,
            negative_tokens=negative_tokens, layers=validation_layers, class_token_ids=class_token_ids,
        )
        for layer, r in clamp_results.items():
            print(f"  layer={layer:2d}  unclamped CFR={r['unclamped_cfr']:.2f}  clamped CFR={r['clamped_cfr']:.2f}  (n={r['n_cases']})")

    metrics = {
        "model": spec.display_name,
        "attribute": args.attribute,
        "signal_type": args.signal_type,
        "n_pairs_used": len(case_ids),
        "validation_layers": validation_layers,
        "coefficients": coefficients,
        "layers_passing_opposite_sign_test": layer_pass_count,
        "layers_tested": len(validation_layers),
        "clamp_effectiveness": clamp_results,
        "note": (
            "opposite-sign pass = mean ESI expected-ordinal-value shift under "
            "positive steering and negative steering have opposite signs, "
            "relative to this model's own unsteered neutral baseline. This is "
            "this project's own causal-validation criterion (design choice, "
            "see mech/ace_sensor.py module docstring), not a threshold "
            "specified by Karvonen & Marks."
        ),
    }
    export_metrics_json(metrics, f"paper2_results/phase2/metrics/phase2_metrics_{spec.display_name}_{args.attribute}.json")
    print(f"\nSummary: {layer_pass_count}/{len(validation_layers)} tested layers pass the opposite-sign causal check.")
    print(f"Full metrics: paper2_results/phase2/metrics/phase2_metrics_{spec.display_name}_{args.attribute}.json")
    print(f"Every steering observation: {STEERING_CHECKPOINT}")
    print(f"Directions: {DIRECTIONS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())