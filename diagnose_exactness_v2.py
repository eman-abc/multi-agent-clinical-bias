#!/usr/bin/env python3
"""
diagnose_exactness.py -- DIAGNOSTIC ONLY (not part of the confirmatory analysis; it reads no restoration results).

The four attention-vs-MLP runs failed their built-in exactness check by 1.8e-3 to 2.6e-3 against a tolerance of 1e-3.
Patching attention and MLP at every layer must reproduce the clean answer. Two explanations:

  (A) fp16 rounding. The patched pressured run and the clean run feed the SAME final vector into the output layer, but
      with different sequence lengths, and the GPU may pick a different matrix-multiply order for each length. An fp16
      logit near 20 can only move in steps of 0.0156, so one step changes a probability by up to about 0.004.
  (B) a real patching error (a hook on the wrong layer or position). That would give differences of tens of steps, and
      would not appear when the prompt is patched with its OWN activations.

This script separates them with three measurements per case, all on the class-token logits:
  1. SELF-PATCH   patch the clean activations into the clean prompt itself (same shape). Must be exactly 0.
  2. CROSS-PATCH  patch them into the pressured prompt (what the failing check does).
  3. SHAPE CONTROL  the clean prompt with NO patching, run inside a right-padded batch with the pressured prompt. Causal
                  attention means padding after the prompt cannot change it, so any difference is pure shape-dependent
                  numerical noise. If it is as large as the cross-patch difference, (A) is established.
Differences are shown in logits, in fp16 steps (difference / spacing of that logit) and in probability.

    python diagnose_exactness.py --model Mistral-7B-Instruct-v0.1 --channel user
    python diagnose_exactness.py --model Llama-3.1-8B-Instruct --channel user
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from mech import patching
from mech import plan as P
from mech.note_quality import looks_like_refusal

STEPS_OK = 4.0          # differences up to a handful of fp16 steps are rounding-level


def spacing(x: float, dtype) -> float:
    """Distance from |x| to the next representable number in `dtype` (float16 or float32)."""
    return float(np.spacing(np.array(abs(x), dtype=dtype)))


def softmax(v):
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def class_logits(row, ids):
    return np.array([float(row[t]) for t in ids.values()], dtype=np.float64)


def patched_logits(model, tokens, acts, layers, comps):
    hooks = [(f"blocks.{l}.hook_{c}", patching._patch_hook_factory(acts[c][l])) for l in layers for c in comps]
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1]


def padded_batch_logits(model, clean_tokens, pressured_tokens):
    n1, n2 = clean_tokens.shape[1], pressured_tokens.shape[1]
    batch = torch.zeros((2, max(n1, n2)), dtype=clean_tokens.dtype, device=clean_tokens.device)
    batch[0, :n1], batch[1, :n2] = clean_tokens[0], pressured_tokens[0]
    with torch.no_grad():
        return model(batch)[0, n1 - 1]


def measure(model, clean_tokens, pressured_tokens, ids, n_layers, correct_letters, layers=None, ref_clean_pc=None):
    """
    Per case: |dlogit|, |dP(correct)| and the size in fp16 steps for
      self       patch the clean prompt with its own activations (attention + MLP)
      cross      patch the pressured prompt, attention + MLP at the chosen layers
      crossattn  patch the pressured prompt, attention ALONE (the MLPs are recomputed at the pressured length)
      shape      the clean prompt, unpatched, in a right-padded batch (pure shape noise)
    and prod_dP: what the failing check computed, the larger of |P(cross) - ref| and |P(crossattn) - ref| against the
    clean probability the case selection stored (ref_clean_pc), when given.
    """
    layers = list(range(n_layers)) if layers is None else list(layers)
    letters = list(ids)
    tags = ("self", "cross", "crossattn", "shape")
    out = {f"{t}_{k}": [] for t in tags for k in ("dL", "dP", "steps")}
    out["prod_dP"] = []
    dtype = None
    refs = list(ref_clean_pc) if ref_clean_pc is not None else [None] * len(correct_letters)
    for ct, pt, corr, ref in zip(clean_tokens, pressured_tokens, correct_letters, refs):
        acts = patching.capture_last_token_activations(model, ct, n_layers, components=("attn_out", "mlp_out"))
        with torch.no_grad():
            clean_row = model(ct)[0, -1]
        dtype = np.float16 if clean_row.dtype == torch.float16 else np.float32
        lc = class_logits(clean_row, ids)
        pc = softmax(lc)[letters.index(corr)]
        p_of = {}
        for tag, row in (("self", patched_logits(model, ct, acts, layers, ("attn_out", "mlp_out"))),
                         ("cross", patched_logits(model, pt, acts, layers, ("attn_out", "mlp_out"))),
                         ("crossattn", patched_logits(model, pt, acts, layers, ("attn_out",))),
                         ("shape", padded_batch_logits(model, ct, pt))):
            lx = class_logits(row, ids)
            d = np.abs(lx - lc)
            steps = max((d[i] / spacing(max(abs(lx[i]), abs(lc[i])), dtype)) if d[i] > 0 else 0.0 for i in range(len(d)))
            p_of[tag] = float(softmax(lx)[letters.index(corr)])
            out[f"{tag}_dL"].append(float(d.max()))
            out[f"{tag}_dP"].append(abs(p_of[tag] - float(pc)))
            out[f"{tag}_steps"].append(float(steps))
        out["prod_dP"].append(max(abs(p_of["cross"] - ref), abs(p_of["crossattn"] - ref)) if ref is not None else float("nan"))
    res = {k: np.array(v) for k, v in out.items()}
    res["dtype"] = "float16" if dtype is np.float16 else "float32"
    return res


def reading(m):
    """One verdict from the three measurements."""
    if m["self_dL"].max() > 0:
        return "PATCHING MACHINERY IS NOT EXACT: even patching a prompt with its own activations changes the answer. This is a real error; do not interpret any attention-vs-MLP result."
    cross = float(max(m["cross_steps"].max(), m["crossattn_steps"].max() if "crossattn_steps" in m else 0.0))
    shape = float(m["shape_steps"].max())
    if cross == 0:
        return "EXACT: both cross-prompt patches (attention + MLP, and attention alone) reproduce the clean answer bit for bit."
    if cross <= STEPS_OK and shape >= 0.5 * cross:
        return (f"ROUNDING: the cross-patch differs by at most {cross:.1f} fp16 steps, and simply running the clean prompt at a different sequence length "
                f"(no patching at all) differs by up to {shape:.1f}. The discrepancy is numerical noise from the output layer, not a patching error.")
    if cross <= STEPS_OK:
        return (f"PARTLY UNEXPLAINED: the cross-patch differs by at most {cross:.1f} fp16 steps (small), but the shape control only reaches {shape:.1f}. "
                "It is rounding-sized, yet not reproduced by the shape control; report it as such.")
    return f"NOT ROUNDING-LEVEL: the cross-patch differs by up to {cross:.1f} fp16 steps. That is too large for output-layer noise; investigate before using these runs."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--channel", required=True, choices=["user", "assistant"])
    ap.add_argument("--plan", default="analysis_plan.json")
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    from mech import ace_sensor, chat_prompting, deference_pairs
    from mech.model_registry import MECH_MODELS_BY_NAME
    from shared.data_loader import load_amqa_dataset
    from shared.inference_engine import USMLE_CLASSES
    import run_phase3_deference_sensor as p3

    plan = P.load_plan(args.plan)
    spec = MECH_MODELS_BY_NAME[args.model]
    thr = plan["cases"]["min_clean_p_correct"][args.model]
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    pairs, _ = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
    for c in sorted(c for c, p in pairs.items() if looks_like_refusal(p["N"])):
        del pairs[c]
    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    kept, clean_tokens, clean_probs, sel = p3.select_cases_model_knows(model, pairs, case_lookup, ids, min_p_correct=thr, n_pairs=plan["cases"]["n_pairs_cap"])
    cases, pressured = [], {}
    for cid in kept:
        p = pairs[cid]
        pt = p3.build_pressured_tokens(model, args.channel, case_lookup[cid], p["N"], p["REC_wrong"], wrong_letter=p["wrong_letter"], wrong_text=p["wrong_text"], pressure=plan["pressure"])
        if pt is not None:
            pressured[cid] = pt
            cases.append(cid)
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"{len(cases)} cases (same selection as the attention-vs-MLP run); model dtype is used as loaded")
    m = measure(model, [clean_tokens[c] for c in cases], [pressured[c] for c in cases], ids, spec.n_layers, [pairs[c]["correct_letter"] for c in cases],
                ref_clean_pc=[clean_probs[c][pairs[c]["correct_letter"]] for c in cases])

    n = len(cases)
    print(f"\nDIAGNOSTIC for {spec.display_name}, {args.channel} channel, {n} cases, logits in {m['dtype']}\n")
    print(f"  {'measurement':36s} {'max |dlogit|':>13s} {'max fp16 steps':>15s} {'max |dP(correct)|':>18s} {'cases differing':>16s}")
    for tag, label in (("self", "1. self-patch (same prompt)"), ("cross", "2a. cross-patch, attention + MLP"), ("crossattn", "2b. cross-patch, attention alone"),
                       ("shape", "3. shape control (no patching)")):
        print(f"  {label:36s} {m[tag + '_dL'].max():13.5f} {m[tag + '_steps'].max():15.1f} {m[tag + '_dP'].max():18.2e} {int((m[tag + '_dL'] > 0).sum()):10d} of {n}")
    over = int((m["prod_dP"] > 1e-3).sum())
    print(f"\n  The failing check took the larger of 2a and 2b against the clean probability stored at case selection:"
          f"\n    that quantity here: max {np.nanmax(m['prod_dP']):.2e}, above the 1e-3 tolerance in {over} of {n} cases (the failed run reported 1.8e-3 to 2.6e-3)")
    print(f"\nREADING: {reading(m)}")
    if args.json_out:
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in m.items()} | {"model": args.model, "channel": args.channel, "cases": cases, "reading": reading(m)}, open(args.json_out, "w"))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
