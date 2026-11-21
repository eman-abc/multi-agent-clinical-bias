#!/usr/bin/env python3
"""
collect_interaction.py -- GPU. One forward pass per UNIQUE prompt of the interaction run; saves the probabilities of the four options and the final-token residual
states. It prints no contrast statistic: what the numbers say is decided by analyze_interaction.py under its own frozen rules.

    python collect_interaction.py --model Mistral-7B-Instruct-v0.1                        # everything (about 9,000 prompts, 10 to 15 minutes)
    python collect_interaction.py --model Llama-3.1-8B-Instruct --base-only               # only the no-footer prompts: enough to recompute yields (about 1 minute)
    python collect_interaction.py --model Mistral-7B-Instruct-v0.1 --max-unique 40 --out-dir /tmp/smoke_int      # smoke test

For every record of the bias pilot's pairs file (the plain note, and the plain note plus one footer line) it builds the prompt variants of mech/interaction_prompts.py
(clean, user_ff, user_fl, assistant) for the case's fixed wrong option. A prompt with NO footer is built from the raw plain note of the pool when --pool exists, so that it is
token-identical to the prompts of the deference runs. Identical prompts are computed once. Resumable.

Saved per unique prompt: probs (the four options, renormalized), acts (the residual stream after each saved layer at the FINAL token, float16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from collect_bias_activations import atomic_savez, parse_layers

DEFAULT_VARIANTS = "clean,user_ff,user_fl,assistant"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="bias_stage0/pairs.jsonl")
    ap.add_argument("--pool", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--layers", default="14-24")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--base-only", action="store_true", help="only the prompts without a footer (for the yields)")
    ap.add_argument("--out-dir", default="bias_stage1/acts")
    ap.add_argument("--max-unique", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=800)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    from mech import ace_sensor, chat_prompting, deference_pairs as dp, interaction_prompts as IP
    from mech.model_registry import MECH_MODELS_BY_NAME
    from shared.data_loader import load_amqa_dataset
    from shared.inference_engine import USMLE_CLASSES

    spec = MECH_MODELS_BY_NAME[args.model]
    layers = parse_layers(args.layers)
    variants = [v for v in args.variants.split(",") if v]
    bad = [v for v in variants if v not in IP.VARIANTS]
    if bad:
        print(f"ERROR: unknown variants {bad}; choose from {IP.VARIANTS}", file=sys.stderr)
        return 2
    records = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    pairs_sha = hashlib.sha256(open(args.pairs, "rb").read()).hexdigest()
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    raw_notes = dp.load_plain_n_notes(args.pool) if os.path.exists(args.pool) else {}
    bases = {r["case_id"]: r["text"] for r in records if r["kind"] == "base"}

    # the wrong option of every case, by the same rule as the deference pairs
    case_meta, skipped = {}, []
    for cid in sorted(bases):
        rec = case_lookup.get(cid)
        w = IP.wrong_option_for(rec) if rec is not None else None
        if w is None:
            skipped.append(cid)
        else:
            case_meta[cid] = w
    if skipped:
        print(f"WARNING: {len(skipped)} cases have no usable wrong option or AMQA record and are skipped: {skipped[:8]}")

    specs, rec_map = {}, []                                              # key -> {messages}; (record id, variant, key)
    for r in records:
        cid = r["case_id"]
        if cid not in case_meta:
            continue
        corr, wl, wt = case_meta[cid]
        if r["kind"] == "base":
            base_text, footer = (raw_notes[cid][0] if cid in raw_notes else bases[cid]), None
        elif args.base_only:
            continue
        else:
            base_text, footer = bases[cid], r["footer"]
        for v in variants:
            messages = IP.build_messages(v, case_lookup[cid], base_text, footer, wl, wt)
            key = hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()
            specs.setdefault(key, {"messages": messages, "case_id": cid})
            rec_map.append((r["id"], v, key))
    keys = sorted(specs)
    if args.max_unique:
        keys = keys[:args.max_unique]
    keep = set(keys)
    rec_map = [m for m in rec_map if m[2] in keep]
    print(f"{len(records)} records, {len(variants)} variants -> {len(specs)} unique prompts ({len(keys)} to run); raw plain notes used for base prompts: {bool(raw_notes)}")

    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    letters = list(ids.keys())
    d_model = int(model.cfg.d_model)
    names = [f"blocks.{l}.hook_resid_post" for l in layers]
    name_set = set(names)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"interaction_acts_{spec.display_name}.npz")
    meta = {"model": spec.display_name, "layers": layers, "variants": variants, "base_only": bool(args.base_only), "pairs_sha256": pairs_sha, "n_unique": len(keys),
            "keys_sha1": hashlib.sha1("".join(keys).encode()).hexdigest(), "raw_plain_notes": bool(raw_notes)}
    U = len(keys)
    acts = np.zeros((U, len(layers), d_model), dtype=np.float16)
    probs = np.full((U, len(letters)), np.nan, dtype=np.float32)
    done = np.zeros(U, dtype=bool)
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=False)
        if json.loads(str(z["meta"])) == meta:
            acts, probs, done = z["acts"].copy(), z["probs"].copy(), z["done"].copy()
            print(f"RESUMING: {int(done.sum())} of {U} prompts already done")
        else:
            print(f"ERROR: {out_path} exists with different settings; move it away or use another --out-dir", file=sys.stderr)
            return 1
    cids = sorted(case_meta)

    def save():
        atomic_savez(out_path, meta=np.array(json.dumps(meta)), keys=np.array(keys), acts=acts, probs=probs, done=done, layers=np.array(layers), class_letters=np.array(letters),
                     map_rec=np.array([m[0] for m in rec_map]), map_variant=np.array([m[1] for m in rec_map]), map_key=np.array([m[2] for m in rec_map]),
                     case_ids=np.array(cids), case_correct=np.array([case_meta[c][0] for c in cids]), case_wrong=np.array([case_meta[c][1] for c in cids]))

    t0, n_new = time.time(), 0
    for u, k in enumerate(keys):
        if done[u]:
            continue
        tokens = chat_prompting.tokenize_chat_prompt(model, specs[k]["messages"], chat_prompting.USMLE_SUFFIX)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in name_set)
        acts[u] = np.stack([cache[nm][0, -1, :].detach().float().cpu().numpy() for nm in names]).astype(np.float16)
        cl = np.array([float(logits[0, -1, ids[c]].item()) for c in letters])
        e = np.exp(cl - cl.max())
        probs[u] = (e / e.sum()).astype(np.float32)
        done[u] = True
        n_new += 1
        if n_new % 100 == 0:
            rate = n_new / (time.time() - t0)
            print(f"  {int(done.sum())}/{U} prompts   {rate:.1f}/s   ~{(U - int(done.sum())) / rate / 60:.0f} min left")
        if n_new % args.save_every == 0:
            save()
    save()
    print(f"\nfinished: {out_path}")
    print(f"{int(done.sum())}/{U} prompts done; {len(cids)} cases with a usable wrong option; {len(rec_map)} (record, variant) entries")
    print("next: python analyze_interaction.py   (rules frozen first; nothing about the contrasts was printed here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
