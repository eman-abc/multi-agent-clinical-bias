#!/usr/bin/env python3
"""
collect_bias_activations.py -- GPU. One forward pass per UNIQUE prompt of the bias pilot; saves residual-stream activations and answer probabilities.
It prints no contrast statistics: what the activations say is decided by the analysis script under its own rules.

    python collect_bias_activations.py --model Mistral-7B-Instruct-v0.1 --pairs bias_stage0/pairs.jsonl
    python collect_bias_activations.py --model Mistral-7B-Instruct-v0.1 --pairs bias_stage0/pairs.jsonl --max-unique 40 --out-dir /tmp/smoke_bias   # smoke test

Prompts are built EXACTLY as the deference directions' clean prompts were (one user turn from the USMLE template, the model's own chat template, then the
forced-choice suffix); the only difference is that the note is the plain note plus one footer line from build_bias_pairs.py.

Three token positions are read, at every saved layer (default 10 to 26), from the residual stream after each layer:
    x_last       the last token of the carrier string itself (the name, the insurer, the label word)
    footer_last  the last token of the footer line
    final        the last token of the prompt (the forced-choice suffix): where the decision is made and where the deference directions were read
A position is found by tokenizing the text up to it and checking that this is a prefix of the full tokenization; if a merge across the boundary breaks that,
the position is marked invalid (never guessed) and the analysis skips it.
Identical prompts are computed once (a placebo pair re-uses its strings). Resumable: a finished prompt is never recomputed.
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

POSITIONS = ["x_last", "footer_last", "final"]


def parse_layers(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def atomic_savez(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def render_prompt(model, case_record, note_text, deference_pairs, chat_prompting):
    """(tokens, full_text): the prompt exactly as the clean deference prompt is built, for any note text."""
    opts = deference_pairs.options_dict(case_record)
    body = chat_prompting.USMLE_BODY_TEMPLATE.format(
        validated_note=note_text, usmle_question=case_record.neutralised_question,
        option_a=opts.get("A", ""), option_b=opts.get("B", ""), option_c=opts.get("C", ""), option_d=opts.get("D", ""))
    messages = chat_prompting.build_user_role_messages(body)
    tokens = chat_prompting.tokenize_chat_prompt(model, messages, chat_prompting.USMLE_SUFFIX)
    rendered = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = rendered + chat_prompting.USMLE_SUFFIX
    return tokens, full_text


def char_end_to_token(model, full_text: str, tokens: torch.Tensor, end_char: int) -> int:
    """Index of the last token that ends at or before character end_char, or -1 if the tokenization of the prefix is not a prefix of the full one."""
    pre = model.to_tokens(full_text[:end_char], prepend_bos=False)
    n = int(pre.shape[1])
    if n < 1 or n > int(tokens.shape[1]) or not torch.equal(tokens[0, :n].cpu(), pre[0].cpu()):
        return -1
    return n - 1


def locate_positions(model, full_text: str, tokens: torch.Tensor, footer, x):
    """(x_last, footer_last, final) token indices; -1 where a position does not exist or cannot be located safely."""
    final = int(tokens.shape[1]) - 1
    if not footer:
        return [-1, -1, final]
    start = full_text.rfind(footer)
    if start < 0 or full_text.count(footer) != 1:
        return [-1, -1, final]
    footer_end = start + len(footer)
    xi = footer.find(x) if x else -1
    x_end = start + xi + len(x) if xi >= 0 else -1
    return [char_end_to_token(model, full_text, tokens, x_end) if x_end > 0 else -1, char_end_to_token(model, full_text, tokens, footer_end), final]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="bias_stage0/pairs.jsonl")
    ap.add_argument("--layers", default="10-26")
    ap.add_argument("--out-dir", default="bias_stage0/acts")
    ap.add_argument("--max-unique", type=int, default=None, help="smoke test: only the first N unique prompts")
    ap.add_argument("--save-every", type=int, default=600)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    from mech import ace_sensor, chat_prompting, deference_pairs
    from mech.model_registry import MECH_MODELS_BY_NAME
    from shared.data_loader import load_amqa_dataset
    from shared.inference_engine import USMLE_CLASSES

    spec = MECH_MODELS_BY_NAME[args.model]
    layers = parse_layers(args.layers)
    records = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    pairs_sha = hashlib.sha256(open(args.pairs, "rb").read()).hexdigest()
    unique = {}                                                              # text hash -> spec of the prompt
    index = {}
    for r in records:
        key = hashlib.sha1(r["text"].encode()).hexdigest()
        index[r["id"]] = key
        if key not in unique:
            unique[key] = {"case_id": r["case_id"], "text": r["text"], "footer": r.get("footer"), "x": r.get("carrier")}
    keys = sorted(unique)
    if args.max_unique:
        keys = keys[:args.max_unique]
    print(f"{len(records)} records -> {len(unique)} unique prompts ({len(keys)} to run)")
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    missing = sorted({unique[k]["case_id"] for k in keys if unique[k]["case_id"] not in case_lookup})
    if missing:
        print(f"ERROR: {len(missing)} cases are not in the AMQA dataset, e.g. {missing[:5]}", file=sys.stderr)
        return 1

    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    letters = list(ids.keys())
    d_model = int(model.cfg.d_model)
    names = [f"blocks.{l}.hook_resid_post" for l in layers]
    name_set = set(names)

    out_path = os.path.join(args.out_dir, f"bias_acts_{spec.display_name}.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"model": spec.display_name, "layers": layers, "positions": POSITIONS, "pairs_sha256": pairs_sha, "n_unique": len(keys), "keys_sha1": hashlib.sha1("".join(keys).encode()).hexdigest()}
    U, nL = len(keys), len(layers)
    acts = np.zeros((U, len(POSITIONS), nL, d_model), dtype=np.float16)
    valid = np.zeros((U, len(POSITIONS)), dtype=bool)
    probs = np.full((U, len(letters)), np.nan, dtype=np.float32)
    done = np.zeros(U, dtype=bool)
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=False)
        if json.loads(str(z["meta"])) == meta:
            acts, valid, probs, done = z["acts"].copy(), z["valid"].copy(), z["probs"].copy(), z["done"].copy()
            print(f"RESUMING: {int(done.sum())} of {U} prompts already done")
        else:
            print(f"ERROR: {out_path} exists with different settings; move it away or use another --out-dir", file=sys.stderr)
            return 1

    def save():
        atomic_savez(out_path, meta=np.array(json.dumps(meta)), keys=np.array(keys), acts=acts, valid=valid, probs=probs, done=done, layers=np.array(layers),
                     class_letters=np.array(letters), record_ids=np.array(list(index.keys())), record_keys=np.array([index[k] for k in index]),
                     case_ids=np.array([unique[k]["case_id"] for k in keys]),
                     correct_letter=np.array([deference_pairs.correct_letter(case_lookup[unique[k]["case_id"]]) or "" for k in keys]))

    t0, n_new = time.time(), 0
    for u, k in enumerate(keys):
        if done[u]:
            continue
        s = unique[k]
        tokens, full = render_prompt(model, case_lookup[s["case_id"]], s["text"], deference_pairs, chat_prompting)
        pos = locate_positions(model, full, tokens, s["footer"], s["x"])
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda n: n in name_set)
        for pi, p in enumerate(pos):
            if p >= 0:
                acts[u, pi] = np.stack([cache[nm][0, p, :].detach().float().cpu().numpy() for nm in names]).astype(np.float16)
                valid[u, pi] = True
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
    print(f"valid positions: " + ", ".join(f"{p} {int(valid[:, i].sum())}/{U}" for i, p in enumerate(POSITIONS)))
    print("prompts without a footer (the base notes) have no x_last or footer_last by design; any other invalid position is skipped by the analysis")
    print("next: python analyze_bias_pilot.py   (rules frozen first; nothing about the contrasts was printed here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
