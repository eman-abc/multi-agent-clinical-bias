#!/usr/bin/env python3
"""
collect_stage3.py -- GPU. One forward pass per UNIQUE prompt; saves the probabilities of the answer options (no hidden states: stage 3 is behavioral, which keeps the files small and
the runs fast). It prints no contrast statistic.

    python collect_stage3.py --model Mistral-7B-Instruct-v0.1 --task usmle                       # every family, every pressure (about 45,000 prompts)
    python collect_stage3.py --model Mistral-7B-Instruct-v0.1 --task esi                         # triage task
    python collect_stage3.py --model Llama-3.1-8B-Instruct --task usmle --families footer        # only the pressure-dial rerun of the bias pilot's footers
    python collect_stage3.py --model Mistral-7B-Instruct-v0.1 --task usmle --max-cases 4 --out-dir /tmp/smoke3     # smoke test

Families (stage3_cues.py): sl (stigmatizing language), stack (stacked cues), dialect (minimal pairs), footer (the bias pilot's own footer pairs, for the pressure dial),
para (token-edit-matched paraphrases of the note, from make_paraphrase_controls.py, if --paraphrases exists). 'base' and 'base_nl' (no cue; the plain and the
newline join) are always included. Pressures: USMLE user_single, user_peer, user_consensus, assistant_consensus (plus the clean prompt); ESI user and assistant consensus with the
stated level 4 (a downgrade) and 2 (an upgrade). Resumable.
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

import stage3_cues as S
from collect_bias_activations import atomic_savez

DEFAULT_PRESSURES = {"usmle": "user_single,user_peer,user_consensus,assistant_consensus", "esi": "user_consensus@4,user_consensus@2,assistant_consensus@4,assistant_consensus@2"}
DEFAULT_FAMILIES = {"usmle": "sl,stack,dialect,footer,para", "esi": "sl,stack,footer"}


def load_footer_conditions(records: list, cases: list) -> dict:
    """The bias pilot's own footer pairs (name or insurer form, the first carrier set), as per-case conditions ftr_<attr>_pos / ftr_<attr>_neg."""
    out = {}
    for attr in ("race", "sex", "ses"):
        for role in ("pos", "neg"):
            per = {}
            for r in records:
                if r["kind"] == "pair" and r["attribute"] == attr and r["form"] == "PRX" and r["role"] == role and r["case_id"] in cases and r["case_id"] not in per:
                    per[r["case_id"]] = r["footer"]
            if per:
                out[f"ftr_{attr}_{role}"] = S.Condition(f"ftr_{attr}_{role}", "footer", "footer", per_case=per)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=("usmle", "esi"), default="usmle")
    ap.add_argument("--families", default=None)
    ap.add_argument("--pressures", default=None)
    ap.add_argument("--pairs", default="bias_stage0/pairs.jsonl")
    ap.add_argument("--pool", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--paraphrases", default="stage3/paraphrases.jsonl")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out-dir", default="stage3/acts")
    ap.add_argument("--max-unique", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=1500)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()

    from mech import ace_sensor, chat_prompting as CP, deference_pairs as dp, interaction_prompts as IP, stage3_prompts as SP
    from mech.model_registry import MECH_MODELS_BY_NAME
    from shared.data_loader import load_amqa_dataset

    spec = MECH_MODELS_BY_NAME[args.model]
    families = [f for f in (args.families or DEFAULT_FAMILIES[args.task]).split(",") if f]
    pressures = [p for p in (args.pressures or DEFAULT_PRESSURES[args.task]).split(",") if p]
    records = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    pairs_sha = hashlib.sha256(open(args.pairs, "rb").read()).hexdigest()
    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    raw_notes = dp.load_plain_n_notes(args.pool) if os.path.exists(args.pool) else {}
    bases = {r["case_id"]: r["text"] for r in records if r["kind"] == "base"}
    case_meta, skipped = {}, []
    for cid in sorted(bases):
        rec = case_lookup.get(cid)
        w = IP.wrong_option_for(rec) if rec is not None else None
        (case_meta.__setitem__(cid, w) if w is not None else skipped.append(cid))
    cases = sorted(case_meta)[:args.max_cases] if args.max_cases else sorted(case_meta)
    if skipped:
        print(f"WARNING: {len(skipped)} cases have no usable wrong option or AMQA record and are skipped: {skipped[:8]}")

    # paraphrases (token-edit-matched controls), if they were generated
    para, targets = {}, set()
    if "para" in families and os.path.exists(args.paraphrases):
        for line in open(args.paraphrases, encoding="utf-8"):
            r = json.loads(line)
            if r.get("valid"):
                para[(r["case_id"], int(r["target_pct"]))] = r["paraphrase"]
                targets.add(int(r["target_pct"]))
    elif "para" in families:
        print(f"NOTE: no paraphrase file at {args.paraphrases}; the 'para' family is skipped (run make_paraphrase_controls.py first)")
    allc = S.build_conditions(tuple(sorted(targets)))
    wanted = {"base", "base_nl"} | {cid for cid, c in allc.items() if c.family in families and c.family != "footer"}
    conds = {cid: allc[cid] for cid in sorted(wanted)}
    if "footer" in families:
        conds.update(load_footer_conditions(records, set(cases)))

    # every prompt
    specs, rows = {}, []
    for ci, cid in enumerate(cases):
        corr, wl, wt = case_meta[cid]
        base_note = raw_notes[cid][0] if cid in raw_notes else bases[cid]
        for cond_id, cond in conds.items():
            if cond.family == "para":
                note = para.get((cid, int(cond_id.split("_")[1])))
                if note is None:
                    continue
            else:
                text = cond.text_for(cid, ci)
                if cond.family not in ("base",) and text is None:
                    continue
                note = S.insert_cue(base_note, cond, text)
            for pr in [None] + pressures:
                if args.task == "usmle":
                    messages = SP.usmle_messages(case_lookup[cid], note, pr, wl, wt, newline_join=cond.newline_join)
                else:
                    channel, _, level = (pr or "").partition("@")
                    messages = SP.esi_messages(note, channel or None, level or None, newline_join=cond.newline_join)
                key = hashlib.sha1((args.task + json.dumps(messages, sort_keys=True)).encode()).hexdigest()
                specs.setdefault(key, messages)
                rows.append((cid, cond_id, pr or "clean", key))
    keys = sorted(specs)
    if args.max_unique:
        keys = keys[:args.max_unique]
    keep = set(keys)
    rows = [r for r in rows if r[3] in keep]
    print(f"{len(cases)} cases x {len(conds)} conditions x {len(pressures) + 1} pressures -> {len(specs)} unique prompts ({len(keys)} to run); task {args.task}; families {families}")

    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    suffix, classes = (CP.USMLE_SUFFIX, ("A", "B", "C", "D")) if args.task == "usmle" else (CP.ESI_SUFFIX, SP.ESI_CLASSES)
    ids = CP.resolve_class_token_ids(model, suffix, classes)
    letters = list(ids.keys())
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"stage3_{args.task}_{spec.display_name}.npz")
    meta = {"model": spec.display_name, "task": args.task, "families": families, "pressures": pressures, "cues_version": S.CUES_VERSION, "pairs_sha256": pairs_sha,
            "n_unique": len(keys), "keys_sha1": hashlib.sha1("".join(keys).encode()).hexdigest(), "raw_plain_notes": bool(raw_notes), "paraphrase_targets": sorted(targets), "n_cases": len(cases)}
    U = len(keys)
    probs = np.full((U, len(letters)), np.nan, dtype=np.float32)
    done = np.zeros(U, dtype=bool)
    if os.path.exists(out_path):
        z = np.load(out_path, allow_pickle=False)
        if json.loads(str(z["meta"])) == meta:
            probs, done = z["probs"].copy(), z["done"].copy()
            print(f"RESUMING: {int(done.sum())} of {U} prompts already done")
        else:
            print(f"ERROR: {out_path} exists with different settings; move it away or use another --out-dir", file=sys.stderr)
            return 1

    def save():
        atomic_savez(out_path, meta=np.array(json.dumps(meta)), keys=np.array(keys), probs=probs, done=done, class_letters=np.array(letters),
                     map_case=np.array([r[0] for r in rows]), map_cond=np.array([r[1] for r in rows]), map_pressure=np.array([r[2] for r in rows]), map_key=np.array([r[3] for r in rows]),
                     case_ids=np.array(cases), case_correct=np.array([case_meta[c][0] for c in cases]), case_wrong=np.array([case_meta[c][1] for c in cases]))

    t0, n_new = time.time(), 0
    for u, k in enumerate(keys):
        if done[u]:
            continue
        tokens = CP.tokenize_chat_prompt(model, specs[k], suffix)
        with torch.no_grad():
            logits = model(tokens)
        cl = np.array([float(logits[0, -1, ids[c]].item()) for c in letters])
        e = np.exp(cl - cl.max())
        probs[u] = (e / e.sum()).astype(np.float32)
        done[u] = True
        n_new += 1
        if n_new % 200 == 0:
            rate = n_new / (time.time() - t0)
            print(f"  {int(done.sum())}/{U} prompts   {rate:.1f}/s   ~{(U - int(done.sum())) / rate / 60:.0f} min left", flush=True)
        if n_new % args.save_every == 0:
            save()
    save()
    print(f"\nfinished: {out_path}\n{int(done.sum())}/{U} prompts done; {len(cases)} cases; {len(rows)} (case, condition, pressure) entries")
    print("next: python analyze_stage3.py   (rules frozen first; nothing about the contrasts was printed here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
