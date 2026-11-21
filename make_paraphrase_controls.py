#!/usr/bin/env python3
"""
make_paraphrase_controls.py -- GPU, NO API KEY. Generates the token-edit-matched PARAPHRASE controls of 'Compared to What?' with a local open-weight model and the project's vendored cfprompt functions
(shared/vendor/cfprompt_paraphrase/), then writes stage3/paraphrases.jsonl for collect_stage3.py (the 'para' family).

    python make_paraphrase_controls.py --report-cue-edits                           # CPU: how many tokens (percent of the note) each cue adds? choose the targets from this
    python make_paraphrase_controls.py --targets 6,12,18 --model meta-llama/Llama-3.1-8B-Instruct   # GPU: about 1 to 2 hours for 135 cases x 3 targets

For each case and target percentage the paraphrase must keep every number of the note (vital signs and so on), must not be a refusal, must differ from the note, and must land within --accept
percentage points of the target; otherwise it is written with valid = false and excluded downstream (the cases drop out of the paraphrase contrasts, never silently substituted).
A paraphrase written by an 8B model can drift clinically; the digit check is a floor, not a guarantee: read a sample before using the control.
Edit distance is measured with the paraphrase model's tokenizer (cfprompt's definition: Levenshtein over token ids divided by the original's length).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

NUM = re.compile(r"\d+(?:\.\d+)?")


def facts_preserved(original: str, paraphrase: str) -> bool:
    """Every number of the original appears in the paraphrase."""
    return set(NUM.findall(original)) <= set(NUM.findall(paraphrase))


def run(notes: dict, targets, paraphrase_model, tokenizer, generate_fn, out_path: str, tolerance: float = 1.5, accept: float = 3.0, max_retries: int = 12, seed: int = 0, model_name: str = "") -> dict:
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["case_id"], int(r["target_pct"])))
    n = {"written": 0, "valid": 0, "skipped_existing": 0}
    with open(out_path, "a", encoding="utf-8") as fh:
        for cid in sorted(notes):
            for t in targets:
                if (cid, int(t)) in done:
                    n["skipped_existing"] += 1
                    continue
                res = generate_fn(notes[cid], float(t), paraphrase_model, tokenizer, tolerance=tolerance, max_retries=max_retries, seed=seed)
                facts = (not res.refused) and facts_preserved(notes[cid], res.paraphrase)
                valid = bool((not res.refused) and facts and res.paraphrase.strip() != notes[cid].strip() and abs(res.actual_edit_pct - float(t)) <= accept)
                fh.write(json.dumps({"case_id": cid, "target_pct": int(t), "paraphrase": res.paraphrase, "actual_edit_pct": res.actual_edit_pct, "retries_used": res.retries_used,
                                     "refused": res.refused, "facts_preserved": facts, "valid": valid, "paraphrase_model": model_name}) + "\n")
                fh.flush()
                n["written"] += 1
                n["valid"] += int(valid)
    return n


def cue_edit_report(notes: dict, tokenizer, edit_pct_fn) -> list:
    import stage3_cues as S
    rows = []
    C = S.build_conditions()
    for cid, cond in sorted(C.items()):
        if cond.family not in ("sl", "stack", "dialect"):
            continue
        pcts = []
        for i, (case, note) in enumerate(sorted(notes.items())):
            text = cond.text_for(case, i)
            if text:
                pcts.append(edit_pct_fn(tokenizer.encode(note), tokenizer.encode(S.insert_cue(note, cond, text))))
        if pcts:
            rows.append((cid, sum(pcts) / len(pcts), min(pcts), max(pcts)))
    return rows


def _load_notes(pool: str) -> dict:
    from mech import deference_pairs as dp
    return {cid: txt for cid, (txt, _) in dp.load_plain_n_notes(pool).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--pairs", default="bias_stage0/pairs.jsonl", help="only the cases of this file get paraphrases")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--targets", default="6,12,18")
    ap.add_argument("--tolerance", type=float, default=1.5)
    ap.add_argument("--accept", type=float, default=3.0)
    ap.add_argument("--max-retries", type=int, default=12)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out", default="stage3/paraphrases.jsonl")
    ap.add_argument("--report-cue-edits", action="store_true")
    args = ap.parse_args()
    try:
        from shared.vendor.cfprompt_paraphrase.paraphrase import generate_adjusted_paraphrase
        from shared.vendor.cfprompt_paraphrase.tokenization import token_edit_distance_pct
    except ImportError as e:
        print(f"ERROR: the vendored cfprompt functions are not importable ({e}). They live in shared/vendor/cfprompt_paraphrase/ in the project.", file=sys.stderr)
        return 2
    case_ids = {json.loads(l)["case_id"] for l in open(args.pairs, encoding="utf-8") if l.strip()}
    notes = {c: t for c, t in _load_notes(args.pool).items() if c in case_ids}
    if args.max_cases:
        notes = dict(sorted(notes.items())[:args.max_cases])
    from transformers import AutoTokenizer
    from local_paraphrase import _Tok
    tokenizer = _Tok(AutoTokenizer.from_pretrained(args.model))
    if args.report_cue_edits:
        print(f"{'condition':28s} {'mean %':>8s} {'min %':>7s} {'max %':>7s}   (inserted tokens as a percent of the note's tokens, {args.model} tokenizer)")
        for cid, mean, lo, hi in cue_edit_report(notes, tokenizer, token_edit_distance_pct):
            print(f"{cid:28s} {mean:8.1f} {lo:7.1f} {hi:7.1f}")
        print("\nChoose --targets near the level-3 stigmatizing-language sentences and the three-line stacks (a paraphrase should edit about as many tokens as the cue does).")
        return 0
    from local_paraphrase import LocalParaphraseModel
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pm = LocalParaphraseModel.from_pretrained(args.model)
    n = run(notes, [int(t) for t in args.targets.split(",")], pm, pm.tokenizer, generate_adjusted_paraphrase, args.out, args.tolerance, args.accept, args.max_retries, 0, args.model)
    print(f"{n['written']} written ({n['valid']} valid), {n['skipped_existing']} already present. Read a sample of {args.out} before using it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
