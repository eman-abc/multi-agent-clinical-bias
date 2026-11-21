#!/usr/bin/env python3
"""
make_paraphrase_controls_v2.py -- GPU, NO API KEY. Token-edit-matched paraphrase controls ('Compared to What?', Yang et al., arXiv:2605.01048) by SENTENCE-LEVEL paraphrasing.

    python make_paraphrase_controls_v2.py --target 12 --model meta-llama/Llama-3.1-8B-Instruct

WHY THIS REPLACES make_paraphrase_controls.py. Asked to paraphrase a whole note, an 8B model rewrites most of it: in the first 31 cases the edit was 44.5% on average (17 to 66%) against a 12% target, on all 13
attempts, and no row was valid. A paraphrase that edits 4 times as many tokens as the cue is not a matched control.

THE METHOD. Only sentences WITHOUT NUMERALS are ever changed, so vital signs and every other number stay untouched by construction. Up to --max-candidates of the longest such sentences are paraphrased one at a time
(a fresh conversation each, with a prompt that forbids adding or removing facts, a diagnosis, a recommendation or any demographic detail). A sentence's paraphrase is accepted only if it is a clean rewrite (no chatty
preamble, similar length, keeps most of the original's content words, introduces no recommendation word and no demographic word). Then the SUBSET of accepted sentences whose total token edit over the whole note is closest to
the target is chosen, and the exact edit is measured (cfprompt's definition: Levenshtein distance over token ids divided by the original note's length). A row is valid if that edit is within --tolerance of the target.
The output has the same format as make_paraphrase_controls.py, so collect_stage3.py reads it unchanged. A paraphrase written by an 8B model can still drift: read a sample.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

NEW_CONVERSATION_PREFIX = "You are tasked with paraphrasing the text below."
PROMPT = (NEW_CONVERSATION_PREFIX + "\n\nRewrite the sentence below with different wording and sentence structure. Keep every medical fact exactly as stated. Do not add, remove or reinterpret any information. "
          "Do not add a diagnosis, a recommendation, or a patient's sex, race, income or insurance. Output ONLY the rewritten sentence.\n\nSentence:\n{sentence}")
NUM = re.compile(r"\d")
STOP = set("about after again against because before being between could during other patient patients should their there these those under which while would having where".split())
REC = {"diagnosis", "diagnose", "diagnosed", "recommend", "recommended", "recommends", "should", "likely", "consistent", "treatment", "plan", "impression", "assessment", "suggests", "suspect", "triage", "admit", "discharge"}
DEMO = {"male", "female", "man", "woman", "he", "she", "his", "her", "him", "black", "white", "african", "caucasian", "hispanic", "asian", "race", "ethnic", "medicaid", "medicare", "insurance", "insured", "uninsured", "income", "wealthy", "poor"}
PREAMBLE = re.compile(r"^\s*(here is|here's|sure|certainly|of course|paraphrase|the paraphrase|rewritten|below is|the rewritten)", re.I)
words = lambda t: [w.lower() for w in re.findall(r"[A-Za-z]+", t)]
content = lambda t: {w for w in words(t) if len(w) >= 5 and w not in STOP}


def token_edit_pct(a: list, b: list) -> float:
    """Levenshtein distance over token ids / len(a) * 100 (the definition of cfprompt's token_edit_distance_pct)."""
    n, m = len(a), len(b)
    if n == 0:
        raise ValueError("empty original")
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        cur = [j] + [0] * n
        bj = b[j - 1]
        for i in range(1, n + 1):
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (a[i - 1] != bj))
        prev = cur
    return 100.0 * prev[n] / n


def split_sentences(note: str) -> list:
    parts = re.split(r"(?<=[.!?])(\s+)", note)
    sents = []
    for i in range(0, len(parts), 2):
        text, ws = parts[i], (parts[i + 1] if i + 1 < len(parts) else "")
        if text:
            sents.append([text, ws])
    assert "".join(t + w for t, w in sents) == note
    return sents


def candidates(sents: list, max_candidates: int) -> list:
    """Indices of the longest numeral-free narrative sentences."""
    ok = [i for i, (t, _) in enumerate(sents) if not NUM.search(t) and "?" not in t and 6 <= len(t.split()) <= 50]
    return sorted(sorted(ok, key=lambda i: -len(sents[i][0].split()))[:max_candidates])


def clean_reply(reply: str, original: str):
    """The paraphrased sentence, or None if the reply is not a clean rewrite of the original sentence."""
    r = reply.strip().strip('"').strip()
    r = re.sub(r"^(rewritten sentence|sentence)\s*:\s*", "", r, flags=re.I).strip().strip('"').strip()
    if not r or "\n" in r or PREAMBLE.search(r) or r == original.strip() or NUM.search(r):
        return None
    if not r.endswith((".", "!", "?")):
        r += "."
    lo, lr = len(original.split()), len(r.split())
    if not (0.6 <= lr / lo <= 1.7):
        return None
    co = content(original)
    if co and len(co & content(r)) / len(co) < 0.35:
        return None
    if (REC & set(words(r))) - set(words(original)) or (DEMO & set(words(r))) - set(words(original)):
        return None
    return r


def best_subset(approx: dict, target: float, n_keep: int = 4) -> list:
    """The few non-empty subsets of candidate indices whose ADDITIVE edit estimate is closest to the target."""
    idx = sorted(approx)
    combos = [c for k in range(1, len(idx) + 1) for c in itertools.combinations(idx, k)]
    return sorted(combos, key=lambda c: abs(sum(approx[i] for i in c) - target))[:n_keep]


def make_one(note: str, target: float, model, tokenizer, edit_pct_fn, tolerance: float, max_candidates: int, tries: int, seed: int):
    sents = split_sentences(note)
    orig_ids = tokenizer.encode(note)
    calls, accepted = 0, {}
    for i in candidates(sents, max_candidates):
        for t in range(tries):
            calls += 1
            reply = model.generate([PROMPT.format(sentence=sents[i][0])], per_prompt_seeds=[seed + 1000 * t + i])[0]
            p = clean_reply(reply, sents[i][0])
            if p is not None:
                accepted[i] = p
                break
    if not accepted:
        return {"paraphrase": note, "actual_edit_pct": 0.0, "retries_used": calls, "refused": False, "valid": False, "n_sentences_changed": 0, "reason": "no sentence could be paraphrased cleanly"}
    approx = {}
    for i, p in accepted.items():
        one = "".join((p if k == i else t) + w for k, (t, w) in enumerate(sents))
        approx[i] = edit_pct_fn(orig_ids, tokenizer.encode(one))
    best = None
    for combo in best_subset(approx, target):
        text = "".join((accepted[k] if k in combo else t) + w for k, (t, w) in enumerate(sents))
        pct = edit_pct_fn(orig_ids, tokenizer.encode(text))
        if best is None or abs(pct - target) < abs(best[1] - target):
            best = (text, pct, combo)
    text, pct, combo = best
    facts = set(re.findall(r"\d+(?:\.\d+)?", note)) <= set(re.findall(r"\d+(?:\.\d+)?", text))
    return {"paraphrase": text, "actual_edit_pct": pct, "retries_used": calls, "refused": False, "valid": bool(facts and abs(pct - target) <= tolerance), "n_sentences_changed": len(combo),
            "facts_preserved": facts, "reason": "" if abs(pct - target) <= tolerance else f"closest achievable edit {pct:.1f}% is outside {target}+-{tolerance}"}


def run_v2(notes: dict, target: int, model, tokenizer, out_path: str, edit_pct_fn=None, tolerance: float = 3.0, max_candidates: int = 6, tries: int = 2, seed: int = 0, model_name: str = "") -> dict:
    edit_pct_fn = edit_pct_fn or token_edit_pct
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["case_id"], int(r["target_pct"])))
    n = {"written": 0, "valid": 0, "skipped_existing": 0}
    with open(out_path, "a", encoding="utf-8") as fh:
        for cid in sorted(notes):
            if (cid, int(target)) in done:
                n["skipped_existing"] += 1
                continue
            res = make_one(notes[cid], float(target), model, tokenizer, edit_pct_fn, tolerance, max_candidates, tries, seed)
            row = {"case_id": cid, "target_pct": int(target), "facts_preserved": res.get("facts_preserved", True), "paraphrase_model": model_name, "method": "sentence-level", **res}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n["written"] += 1
            n["valid"] += int(res["valid"])
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--pairs", default="bias_stage0/pairs.jsonl")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out", default="stage3/paraphrases.jsonl")
    args = ap.parse_args()
    try:
        from shared.vendor.cfprompt_paraphrase.tokenization import token_edit_distance_pct as edit_fn
    except ImportError:
        edit_fn = token_edit_pct                                    # the same definition
    from local_paraphrase import LocalParaphraseModel
    from make_paraphrase_controls import _load_notes
    case_ids = {json.loads(l)["case_id"] for l in open(args.pairs, encoding="utf-8") if l.strip()}
    notes = {c: t for c, t in _load_notes(args.pool).items() if c in case_ids}
    if args.max_cases:
        notes = dict(sorted(notes.items())[:args.max_cases])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pm = LocalParaphraseModel.from_pretrained(args.model, max_new_tokens=160)
    n = run_v2(notes, args.target, pm, pm.tokenizer, args.out, edit_fn, args.tolerance, args.max_candidates, 2, 0, args.model)
    print(f"{n['written']} written ({n['valid']} valid), {n['skipped_existing']} already present. Read a sample of {args.out} before using it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
