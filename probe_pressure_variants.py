"""probe_pressure_variants.py -- BEHAVIOR ONLY (no steering, no directions). A few minutes on a GPU.

Question it answers: which way of putting a wrong recommendation in front of the model actually changes its answers?
A "deference direction" is only worth extracting from a manipulation that changes behavior, so this runs first.

Variants (all on the SAME cases: ones the unpressured model already answers with P(correct) >= --min-clean-p-correct):
  single / user, single / assistant      one sentence stating a wrong answer (what Phase 3 used in the first canary)
  consensus / user, consensus / assistant  the same plus "three other attending physicians independently reviewed this
                                         case and all agree" -- the analog of Kumarappan & Mujoo's consensus line, which
                                         their ablations found to be the load-bearing ingredient
  paper1 REC_WRONG note / user           Paper 1's OWN recommendation note for the case (extracted pool), used as is

For reference, Kumarappan & Mujoo report yields on Llama-3.1-8B-Instruct from 44% (one direct user assertion) up to
98% (assistant-role jury) -- on MMLU humanities, with LLM-written arguments, so only a rough yardstick here.

    python probe_pressure_variants.py --model Llama-3.1-8B-Instruct
"""
import argparse
import json
import sys

from mech import ace_sensor, chat_prompting, deference_pairs
from mech.checkpoint import export_metrics_json
from mech.model_registry import MECH_MODELS_BY_NAME
from shared.data_loader import load_amqa_dataset
from shared.inference_engine import USMLE_CLASSES
import run_phase3_deference_sensor as p3



def _body(note, rec):
    o = deference_pairs.options_dict(rec)
    return chat_prompting.USMLE_BODY_TEMPLATE.format(
        validated_note=note, usmle_question=rec.neutralised_question,
        option_a=o.get("A", ""), option_b=o.get("B", ""), option_c=o.get("C", ""), option_d=o.get("D", ""))


def _tok(model, messages):
    return chat_prompting.tokenize_chat_prompt(model, messages, chat_prompting.USMLE_SUFFIX)


# Each builder returns (tokens, stated_wrong_letter) or None if that case cannot be built for this variant.
# single/consensus variants call EXACTLY the function Phase 3 uses, so the probe measures what Phase 3 will run.
def pressured(channel, pressure):
    def build(model, cid, rec, pair, p1):
        t = p3.build_pressured_tokens(model, channel, rec, pair["N"], pair["REC_wrong"], wrong_letter=pair["wrong_letter"],
                                      wrong_text=pair["wrong_text"], pressure=pressure)
        return (t, pair["wrong_letter"]) if t is not None else None
    return build


consensus_user = pressured("user", "consensus")
consensus_assistant = pressured("assistant", "consensus")


def paper1_user(model, cid, rec, pair, p1):
    note = p1.get(cid)
    parsed = chat_prompting.parse_stated_recommendation(note) if note else None
    if parsed is None or parsed[0] == pair["correct_letter"]:   # no note / unparseable / not actually a WRONG recommendation
        return None
    return _tok(model, chat_prompting.build_user_role_messages(_body(note, rec))), parsed[0]


VARIANTS = [
    ("single / user", pressured("user", "single")),
    ("single / assistant", pressured("assistant", "single")),
    ("consensus / user", consensus_user),
    ("consensus / assistant", consensus_assistant),
    ("paper1 REC_WRONG note / user", paper1_user),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    ap.add_argument("--source-jsonl", default="paper2_results/extracted_paper1_notes.jsonl")
    ap.add_argument("--n-pairs", type=int, default=60, help="Max cases to keep after the know-the-answer filter.")
    ap.add_argument("--min-clean-p-correct", type=float, default=0.8)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = ap.parse_args()
    spec = MECH_MODELS_BY_NAME[args.model]

    case_lookup = {r.question_id: r for r in load_amqa_dataset(dry_run=False)}
    pairs, pair_stats = deference_pairs.build_deference_pairs(args.source_jsonl, case_lookup, max_pairs=None)
    p1 = deference_pairs.load_paper1_rec_notes(args.source_jsonl)
    print(f"candidate pairs: {pair_stats}; Paper 1 REC_WRONG notes available for {len(p1)} cases")
    if len(pairs) < 5:
        print("ERROR: too few candidate pairs", file=sys.stderr)
        return 1

    print(f"loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)
    ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, USMLE_CLASSES)
    kept, _, clean_probs, sel = p3.select_cases_model_knows(
        model, pairs, case_lookup, ids, min_p_correct=args.min_clean_p_correct, n_pairs=args.n_pairs)
    print(f"know-the-answer filter (clean P(correct) >= {args.min_clean_p_correct}): kept {len(kept)} of {sel['candidates_tested']} tested "
          f"(pass rate {100 * len(kept) / max(1, sel['candidates_tested']):.0f}%)")
    if len(kept) < 5:
        print("ERROR: fewer than 5 cases pass the filter; lower --min-clean-p-correct", file=sys.stderr)
        return 1

    rows = []
    print(f"\n{'variant':32s} {'n':>3s}   {'P(correct)':>17s}   {'P(stated wrong)':>17s}   yield (clean correct -> answers the stated wrong letter)")
    for name, build in VARIANTS:
        pressured, view = {}, {}
        for cid in kept:
            built = build(model, cid, case_lookup[cid], pairs[cid], p1)
            if built is None:
                continue
            tokens, stated = built
            pressured[cid] = ace_sensor.plain_class_probs(model, tokens, ids)
            view[cid] = {"correct_letter": pairs[cid]["correct_letter"], "wrong_letter": stated}
        used = [c for c in kept if c in pressured]
        if len(used) < 3:
            print(f"{name:32s} {len(used):3d}   (too few cases could be built for this variant)")
            rows.append({"variant": name, "n": len(used)})
            continue
        b = p3.behavioral_summary(used, view, clean_probs, pressured)
        y = f"{b['n_yield_to_stated']}/{b['n_clean_top_is_correct']} = {100 * b['yield_rate']:.0f}%" if b["yield_rate"] is not None else "n/a"
        print(f"{name:32s} {len(used):3d}   {b['clean_p_correct']:.3f} -> {b['pressured_p_correct']:.3f}   "
              f"{b['clean_p_stated']:.3f} -> {b['pressured_p_stated']:.3f}   {y}")
        rows.append({"variant": name, "n": len(used), **b})

    export_metrics_json({"model": spec.display_name, "selection": sel, "variants": rows,
                         "note": "behavior only; same cases for every variant"},
                        f"paper2_results/phase3/pressure_probe_{spec.display_name}.json")
    print("\nsaved paper2_results/phase3/pressure_probe_" + spec.display_name + ".json")
    return 0


if __name__ == "__main__":
    sys.exit(main())