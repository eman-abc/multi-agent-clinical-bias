#!/usr/bin/env python3
"""
show_stage3_json.py -- CPU. Reads a stage-3 results file (stage3/results/stage3_<task>_<model>.json) and prints (1) the reproducibility gate WITH ITS NUMBERS, which the report shows only as OK or FAILED, and
(2) the interactions of the SECONDARY contrasts against the no-cue note ('vs_base_nl') and against a paraphrase ('vs_paraphrase'), which the main report leaves in the JSON.

    python show_stage3_json.py stage3\\results\\stage3_usmle_Llama-3.1-8B-Instruct.json
    python show_stage3_json.py stage3\\results\\stage3_esi_Mistral-7B-Instruct-v0.1.json --kinds vs_base_nl vs_paraphrase

Why (2): the primary contrasts compare a cue with a length-matched NEUTRAL control sentence. If the control sentence itself moves the decision, the contrast mixes the cue's effect with the control's. The comparison
with the note carrying no sentence (vs_base_nl) has no control sentence in it.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--kinds", nargs="+", default=["vs_base_nl", "vs_paraphrase"])
a = ap.parse_args()
d = json.load(open(a.path))
res = d["result"]
print(f"{a.path}\nplan status: {d.get('plan_status')}\n")
print("GATE (the largest per-case difference between the no-cue prompts and the confirmatory run's stored probabilities; the limit is 0.01):")
for ch, v in res["gate"].items():
    print(f"  {ch}: {v}")
print(f"\nINTERACTIONS of the contrasts of kind {a.kinds} (a = no-cue or paraphrased note, b = the cue; rho > 0 means the pressure moves the model MORE on the no-cue side)")
print(f"{'contrast | pressure':58s} {'n':>4s} {'rho':>8s} {'95% CI':>20s}  reading")
n = 0
for k, v in sorted(res["interaction"].items()):
    if v["kind"] in a.kinds:
        ci = v["ci"]
        print(f"{k:58s} {v['n']:4d} {v['rho']:+8.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  {v['reading']}")
        n += 1
print(f"\n{n} cells.")
