"""Mock dataset with the REAL structure of p1v2 (stimulus factors, answer records, duplicates, invalid answers) and PLANTED effects."""
import json, os, random

MODELS = ["M1", "M2", "M3"]


def make(root, n_cases=150, seed=0):
    rng = random.Random(seed)
    os.makedirs(os.path.join(root, "p1v2_data", "banks", "p1v2_bank_main"), exist_ok=True)
    correct = {str(c): rng.choice("ABCD") for c in range(n_cases)}
    with open(os.path.join(root, "p1v2_data", "banks", "p1v2_bank_main", "cases_clean.jsonl"), "w") as f:
        for c, l in correct.items():
            f.write(json.dumps(dict(case_id=c, correct=l)) + "\n")
    S = []
    def add(case, task, **fac):
        S.append(dict(stimulus_id=f"s{len(S):06d}", case_id=case, task=task, arm=fac["arm"], factors=fac, inserted=[], target=fac.get("target"), bank="luna"))
    for c in correct:
        for task in ("esi", "usmle"):
            add(c, task, arm="bias", condition="N", side="none")
            for attr, form, kind, pair, vals in (("race", "EXP", "label", -1, ("White", "Black")), ("race", "PRX", "clinic", 0, ("MGH", "BMC")), ("ses", "EXP", "label", -1, ("high", "low")),
                                                 ("sex", "EXP", "label", -1, ("male", "female")), ("placebo", "PRX", "bay", 0, ("Bay 2", "Bay 4")), ("placebo", "IMP", "bay", 0, ("seen bay 2", "seen bay 4"))):
                for side, v in zip(("plus", "minus"), vals):
                    add(c, task, arm="bias", condition="carrier", attribute=attr, form=form, kind=kind, side=side, pair_index=pair, value=v)
            srcs = [("user", s) for s in ("none", "author", "peer", "consensus")] + [("assistant", s) for s in ("own", "peer", "consensus")]
            for chan, src in srcs:
                if task == "esi":
                    for T in ("1", "5"):
                        add(c, task, arm="conformity", condition="rec", channel=chan, source=src, target_type="level", target=T)
                else:
                    wrong = rng.choice([x for x in "ABCD" if x != correct[c]])
                    add(c, task, arm="conformity", condition="rec", channel=chan, source=src, target_type="wrong", target=wrong)
                    add(c, task, arm="conformity", condition="rec", channel=chan, source=src, target_type="correct", target=correct[c])
            add(c, "esi", arm="conformity", condition="rec", channel="user", source="author", target_type="level", target="2") if task == "esi" else None
    with open(os.path.join(root, "p1v2_data", "stimuli_luna_p0.jsonl"), "w") as f:
        for s in S:
            f.write(json.dumps(s) + "\n")
    P_ADOPT = {"none": .03, "author": .25, "peer": .35, "consensus": .55, "own": .6}
    P_YIELD = {"none": .02, "author": .10, "peer": .20, "consensus": .35, "own": .40}
    steps = os.path.join(root, "results", "p1v2_run1", "steps")
    for mi, m in enumerate(MODELS):
        base = {}
        for c in correct:
            base[(c, "esi")] = rng.choice([2, 3, 4])
            base[(c, "usmle")] = correct[c] if rng.random() < .8 else rng.choice([x for x in "ABCD" if x != correct[c]])
        recs = []
        for s in S:
            c, task, fac = s["case_id"], s["task"], s["factors"]
            b = base[(c, task)]
            a = b
            if fac["arm"] == "bias" and fac["condition"] == "carrier":
                if rng.random() < 0.03:                                    # symmetric noise
                    a = rng.choice([2, 3, 4]) if task == "esi" else rng.choice("ABCD")
                if m == "M2" and fac["attribute"] == "race" and fac["side"] == "minus":     # planted: M2 under-triages / errs for the 'Black' side
                    if task == "esi" and rng.random() < 0.15:
                        a = min(5, int(a) + 1)
                    if task == "usmle" and a == correct[c] and rng.random() < 0.08:
                        a = rng.choice([x for x in "ABCD" if x != correct[c]])
            elif fac["arm"] == "conformity":
                src, T = fac["source"], fac["target"]
                if task == "esi":
                    if rng.random() < P_ADOPT[src] * (1.0 if fac["channel"] == "user" else 0.9):
                        a = int(T)
                else:
                    if fac["target_type"] == "wrong" and b == correct[c] and rng.random() < P_YIELD[src]:
                        a = T
                    if fac["target_type"] == "correct" and b != correct[c] and rng.random() < 0.7:
                        a = T
            ans = None if rng.random() < 0.03 else str(a)
            conf = "exact" if ans else rng.choice(["refusal", "unparseable"])
            err = None if ans or conf == "unparseable" else ("content_filter: finish_reason=content_filter" if rng.random() < .5 else None)
            recs.append(dict(stimulus_id=s["stimulus_id"], model=m, draw=0, task=task, arm=fac["arm"], case_id=c, answer=ans, confidence=conf, raw_text="x",
                             finish_reason="stop", reasoning_chars=0, usage={}, provider="x", error=err, call_id="c"))
        # split into steps; first step also holds 40 FAILED duplicates (rate-limit errors) that later succeed
        d1 = os.path.join(steps, f"{2*mi+1:02d}_agent_b_{m}"); d2 = os.path.join(steps, f"{2*mi+2:02d}_agent_b_{m}")
        os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
        cut = int(len(recs) * 0.6)
        with open(os.path.join(d1, "output.jsonl"), "w") as f:
            for r in recs[:cut]:
                f.write(json.dumps(r) + "\n")
            for r in recs[cut:cut + 40]:
                f.write(json.dumps(dict(r, answer=None, confidence="unparseable", finish_reason=None, error="failed after 5 attempts: Error code: 429")) + "\n")
        with open(os.path.join(d2, "output.jsonl"), "w") as f:
            for r in recs[cut:]:
                f.write(json.dumps(r) + "\n")
    return len(S)
