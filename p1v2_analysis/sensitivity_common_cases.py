"""Sensitivity: do the headline results change if only cases that EVERY model answered cleanly are used? Read-only; writes new files only.
  python p1v2_analysis\\sensitivity_common_cases.py
'Clean case' = no invalid answer (refusal, content filter, unparseable) for any stimulus of that case in any model. The registered analysis is unchanged;
this is a derived robustness check to be reported as such."""
import argparse, collections as C, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_p1v2 as A
import pool_p1v2 as P


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--run", default="p1v2_run1")
    ap.add_argument("--stimuli", default=os.path.join("p1v2_data", "stimuli_luna_p0.jsonl"))
    ap.add_argument("--cases", default=os.path.join("p1v2_data", "banks", "p1v2_bank_main", "cases_clean.jsonl"))
    ap.add_argument("--out", default=os.path.join("p1v2_analysis", "results")); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    S = A.load_stimuli(os.path.join(a.root, a.stimuli)); ANS = A.load_answers(a.root, a.run)
    correct = {str(r["case_id"]): r["correct"] for r in A.rd(os.path.join(a.root, a.cases))}
    bad = set()
    for m, recs in ANS.items():
        for sid, r in recs.items():
            if sid in S and A.status(r, S[sid]["task"])[1] != "ok":
                bad.add(S[sid]["case"])
    allc = {s["case"] for s in S.values()}
    keep = allc - bad
    print(f"cases: {len(allc)} | with at least one invalid answer in any model: {len(bad)} | clean cases kept: {len(keep)} ({100*len(keep)/len(allc):.0f}%)")
    def run_set(Ssub):
        V, ST, BASE = A.build(Ssub, ANS, correct)
        cell, pool = A.rq1(Ssub, V, correct, a.boot, a.seed)
        obs = P.build_obs(Ssub, V, BASE, correct)
        rates = {}
        for ck, mets in obs.items():
            for metric, o in mets.items():
                rates[ck + (metric,)] = P.cluster_ci(o, a.boot, a.seed)
        return pool, rates
    full_pool, full_rates = run_set(S)
    sub_pool, sub_rates = run_set({k: v for k, v in S.items() if v["case"] in keep})
    rows = []
    # RQ1: pooled carrier effects, full vs clean cases
    fp = {(r["model"], r["task"], r["attr"], r["form"]): r for r in full_pool}
    for r in sub_pool:
        k = (r["model"], r["task"], r["attr"], r["form"]); f = fp[k]
        rows.append(dict(analysis="RQ1 carrier effect", model=r["model"], cell=f"{r['task']} {r['attr']} {r['form']}", full=f["effect"], full_lo=f["lo"], full_hi=f["hi"], clean=r["effect"], clean_lo=r["lo"], clean_hi=r["hi"],
                         n_full=f["n"], n_clean=r["n"], label_full=f["equiv"], label_clean=r["equiv"], changed=f["equiv"] != r["equiv"], shift=r["effect"] - f["effect"]))
    # RQ2 headline rates
    heads = [("usmle", "user", "none", "wrong", "all", "yield"), ("usmle", "user", "consensus", "wrong", "all", "yield"), ("usmle", "user", "peer", "wrong", "all", "yield"), ("usmle", "assistant", "consensus", "wrong", "all", "yield"),
             ("usmle", "user", "consensus", "correct", "all", "rescue"), ("esi", "user", "none", "level", "1", "adopted"), ("esi", "user", "consensus", "level", "1", "adopted"), ("esi", "assistant", "consensus", "level", "1", "adopted"),
             ("esi", "user", "none", "level", "5", "adopted"), ("esi", "user", "consensus", "level", "5", "adopted")]
    for m in sorted({k[0] for k in full_rates}):
        for (task, chan, src, tt, tg, met) in heads:
            k = (m, task, chan, src, tt, tg, met)
            if k in full_rates and k in sub_rates:
                f, c = full_rates[k], sub_rates[k]
                rows.append(dict(analysis="RQ2 rate", model=m, cell=f"{task} {chan} {src} {tt} {tg} {met}", full=f["est"], full_lo=f["lo"], full_hi=f["hi"], clean=c["est"], clean_lo=c["lo"], clean_hi=c["hi"],
                                 n_full=f["n_cases"], n_clean=c["n_cases"], label_full="", label_clean="", changed=False, shift=c["est"] - f["est"]))
    out = os.path.join(a.root, a.out); os.makedirs(out, exist_ok=True)
    A.write_csv(rows, os.path.join(out, "sensitivity_common_cases.csv"))
    r1 = [r for r in rows if r["analysis"] == "RQ1 carrier effect"]; r2 = [r for r in rows if r["analysis"] == "RQ2 rate"]
    print(f"RQ1: {sum(r['changed'] for r in r1)} of {len(r1)} cells change their equivalence label; largest shift in an effect {max(abs(r['shift']) for r in r1):.3f} (triage levels or diagnosis accuracy units)")
    print(f"RQ2: largest shift in a headline rate {100*max(abs(r['shift']) for r in r2):.1f} points")
    for r in r2:
        if r["cell"].startswith("usmle user consensus wrong") or r["cell"].startswith("usmle user none wrong") or r["cell"].startswith("esi user consensus level 1"):
            print(f"  {r['model']:18s} {r['cell']:40s} full {100*r['full']:5.1f}%  clean {100*r['clean']:5.1f}%  (n {r['n_full']} -> {r['n_clean']})")
    for r in r1:
        if r["changed"]:
            print(f"  label change: {r['model']} {r['cell']}: {r['label_full']} -> {r['label_clean']}")
    print("wrote sensitivity_common_cases.csv to", out)


if __name__ == "__main__":
    main()
