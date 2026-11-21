"""Do invalid answers (refusal, content filter, unparseable) concentrate on particular cases or conditions? Read-only; writes new files only.
  python p1v2_analysis\\check_invalid_by_case.py
Reports per model: how many cases are affected, the share of all invalid answers that fall in the 10 worst cases against what chance would give
(permutation, 500 shuffles of the invalid flags across stimuli), the invalid rate by condition, and the overlap between models."""
import argparse, collections as C, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_p1v2 as A


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--run", default="p1v2_run1")
    ap.add_argument("--stimuli", default=os.path.join("p1v2_data", "stimuli_luna_p0.jsonl"))
    ap.add_argument("--out", default=os.path.join("p1v2_analysis", "results")); ap.add_argument("--perm", type=int, default=500)
    a = ap.parse_args(argv)
    S = A.load_stimuli(os.path.join(a.root, a.stimuli)); ANS = A.load_answers(a.root, a.run)
    rng = np.random.default_rng(0)
    out = os.path.join(a.root, a.out); os.makedirs(out, exist_ok=True)
    rows_case, rows_cond, bad_sets, case_sets = [], [], {}, {}
    for m, recs in ANS.items():
        sids = [s for s in S if s in recs]
        flags = np.array([A.status(recs[s], S[s]["task"])[1] != "ok" for s in sids])
        cases = np.array([S[s]["case"] for s in sids])
        uc, inv = np.unique(cases, return_inverse=True)
        per_case = np.bincount(inv, weights=flags.astype(float), minlength=len(uc))
        n_per_case = np.bincount(inv, minlength=len(uc))
        tot = flags.sum()
        top10 = np.sort(per_case)[::-1][:10].sum() / max(tot, 1)
        null = []
        for _ in range(a.perm):
            f2 = rng.permutation(flags)
            pc = np.bincount(inv, weights=f2.astype(float), minlength=len(uc))
            null.append(np.sort(pc)[::-1][:10].sum() / max(tot, 1))
        null = np.array(null)
        affected = int((per_case > 0).sum())
        print(f"{m}: {int(tot)} invalid of {len(flags)} ({100*tot/len(flags):.1f}%); {affected} of {len(uc)} cases affected; the 10 worst cases hold {100*top10:.1f}% of them (chance: {100*null.mean():.1f}%, 95th pct {100*np.percentile(null,95):.1f}%)")
        worst = np.argsort(-per_case)[:10]
        rows_case.append(dict(model=m, invalid=int(tot), calls=len(flags), cases_affected=affected, cases=len(uc), top10_share=top10, chance_mean=float(null.mean()), chance_p95=float(np.percentile(null, 95)),
                              concentrated=bool(top10 > np.percentile(null, 95)), worst_cases=" ".join(f"{uc[i]}:{int(per_case[i])}/{int(n_per_case[i])}" for i in worst)))
        bad_sets[m] = {s for s, f in zip(sids, flags) if f}; case_sets[m] = {c for c, k in zip(uc, per_case) if k > 0}
        cond = C.defaultdict(lambda: [0, 0])
        for s, f in zip(sids, flags):
            x = S[s]
            key = (x["arm"], x["task"], x["cond"] if x["arm"] == "bias" and x["cond"] == "N" else (x["attr"] + "/" + x["form"] if x["arm"] == "bias" else x["chan"] + "/" + x["src"] + "/" + x["ttype"]))
            cond[key][0] += int(f); cond[key][1] += 1
        for (arm, task, c), (k, n) in sorted(cond.items()):
            rows_cond.append(dict(model=m, arm=arm, task=task, condition=c, invalid=k, calls=n, rate=k / n))
    rows_ov = []
    ms = list(bad_sets)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a_, b_ = ms[i], ms[j]
            rows_ov.append(dict(model_a=a_, model_b=b_, stimuli_overlap=len(bad_sets[a_] & bad_sets[b_]), stimuli_union=len(bad_sets[a_] | bad_sets[b_]),
                                cases_overlap=len(case_sets[a_] & case_sets[b_]), cases_union=len(case_sets[a_] | case_sets[b_])))
    for name, rows in (("invalid_by_case", rows_case), ("invalid_by_condition", rows_cond), ("invalid_overlap", rows_ov)):
        A.write_csv(rows, os.path.join(out, name + ".csv"))
    for r in rows_ov:
        print(f"overlap {r['model_a']} / {r['model_b']}: {r['stimuli_overlap']} of {r['stimuli_union']} invalid stimuli shared; {r['cases_overlap']} of {r['cases_union']} affected cases shared")
    print("wrote invalid_by_case.csv, invalid_by_condition.csv, invalid_overlap.csv to", out)


if __name__ == "__main__":
    main()
