#!/usr/bin/env python3
"""
analyze_crossfit.py -- CPU only. Applies the FROZEN rules (ANALYSIS_PLAN.md section 3) to the raw held-out results from
run_crossfit.py and prints one report per model and channel, every layer included, not only the ones that pass.

    python3 analyze_crossfit.py                                 # every finished run in paper2_results/crossfit
    python3 analyze_crossfit.py --model Mistral-7B-Instruct-v0.1 --channel assistant

The plan status (FROZEN-OK / MODIFIED / NOT-FROZEN) is printed at the top and stored in the JSON summary. If the plan is
not FROZEN-OK the result is exploratory and must be described that way.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from mech import crossfit
from mech import plan as P

DEFAULT_CFG = {"primary_strength": 2.0, "secondary_strength": 1.0, "alpha_fwe": 0.05, "min_band_layers": 3,
               "bootstrap_resamples": 2000, "ci_level": 0.95, "min_pressure_effect_for_shares": 0.05}


def load_run(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def insample_raw(phase3_dir, model, pressure, channel, layers, strength):
    """In-sample raw effect at the primary strength from the Phase 3 metrics file (None where unavailable)."""
    path = os.path.join(phase3_dir, f"phase3_metrics_{model}_{pressure}_{channel}.json")
    if not os.path.exists(path):
        return [None] * len(layers)
    m = json.load(open(path))
    out = []
    for l in layers:
        try:
            v = m["per_layer"][str(l)]["d_p_stated"][str(float(strength))]
            out.append(None if v is None else -float(v))
        except (KeyError, TypeError):
            out.append(None)
    return out


def analyze_file(path, cfg, phase3_dir, status):
    z, meta = load_run(path)
    layers = [int(x) for x in z["layers"]]
    done = z["done"]
    res = {"file": os.path.basename(path), "model": meta["model"], "channel": meta["channel"], "pressure": meta["pressure"],
           "folds": meta["folds"], "plan_status": status["status"], "plan_frozen_at": status["frozen_at_utc"], "layers_done": int(done.sum()), "layers_total": len(layers)}
    if not done.all():
        res["incomplete"] = True
        return res
    real, rand = z["real"].astype(float), z["rand"].astype(float)
    base, clean = z["p_stated_pressured"].astype(float), z["p_stated_clean"].astype(float)
    press = base - clean
    rc, rrc = [float(x) for x in z["real_coefficients"]], [float(x) for x in z["random_coefficients"]]
    stats = crossfit.crossfit_statistics(real, rand, base, press, layers, rc, rrc, cfg)
    stats["insample_raw"] = insample_raw(phase3_dir, meta["model"], meta["pressure"], meta["channel"], layers, cfg["primary_strength"])
    stats["shrinkage"] = [None if (i is None or i <= 0) else t / i for t, i in zip(stats["T_primary"], stats["insample_raw"])]
    res.update(stats)
    res["has_heldout_direction"] = bool(stats["bands"])
    res["meta"] = {k: v for k, v in meta.items() if k != "case_ids"}
    return res


def render(r, cfg):
    head = f"=== {r['model']} | {r['pressure']} | {r['channel']} | {r.get('n_cases', '?')} held-out cases | {r['folds']}-fold | plan: {r['plan_status']} ==="
    print(head)
    if r.get("incomplete"):
        print(f"  INCOMPLETE: {r['layers_done']}/{r['layers_total']} layers done. Rerun run_crossfit.py (it resumes).\n")
        return
    if r["min_attainable_p"] > cfg["alpha_fwe"]:
        print(f"  WARNING: only {r['n_random']} random directions: the smallest attainable p is {r['min_attainable_p']:.3f}, above alpha {cfg['alpha_fwe']}; nothing can be called supported.")
    print(f"  mean pressure effect on P(stated): {r['mean_pressure_effect']:.3f}" + ("" if r["shares_reported"] else "   (below the minimum: shares not reported)"))
    print(f"  held-out change in P(stated) at +{cfg['primary_strength']:g} gaps (T > 0: the real direction lowers it); p_FWE = max-T over {len(r['layers'])} layers")
    print(f"  {'layer':>5s} {'T(+2)':>7s} {'T(+1)':>7s} {'null mean':>9s} {'null max':>8s} {'p':>6s} {'p_FWE':>6s} {'supported':>10s} {'raw share':>10s} {'NET share [95% CI]':>24s} {'in-sample':>10s} {'held/in':>8s}")
    for i, l in enumerate(r["layers"]):
        net = f"{100 * r['net_share'][i]:5.0f}% [{100 * r['net_share_ci'][i][0]:4.0f},{100 * r['net_share_ci'][i][1]:4.0f}]" if r["shares_reported"] else "-"
        raw = f"{100 * r['raw_share'][i]:7.0f}%" if r["shares_reported"] else "-"
        ins = f"{r['insample_raw'][i]:.3f}" if r["insample_raw"][i] is not None else "-"
        shr = f"{r['shrinkage'][i]:.2f}" if r["shrinkage"][i] is not None else "-"
        print(f"  L{l:<4d} {r['T_primary'][i]:+7.3f} {r['T_secondary'][i]:+7.3f} {r['null_mean_by_layer'][i]:9.3f} {r['null_max_by_layer'][i]:8.3f} {r['p_unc'][i]:6.3f} {r['p_fwe'][i]:6.3f} {('YES' if r['supported'][i] else 'no'):>10s} {raw:>10s} {net:>24s} {ins:>10s} {shr:>8s}")
    if r["bands"]:
        for b in r.get("band_net_share", []):
            print(f"  BAND L{b['band'][0]}-L{b['band'][1]}: mean net share {100 * b['mean']:.0f}% (95% CI {100 * b['ci'][0]:.0f} to {100 * b['ci'][1]:.0f}%)")
        print("  VERDICT (frozen rule: a band of >= %d consecutive supported layers): this model/channel HAS a held-out deference direction." % cfg["min_band_layers"])
    else:
        print("  VERDICT (frozen rule): NO band of >= %d consecutive supported layers; no held-out deference direction is claimed." % cfg["min_band_layers"])
    if r["isolated"]:
        print(f"  isolated supported layers (reported, NOT interpreted): {', '.join('L%d' % l for l in r['isolated'])}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="paper2_results/crossfit")
    ap.add_argument("--phase3-dir", default="paper2_results/phase3/metrics")
    ap.add_argument("--plan", default=None, help="default: analysis_plan.json if it exists")
    ap.add_argument("--model", default=None)
    ap.add_argument("--channel", default=None)
    args = ap.parse_args()
    plan_path = args.plan or ("analysis_plan.json" if os.path.exists("analysis_plan.json") else None)
    cfg = P.load_plan(plan_path)["crossfit"] if plan_path else DEFAULT_CFG
    status = P.plan_status(plan_path) if plan_path else {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": ["no plan file"]}
    print(P.banner(status) + "\n")
    files = sorted(f for f in glob.glob(os.path.join(args.dir, "crossfit_*.npz")) if not f.endswith("_acts.npz") and ".tmp" not in f)
    if args.model:
        files = [f for f in files if f"crossfit_{args.model}_" in f]
    if args.channel:
        files = [f for f in files if f.endswith(f"_{args.channel}.npz")]
    if not files:
        print("no crossfit results found in", args.dir)
        return 0
    for f in files:
        r = analyze_file(f, cfg, args.phase3_dir, status)
        render(r, cfg)
        out = os.path.join(args.dir, "summary_" + os.path.basename(f).replace(".npz", ".json"))
        json.dump(r, open(out, "w"), indent=1, default=float)
    print("Intervals are over cases; randomization p-values use the random-direction null; see ANALYSIS_PLAN.md section 6 for the design's known limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
