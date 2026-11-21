#!/usr/bin/env python3
"""
analyze_attn_mlp.py -- CPU only. Applies the FROZEN rules (ANALYSIS_PLAN.md section 4) to the raw results of
run_attn_mlp_split.py.

    python3 analyze_attn_mlp.py                                  # every finished run in paper2_results/attn_mlp
    python3 analyze_attn_mlp.py --model Mistral-7B-Instruct-v0.1

Restoration = (mean patched P(correct) - mean pressured) / (mean clean - mean pressured); intervals resample CASES.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from mech import plan as P

DEFAULT_CFG = {"dominance_margin": 0.10, "min_dominant_restoration": 0.15, "min_gap": 0.10, "bootstrap_resamples": 2000, "ci_level": 0.95}
TYPES = ["attn_out", "mlp_out", "both"]


def _pct(x, ci, axis=-1):
    return np.percentile(x, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=axis)


def analyze_file(path, cfg, status, seed=0):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    clean, press = z["clean_pc"].astype(float), z["pressured_pc"].astype(float)
    n = len(clean)
    res = {"file": os.path.basename(path), "model": meta["model"], "channel": meta["channel"], "pressure": meta["pressure"], "n_cases": n,
           "plan_status": status["status"], "plan_frozen_at": status["frozen_at_utc"], "exactness_max_abs_diff": meta["exactness_max_abs_diff"], "valid": meta["valid"],
           "gap": float(clean.mean() - press.mean())}
    if not meta["valid"]:
        res["invalid"] = True
        return res
    if res["gap"] < cfg["min_gap"]:
        res["uninformative"] = True
        return res
    ci, B = cfg["ci_level"], cfg["bootstrap_resamples"]
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    cb, pb = clean[idx].mean(axis=1), press[idx].mean(axis=1)          # (B,)
    den_b = cb - pb
    gap = res["gap"]

    pwin = z["patched_window"].astype(float)                           # (W, T, n)
    Rp = (pwin.mean(axis=-1) - press.mean()) / gap                     # (W, T)
    Rb = (pwin[..., idx].mean(axis=-1) - pb) / den_b                   # (W, T, B)
    lo, hi = _pct(Rb, ci, axis=-1)
    ia, im = TYPES.index("attn_out"), TYPES.index("mlp_out")
    wins = []
    for w, (a, b) in enumerate(z["windows"].tolist()):
        D = Rp[w, ia] - Rp[w, im]
        Dl, Dh = _pct(Rb[w, ia] - Rb[w, im], ci)
        if Dl > cfg["dominance_margin"] and lo[w, ia] > cfg["min_dominant_restoration"]:
            reading = "ATTENTION-DOMINANT"
        elif Dh < -cfg["dominance_margin"] and lo[w, im] > cfg["min_dominant_restoration"]:
            reading = "MLP-DOMINANT"
        else:
            reading = "neither dominant"
        wins.append({"window": [a, b], "role": meta["roles"][w],
                     "restoration": {t: {"point": float(Rp[w, k]), "ci": [float(lo[w, k]), float(hi[w, k])]} for k, t in enumerate(TYPES)},
                     "attn_minus_mlp": {"point": float(D), "ci": [float(Dl), float(Dh)]}, "reading": reading})
    res["windows"] = wins
    # specificity of the primary window's attention effect against each control window (paired)
    spec = []
    for w in range(1, len(wins)):
        d = Rb[0, ia] - Rb[w, ia]
        l, h = _pct(d, ci)
        spec.append({"control": wins[w]["window"], "primary_minus_control": float(Rp[0, ia] - Rp[w, ia]), "ci": [float(l), float(h)], "specific": bool(l > 0)})
    res["specificity"] = spec
    res["primary_is_specific"] = bool(spec) and all(s["specific"] for s in spec)

    psw = z["patched_sweep"].astype(float)                             # (S, 2, n)
    Rs = (psw.mean(axis=-1) - press.mean()) / gap
    Rsb = (psw[..., idx].mean(axis=-1) - pb) / den_b
    sl, sh = _pct(Rsb, ci)
    res["sweep"] = [{"layer": int(l), "attn_out": {"point": float(Rs[i, 0]), "ci": [float(sl[i, 0]), float(sh[i, 0])]},
                     "mlp_out": {"point": float(Rs[i, 1]), "ci": [float(sl[i, 1]), float(sh[i, 1])]}} for i, l in enumerate(z["sweep_layers"].tolist())]
    return res


def render(r, cfg):
    print(f"=== {r['model']} | {r['pressure']} | {r['channel']} | {r['n_cases']} cases | plan: {r['plan_status']} ===")
    if r.get("invalid"):
        print(f"  INVALID RUN: the exactness check failed (max difference {r['exactness_max_abs_diff']:.2e}); patching is not behaving as an identity. Do not use.\n")
        return
    if r.get("uninformative"):
        print(f"  UNINFORMATIVE: the clean-minus-pressured gap in P(correct) is {r['gap']:.3f}, below the minimum {cfg['min_gap']}; restoration is not defined.\n")
        return
    print(f"  gap in P(correct) (clean - pressured): {r['gap']:.3f}; exactness check passed (max difference {r['exactness_max_abs_diff']:.1e})")
    print(f"  restoration of P(correct) when the clean last-token output is patched into the pressured run, all layers of the window at once")
    print(f"  {'window':>8s} {'role':>9s} {'attention only':>22s} {'MLP only':>22s} {'both':>22s} {'attn - MLP':>22s}  reading")
    f = lambda d: f"{d['point']:+5.2f} [{d['ci'][0]:+5.2f},{d['ci'][1]:+5.2f}]"
    for w in r["windows"]:
        rr = w["restoration"]
        print(f"  L{w['window'][0]}-{w['window'][1]:<4d} {w['role']:>9s} {f(rr['attn_out']):>22s} {f(rr['mlp_out']):>22s} {f(rr['both']):>22s} {f(w['attn_minus_mlp']):>22s}  {w['reading']}")
    for s in r["specificity"]:
        print(f"  specificity vs L{s['control'][0]}-{s['control'][1]}: attention restoration is higher in the primary window by {s['primary_minus_control']:+.2f} [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}] -> {'specific' if s['specific'] else 'NOT shown to be specific'}")
    prim = r["windows"][0]
    print(f"  VERDICT (frozen rules): primary window L{prim['window'][0]}-{prim['window'][1]} is {prim['reading']}; the window is {'SPECIFIC' if r['primary_is_specific'] else 'not shown to be specific'} relative to the control windows.")
    print(f"  single-layer sweep (descriptive, no rule): restoration by one layer's output")
    print(f"  {'layer':>6s} {'attention':>22s} {'MLP':>22s}")
    for s in r["sweep"]:
        print(f"  L{s['layer']:<5d} {f(s['attn_out']):>22s} {f(s['mlp_out']):>22s}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="paper2_results/attn_mlp")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--channel", default=None)
    args = ap.parse_args()
    plan_path = args.plan or ("analysis_plan.json" if os.path.exists("analysis_plan.json") else None)
    cfg = P.load_plan(plan_path)["attn_mlp_split"] if plan_path else DEFAULT_CFG
    status = P.plan_status(plan_path) if plan_path else {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": ["no plan file"]}
    print(P.banner(status) + "\n")
    files = sorted(f for f in glob.glob(os.path.join(args.dir, "attn_mlp_*.npz")) if ".tmp" not in f)
    if args.model:
        files = [f for f in files if f"attn_mlp_{args.model}_" in f]
    if args.channel:
        files = [f for f in files if f.endswith(f"_{args.channel}.npz")]
    if not files:
        print("no attn_mlp results found in", args.dir)
        return 0
    for f in files:
        r = analyze_file(f, cfg, status)
        render(r, cfg)
        json.dump(r, open(os.path.join(args.dir, "summary_" + os.path.basename(f).replace(".npz", ".json")), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
