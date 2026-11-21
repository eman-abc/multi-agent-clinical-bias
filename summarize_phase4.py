#!/usr/bin/env python3
"""summarize_phase4.py -- READ-ONLY, CPU only. Reads the localization and Phase 4 outputs.

    python3 summarize_phase4.py                  # everything finished so far
    python3 summarize_phase4.py --window 14,24   # layer window for the cosine summary (default 14,24)

LOCALIZATION: patch the clean run's activation into the pressured run at ONE layer (last token) and see how much of the
lost P(correct) comes back. Restoration usually rises with depth, because late last-token states already encode the
answer; the informative thing is WHERE it jumps, so this prints the first layer reaching 50% / 90% and the biggest
single-layer jump.

PHASE 4: cosine between the saved bias direction and the saved deference direction at each layer.
CAUTION printed with every table: these bias directions come from ESI prompts and these deference directions from
USMLE prompts, so a low cosine cannot be called "separable" -- the formats differ. The matched-format grid with
reliability ceilings (run_phase4_grid.py) is the version to interpret.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

LOC_RE = re.compile(r"localization_(?P<model>.+)_(?P<pressure>consensus|single)_(?P<channel>user|assistant)\.json$")
ENT_RE = re.compile(r"entanglement_(?P<model>.+)_(?P<attr>ses|race|sex)_(?P<pressure>consensus|single)_(?P<channel>user|assistant)\.json$")


def first_at(layers, vals, thr):
    for l, v in zip(layers, vals):
        if v is not None and v >= thr:
            return l
    return None


def analyze_localization(path):
    m = LOC_RE.search(os.path.basename(path))
    r = json.load(open(path))
    rows = sorted((x for x in r.get("stage1_resid_post", []) if x.get("component") == "resid_post"), key=lambda x: x["layer"])
    layers = [x["layer"] for x in rows]
    vals = [x["restoration_fraction"] for x in rows]
    out = {"model": m["model"] if m else "?", "pressure": r.get("pressure"), "channel": r.get("channel"), "n_cases": r.get("n_cases"),
           "gap": r.get("clean_minus_pressured_gap"), "layers": layers, "restoration": vals}
    out["first_50"], out["first_90"] = first_at(layers, vals, 0.5), first_at(layers, vals, 0.9)
    jumps = [(vals[i] - vals[i - 1], layers[i - 1], layers[i]) for i in range(1, len(vals)) if vals[i] is not None and vals[i - 1] is not None]
    out["biggest_jump"] = max(jumps) if jumps else None
    beh = r.get("behavioral") or {}
    out["p_correct"] = beh.get("clean_p_correct"), beh.get("pressured_p_correct")
    return out


def render_localization(o):
    print(f"=== {o['model']} | {o.get('pressure')} | {o.get('channel')} | {o['n_cases']} cases ===")
    if o["gap"] is None:
        print("  no gap recorded\n")
        return
    pc = o["p_correct"]
    print(f"  gap      : clean P(correct) {pc[0]:.3f} - pressured {pc[1]:.3f} = {o['gap']:.3f}" + ("   <-- under 0.10: restoration fractions divide by a tiny number, ignore them" if o["gap"] < 0.10 else ""))
    print(f"  restores : first >= 50% at {'L%d' % o['first_50'] if o['first_50'] is not None else 'never'}, "
          f">= 90% at {'L%d' % o['first_90'] if o['first_90'] is not None else 'never'}")
    if o["biggest_jump"]:
        j = o["biggest_jump"]
        print(f"  biggest single-layer jump: L{j[1]} -> L{j[2]} (+{j[0]:.2f})")
    pick = [i for i, l in enumerate(o["layers"]) if l % 4 == 0 or i == len(o["layers"]) - 1]
    print("  profile  : " + " | ".join(f"L{o['layers'][i]} {o['restoration'][i]:.2f}" for i in pick if o["restoration"][i] is not None))
    print()


def analyze_entanglement(path, window):
    m = ENT_RE.search(os.path.basename(path))
    r = json.load(open(path))
    pl = [x for x in r["per_layer"] if x["cosine_similarity"] is not None and not math.isnan(x["cosine_similarity"])]
    cos = np.array([x["cosine_similarity"] for x in pl])
    ls = np.array([x["layer"] for x in pl])
    inwin = (ls >= window[0]) & (ls <= window[1])
    return {"model": r["model"], "attr": r["attribute"], "pressure": r.get("deference_pressure"), "channel": r.get("deference_channel"),
            "mean_abs": float(np.abs(cos).mean()) if len(cos) else None,
            "mean_abs_window": float(np.abs(cos[inwin]).mean()) if inwin.any() else None,
            "max_abs": float(np.abs(cos).max()) if len(cos) else None,
            "max_layer": int(ls[np.abs(cos).argmax()]) if len(cos) else None,
            "signed_window": float(cos[inwin].mean()) if inwin.any() else None}


def render_entanglement(rows, window):
    print(f"=== PHASE 4: cosine(bias direction, deference direction), layers {window[0]}-{window[1]} ===")
    print(f"  {'model':26s} {'attr':5s} {'pressure/channel':20s} {'mean|cos| all':>13s} {'mean|cos| window':>17s} {'signed (window)':>16s} {'max|cos| (layer)':>18s}")
    for o in sorted(rows, key=lambda o: (o["model"], o["attr"], str(o["channel"]))):
        print(f"  {o['model']:26s} {o['attr']:5s} {str(o['pressure']) + '/' + str(o['channel']):20s} {o['mean_abs']:13.3f} {o['mean_abs_window']:17.3f} "
              f"{o['signed_window']:+16.3f} {o['max_abs']:12.3f} (L{o['max_layer']})")
    print(f"  reference: two random unit vectors in 4096 dimensions give a mean |cos| of about {math.sqrt(2 / (math.pi * 4096)):.4f}")
    print("  CAUTION: bias = ESI prompts, deference = USMLE prompts. A low cosine here does NOT show the mechanisms are separable; "
          "use run_phase4_grid.py (matched formats + reliability ceilings).\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="paper2_results")
    ap.add_argument("--window", default="14,24")
    args = ap.parse_args()
    window = tuple(int(x) for x in args.window.split(","))
    locs = sorted(glob.glob(os.path.join(args.dir, "phase3", "metrics", "localization_*.json")))
    ents = sorted(glob.glob(os.path.join(args.dir, "phase4", "entanglement_*.json")))
    if not locs and not ents:
        print("no localization or Phase 4 files yet")
        return
    for p in locs:
        render_localization(analyze_localization(p))
    if ents:
        render_entanglement([analyze_entanglement(p, window) for p in ents], window)


if __name__ == "__main__":
    main()
