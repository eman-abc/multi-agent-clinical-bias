#!/usr/bin/env python3
"""
run_phase4_grid.py -- Stage C3: the comparison grid. CPU only, reads saved directions; run it whenever new ones exist.

For each model it compares every pair of saved directions that belongs to a question the paper asks, per layer, and puts
each cosine next to what it could possibly be:

  relation                                         why it is in the grid
  bias vs deference, USMLE (matched format)        THE comparison, once C1 has built bias on USMLE prompts
  bias vs deference, ESI (matched format)          THE comparison on the ESI side, once C2 exists
  bias vs deference, ESI bias x USMLE deference    what the original Phase 4 computed; format-MISMATCHED
  same bias, ESI vs USMLE prompts                  how much the FORMAT alone moves a direction (a ceiling in practice)
  same deference, ESI vs USMLE prompts             same, for deference (needs C2)
  deference, user vs assistant channel             do the two channels share one direction?
  bias across attributes (SES / race / sex)        is there one demographic direction or three?

Columns: mean |cos| over a layer window (default 14-24), the strongest layer, and when per-direction reliability files
exist (run_reliability.py, run_bias_usmle.py):
  noise max : the largest window-mean |cos| any of the sign-flip null directions reached with the other direction
  ceiling   : sqrt(r_A * r_B), what two NOISY copies of the SAME direction could reach (Spearman-Brown split-half)
  / ceiling : the observed |cos| as a fraction of that ceiling (a disattenuated cosine)

READ IT LIKE THIS (my cutoffs, not published ones): well above noise max AND a large fraction of the ceiling = consistent
with a shared direction; at or below noise max = nothing shared that this data can detect; in between = partial overlap.
The ceiling treats each direction as a noisy measurement of one true direction -- an assumption.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

from mech import reliability as rel
from mech.checkpoint import DirectionStore, export_metrics_json

MODELS = ["Mistral-7B-Instruct-v0.1", "Llama-3.1-8B-Instruct"]
ATTRS = ["ses", "race", "sex"]
CHANNELS = ["user", "assistant"]
PRESSURE = "consensus"
STORE_SUBDIR = {"bias_esi": "phase2/directions", "bias_usmle": "phase2usmle/directions", "def_usmle": "phase3/directions", "def_esi": "phase3esi/directions"}


def load_dirs(root: str, kind: str, model: str, key: str):
    """(n_layers, d) saved directions, or None. key: attribute for bias_*, '<pressure>_<channel>' for def_*."""
    store = DirectionStore(os.path.join(root, STORE_SUBDIR[kind]))
    attr = key if kind.startswith("bias") else f"deference_{key}"
    layers = sorted(store.list_layers(model, attr))
    if not layers or layers != list(range(len(layers))):
        return None
    return np.stack([store.load(model, attr, l)["direction"] for l in layers])


def relations():
    """(relation, (kindA, keyA), (kindB, keyB)) for every pair the grid reports."""
    out = []
    for a in ATTRS:
        for c in CHANNELS:
            d = f"{PRESSURE}_{c}"
            out.append(("bias vs deference, USMLE (matched)", ("bias_usmle", a), ("def_usmle", d)))
            out.append(("bias vs deference, ESI (matched)", ("bias_esi", a), ("def_esi", d)))
            out.append(("bias vs deference, ESI bias x USMLE deference (MISMATCHED)", ("bias_esi", a), ("def_usmle", d)))
    for a in ATTRS:
        out.append(("same bias, ESI vs USMLE prompts", ("bias_esi", a), ("bias_usmle", a)))
    for c in CHANNELS:
        d = f"{PRESSURE}_{c}"
        out.append(("same deference, ESI vs USMLE prompts", ("def_esi", d), ("def_usmle", d)))
    for kind in ("def_usmle", "def_esi"):
        out.append((f"deference, user vs assistant channel ({kind.split('_')[1].upper()})", (kind, f"{PRESSURE}_user"), (kind, f"{PRESSURE}_assistant")))
    for kind in ("bias_esi", "bias_usmle"):
        for a, b in itertools.combinations(ATTRS, 2):
            out.append((f"bias across attributes ({kind.split('_')[1].upper()})", (kind, a), (kind, b)))
    return out


def compare(dirs_a, dirs_b, rel_a, rel_b, window):
    L = min(len(dirs_a), len(dirs_b))
    a, b = dirs_a[:L], dirs_b[:L]
    cos = rel.layer_cosines(a, b)
    lo, hi = window
    w = np.arange(L)
    inwin = (w >= lo) & (w <= min(hi, L - 1))
    res = {"cos_by_layer": [float(x) for x in cos], "mean_abs_window": float(np.abs(cos[inwin]).mean()), "signed_window": float(cos[inwin].mean()),
           "max_abs": float(np.abs(cos).max()), "max_layer": int(np.abs(cos).argmax()), "noise_max": None, "ceiling": None, "frac_of_ceiling": None}
    if rel_a is not None and rel_b is not None:
        ra = rel.spearman_brown(rel_a["split_half"].mean(axis=0))[:L]
        rb = rel.spearman_brown(rel_b["split_half"].mean(axis=0))[:L]
        ceil = float(rel.attenuation_ceiling(ra, rb)[inwin].mean())
        n1 = rel.null_abs_cosines(rel_a["null_dirs"][:, :L, :], b)[:, inwin].mean(axis=1)
        n2 = rel.null_abs_cosines(rel_b["null_dirs"][:, :L, :], a)[:, inwin].mean(axis=1)
        res["noise_max"] = float(max(n1.max(), n2.max()))
        res["ceiling"] = ceil
        res["frac_of_ceiling"] = float(res["mean_abs_window"] / ceil) if ceil > 0.05 else None
    return res


def reading(res):
    if res["noise_max"] is None:
        return "no reliability files: cannot calibrate"
    if res["mean_abs_window"] <= res["noise_max"]:
        return "at or below noise"
    f = res["frac_of_ceiling"]
    if f is None:
        return "above noise; ceiling too low to scale"
    return "near the ceiling" if f >= 0.7 else ("partial overlap" if f >= 0.25 else "above noise, far below ceiling")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="paper2_results")
    ap.add_argument("--window", default="14,24")
    ap.add_argument("--out", default=None, help="default <root>/phase4/grid.json")
    args = ap.parse_args()
    window = tuple(int(x) for x in args.window.split(","))
    rows = []
    for model in MODELS:
        printed_header = False
        for name, (ka, ya), (kb, yb) in relations():
            da, db = load_dirs(args.root, ka, model, ya), load_dirs(args.root, kb, model, yb)
            if da is None or db is None:
                continue
            ra = rel.load_reliability(rel.reliability_path(ka, model, ya, args.root))
            rb = rel.load_reliability(rel.reliability_path(kb, model, yb, args.root))
            res = compare(da, db, ra, rb, window)
            res.update({"model": model, "relation": name, "a": f"{ka}:{ya}", "b": f"{kb}:{yb}"})
            rows.append(res)
            if not printed_header:
                print(f"\n=== {model}: cosine between saved directions, layers {window[0]}-{window[1]} ===")
                print(f"  {'relation':62s} {'A':>22s} {'B':>26s} {'mean|cos|':>10s} {'peak (layer)':>14s} {'noise max':>10s} {'ceiling':>8s} {'/ceiling':>9s}  reading")
                printed_header = True
            f = lambda v, fmt: ("n/a" if v is None else format(v, fmt))
            print(f"  {name:62s} {res['a']:>22s} {res['b']:>26s} {res['mean_abs_window']:10.3f} {res['max_abs']:8.3f} (L{res['max_layer']:<2d}) "
                  f"{f(res['noise_max'], '10.3f'):>10s} {f(res['ceiling'], '8.3f'):>8s} {f(res['frac_of_ceiling'], '9.2f'):>9s}  {reading(res)}")
    if not rows:
        print("no pair of saved directions found under", args.root)
        return 0
    out = args.out or os.path.join(args.root, "phase4", "grid.json")
    export_metrics_json({"window": list(window), "rows": rows, "random_reference_cos_d4096": math.sqrt(2 / (math.pi * 4096))}, out)
    print(f"\nrandom unit vectors in 4096 dimensions: mean |cos| about {math.sqrt(2 / (math.pi * 4096)):.4f}.  Saved {out}")
    print("Cells are blank where a direction has not been built yet (C1 builds bias on USMLE; C2 builds deference on ESI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
