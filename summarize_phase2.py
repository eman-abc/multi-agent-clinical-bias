#!/usr/bin/env python3
"""summarize_phase2.py -- compact, READ-ONLY summary of Phase 2 logs.

    python3 summarize_phase2.py                      # every run_logs/p2_*.log
    python3 summarize_phase2.py run_logs/p2_llama_race.log --table   # one log, per-layer table

For each log it reports, per layer: whether the sign check passed, how big the steering effect is
("swing" = mean ESI shift at coefficient -8 minus shift at +8, in ESI points on the 1-5 scale), whether the
response is monotone across all four coefficients (within 0.05 ESI points), and the clamp test's CFR before/after. The sign check
alone only says the two signs differ -- the swing is what says whether the effect is big enough to matter.
No thresholds are invented: it ranks and counts, and the one count with a cutoff says so.
"""
import argparse
import ast
import glob
import os
import re

SHIFT = re.compile(r"layer=\s*(\d+)\s+mean ESI shift by coefficient:\s*(\{.*?\})\s*->\s*opposite-sign pass:\s*(True|False)")
CLAMP = re.compile(r"layer=\s*(\d+)\s+unclamped CFR=([0-9.]+)\s+clamped CFR=([0-9.]+)\s+\(n=(\d+)\)")
COEFS = (-8.0, -4.0, 4.0, 8.0)


def parse_log(path):
    """layer -> ({coef: shift}, passed) and layer -> (unclamped, clamped); last occurrence wins if a log was appended to."""
    shifts, clamp, n = {}, {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = SHIFT.search(line)
            if m:
                d = {float(k): v for k, v in ast.literal_eval(m.group(2)).items()}
                shifts[int(m.group(1))] = (d, m.group(3) == "True")
                continue
            m = CLAMP.search(line)
            if m:
                clamp[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                n = int(m.group(4))
    return shifts, clamp, n


def swing(d):
    if any(d.get(c) is None for c in COEFS):
        return None
    return d[-8.0] - d[8.0]


def monotone(d, tol=0.05):
    """True if the shift moves one way across -8,-4,+4,+8, ignoring wobbles smaller than `tol` ESI points
    (a strict check would fail a layer over a 0.002 wobble, which is noise, not a real reversal)."""
    if any(d.get(c) is None for c in COEFS):
        return False
    v = [d[c] for c in COEFS]
    return all(b >= a - tol for a, b in zip(v, v[1:])) or all(b <= a + tol for a, b in zip(v, v[1:]))


def longest_streak(flags):
    """flags: sorted [(layer, bool)] -> (length, first_layer, last_layer) of the longest run of consecutive passing layers."""
    best, cur_len, cur_start, prev = (0, None, None), 0, None, None
    for layer, ok in flags:
        if ok:
            if cur_len > 0 and prev is not None and layer == prev + 1:
                cur_len += 1
            else:
                cur_start, cur_len = layer, 1
            if cur_len > best[0]:
                best = (cur_len, cur_start, layer)
        else:
            cur_len = 0
        prev = layer
    return best


def summarize(path, table=False):
    shifts, clamp, n = parse_log(path)
    name = os.path.basename(path)
    if not shifts:
        print(f"=== {name}: no layer lines found (run not finished, or not a Phase 2 log) ===\n")
        return
    layers = sorted(shifts)
    passes = [l for l in layers if shifts[l][1]]
    length, a, b = longest_streak([(l, shifts[l][1]) for l in layers])
    mono_pass = [l for l in passes if monotone(shifts[l][0])]
    sw = {l: swing(shifts[l][0]) for l in layers if swing(shifts[l][0]) is not None}
    top = sorted(sw, key=lambda l: -abs(sw[l]))[:5]

    print(f"=== {name}  (n={n if n else '?'}) ===")
    print(f"  sign check passed: {len(passes)}/{len(layers)} layers; longest run: {length} layers"
          + (f" (L{a}-L{b})" if length else ""))
    print(f"  of the passing layers, monotone across all 4 coefficients: {len(mono_pass)}/{len(passes)}")
    print("  strongest steering swing (ESI points, shift@-8 minus shift@+8): "
          + ", ".join(f"L{l} {sw[l]:+.2f}" for l in top))
    if clamp:
        base = sorted({c[0] for c in clamp.values()})
        best_cfr = max(c[1] for c in clamp.values())
        best_layers = [l for l in sorted(clamp) if clamp[l][1] == best_cfr]
        lift5 = [l for l in sorted(clamp) if clamp[l][1] - clamp[l][0] >= 0.05]
        print(f"  clamp test: baseline CFR (no edit) {', '.join(f'{b:.2f}' for b in base)}; best clamped CFR {best_cfr:.2f} at "
              f"{', '.join('L' + str(l) for l in best_layers[:6])}{' ...' if len(best_layers) > 6 else ''}")
        print(f"  layers where clamping raised CFR by >= 0.05 (my cutoff, = {max(1, round(0.05 * (n or 100)))} cases at this n): "
              f"{len(lift5)}/{len(clamp)}" + (f"  [L{lift5[0]}-L{lift5[-1]}]" if lift5 else ""))
    else:
        print("  clamp test: no clamp lines in this log (was --test-clamp used? did the run finish?)")
    if table:
        print("\n  layer  pass  swing   shift@-8  shift@+8  mono   unclamped  clamped")
        for l in layers:
            d, ok = shifts[l]
            s = sw.get(l)
            cl = clamp.get(l)
            print(f"  {l:5d}  {str(ok):5s} {('%+.2f' % s) if s is not None else '  n/a':>6s}  "
                  f"{d.get(-8.0, float('nan')):+8.2f}  {d.get(8.0, float('nan')):+8.2f}  {str(monotone(d)):5s}  "
                  f"{(f'{cl[0]:.2f}' if cl else '-'):>9s}  {(f'{cl[1]:.2f}' if cl else '-'):>7s}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="log files (default: run_logs/p2_*.log)")
    ap.add_argument("--table", action="store_true", help="also print the per-layer table")
    args = ap.parse_args()
    paths = args.logs or sorted(glob.glob("run_logs/p2_*.log"))
    if not paths:
        print("no logs found")
        return
    for p in paths:
        summarize(p, table=args.table)


if __name__ == "__main__":
    main()