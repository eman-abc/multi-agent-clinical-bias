#!/usr/bin/env python3
"""summarize_phase3.py -- READ-ONLY, CPU only, safe to run while the queue is running.

    python3 summarize_phase3.py                 # every finished Phase 3 step
    python3 summarize_phase3.py --table         # also a per-layer table
    python3 summarize_phase3.py --file paper2_results/phase3/metrics/phase3_metrics_<model>_consensus_user.json

Reads, per finished step: the metrics file (per-case baselines) and steering_validation.jsonl (per-case steered
probabilities for the deference direction AND for the random-direction control). From those it computes 95% bootstrap
intervals over CASES, instead of comparing bare averages.

A layer is called "restoring" only if, at EVERY positive strength tested (+1 and +2 gaps):
  (1) the mean change in P(stated wrong letter) is below zero, its 95% interval excludes zero, and it exceeds the run's
      minimum-effect floor; and
  (2) it lowers P(stated) MORE than the random-direction baseline does, with the 95% interval of that difference
      (random minus real, paired by case) excluding zero.

Read the result with these limits in mind (printed again at the end):
  * 32 layers x 2 strengths at 95% intervals -> expect 1-2 layers to pass BY CHANCE. Trust contiguous runs of layers,
    not a single isolated layer.
  * The intervals reflect which CASES were sampled. The random baseline is ONE random direction per layer, so it does
    not reflect which random direction was drawn.
  * The direction was fit on the same cases it is tested on (in-sample), so effect sizes are probably optimistic.
"""
import argparse
import glob
import json
import os

import numpy as np


def wilson(x, n, z=1.96):
    """95% Wilson interval for a proportion x/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def ranges(layers):
    """[14,15,16,19] -> 'L14-L16, L19'"""
    layers = sorted(layers)
    if not layers:
        return "none"
    out, start, prev = [], layers[0], layers[0]
    for l in layers[1:] + [None]:
        if l is not None and l == prev + 1:
            prev = l
            continue
        out.append(f"L{start}" if start == prev else f"L{start}-L{prev}")
        if l is not None:
            start = prev = l
    return ", ".join(out)


def load_steering_rows(jsonl_path):
    """{(model, attribute): {(layer, coefficient, case_id): class_probabilities}}; the last row wins on duplicates."""
    rows = {}
    if not os.path.exists(jsonl_path):
        return rows
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("class_probabilities"):
                continue
            rows.setdefault((r["model"], r["attribute"]), {})[(int(r["layer"]), float(r["coefficient"]), r["case_id"])] = r["class_probabilities"]
    return rows


def analyze(metrics_path, rows, boot=2000, alpha=0.05, seed=0):
    m = json.load(open(metrics_path))
    out = {"file": os.path.basename(metrics_path), "model": m["model"], "pressure": m.get("pressure"), "channel": m.get("channel")}
    pc = m.get("per_case")
    if not pc:
        out["error"] = "no per-case baselines in this metrics file (it was written by a run that predates the per-case logging)"
        return out
    rng = np.random.default_rng(seed)
    cases = sorted(pc)
    stated = {c: pc[c]["stated"] for c in cases}
    correct = {c: pc[c]["correct"] for c in cases}
    base = {c: pc[c]["pressured_probs"][stated[c]] for c in cases}

    # ---- behavior, with an interval on the yield
    top = lambda p: max(p, key=p.get)
    knew = [c for c in cases if top(pc[c]["clean_probs"]) == correct[c]]
    flipped = [c for c in knew if top(pc[c]["pressured_probs"]) == stated[c]]
    out["behavior"] = {
        "n": len(cases),
        "p_correct": (float(np.mean([pc[c]["clean_probs"][correct[c]] for c in cases])), float(np.mean([pc[c]["pressured_probs"][correct[c]] for c in cases]))),
        "p_stated": (float(np.mean([pc[c]["clean_probs"][stated[c]] for c in cases])), float(np.mean([pc[c]["pressured_probs"][stated[c]] for c in cases]))),
        "yield": (len(flipped), len(knew)), "yield_ci": wilson(len(flipped), len(knew)),
    }

    # ---- steering
    attr = m["attribute"]
    dim_rows = rows.get((m["model"], attr), {})
    rnd_rows = rows.get((m["model"], attr + "__random"), {})
    out["has_random_rows"] = bool(rnd_rows)
    coefs = [float(c) for c in m["coefficients"]]
    pos = sorted(c for c in coefs if c > 0)
    min_effect = float(m.get("min_effect", 0.0) or 0.0)
    pressure_effect = {c: pc[c]["pressured_probs"][stated[c]] - pc[c]["clean_probs"][stated[c]] for c in cases}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    layers_out = {}
    for l in [int(x) for x in m["validation_layers"]]:
        use = [c for c in cases if all((l, p, c) in dim_rows for p in pos)]
        if len(use) < 5:
            continue
        idx = rng.integers(0, len(use), size=(boot, len(use)))
        press = np.array([pressure_effect[c] for c in use])
        res = {"n": len(use), "pos": {}}
        control_ok = bool(rnd_rows)
        ok_all = True
        for p in pos:
            dim = np.array([dim_rows[(l, p, c)][stated[c]] - base[c] for c in use])
            dmeans = dim[idx].mean(axis=1)
            entry = {"dim": float(dim.mean()), "dim_ci": tuple(np.percentile(dmeans, [lo_q, hi_q])), "rand": None, "exc": None, "exc_ci": None, "frac": None, "frac_ci": None, "frac_net": None, "frac_net_ci": None}
            den = press[idx].mean(axis=1)
            if abs(press.mean()) >= 0.05:
                fr = -dmeans / den
                entry["frac"] = float(-dim.mean() / press.mean())
                entry["frac_ci"] = tuple(np.percentile(fr, [lo_q, hi_q]))
            if control_ok:
                rvals = []
                for c in use:
                    vs = [rnd_rows[(l, s * p, c)][stated[c]] - base[c] for s in (1, -1) if (l, s * p, c) in rnd_rows]
                    if not vs:
                        control_ok = False
                        break
                    rvals.append(float(np.mean(vs)))
                if control_ok:
                    r = np.array(rvals)
                    exc = r - dim
                    emeans = exc[idx].mean(axis=1)
                    entry["rand"] = float(r.mean())
                    entry["exc"] = float(exc.mean())
                    entry["exc_ci"] = tuple(np.percentile(emeans, [lo_q, hi_q]))
                    if abs(press.mean()) >= 0.05:
                        entry["frac_net"] = float(exc.mean() / press.mean())
                        entry["frac_net_ci"] = tuple(np.percentile(emeans / den, [lo_q, hi_q]))
            res["pos"][p] = entry
            clear = entry["dim_ci"][1] < 0 and entry["dim"] < -min_effect
            ok_all = ok_all and clear and (not control_ok or (entry["exc_ci"] is not None and entry["exc_ci"][0] > 0))
        res["control"] = control_ok
        res["restoring"] = bool(ok_all) if control_ok else None
        res["dim_only_clear"] = bool(ok_all) if not control_ok else None
        layers_out[l] = res
    out["layers"] = layers_out
    out["positive_coefs"] = pos
    out["min_effect"] = min_effect
    out["n_layers"] = len(layers_out)

    # ---- clamp test (aggregate numbers only; the metrics file holds no per-case clamp outcomes)
    ce = m.get("clamp_effectiveness") or {}
    if ce:
        items = sorted((int(k), v) for k, v in ce.items())
        base_y = items[0][1]["unclamped_yield_rate"]
        best = min(v["clamped_yield_rate"] for _, v in items)
        out["clamp"] = {"baseline": base_y, "best": best, "best_layers": [l for l, v in items if v["clamped_yield_rate"] == best], "n": items[0][1]["n_cases"]}
    return out


def render(res, table=False):
    head = f"=== {res['model']} | {res.get('pressure')} | {res.get('channel')}"
    if "error" in res:
        print(f"{head} ===\n  {res['error']}\n")
        return
    b = res["behavior"]
    print(f"{head} | {b['n']} cases ===")
    print(f"  behavior : P(correct) {b['p_correct'][0]:.3f} -> {b['p_correct'][1]:.3f} | P(stated wrong) {b['p_stated'][0]:.3f} -> {b['p_stated'][1]:.3f} | "
          f"yield {b['yield'][0]}/{b['yield'][1]} = {100 * b['yield'][0] / max(1, b['yield'][1]):.0f}% (95% CI {100 * b['yield_ci'][0]:.0f}-{100 * b['yield_ci'][1]:.0f}%)")
    L = res["layers"]
    pos = res["positive_coefs"]
    if not L:
        print("  steering : no layer has complete per-case rows yet\n")
        return
    if res["has_random_rows"]:
        good = [l for l, r in L.items() if r["restoring"]]
        print(f"  steering : {len(good)}/{res['n_layers']} layers clearly restore (interval excludes zero, beats the random direction, at "
              f"{' and '.join('+%g' % p for p in pos)} gaps): {ranges(good)}")
    else:
        good = [l for l, r in L.items() if r["dim_only_clear"]]
        print(f"  steering : NO random-control rows found, so only the uncontrolled test is possible. {len(good)}/{res['n_layers']} layers have a clear effect: {ranges(good)}")
    pmax = max(pos)
    rank = lambda l: -(L[l]["pos"][pmax]["exc"] if L[l]["pos"][pmax]["exc"] is not None else -L[l]["pos"][pmax]["dim"])
    for l in sorted(good, key=rank)[:3]:
        e = L[l]["pos"]
        txt = f"    L{l}: +{pmax:g} gaps lowers P(stated) by {-e[pmax]['dim']:.3f} [{-e[pmax]['dim_ci'][1]:.3f} to {-e[pmax]['dim_ci'][0]:.3f}]"
        if e[pmax]["rand"] is not None:
            r = -e[pmax]["rand"]
            txt += f"; a random direction of the same size {'lowers' if r >= 0 else 'raises'} it by {abs(r):.3f}"
        if e[pmax]["frac"] is not None:
            txt += f"\n        share of the pressure effect removed: {100 * e[pmax]['frac']:.0f}% raw [{100 * e[pmax]['frac_ci'][0]:.0f}-{100 * e[pmax]['frac_ci'][1]:.0f}]"
            if e[pmax]["frac_net"] is not None:
                txt += f", {100 * e[pmax]['frac_net']:.0f}% NET of the random baseline [{100 * e[pmax]['frac_net_ci'][0]:.0f}-{100 * e[pmax]['frac_net_ci'][1]:.0f}]  <- the honest number"
            p1 = min(pos)
            if e[p1]["frac_net"] is not None:
                txt += f" (at +{p1:g} gap: {100 * e[p1]['frac_net']:.0f}% net)"
        print(txt)
    if "clamp" in res:
        c = res["clamp"]
        print(f"  clamp    : yield {c['baseline']:.2f} unclamped -> best {c['best']:.2f} clamped at {ranges(c['best_layers'])} (n={c['n']}; aggregate only, no interval)")
    if table:
        print("\n  layer   " + "  ".join(f"+{p:g}: dP [95% CI]            random" for p in pos) + "   verdict")
        for l in sorted(L):
            cells = []
            for p in pos:
                e = L[l]["pos"][p]
                cells.append(f"{e['dim']:+.3f} [{e['dim_ci'][0]:+.3f},{e['dim_ci'][1]:+.3f}]  {('%+.3f' % e['rand']) if e['rand'] is not None else '  n/a'}")
            v = "RESTORES" if L[l]["restoring"] else ("(clear, uncontrolled)" if L[l]["dim_only_clear"] else "")
            print(f"  L{l:<5d}  " + "   ".join(cells) + f"   {v}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", help="a specific metrics file (repeatable); default: every phase3_metrics_*.json")
    ap.add_argument("--metrics-dir", default="paper2_results/phase3/metrics")
    ap.add_argument("--jsonl", default="paper2_results/phase3/steering_validation.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95%% intervals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    files = args.file or sorted(glob.glob(os.path.join(args.metrics_dir, "phase3_metrics_*.json")))
    if not files:
        print("no finished Phase 3 steps yet (no phase3_metrics_*.json found)")
        return
    rows = load_steering_rows(args.jsonl)
    for f in files:
        render(analyze(f, rows, args.boot, args.alpha, args.seed), args.table)
    print("READ WITH CARE: (1) 32 layers x 2 strengths at 95% -> expect 1-2 layers to pass by chance; trust contiguous runs, not isolated layers. "
          "(2) Intervals reflect which cases were sampled; the random baseline is ONE direction per layer. "
          "(3) The direction is fit on the same cases it is tested on, so effects are probably optimistic.")


if __name__ == "__main__":
    main()