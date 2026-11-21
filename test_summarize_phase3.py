"""test_summarize_phase3.py -- CPU only. Synthetic data with a KNOWN planted effect:

  layers 4-7 : real effect, P(stated) falls 0.10 per gap toward clean (and rises toward pressured)   -> must be RESTORING
  layer 8    : effect at +2 gaps only, nothing at +1                                                 -> must NOT restore
  layer 9    : generic disruption: real AND random directions lower P(stated) equally                -> must NOT restore
  layer 10   : wrong sign (toward clean RAISES P(stated))                                            -> must NOT restore
  layers 0-3, 11 : pure noise                                                                        -> must NOT restore
  layer 12   : real effect 0.15 per gap, but a random direction ALSO lowers P(stated) by 0.075 per gap (the Mistral-assistant
               situation)                                                                            -> restores, but the NET share is half the raw share

    python test_summarize_phase3.py
"""
import contextlib
import io
import json
import os
import tempfile

import numpy as np

import summarize_phase3 as s3

MODEL, ATTR = "TestModel", "deference_consensus_user"
N, LAYERS, COEFS = 40, list(range(13)), [-2.0, -1.0, 1.0, 2.0]


def probs_with(stated_p, correct="A", stated="B"):
    p = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    p[stated] = float(np.clip(stated_p, 0.0, 1.0))
    rest = 1.0 - p[stated]
    p[correct] = rest * 0.6
    p["C"], p["D"] = rest * 0.2, rest * 0.2
    return p


def build(tmp, with_random=True, with_per_case=True):
    rng = np.random.default_rng(7)
    per_case, base = {}, {}
    for i in range(N):
        b = 0.90 + rng.uniform(-0.05, 0.05)
        base[f"q{i}"] = b
        per_case[f"q{i}"] = {"stated": "B", "correct": "A",
                             "clean_probs": probs_with(0.04), "pressured_probs": probs_with(b)}
    metrics = {"model": MODEL, "attribute": ATTR, "pressure": "consensus", "channel": "user", "coefficients": COEFS,
               "validation_layers": LAYERS, "min_effect": 0.01,
               "clamp_effectiveness": {"3": {"unclamped_yield_rate": 0.9, "clamped_yield_rate": 0.7, "n_cases": N},
                                       "5": {"unclamped_yield_rate": 0.9, "clamped_yield_rate": 0.4, "n_cases": N}}}
    if with_per_case:
        metrics["per_case"] = per_case
    mpath = os.path.join(tmp, "phase3_metrics_TestModel_consensus_user.json")
    json.dump(metrics, open(mpath, "w"))

    def real_effect(l, c):
        if 4 <= l <= 7:
            return -0.10 * c
        if l == 8:
            return {2.0: -0.20, -2.0: 0.20}.get(c, 0.0)
        if l == 9:
            return -0.10 * abs(c)
        if l == 10:
            return +0.10 * c
        if l == 12:
            return -0.15 * c
        return 0.0

    def random_effect(l, c):
        if l == 9:
            return -0.10 * abs(c)
        if l == 12:
            return -0.075 * abs(c)
        return 0.0

    jpath = os.path.join(tmp, "steering_validation.jsonl")
    with open(jpath, "w") as f:
        for l in LAYERS:
            for c in COEFS:
                for cid in per_case:
                    for attr, eff in ((ATTR, real_effect(l, c)), (ATTR + "__random", random_effect(l, c))):
                        if attr.endswith("__random") and not with_random:
                            continue
                        p = probs_with(base[cid] + eff + rng.normal(0, 0.02))
                        f.write(json.dumps({"model": MODEL, "attribute": attr, "layer": l, "coefficient": c, "case_id": cid, "task": "usmle",
                                            "condition_source": "x", "top_class": max(p, key=p.get), "class_probabilities": p, "raw_text": None}) + "\n")
    return mpath, jpath


def test_helpers():
    lo, hi = s3.wilson(43, 45)                                           # hand-computed: (0.8517, 0.9877)
    assert abs(lo - 0.8517) < 0.001 and abs(hi - 0.9877) < 0.001, (lo, hi)
    assert s3.ranges([14, 15, 16, 19]) == "L14-L16, L19" and s3.ranges([]) == "none" and s3.ranges([5]) == "L5"
    assert s3.ranges([3, 1, 2, 7, 8]) == "L1-L3, L7-L8"
    print("PASS: Wilson interval matches the hand-computed value; layer-range formatting")


def test_planted_effect_is_found_and_decoys_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        mpath, jpath = build(tmp)
        res = s3.analyze(mpath, s3.load_steering_rows(jpath), boot=1000, seed=0)
        good = sorted(l for l, r in res["layers"].items() if r["restoring"])
        assert good == [4, 5, 6, 7, 12], good
        e = res["layers"][5]["pos"][2.0]
        assert -0.22 < e["dim"] < -0.18 and e["dim_ci"][1] < 0 and (e["dim_ci"][1] - e["dim_ci"][0]) < 0.02, e
        assert 0.21 <= e["frac"] <= 0.26 and e["frac_ci"][0] < e["frac"] < e["frac_ci"][1], e          # 0.20 / ~0.86 = 0.23
        assert abs(res["layers"][5]["pos"][1.0]["frac"] - 0.116) < 0.02
        for l in (5, 7):                                      # random baseline ~0 -> net share ~ raw share
            e5 = res["layers"][l]["pos"][2.0]
            assert abs(e5["frac_net"] - e5["frac"]) < 0.01 and e5["frac_net_ci"][0] < e5["frac_net"] < e5["frac_net_ci"][1], e5
        e9 = res["layers"][9]["pos"][2.0]                      # generic disruption: raw share looks big, NET share is ~0
        assert 0.20 < e9["frac"] < 0.26 and abs(e9["frac_net"]) < 0.02, e9
        e12 = res["layers"][12]["pos"][2.0]                    # 0.30 raw, 0.15 net, over a pressure effect of ~0.86 -> 0.35 raw, 0.17 net
        assert 0.32 < e12["frac"] < 0.38 and 0.15 < e12["frac_net"] < 0.20, e12
        assert 0.075 - 0.01 < res["layers"][12]["pos"][1.0]["exc"] < 0.075 + 0.01      # +1 gap: dim -0.15 vs random -0.075 -> excess 0.075
        assert not res["layers"][8]["restoring"]            # only +2 -> fails the "every strength" rule
        assert not res["layers"][9]["restoring"]            # the random direction does just as much
        assert not res["layers"][10]["restoring"]           # wrong sign
        b = res["behavior"]
        assert b["n"] == N and b["yield"] == (N, N) and abs(b["p_stated"][0] - 0.04) < 1e-9
        assert res["clamp"] == {"baseline": 0.9, "best": 0.4, "best_layers": [5], "n": N}
        print("PASS: planted effect (layers 4-7) found with a correct interval and a correct share-removed; the +2-only, generic-disruption and wrong-sign decoys are rejected")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s3.render(res, table=True)
        text = out.getvalue()
        assert "L4-L7, L12" in text and "RESTORES" in text and "yield 40/40" in text
        assert "NET of the random baseline" in text and "a random direction of the same size lowers it by" in text
        print("PASS: report renders (summary + per-layer table)")


def test_missing_random_rows_and_missing_per_case():
    with tempfile.TemporaryDirectory() as tmp:
        mpath, jpath = build(tmp, with_random=False)
        res = s3.analyze(mpath, s3.load_steering_rows(jpath), boot=500, seed=0)
        assert res["has_random_rows"] is False
        assert all(r["restoring"] is None for r in res["layers"].values())
        clear = sorted(l for l, r in res["layers"].items() if r["dim_only_clear"])
        assert clear == [4, 5, 6, 7, 9, 12], clear           # without the control, generic disruption (layer 9) cannot be told apart
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s3.render(res)
        assert "NO random-control rows" in out.getvalue()
    with tempfile.TemporaryDirectory() as tmp:
        mpath, jpath = build(tmp, with_per_case=False)
        res = s3.analyze(mpath, s3.load_steering_rows(jpath))
        assert "no per-case baselines" in res["error"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s3.render(res)
        assert "predates the per-case logging" in out.getvalue()
    print("PASS: without the random control only the uncontrolled test is reported (and says so); an old-format metrics file gives a clear message")


if __name__ == "__main__":
    test_helpers()
    test_planted_effect_is_found_and_decoys_rejected()
    test_missing_random_rows_and_missing_per_case()
    print("\nALL SUMMARIZER TESTS PASSED")