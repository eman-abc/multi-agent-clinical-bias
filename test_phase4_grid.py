"""test_phase4_grid.py -- CPU only. Synthetic direction stores with KNOWN cosines.   python test_phase4_grid.py"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

import run_phase4_grid as grid
from mech import reliability as rel
from mech.checkpoint import DirectionStore

L, D = 4, 64
MODEL = "Mistral-7B-Instruct-v0.1"


def unit(v):
    return v / np.linalg.norm(v)


def put(root, kind, key, dirs):
    store = DirectionStore(os.path.join(root, grid.STORE_SUBDIR[kind]))
    attr = key if kind.startswith("bias") else f"deference_{key}"
    for l in range(dirs.shape[0]):
        d = dirs[l]
        store.save(model=MODEL, attribute=attr, layer=l, direction=d, mean_pos=d, mean_neg=-d, std=np.ones(D), bias_scalar=0.0, n_pairs=10, extraction_point="last_token")


def put_rel(root, kind, key, r_half, seed):
    sh = np.full((6, L), r_half, dtype=np.float32)
    nd = np.random.default_rng(seed).normal(size=(5, L, D)).astype(np.float32)
    nd /= np.linalg.norm(nd, axis=2, keepdims=True)
    rel.save_reliability(rel.reliability_path(kind, MODEL, key, root), split_half=sh, null_dirs=nd, n_pairs=10)


def test_known_cosines_ceilings_and_readings():
    rng = np.random.default_rng(0)
    u = np.stack([unit(rng.normal(size=D)) for _ in range(L)])
    orth = np.stack([unit(rng.normal(size=D) - (rng.normal(size=D) @ u[l]) * u[l]) for l in range(L)])
    orth = np.stack([unit(orth[l] - (orth[l] @ u[l]) * u[l]) for l in range(L)])
    dev = 0.6 * u + 0.8 * orth                         # cosine with u is exactly 0.6 at every layer
    with tempfile.TemporaryDirectory() as root:
        put(root, "bias_usmle", "ses", u)
        put(root, "def_usmle", "consensus_user", dev)
        put(root, "bias_esi", "ses", u)                # the same direction in the other format: cosine 1
        put(root, "def_usmle", "consensus_assistant", unit(rng.normal(size=(L, D))[0]) * np.ones((L, 1)))
        put_rel(root, "bias_usmle", "ses", 0.64, 1)
        put_rel(root, "def_usmle", "consensus_user", 0.64, 2)
        out = os.path.join(root, "grid.json")
        old = sys.argv
        sys.argv = ["x", "--root", root, "--window", "0,3", "--out", out]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                assert grid.main() == 0
        finally:
            sys.argv = old
        text = buf.getvalue()
        res = json.load(open(out))
        rows = {(r["relation"], r["a"], r["b"]): r for r in res["rows"] if r["model"] == MODEL}

        matched = rows[("bias vs deference, USMLE (matched)", "bias_usmle:ses", "def_usmle:consensus_user")]
        assert abs(matched["mean_abs_window"] - 0.6) < 1e-5 and abs(matched["signed_window"] - 0.6) < 1e-5
        r_full = 2 * 0.64 / 1.64                                                    # Spearman-Brown of 0.64 = 0.7805
        assert abs(matched["ceiling"] - r_full) < 1e-5                              # sqrt(r_full * r_full) with equal reliabilities
        assert abs(matched["frac_of_ceiling"] - 0.6 / r_full) < 1e-4                # 0.769
        assert matched["noise_max"] < 0.35 and reading_of(matched) == "near the ceiling", matched
        fmt = rows[("same bias, ESI vs USMLE prompts", "bias_esi:ses", "bias_usmle:ses")]
        assert abs(fmt["mean_abs_window"] - 1.0) < 1e-5 and fmt["ceiling"] is None and reading_of(fmt) == "no reliability files: cannot calibrate"
        chan = [r for r in res["rows"] if r["relation"].startswith("deference, user vs assistant channel (USMLE)")]
        assert len(chan) == 1 and chan[0]["mean_abs_window"] < 0.5
        assert not any(r["relation"] == "bias vs deference, ESI (matched)" for r in res["rows"]), "pairs with a missing direction must be skipped, not invented"
        assert "Cells are blank where a direction has not been built yet" in text and "random unit vectors" in text
    print("PASS: cosines, Spearman-Brown ceiling, share of ceiling and readings match hand-computed values; missing directions are skipped")


def reading_of(res):
    return grid.reading(res)


def test_reading_rules():
    base = {"mean_abs_window": 0.1, "noise_max": 0.1, "frac_of_ceiling": 0.9}
    assert grid.reading(base) == "at or below noise"                              # not strictly above the null -> nothing detectable
    assert grid.reading({**base, "mean_abs_window": 0.5, "frac_of_ceiling": 0.8}) == "near the ceiling"
    assert grid.reading({**base, "mean_abs_window": 0.5, "frac_of_ceiling": 0.4}) == "partial overlap"
    assert grid.reading({**base, "mean_abs_window": 0.5, "frac_of_ceiling": 0.1}) == "above noise, far below ceiling"
    assert grid.reading({**base, "mean_abs_window": 0.5, "frac_of_ceiling": None}) == "above noise; ceiling too low to scale"
    assert grid.reading({"mean_abs_window": 0.5, "noise_max": None, "frac_of_ceiling": None}) == "no reliability files: cannot calibrate"
    print("PASS: the reading rules behave at their boundaries")


def test_low_reliability_makes_the_ceiling_collapse():
    rng = np.random.default_rng(5)
    u = np.stack([unit(rng.normal(size=D)) for _ in range(L)])
    with tempfile.TemporaryDirectory() as root:
        put(root, "bias_usmle", "race", u)
        put(root, "def_usmle", "consensus_user", u)                                # identical directions: cosine exactly 1
        put_rel(root, "bias_usmle", "race", 0.02, 3)                               # but each is barely repeatable
        put_rel(root, "def_usmle", "consensus_user", 0.02, 4)
        da, db = grid.load_dirs(root, "bias_usmle", MODEL, "race"), grid.load_dirs(root, "def_usmle", MODEL, "consensus_user")
        res = grid.compare(da, db, rel.load_reliability(rel.reliability_path("bias_usmle", MODEL, "race", root)),
                           rel.load_reliability(rel.reliability_path("def_usmle", MODEL, "consensus_user", root)), (0, 3))
        assert res["ceiling"] < 0.05 and res["frac_of_ceiling"] is None             # too noisy to scale: say so instead of dividing by ~0
    print("PASS: when both directions are barely repeatable the ceiling collapses and no misleading ratio is reported")


if __name__ == "__main__":
    test_known_cosines_ceilings_and_readings()
    test_reading_rules()
    test_low_reliability_makes_the_ceiling_collapse()
    print("\nALL GRID TESTS PASSED")
