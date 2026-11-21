"""test_attn_mlp.py -- CPU.   python test_attn_mlp.py   (needs transformers + transformer_lens + the Mistral tokenizer, like the other model tests)"""
import json
import os
import tempfile

import numpy as np
import torch

import analyze_attn_mlp as A
from mech import patching
from mech import patching_window as pw

CFG = {"dominance_margin": 0.10, "min_dominant_restoration": 0.15, "min_gap": 0.10, "bootstrap_resamples": 600, "ci_level": 0.95}
STATUS = {"status": "NOT-FROZEN", "frozen_at_utc": None, "differences": []}


def build_rotary_model(n_layers=4, d_model=32):
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    cfg = HookedTransformerConfig(n_layers=n_layers, d_model=d_model, d_head=8, n_heads=4, d_mlp=64, n_ctx=2048, d_vocab=len(tok), act_fn="silu",
                                  positional_embedding_type="rotary", rotary_dim=8, normalization_type="RMS", gated_mlp=True, tokenizer_name=None, seed=0)
    return HookedTransformer(cfg, tokenizer=tok).eval(), tok


def test_window_patching_identities():
    model, tok = build_rotary_model()
    enc = lambda s: torch.tensor([tok(s, add_special_tokens=False)["input_ids"]])
    clean_t = enc("The patient has a fever and a cough. The correct answer is: (")
    corr_t = enc("A different and much longer clinical note about something else, with extra sentences. Three attendings agree. The correct answer is: (")
    assert clean_t.shape[1] != corr_t.shape[1] and int(clean_t[0, -1]) == int(corr_t[0, -1])
    ids = {"A": 100, "B": 200, "C": 300, "D": 400}
    acts = patching.capture_last_token_activations(model, clean_t, 4, components=("attn_out", "mlp_out"))

    def probs_of(tokens):
        with torch.no_grad():
            lg = model(tokens)[0, -1]
        v = np.array([float(lg[t]) for t in ids.values()])
        e = np.exp(v - v.max())
        return dict(zip(ids, e / e.sum()))

    clean, corrupted = probs_of(clean_t), probs_of(corr_t)
    assert max(abs(clean[k] - corrupted[k]) for k in ids) > 1e-4, "the two prompts must give different answers or the test proves nothing"
    both_all = pw.patch_window_and_predict(model, corr_t, acts, layers=range(4), components=pw.TYPES["both"], class_token_ids=ids)
    assert max(abs(both_all[k] - clean[k]) for k in ids) < 1e-6, both_all                    # EXACT: attention + MLP at every layer = the clean answer
    none = pw.patch_window_and_predict(model, corr_t, acts, layers=[], components=pw.TYPES["both"], class_token_ids=ids)
    assert max(abs(none[k] - corrupted[k]) for k in ids) < 1e-6                              # patching nothing = the corrupted answer
    attn_all = pw.patch_window_and_predict(model, corr_t, acts, layers=range(4), components=pw.TYPES["attn_out"], class_token_ids=ids)
    mlp_all = pw.patch_window_and_predict(model, corr_t, acts, layers=range(4), components=pw.TYPES["mlp_out"], class_token_ids=ids)
    # Attention is the ONLY route by which earlier tokens reach the last position (MLPs and norms see that position alone), so
    # patching the last token's attention output at every layer cuts the pressured prompt off completely: also an exact identity.
    assert max(abs(attn_all[k] - clean[k]) for k in ids) < 1e-6, attn_all
    # MLP-only is NOT: the pressured prompt still reaches the last token through attention at every layer.
    assert max(abs(mlp_all[k] - clean[k]) for k in ids) > 1e-6 and max(abs(mlp_all[k] - corrupted[k]) for k in ids) > 1e-9
    half_attn = pw.patch_window_and_predict(model, corr_t, acts, layers=range(2), components=pw.TYPES["attn_out"], class_token_ids=ids)
    assert max(abs(half_attn[k] - clean[k]) for k in ids) > 1e-6                                                          # a partial window is not the identity
    for layer, comp in ((2, "attn_out"), (1, "mlp_out")):                                      # a one-layer window equals the existing single-layer patch
        _, ref = patching.patch_and_predict(model, corr_t, acts[comp][layer], layer=layer, component=comp, class_token_ids=ids)
        got = pw.patch_window_and_predict(model, corr_t, acts, layers=[layer], components=(comp,), class_token_ids=ids)
        assert max(abs(got[k] - ref[k]) for k in ids) < 1e-9
    print("PASS: window patching is an exact identity for attention+MLP AND for attention alone at every layer (attention is the only route to the last token), a no-op when nothing is patched, not an identity for MLP alone or a partial window, and equals the single-layer patch for a one-layer window")


def write_run(path, clean, press, win, sweep, valid=True, worst=0.0, roles=("primary", "control1", "control2"), windows=((14, 18), (8, 12), (26, 30)), sweep_layers=(10, 11)):
    n = len(clean)
    meta = {"model": "M", "channel": "assistant", "pressure": "consensus", "roles": list(roles), "types": ["attn_out", "mlp_out", "both"], "windows": [list(w) for w in windows],
            "sweep_layers": list(sweep_layers), "exactness_max_abs_diff": worst, "valid": valid, "plan_status": "NOT-FROZEN", "plan_frozen_at": None, "n_layers": 32}
    np.savez(path, meta=np.array(json.dumps(meta)), case_ids=np.array([f"q{i}" for i in range(n)]), clean_pc=np.array(clean), pressured_pc=np.array(press),
             windows=np.array(windows), patched_window=np.array(win), sweep_layers=np.array(sweep_layers), patched_sweep=np.array(sweep))


def const(n, v):
    return np.full(n, v)


def test_restoration_and_decision_rules_by_hand():
    n = 10
    clean, press = const(n, 0.9), const(n, 0.1)                       # gap 0.8
    R = lambda r: 0.1 + r * 0.8                                        # the patched P(correct) that gives restoration r
    win = [[const(n, R(0.60)), const(n, R(0.05)), const(n, R(0.80))],  # primary: attention 0.60, MLP 0.05, both 0.80
           [const(n, R(0.10)), const(n, R(0.08)), const(n, R(0.15))],  # control 1
           [const(n, R(0.02)), const(n, R(0.01)), const(n, R(0.03))]]  # control 2
    sweep = [[const(n, R(0.20)), const(n, R(0.10))], [const(n, R(0.30)), const(n, R(0.00))]]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "attn_mlp_M_consensus_assistant.npz")
        write_run(p, clean, press, win, sweep)
        r = A.analyze_file(p, CFG, STATUS)
    assert abs(r["gap"] - 0.8) < 1e-12 and not r.get("invalid") and not r.get("uninformative")
    w0, w1, w2 = r["windows"]
    assert abs(w0["restoration"]["attn_out"]["point"] - 0.60) < 1e-9 and abs(w0["restoration"]["mlp_out"]["point"] - 0.05) < 1e-9 and abs(w0["restoration"]["both"]["point"] - 0.80) < 1e-9
    assert abs(w0["attn_minus_mlp"]["point"] - 0.55) < 1e-9 and w0["reading"] == "ATTENTION-DOMINANT"
    assert w1["reading"] == "neither dominant" and abs(w1["attn_minus_mlp"]["point"] - 0.02) < 1e-9            # margin 0.10 not met
    assert [s["specific"] for s in r["specificity"]] == [True, True] and r["primary_is_specific"]
    assert abs(r["specificity"][0]["primary_minus_control"] - 0.50) < 1e-9 and abs(r["specificity"][1]["primary_minus_control"] - 0.58) < 1e-9
    assert abs(r["sweep"][0]["attn_out"]["point"] - 0.20) < 1e-9 and abs(r["sweep"][1]["attn_out"]["point"] - 0.30) < 1e-9 and abs(r["sweep"][1]["mlp_out"]["point"]) < 1e-9
    print("PASS: restorations (0.60 / 0.05 / 0.80), the difference 0.55, ATTENTION-DOMINANT, 'neither' when the margin is missed, specificity and the sweep all match hand values")

    # the mirror image: MLP-dominant
    win_m = [[const(n, R(0.05)), const(n, R(0.60)), const(n, R(0.8))]] + win[1:]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "attn_mlp_M_consensus_assistant.npz")
        write_run(p, clean, press, win_m, sweep)
        r2 = A.analyze_file(p, CFG, STATUS)
    assert r2["windows"][0]["reading"] == "MLP-DOMINANT" and not r2["primary_is_specific"] or r2["windows"][0]["reading"] == "MLP-DOMINANT"
    print("PASS: the mirror pattern is labelled MLP-DOMINANT")


def test_rules_refuse_to_over_read():
    n = 12
    rng = np.random.default_rng(0)
    clean, press = const(n, 0.9), const(n, 0.1)
    R = lambda r: 0.1 + r * 0.8
    # restoration needs both components' lower bounds and margin; a large attention effect with enormous case-to-case noise cannot be called dominant
    noisy = lambda r: np.clip(R(r) + rng.normal(0, 0.45, n), 0, 1)
    win = [[noisy(0.35), noisy(0.25), noisy(0.5)], [noisy(0.1), noisy(0.1), noisy(0.1)], [noisy(0.1), noisy(0.1), noisy(0.1)]]
    sweep = [[noisy(0.1), noisy(0.1)], [noisy(0.1), noisy(0.1)]]
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.npz")
        write_run(p, clean, press, win, sweep)
        r = A.analyze_file(p, CFG, STATUS)
    assert r["windows"][0]["reading"] == "neither dominant", r["windows"][0]
    lo, hi = r["windows"][0]["attn_minus_mlp"]["ci"]
    assert hi - lo > 0.3                                                                    # the interval is honestly wide
    # a tiny gap makes restoration undefined; an invalid exactness check voids the run
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.npz")
        write_run(p, const(n, 0.5), const(n, 0.45), [[const(n, 0.5)] * 3] * 3, [[const(n, 0.5)] * 2] * 2)
        assert A.analyze_file(p, CFG, STATUS).get("uninformative") is True
        write_run(p, clean, press, win, sweep, valid=False, worst=0.2)
        r = A.analyze_file(p, CFG, STATUS)
        assert r.get("invalid") is True and "windows" not in r
    print("PASS: heavy noise gives 'neither dominant' with a wide interval; a gap below 0.10 is 'uninformative'; a failed exactness check voids the run")


def test_interval_covers_truth():
    n = 80
    rng = np.random.default_rng(1)
    clean, press = const(n, 0.9) + rng.normal(0, 0.02, n), const(n, 0.1) + rng.normal(0, 0.02, n)
    R = lambda r: 0.1 + r * 0.8
    win = [[R(0.5) + rng.normal(0, 0.1, n), R(0.1) + rng.normal(0, 0.1, n), R(0.7) + rng.normal(0, 0.1, n)]] * 3
    sweep = [[R(0.1) + rng.normal(0, 0.1, n)] * 2] * 2
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.npz")
        write_run(p, clean, press, win, sweep)
        r = A.analyze_file(p, {**CFG, "bootstrap_resamples": 1500}, STATUS)
    c = r["windows"][0]["restoration"]
    assert c["attn_out"]["ci"][0] < 0.5 < c["attn_out"]["ci"][1] and c["mlp_out"]["ci"][0] < 0.1 < c["mlp_out"]["ci"][1]
    assert r["windows"][0]["attn_minus_mlp"]["ci"][0] < 0.4 < r["windows"][0]["attn_minus_mlp"]["ci"][1]
    print("PASS: with realistic noise the case-bootstrap intervals contain the planted restorations")


if __name__ == "__main__":
    test_window_patching_identities()
    test_restoration_and_decision_rules_by_hand()
    test_rules_refuse_to_over_read()
    test_interval_covers_truth()
    print("\nALL ATTENTION-VS-MLP TESTS PASSED")
