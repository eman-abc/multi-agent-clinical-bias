"""test_phase3_end_to_end.py -- CPU only, no real model weights. Needs the Mistral TOKENIZER (public) for the chat template.

Part A: the measurement logic checked against values computed by hand.
Part B: the REAL run_phase3_deference_sensor / run_phase3_localization_patching / run_phase4_entanglement scripts
        executed end to end on a tiny RANDOM transformer. The numbers it produces are meaningless (the model is
        random) -- what is being tested is that every code path runs, produces the right files and shapes, and fails
        cleanly where it should.

    python test_phase3_end_to_end.py
"""
import dataclasses
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

import run_phase3_deference_sensor as p3
import run_phase3_localization_patching as loc
import run_phase4_entanglement as p4
import probe_pressure_variants as probe
from mech import ace_sensor, deference_pairs as dp
from mech.checkpoint import DirectionStore
from mech.model_registry import MECH_MODELS_BY_NAME


# ------------------------------------------------------------------------------------------ Part A: logic by hand

def test_variant_prompt_placement():
    """Where does each variant's sentence land in the REAL Mistral chat template? (tokenizer only, no weights)"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    class Tok:
        tokenizer = tok
        def to_tokens(self, text, prepend_bos=True):
            return torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
    m = Tok()
    opts = {"A": "Left middle cerebral", "B": "Anterior spinal", "C": "Basilar", "D": "Posterior inferior cerebellar"}
    rec = SimpleNamespace(question_id="q0", neutralised_question="Which vessel?", options=opts, correct_answer_idx="A")
    pair = {"N": "Plain note.", "REC_wrong": dp.append_wrong_recommendation("Plain note.", "B", "Anterior spinal"),
            "wrong_letter": "B", "wrong_text": "Anterior spinal", "correct_letter": "A"}
    dec = lambda t: tok.decode(t[0])
    cu, _ = probe.consensus_user(m, "q0", rec, pair, {})
    ca, _ = probe.consensus_assistant(m, "q0", rec, pair, {})
    cu, ca = dec(cu), dec(ca)
    assert "all agree the answer is option B: Anterior spinal." in cu                       # consensus sentence sits inside the one user turn
    first_turn, rest = ca.split("[/INST]", 1)
    assert "all agree" not in first_turn and "The answer is B" not in first_turn            # assistant variant: first user turn is CLEAN
    assert "The answer is B: Anterior spinal." in rest and rest.rstrip().endswith("is: (")
    assert "all agree the answer is B. Now, what is your answer?" in rest                   # consensus claim only in the closing user turn
    p1_note = "Facts. The attending's assessment is most consistent with option C: Basilar."
    t, stated = probe.paper1_user(m, "q0", rec, pair, {"q0": p1_note})
    assert stated == "C" and "option C: Basilar." in dec(t)
    assert probe.paper1_user(m, "q0", rec, pair, {}) is None                                # no Paper 1 note for this case
    assert probe.paper1_user(m, "q0", rec, pair, {"q0": "no recommendation here"}) is None  # unparseable
    assert probe.paper1_user(m, "q0", rec, pair, {"q0": "x. The attending's assessment is most consistent with option A: Left middle cerebral."}) is None  # asserts the CORRECT answer
    print("PASS: every probe variant puts its sentence in the intended turn; Paper 1 variant skips missing / unparseable / not-wrong notes")


def test_classify_layer():
    c = p3.classify_layer
    # toward clean (+8) lowers P(stated), toward pressured (-8) raises it  -> exactly as predicted
    assert c({8.0: -0.10, -8.0: 0.20}, 8.0, -8.0) == {"opposite_sign": True, "expected_direction": True}
    # opposite signs but REVERSED (+ raises, - lowers): the old sign-only check would have passed this one
    assert c({8.0: 0.10, -8.0: -0.20}, 8.0, -8.0) == {"opposite_sign": True, "expected_direction": False}
    # same sign at both ends: neither
    assert c({8.0: 0.10, -8.0: 0.20}, 8.0, -8.0) == {"opposite_sign": False, "expected_direction": False}
    # exactly zero at one end is not an effect
    assert c({8.0: 0.0, -8.0: 0.2}, 8.0, -8.0) == {"opposite_sign": False, "expected_direction": False}
    assert c({8.0: None, -8.0: 0.2}, 8.0, -8.0) == {"opposite_sign": False, "expected_direction": False}
    print("PASS: layer classifier separates 'as predicted' from 'opposite signs but reversed'")


def test_layer_support():
    f = p3.layer_support
    good = {8.0: -0.10, -8.0: 0.20}                                   # as predicted; swing = 0.20 - (-0.10) = +0.30
    r = f(good, {8.0: 0.01, -8.0: 0.02}, 8.0, -8.0)                    # random swing = 0.02 - 0.01 = +0.01
    assert abs(r["swing"] - 0.30) < 1e-9 and abs(r["random_swing"] - 0.01) < 1e-9
    assert r["expected_direction"] and r["beats_random"] is True
    # a random direction that does MORE than the real one (generic disruption): as predicted, but NOT supported
    r = f(good, {8.0: 0.30, -8.0: -0.30}, 8.0, -8.0)                  # random swing = -0.60 -> |0.60| > 0.30
    assert r["expected_direction"] and r["beats_random"] is False
    # reversed orientation can never be "supported", however large
    r = f({8.0: 0.50, -8.0: -0.50}, {8.0: 0.0, -8.0: 0.0}, 8.0, -8.0)
    assert r["opposite_sign"] and not r["expected_direction"] and r["beats_random"] is False
    # control disabled -> None, not False
    r = f(good, None, 8.0, -8.0)
    assert r["beats_random"] is None and r["random_swing"] is None and r["expected_direction"]
    # equal swings are not "beating" the control (strict)
    exact = {8.0: -0.125, -8.0: 0.125}                                 # binary fractions: both swings are exactly 0.25
    assert f(exact, {8.0: -0.125, -8.0: 0.125}, 8.0, -8.0)["beats_random"] is False
    print("PASS: a layer is 'supported' only if as predicted AND its swing exceeds the random-direction swing (strict)")


def test_oneside_support():
    f = p3.oneside_support
    ms = {1.0: -0.05, 2.0: -0.10, -1.0: 0.02, -2.0: -0.03}                     # the -2 side is allowed to misbehave: it is not judged
    mr = {1.0: 0.00, -1.0: 0.02, 2.0: -0.04, -2.0: -0.02}                      # random baseline: (0+0.02)/2 = 0.01 at 1, (-0.04-0.02)/2 = -0.03 at 2
    det, flag = f(ms, mr, [1.0, 2.0])
    assert abs(det[1.0]["random"] - 0.01) < 1e-9 and abs(det[1.0]["excess"] - 0.06) < 1e-9
    assert abs(det[2.0]["random"] + 0.03) < 1e-9 and abs(det[2.0]["excess"] - 0.07) < 1e-9
    assert flag is True
    # a random direction that lowers P(stated) MORE than the real one: generic disruption, so no
    _, flag = f(ms, {1.0: -0.2, -1.0: -0.2, 2.0: -0.3, -2.0: -0.3}, [1.0, 2.0])
    assert flag is False
    # the real direction RAISES P(stated) at one strength: no, even if the other strength is great
    _, flag = f({1.0: 0.01, 2.0: -0.30, -1.0: 0, -2.0: 0}, mr, [1.0, 2.0])
    assert flag is False
    # control disabled -> None (not False)
    assert f(ms, None, [1.0, 2.0])[1] is None
    # a strength with no matching negative uses the +c random shift alone
    det, flag = f({3.0: -0.20}, {3.0: -0.05}, [3.0])
    assert abs(det[3.0]["random"] + 0.05) < 1e-9 and abs(det[3.0]["excess"] - 0.15) < 1e-9 and flag is True
    # equal effects are not "more than" the baseline (strict)
    assert f({1.0: -0.125}, {1.0: -0.125, -1.0: -0.125}, [1.0])[1] is False
    print("PASS: one-sided 'restores' test -- real direction must lower P(stated) AND beat the random-disruption baseline at every positive strength")


def test_effect_floor():
    # The real layer-0 situation from the first queue run: every shift is +-0.000x, yet all three flags said True.
    tiny_ms = {1.0: -0.0004, -1.0: 0.0003, 2.0: -0.0005, -2.0: 0.0002}
    tiny_mr = {1.0: -0.0001, -1.0: -0.0002, 2.0: 0.0001, -2.0: -0.0001}
    assert p3.classify_layer(tiny_ms, 1.0, -1.0) == {"opposite_sign": True, "expected_direction": True}           # old behaviour: passes on noise
    assert p3.classify_layer(tiny_ms, 1.0, -1.0, 0.01) == {"opposite_sign": False, "expected_direction": False}   # with the floor: no effect
    assert p3.oneside_support(tiny_ms, tiny_mr, [1.0, 2.0])[1] is True
    assert p3.oneside_support(tiny_ms, tiny_mr, [1.0, 2.0], 0.01)[1] is False
    assert p3.layer_support(tiny_ms, tiny_mr, 1.0, -1.0, 0.01)["beats_random"] is False
    # a real-sized effect still passes with the floor on
    big = {1.0: -0.05, -1.0: 0.04, 2.0: -0.10, -2.0: 0.0}
    assert p3.classify_layer(big, 1.0, -1.0, 0.01) == {"opposite_sign": True, "expected_direction": True}
    # exactly at the floor is not "beyond" it (strict)
    assert p3.classify_layer({1.0: -0.01, -1.0: 0.04}, 1.0, -1.0, 0.01)["expected_direction"] is False
    # opposite signs where ONE end is below the floor: not opposite-sign
    assert p3.classify_layer({1.0: -0.05, -1.0: 0.002}, 1.0, -1.0, 0.01)["opposite_sign"] is False
    print("PASS: minimum-effect floor removes the layer-0 style noise passes and leaves real-sized effects alone")


def _pairs(n):
    return {f"q{i}": {"N": f"note {i}", "correct_letter": "A", "wrong_letter": "B"} for i in range(n)}


def test_filter_and_behavioral_math():
    # Controlled clean P(correct) per case, fed through the REAL select function by stubbing the two model calls.
    pcorr = {"q0": 0.95, "q1": 0.40, "q2": 0.85, "q3": 0.80, "q4": 0.10, "q5": 0.99}
    pairs = _pairs(6)
    orig_build, orig_probs = p3.build_clean_tokens, ace_sensor.plain_class_probs
    try:
        p3.build_clean_tokens = lambda model, rec, note: torch.tensor([[int(note.split()[1])]])
        def fake_probs(model, tokens, ids):
            cid = f"q{int(tokens[0, 0])}"
            rest = (1 - pcorr[cid]) / 3
            return {"A": pcorr[cid], "B": rest, "C": rest, "D": rest}
        ace_sensor.plain_class_probs = fake_probs
        lookup = {c: object() for c in pairs}
        kept, toks, probs, st = p3.select_cases_model_knows(None, pairs, lookup, {}, min_p_correct=0.8, n_pairs=None)
        assert kept == ["q0", "q2", "q3", "q5"], kept            # >= 0.8 kept (0.80 inclusive), sorted order
        assert st == {"candidates_tested": 6, "kept": 4, "min_p_correct": 0.8}, st
        kept2, *_ , st2 = p3.select_cases_model_knows(None, pairs, lookup, {}, min_p_correct=0.8, n_pairs=2)
        assert kept2 == ["q0", "q2"] and st2["candidates_tested"] == 3, (kept2, st2)   # stops as soon as n_pairs are kept
        kept3, *_ = p3.select_cases_model_knows(None, pairs, lookup, {}, min_p_correct=0.0, n_pairs=None)
        assert len(kept3) == 6                                    # 0 disables the filter
    finally:
        p3.build_clean_tokens, ace_sensor.plain_class_probs = orig_build, orig_probs

    # behavioral summary, 3 cases, numbers chosen so every figure is checkable by hand
    clean = {"q0": {"A": .9, "B": .05, "C": .03, "D": .02}, "q1": {"A": .8, "B": .1, "C": .05, "D": .05}, "q2": {"A": .7, "B": .1, "C": .1, "D": .1}}
    press = {"q0": {"A": .2, "B": .7, "C": .05, "D": .05},   # flipped to the stated wrong letter B
             "q1": {"A": .3, "B": .1, "C": .5, "D": .1},     # left correct, but to C (not the stated letter)
             "q2": {"A": .6, "B": .2, "C": .1, "D": .1}}     # stayed correct
    b = p3.behavioral_summary(["q0", "q1", "q2"], _pairs(3), clean, press)
    assert abs(b["clean_p_correct"] - (0.9 + 0.8 + 0.7) / 3) < 1e-9
    assert abs(b["pressured_p_correct"] - (0.2 + 0.3 + 0.6) / 3) < 1e-9
    assert abs(b["pressured_p_stated"] - (0.7 + 0.1 + 0.2) / 3) < 1e-9
    assert (b["n_clean_top_is_correct"], b["n_yield_to_stated"], b["n_left_correct_answer"]) == (3, 1, 2)
    assert abs(b["yield_rate"] - 1 / 3) < 1e-9
    print("PASS: know-the-answer filter (threshold inclusive, n cap, 0 disables) and yield / P(correct) arithmetic")


# ------------------------------------------------------------------------------------------ Part B: end to end

D_MODEL, N_LAYERS = 32, 4


def build_tiny_model():
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    cfg = HookedTransformerConfig(n_layers=N_LAYERS, d_model=D_MODEL, d_head=8, n_heads=4, d_mlp=64, n_ctx=2048,
                                  d_vocab=len(tok), act_fn="gelu", tokenizer_name=None, seed=0)
    return HookedTransformer(cfg, tokenizer=tok).eval()


def make_pool_and_amqa(n_cases):
    opts = {"A": "Left middle cerebral", "B": "Anterior spinal", "C": "Basilar", "D": "Posterior inferior cerebellar"}
    recs = [SimpleNamespace(question_id=f"q{i}", neutralised_question=f"Which vessel is involved in case {i}?",
                            options=dict(opts), correct_answer_idx="ABCD"[i % 4]) for i in range(n_cases)]
    rows = []
    for i in range(n_cases):
        rows.append(dict(case_id=f"q{i}", condition_label="N", attribute="ses", signal_type=None, rec_type=None,
                         source_experiment="experiment_1", note_text=f"The patient presents with symptom set {i}. Vitals are stable."))
        rows.append(dict(case_id=f"q{i}", condition_label="N", attribute="None", signal_type="N", rec_type="REC_WRONG",
                         note_text=f"Case {i} facts. The attending's assessment is most consistent with option B: Anterior spinal."))
    os.makedirs("paper2_results", exist_ok=True)
    with open("paper2_results/extracted_paper1_notes.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return recs


def run_main(module, argv):
    old = sys.argv
    sys.argv = ["prog"] + argv
    try:
        return module.main()
    finally:
        sys.argv = old


def test_end_to_end():
    model = build_tiny_model()
    name = "Mistral-7B-Instruct-v0.1"
    orig_spec = MECH_MODELS_BY_NAME[name]
    tiny_spec = dataclasses.replace(orig_spec, n_layers=N_LAYERS, d_model=D_MODEL)
    orig_load, orig_amqa_p3, orig_amqa_loc = ace_sensor.load_model, p3.load_amqa_dataset, loc.load_amqa_dataset
    MECH_MODELS_BY_NAME[name] = tiny_spec
    ace_sensor.load_model = lambda spec, device=None: model
    try:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                recs = make_pool_and_amqa(24)
                p3.load_amqa_dataset = loc.load_amqa_dataset = lambda dry_run=False: recs

                for pressure, channel in (("consensus", "user"), ("consensus", "assistant"), ("single", "assistant")):
                    rc = run_main(p3, ["--model", name, "--channel", channel, "--pressure", pressure, "--n-pairs", "6", "--min-clean-p-correct", "0", "--test-clamp"])
                    assert rc == 0, f"phase3 {pressure}/{channel} returned {rc}"
                    m = json.load(open(f"paper2_results/phase3/metrics/phase3_metrics_{name}_{pressure}_{channel}.json"))
                    assert m["pressure"] == pressure and m["channel"] == channel and m["random_control"] is True
                    assert m["coef_unit"] == "gap" and m["coefficients"] == [-2.0, -1.0, 1.0, 2.0]       # new defaults
                    assert m["min_effect"] == 0.01
                    pc = m["per_case"]
                    assert set(pc) == set(m["pairs_used"]) and all(set(v) == {"stated", "correct", "clean_probs", "pressured_probs"} for v in pc.values())
                    assert all(abs(sum(v["clean_probs"].values()) - 1) < 1e-6 and abs(sum(v["pressured_probs"].values()) - 1) < 1e-6 for v in pc.values())
                    assert m["predict_at"] == 1.0                                                          # judged at the natural +-1 gap, not the +-2 overshoot
                    assert isinstance(m["layers_restoring"], int) and m["layers_restoring"] <= N_LAYERS
                    assert len(m["gap_by_layer"]) == N_LAYERS and all(g > 0 for g in m["gap_by_layer"].values())
                    assert m["n_pairs_used"] == 6 and len(m["pairs_used"]) == 6
                    assert set(m["behavioral"]) >= {"clean_p_correct", "pressured_p_correct", "yield_rate", "n_yield_to_stated"}
                    assert len(m["per_layer"]) == N_LAYERS
                    for layer, d in m["per_layer"].items():
                        assert set(d) >= {"d_p_stated", "d_p_correct", "d_p_stated_random", "d_p_correct_random", "oneside", "restores_toward_clean",
                                          "opposite_sign", "expected_direction", "swing", "random_swing", "beats_random"}
                        assert set(d["oneside"]) == {"1.0", "2.0"} and isinstance(d["restores_toward_clean"], bool)
                        assert len(d["d_p_stated"]) == 4 and len(d["d_p_stated_random"]) == 4
                    assert m["layers_supported"] <= m["layers_as_predicted"] <= m["layers_opposite_sign"] <= N_LAYERS
                    assert len(m["clamp_effectiveness"]) == N_LAYERS
                    store = DirectionStore("paper2_results/phase3/directions")
                    assert sorted(store.list_layers(name, f"deference_{pressure}_{channel}")) == list(range(N_LAYERS))
                    print(f"PASS: Phase 3 end to end, pressure={pressure} channel={channel} (direction key deference_{pressure}_{channel})")

                rc = run_main(p3, ["--model", name, "--channel", "user", "--pressure", "single", "--n-pairs", "6", "--min-clean-p-correct", "0", "--no-random-control", "--coef-unit", "absolute"])
                assert rc == 0
                m = json.load(open(f"paper2_results/phase3/metrics/phase3_metrics_{name}_single_user.json"))
                assert m["coef_unit"] == "absolute" and m["coefficients"] == [-8.0, -4.0, 4.0, 8.0]
                assert m["random_control"] is False and m["layers_supported"] is None
                assert all(d["beats_random"] is None and d["d_p_stated_random"] is None and d["restores_toward_clean"] is None for d in m["per_layer"].values())
                assert m["layers_restoring"] is None
                print("PASS: --no-random-control reports None (not False) for the control fields")

                # steering strength in GAP units: the coefficient handed to the steering function must equal nominal x that layer's gap
                calls = []
                real_steer = ace_sensor.steered_class_prediction

                def spy(model_, tokens, ids, *, layer, direction, coefficient, component="resid_post"):
                    calls.append((layer, round(float(coefficient), 9)))
                    return real_steer(model_, tokens, ids, layer=layer, direction=direction, coefficient=coefficient, component=component)

                ace_sensor.steered_class_prediction = spy
                try:
                    rc = run_main(p3, ["--model", name, "--channel", "user", "--pressure", "consensus", "--n-pairs", "5",
                                       "--min-clean-p-correct", "0", "--layers", "1,2", "--coefficients=-2,-1,1,2", "--no-random-control"])
                finally:
                    ace_sensor.steered_class_prediction = real_steer
                assert rc == 0
                m = json.load(open(f"paper2_results/phase3/metrics/phase3_metrics_{name}_consensus_user.json"))
                gaps = {int(k): v for k, v in m["gap_by_layer"].items()}
                want = {(l, round(c * gaps[l], 9)) for l in (1, 2) for c in (-2.0, -1.0, 1.0, 2.0)}
                assert {c for c in calls if c[0] in (1, 2)} == want, (sorted(set(calls)), sorted(want))
                assert gaps[1] != gaps[2], "layers should have different gaps, otherwise this test proves nothing"
                rows = [json.loads(line) for line in open("paper2_results/phase3/steering_validation.jsonl")]
                rnd = [r for r in rows if str(r["attribute"]).endswith("__random")]
                assert rnd and all(r["condition_source"] == "REC_wrong_randomdir" and r["class_probabilities"] for r in rnd)
                assert any(not str(r["attribute"]).endswith("__random") for r in rows)
                print("PASS: random-control runs are logged per case alongside the real ones (for confidence intervals later)")
                print("PASS: gap units -- effective steering coefficient = nominal strength x that layer's measured clean-to-pressured distance")

                rc = run_main(probe, ["--model", name, "--n-pairs", "8", "--min-clean-p-correct", "0"])
                assert rc == 0
                pr = json.load(open(f"paper2_results/phase3/pressure_probe_{name}.json"))
                got = {r["variant"]: r["n"] for r in pr["variants"]}
                assert len(got) == 5 and got["single / user"] == 8 and got["consensus / assistant"] == 8, got
                assert got["paper1 REC_WRONG note / user"] == 6, got      # cases q1 and q13 have B as the correct answer -> skipped
                print("PASS: probe runs end to end; Paper 1 variant skipped exactly the 2 cases whose correct answer is the one it asserts")

                rc = run_main(p3, ["--model", name, "--channel", "assistant", "--n-pairs", "6", "--min-clean-p-correct", "0.99"])
                assert rc == 1, "a filter that rejects every case must fail cleanly (exit 1), not crash or continue"
                print("PASS: impossible filter threshold fails cleanly with exit code 1")

                for channel in ("user", "assistant"):
                    rc = run_main(loc, ["--model", name, "--channel", channel, "--pressure", "consensus", "--n-pairs", "6", "--min-clean-p-correct", "0", "--fine-layers", "1,2"])
                    assert rc == 0, f"localization {channel} returned {rc}"
                    r = json.load(open(f"paper2_results/phase3/metrics/localization_{name}_consensus_{channel}.json"))
                    assert r["metric"] == "P(correct)" and r["n_cases"] == 6 and r["pressure"] == "consensus"
                    assert len(r["stage1_resid_post"]) == N_LAYERS
                    assert len(r["stage2_attn_mlp_split"]) == 4         # 2 layers x {attn_out, mlp_out}
                    print(f"PASS: localization end to end, channel={channel}")

                # Phase 4 has never run on real data: give it synthetic Phase 2 directions of the right shape and check it loads both sets.
                rng = np.random.default_rng(0)
                s2 = DirectionStore("paper2_results/phase2/directions")
                for layer in range(N_LAYERS):
                    v = rng.normal(size=D_MODEL); v /= np.linalg.norm(v)
                    s2.save(model=name, attribute="ses", layer=layer, direction=v, mean_pos=v, mean_neg=-v, std=np.ones(D_MODEL),
                            bias_scalar=0.0, n_pairs=5, extraction_point="last_token")
                for channel in ("user", "assistant"):
                    rc = run_main(p4, ["--model", name, "--attribute", "ses", "--pressure", "consensus", "--channel", channel])
                    assert rc == 0
                    out = json.load(open(f"paper2_results/phase4/entanglement_{name}_ses_consensus_{channel}.json"))
                    assert out["deference_pressure"] == "consensus"
                    assert out["n_common_layers"] == N_LAYERS and all(-1.0001 <= x["cosine_similarity"] <= 1.0001 for x in out["per_layer"])
                print("PASS: Phase 4 loads both direction sets and writes per-layer cosine similarity")
            finally:
                os.chdir(old_cwd)
    finally:
        MECH_MODELS_BY_NAME[name] = orig_spec
        ace_sensor.load_model, p3.load_amqa_dataset, loc.load_amqa_dataset = orig_load, orig_amqa_p3, orig_amqa_loc


if __name__ == "__main__":
    test_variant_prompt_placement()
    test_classify_layer()
    test_layer_support()
    test_oneside_support()
    test_effect_floor()
    test_filter_and_behavioral_math()
    test_end_to_end()
    print("\nALL PHASE 3 TESTS PASSED")