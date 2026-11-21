"""test_final_stages_end_to_end.py -- CPU: the REAL run_crossfit.py / analyze_crossfit.py and run_attn_mlp_split.py /
analyze_attn_mlp.py on a tiny random rotary transformer, through the frozen-plan path. The numbers are meaningless; what is
checked is the plumbing and the properties that matter:

  * a held-out case is never steered by a direction built from it, and the direction used for case i is exactly fold(i)'s
  * random directions run at the same effective size as the real one for that case
  * parameters come from the plan; a conflicting flag, or a modified plan, is refused
  * a partly finished run resumes and reproduces the complete run exactly
  * the exactness check passes on a rotary model and FAILS on a model where the identity cannot hold

    python test_final_stages_end_to_end.py        (needs test_phase3_end_to_end.py and test_attn_mlp.py alongside)
"""
import contextlib
import dataclasses
import io
import json
import os
import sys
import tempfile

import numpy as np

import test_phase3_end_to_end as T3
import test_attn_mlp as TA
import analyze_attn_mlp as aam
import analyze_crossfit as acf
import run_attn_mlp_split as ram
import run_crossfit as rcf
import run_phase3_deference_sensor as p3
from mech import ace_sensor, crossfit
from mech import plan as P
from mech.model_registry import MECH_MODELS_BY_NAME

NAME = "Mistral-7B-Instruct-v0.1"
N_CASES = 24

TEST_PLAN = {
    "plan_version": "test", "models": [NAME], "channels": ["user", "assistant"], "pressure": "consensus",
    "cases": {"source": "paper2_results/extracted_paper1_notes.jsonl", "n_pairs_cap": 16, "min_clean_p_correct": {NAME: 0.0}},
    "crossfit": {"layers": [1, 2], "folds": 4, "seed": 0, "real_coefficients": [1.0, 2.0], "random_coefficients": [2.0], "n_random": 5, "primary_strength": 2.0,
                 "secondary_strength": 1.0, "alpha_fwe": 0.05, "min_band_layers": 3, "bootstrap_resamples": 200, "ci_level": 0.95, "min_pressure_effect_for_shares": 0.0},
    "attn_mlp_split": {"primary_window": [1, 2], "control_windows": [[0, 0], [3, 3]], "single_layer_sweep": [1, 2], "dominance_margin": 0.10, "min_dominant_restoration": 0.15,
                       "min_gap": -1.0, "bootstrap_resamples": 200, "ci_level": 0.95},
    "frozen_files": ["dummy_code.py"], "frozen_data": ["paper2_results/extracted_paper1_notes.jsonl"],
}


def run(module, argv):
    return T3.run_main(module, argv)


def capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def main():
    model, _ = TA.build_rotary_model(T3.N_LAYERS, T3.D_MODEL)
    orig_spec = MECH_MODELS_BY_NAME[NAME]
    tiny = dataclasses.replace(orig_spec, n_layers=T3.N_LAYERS, d_model=T3.D_MODEL)
    orig_load, orig_amqa = ace_sensor.load_model, (rcf.load_amqa_dataset, ram.load_amqa_dataset)
    MECH_MODELS_BY_NAME[NAME] = tiny
    ace_sensor.load_model = lambda spec, device=None: model
    try:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                recs = T3.make_pool_and_amqa(N_CASES)
                rcf.load_amqa_dataset = ram.load_amqa_dataset = lambda dry_run=False: recs
                open("dummy_code.py", "w").write("print('frozen')\n")
                json.dump(TEST_PLAN, open("analysis_plan.json", "w"), indent=1)
                open("ANALYSIS_PLAN.md", "w").write("# test plan\n")
                P.write_lock("analysis_plan.json", ".")
                assert P.plan_status("analysis_plan.json", ".")["status"] == "FROZEN-OK"

                # ======================= cross-fitting =======================
                spy = []
                real_steer = ace_sensor.steered_class_prediction

                def spy_steer(model_, tokens, ids, *, layer, direction, coefficient, component="resid_post"):
                    spy.append((layer, np.array(direction, dtype=np.float64), float(coefficient)))
                    return real_steer(model_, tokens, ids, layer=layer, direction=direction, coefficient=coefficient, component=component)

                ace_sensor.steered_class_prediction = spy_steer
                try:
                    rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json"]))
                finally:
                    ace_sensor.steered_class_prediction = real_steer
                assert rc == 0 and "FROZEN-OK" in out, out[-600:]
                mpath, apath = rcf.out_paths(NAME, "consensus", "user")
                z = np.load(mpath)
                n, nL, R = len(z["case_ids"]), 2, 5
                assert n == 16 and z["real"].shape == (nL, 2, n) and z["rand"].shape == (nL, 1, R, n) and z["done"].all()
                assert not np.isnan(z["real"]).any() and not np.isnan(z["rand"]).any() and ((z["real"] >= 0) & (z["real"] <= 1)).all()
                meta = json.loads(str(z["meta"]))
                assert meta["plan_status"] == "FROZEN-OK" and meta["folds"] == 4 and meta["n_random"] == 5 and meta["layers"] == [1, 2]
                assert sorted(np.bincount(z["folds"])) == [4, 4, 4, 4]

                # the direction handed to the model for case i at layer l IS fold(i)'s direction, and beats every other fold's (no leakage)
                acts = np.load(apath)
                pos, neg = acts["acts_pos"].astype(np.float32), acts["acts_neg"].astype(np.float32)
                dirs, gaps = crossfit.fold_directions(pos, neg, z["folds"], [1, 2])
                assert np.allclose(gaps, z["gaps"], rtol=2e-2), (gaps, z["gaps"])
                # real-direction calls come first for each (layer, case): strengths 1 then 2, then R random at strength 2
                per_case = 2 + R
                calls = [c for c in spy if c[0] in (1, 2)]
                assert len(calls) == nL * n * per_case, len(calls)
                k = 0
                for li, layer in enumerate([1, 2]):
                    for i in range(n):
                        block = calls[k:k + per_case]
                        k += per_case
                        f = int(z["folds"][i])
                        cos = [float(np.dot(block[0][1], dirs[li, g]) / (np.linalg.norm(block[0][1]) * np.linalg.norm(dirs[li, g]))) for g in range(4)]
                        assert int(np.argmax(cos)) == f and cos[f] > 0.999, (layer, i, f, cos)                    # its OWN fold's training-only direction
                        assert abs(block[0][2] - 1.0 * z["gaps"][li, f]) < 1e-6 and abs(block[1][2] - 2.0 * z["gaps"][li, f]) < 1e-6   # strength x training gap
                        for r, c in enumerate(block[2:]):
                            assert abs(c[2] - 2.0 * z["gaps"][li, f]) < 1e-6                                       # random: same effective size as the real one
                            assert abs(np.linalg.norm(c[1]) - 1) < 1e-4 and np.allclose(c[1], crossfit.random_directions(layer, R, 32, 0)[r], atol=1e-6)
                print("PASS: cross-fit -- each held-out case is steered by ITS fold's training-only direction (strength x training gap), and random directions are the deterministic unit vectors at the same effective size")

                # a conflicting flag is refused; a modified plan/code is refused unless explicitly allowed (and then labelled)
                rc, _ = capture(lambda: run(rcf, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json", "--n-random", "9"]))
                assert rc == 2
                open("dummy_code.py", "w").write("print('edited after freezing')\n")
                rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "assistant", "--plan", "analysis_plan.json"]))
                assert rc == 2 and "PLAN MODIFIED" in out and not os.path.exists(str(rcf.out_paths(NAME, "consensus", "assistant")[0]))
                rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "assistant", "--plan", "analysis_plan.json", "--allow-modified-plan", "--out-dir", "paper2_results/crossfit_mod"]))
                assert rc == 0
                zm = np.load(rcf.out_paths(NAME, "consensus", "assistant", "paper2_results/crossfit_mod")[0])
                assert json.loads(str(zm["meta"]))["plan_status"] == "MODIFIED"
                open("dummy_code.py", "w").write("print('frozen')\n")
                assert P.plan_status("analysis_plan.json", ".")["status"] == "FROZEN-OK"
                print("PASS: a flag that conflicts with the plan is refused (exit 2); a modified plan is refused, or runs only when explicitly allowed and is then stamped MODIFIED")

                # resume: knock out layer 2, rerun, get the same numbers; and --stop-after-layers then resume
                full_real, full_rand = z["real"].copy(), z["rand"].copy()
                d = {k_: z[k_] for k_ in z.files}
                d["done"] = np.array([True, False])
                d["real"] = d["real"].copy(); d["rand"] = d["rand"].copy()
                d["real"][1] = np.nan; d["rand"][1] = np.nan
                np.savez(mpath, **d)
                rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json"]))
                assert rc == 0 and "RESUMING: 1 of 2" in out, out[-400:]
                z2 = np.load(mpath)
                assert z2["done"].all() and np.allclose(z2["real"], full_real, atol=1e-6) and np.allclose(z2["rand"], full_rand, atol=1e-6)
                rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json", "--restart", "--stop-after-layers", "1"]))
                assert rc == 0 and int(np.load(mpath)["done"].sum()) == 1
                rc, out = capture(lambda: run(rcf, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json"]))
                z3 = np.load(mpath)
                assert rc == 0 and z3["done"].all() and np.allclose(z3["real"], full_real, atol=1e-6)
                print("PASS: a run knocked out mid-way resumes and reproduces the complete run exactly (also after --stop-after-layers)")

                # the analysis: plan status printed and stored, the too-few-draws warning, a verdict, a JSON summary
                _, out = capture(lambda: T3.run_main(acf, []))
                assert "PLAN FROZEN-OK" in out and "WARNING: only 5 random directions" in out and "NO band" in out and "held-out cases" in out, out[:900]
                sj = json.load(open("paper2_results/crossfit/summary_crossfit_%s_consensus_user.json" % NAME))
                assert sj["plan_status"] == "FROZEN-OK" and sj["supported"] == [False, False] and sj["has_heldout_direction"] is False and abs(sj["min_attainable_p"] - 1 / 6) < 1e-12
                print("PASS: analyze_crossfit prints and stores the plan status, warns that 5 draws cannot reach alpha 0.05, and reports no band")

                # ======================= attention vs MLP =======================
                rc, out = capture(lambda: run(ram, ["--model", NAME, "--channel", "assistant", "--plan", "analysis_plan.json"]))
                assert rc == 0 and "EXACTNESS CHECK" in out and "(valid)" in out, out[-500:]
                zz = np.load("paper2_results/attn_mlp/attn_mlp_%s_consensus_assistant.npz" % NAME)
                mm = json.loads(str(zz["meta"]))
                assert mm["valid"] and mm["windows"] == [[1, 2], [0, 0], [3, 3]] and mm["roles"] == ["primary", "control1", "control2"] and mm["plan_status"] == "FROZEN-OK"
                nn = len(zz["case_ids"])
                assert zz["patched_window"].shape == (3, 3, nn) and zz["patched_sweep"].shape == (2, 2, nn) and np.abs(zz["sanity"] - zz["clean_pc"]).max() < 1e-3 and np.abs(zz["sanity_attn"] - zz["clean_pc"]).max() < 1e-3
                # a window covering ALL layers with attention alone must restore everything: restoration 1.0
                rc, _ = capture(lambda: run(ram, ["--model", NAME, "--channel", "user", "--windows", "0-3,1-1", "--sweep", "1-2", "--min-clean-p-correct", "0", "--n-pairs", "12", "--out-dir", "paper2_results/attn_all"]))
                za = np.load("paper2_results/attn_all/attn_mlp_%s_consensus_user.npz" % NAME)
                assert rc == 0 and np.allclose(za["patched_window"][0, 0], za["clean_pc"], atol=1e-6) and np.allclose(za["patched_window"][0, 2], za["clean_pc"], atol=1e-6)
                assert not np.allclose(za["patched_window"][0, 1], za["clean_pc"], atol=1e-6)               # MLP-only over all layers is NOT the identity
                rc, _ = capture(lambda: run(ram, ["--model", NAME, "--channel", "user", "--plan", "analysis_plan.json", "--sweep", "1-3"]))
                assert rc == 2                                                                              # conflicting flag refused
                _, out = capture(lambda: T3.run_main(aam, []))
                assert "PLAN FROZEN-OK" in out and "VERDICT (frozen rules)" in out and "single-layer sweep" in out, out[:800]
                assert json.load(open("paper2_results/attn_mlp/summary_attn_mlp_%s_consensus_assistant.json" % NAME))["plan_status"] == "FROZEN-OK"
                print("PASS: attention-vs-MLP runs from the plan; attention (alone or with MLP) over ALL layers restores exactly 1.0 and MLP alone does not; conflicting flag refused; analysis stamps the plan status")

                # the exactness check can FAIL. Sabotage the patching function (it ignores the clean activations whenever a window covers
                # every layer) and the run must be flagged INVALID (exit 3), and the analysis must refuse to interpret it.
                from mech import patching_window as pwm
                orig_patch = pwm.patch_window_and_predict

                def broken(model_, tokens, acts_, *, layers, components, class_token_ids):
                    probs = orig_patch(model_, tokens, acts_, layers=layers, components=components, class_token_ids=class_token_ids)
                    if len(list(layers)) == T3.N_LAYERS:
                        return {c: (1.0 if c == "A" else 0.0) for c in probs}
                    return probs

                pwm.patch_window_and_predict = broken
                try:
                    rc, out = capture(lambda: run(ram, ["--model", NAME, "--channel", "user", "--windows", "1-2,0-0", "--sweep", "1-2", "--min-clean-p-correct", "0", "--n-pairs", "12", "--out-dir", "paper2_results/attn_bad"]))
                finally:
                    pwm.patch_window_and_predict = orig_patch
                assert rc == 3 and "INVALID" in out, out[-400:]
                bad_path = "paper2_results/attn_bad/attn_mlp_%s_consensus_user.npz" % NAME
                _, out = capture(lambda: aam.render(aam.analyze_file(bad_path, aam.DEFAULT_CFG, P.plan_status("analysis_plan.json", ".")), aam.DEFAULT_CFG))
                assert "INVALID RUN" in out
                print("PASS: when the identity does not hold the exactness check fails (exit 3, flagged INVALID) and the analysis refuses to interpret the run")
            finally:
                os.chdir(old)
    finally:
        MECH_MODELS_BY_NAME[NAME] = orig_spec
        ace_sensor.load_model = orig_load
        rcf.load_amqa_dataset, ram.load_amqa_dataset = orig_amqa
    print("\nALL FINAL-STAGE END-TO-END TESTS PASSED")


if __name__ == "__main__":
    main()
