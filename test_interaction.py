"""
test_interaction.py -- CPU.   python test_interaction.py
Needs, alongside: mech/, build_bias_pairs.py, collect_bias_activations.py, collect_interaction.py, analyze_bias_pilot.py, analyze_interaction.py, recompute_yields.py,
test_analyze_bias_pilot.py and bias_stage0/carriers_proposed.json. Uses the project's `shared` package if importable, otherwise a stand-in (created by test_pressure_format).

PART A  the collector, end to end, on a tiny random transformer with the REAL Mistral tokenizer and chat template
  * the no-footer prompts reproduce the existing deference prompts' probabilities and states; a footer prompt's final-token states equal a hand-made hook read
  * identical prompts are computed once; --base-only collects only the no-footer prompts; a half-finished run resumes exactly
PART B  the analysis, on a synthetic world with PLANTED answers (including nulls):
  * a planted cue-by-pressure interaction is recovered inside its interval; with none, the interval lies inside the equivalence margin; the footer-position control agrees
  * a planted overlap between the bias and deference directions reads OVERLAP; orthogonal directions do not
  * the recomputed yields equal the planted yield exactly; a prompt mismatch trips the reproducibility gate
  * an unfrozen plan is refused, and the plan's numbers equal the script's defaults
"""
import contextlib
import dataclasses
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

import test_pressure_format as TPF                       # also sets up the stand-in for `shared` when the real package is absent
from types import SimpleNamespace

import analyze_bias_pilot as A
import analyze_interaction as AI
import build_bias_pairs as B
import collect_interaction as CI
import recompute_yields as Y
import test_analyze_bias_pilot as TB
from mech import ace_sensor, chat_prompting, deference_pairs as dp, interaction_prompts as IP
from mech.model_registry import MECH_MODELS_BY_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
CARRIERS = os.path.join(HERE, "bias_stage0", "carriers_proposed.json")
NAME = "Mistral-7B-Instruct-v0.1"
N_LAYERS, D_MODEL = 4, 32


# ================================================================================================ PART A: the collector

def tiny_model():
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    cfg = HookedTransformerConfig(n_layers=N_LAYERS, d_model=D_MODEL, d_head=8, n_heads=4, d_mlp=64, n_ctx=4096, d_vocab=len(tok), act_fn="gelu", tokenizer_name=None, seed=0)
    return HookedTransformer(cfg, tokenizer=tok).eval()


def amqa_records(n):
    opts = {"A": "Left middle cerebral", "B": "Anterior spinal", "C": "Basilar", "D": "Posterior inferior cerebellar"}
    return [SimpleNamespace(question_id=f"q{i}", neutralised_question=f"Which vessel is involved in case {i}?", options=dict(opts), correct_answer_idx="ABCD"[i % 4]) for i in range(n)]


def test_collector_end_to_end():
    model = tiny_model()
    orig_spec = MECH_MODELS_BY_NAME[NAME]
    MECH_MODELS_BY_NAME[NAME] = dataclasses.replace(orig_spec, n_layers=N_LAYERS, d_model=D_MODEL)
    orig_load = ace_sensor.load_model
    import shared.data_loader as dl
    orig_amqa = dl.load_amqa_dataset
    ace_sensor.load_model = lambda spec, device=None: model
    try:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.getcwd()
            os.chdir(tmp)
            try:
                recs = amqa_records(8)
                dl.load_amqa_dataset = lambda dry_run=False: recs
                os.makedirs("paper2_results")
                with open("paper2_results/extracted_paper1_notes.jsonl", "w") as f:
                    for i in range(8):
                        note = f"The patient presents with symptom set {i}. Vitals are stable." + ("\n" if i == 2 else "")      # one raw note has a trailing newline
                        f.write(json.dumps(dict(case_id=f"q{i}", condition_label="N", attribute="ses", signal_type=None, rec_type=None, note_text=note)) + "\n")
                os.makedirs("bias_stage0")
                shutil.copy(CARRIERS, "bias_stage0/carriers.json")
                sys.argv = ["x", "--pool", "paper2_results/extracted_paper1_notes.jsonl", "--carriers", "bias_stage0/carriers.json", "--out", "bias_stage0/pairs.jsonl"]
                with contextlib.redirect_stdout(io.StringIO()):
                    assert B.main() == 0
                # ---- base-only: only the no-footer prompts, user_fl folded into user_ff
                argv = ["x", "--model", NAME, "--base-only", "--layers", "1-3", "--out-dir", "bias_stage1/acts_base"]
                sys.argv = argv
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    assert CI.main() == 0
                z = {k: v for k, v in np.load("bias_stage1/acts_base/interaction_acts_Mistral-7B-Instruct-v0.1.npz", allow_pickle=False).items()}
                U = len(z["keys"])
                assert U == 8 * 3 and z["done"].all(), U                                   # 8 cases x (clean, user_ff, assistant): user_fl is the same prompt as user_ff
                assert z["acts"].shape == (U, 3, D_MODEL) and np.allclose(z["probs"].sum(1), 1, atol=1e-5) and not np.isnan(z["probs"]).any()
                ids = chat_prompting.resolve_class_token_ids(model, chat_prompting.USMLE_SUFFIX, list(z["class_letters"]))
                keys = {str(k): i for i, k in enumerate(z["keys"])}
                mp = {(str(r), str(v)): keys[str(k)] for r, v, k in zip(z["map_rec"], z["map_variant"], z["map_key"])}
                assert mp[("q0|base", "user_ff")] == mp[("q0|base", "user_fl")]
                for i in (0, 2, 5):                                                          # the base prompts equal the EXISTING builders' prompts, raw notes included
                    cid = f"q{i}"
                    rec = recs[i]
                    corr, wl, wt = IP.wrong_option_for(rec)
                    raw = f"The patient presents with symptom set {i}. Vitals are stable." + ("\n" if i == 2 else "")
                    for variant, builder in (("clean", lambda: chat_prompting.tokenize_chat_prompt(model, IP.build_messages("clean", rec, raw, None, wl, wt), chat_prompting.USMLE_SUFFIX)),
                                             ("user_ff", lambda: chat_prompting.tokenize_chat_prompt(model, chat_prompting.build_user_role_messages(chat_prompting.USMLE_BODY_TEMPLATE.format(
                                                 validated_note=dp.append_consensus_recommendation(raw, wl, wt), usmle_question=rec.neutralised_question,
                                                 option_a=rec.options["A"], option_b=rec.options["B"], option_c=rec.options["C"], option_d=rec.options["D"])), chat_prompting.USMLE_SUFFIX)),
                                             ("assistant", lambda: chat_prompting.tokenize_chat_prompt(model, chat_prompting.build_assistant_role_deference_messages(
                                                 raw, rec.neutralised_question, dp.options_dict(rec), wl, wt, closing_text=dp.CONSENSUS_CLOSING.format(letter=wl)), chat_prompting.USMLE_SUFFIX))):
                        toks = builder()
                        pr = ace_sensor.plain_class_probs(model, toks, ids)
                        u = mp[(f"{cid}|base", variant)]
                        assert np.allclose(z["probs"][u], [pr[c] for c in z["class_letters"]], atol=1e-4), (cid, variant)
                        ref = ace_sensor.extract_all_layers_last_token(model, toks, N_LAYERS)[1:4]
                        assert np.allclose(z["acts"][u].astype(np.float32), ref, atol=2e-2 * max(1.0, float(np.abs(ref).max()) / 10)), (cid, variant)
                print("PASS: the no-footer prompts the collector computes reproduce the EXISTING deference prompts' probabilities and final-token states (user and assistant, raw notes with a trailing newline included); "
                      "user_fl is folded into user_ff when there is no footer")
                # ---- the full run on a subset: dedupe, a footer prompt's states, resume
                argv = ["x", "--model", NAME, "--layers", "1-3", "--max-unique", "60", "--save-every", "20", "--out-dir", "bias_stage1/acts"]
                sys.argv = argv
                with contextlib.redirect_stdout(io.StringIO()):
                    assert CI.main() == 0
                path = "bias_stage1/acts/interaction_acts_Mistral-7B-Instruct-v0.1.npz"
                z = {k: v for k, v in np.load(path, allow_pickle=False).items()}
                assert z["done"].all() and len(z["keys"]) == 60
                recs_pairs = [json.loads(l) for l in open("bias_stage0/pairs.jsonl")]
                by_id = {r["id"]: r for r in recs_pairs}
                keys = {str(k): i for i, k in enumerate(z["keys"])}
                checked = 0
                for r_id, v, k in zip(z["map_rec"], z["map_variant"], z["map_key"]):
                    r = by_id[str(r_id)]
                    if r["kind"] == "base" or str(v) != "user_ff" or checked >= 2:
                        continue
                    rec = recs[int(r["case_id"][1:])]
                    corr, wl, wt = IP.wrong_option_for(rec)
                    base = next(x["text"] for x in recs_pairs if x["kind"] == "base" and x["case_id"] == r["case_id"])
                    toks = chat_prompting.tokenize_chat_prompt(model, IP.build_messages("user_ff", rec, base, r["footer"], wl, wt), chat_prompting.USMLE_SUFFIX)
                    _, cache = model.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_resid_post"))
                    hand = np.stack([cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu().numpy() for l in (1, 2, 3)])
                    assert np.allclose(z["acts"][keys[str(k)]].astype(np.float32), hand, atol=2e-2 * max(1.0, float(np.abs(hand).max()) / 10))
                    checked += 1
                assert checked == 2
                d = {k: v.copy() for k, v in z.items()}
                d["done"][40:] = False
                d["acts"][40:] = 0
                np.savez(path, **d)
                buf = io.StringIO()
                sys.argv = argv
                with contextlib.redirect_stdout(buf):
                    assert CI.main() == 0
                z2 = dict(np.load(path, allow_pickle=False))
                assert "RESUMING: 40 of 60" in buf.getvalue() and z2["done"].all() and np.array_equal(z2["acts"], z["acts"]) and np.allclose(z2["probs"], z["probs"], atol=1e-6)
                print("PASS: a footer prompt's final-token states equal a hand-made hook read, and a partly finished run resumes exactly")
            finally:
                os.chdir(old)
    finally:
        MECH_MODELS_BY_NAME[NAME] = orig_spec
        ace_sensor.load_model = orig_load
        dl.load_amqa_dataset = orig_amqa


# ================================================================================================ PART B: the analysis

D, NL, LAYERS_I = 64, 11, list(range(14, 25))
LETTERS = list("ABCD")


def make_world(tmp, n_cases=60, n_defer=40, rho=0.0, race_overlap=0.0, seed=0, perturb_run_case=None, sigma=1.0):
    """A synthetic interaction run. Income cues change the pressure effect by `rho` (a share of it); the race group direction shares `race_overlap` of its norm with the deference axis."""
    rng = np.random.default_rng(seed)
    cfg = B.load_carriers(CARRIERS)
    notes = {f"c{i:03d}": f"Base note number {i}. " * 25 for i in range(n_cases)}
    recs, _, _ = B.build(notes, cfg)
    pairs_path = os.path.join(tmp, "pairs.jsonl")
    with open(pairs_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    traits = TB.traits_by_footer(cfg)
    K = 6                                                                                  # the deference axis lives in the first K dimensions
    u = rng.normal(size=(32, D))
    u[:, K:] = 0.0
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    def off(v):
        v = v.copy()
        v[:K] = 0.0
        return v / np.linalg.norm(v)
    gvec = {}
    for t in ("race", "sex", "ses"):
        w = np.stack([off(rng.normal(size=D)) for _ in range(32)])
        c = race_overlap if t == "race" else 0.0
        gvec[t] = 6.0 * (c * u + np.sqrt(1 - c ** 2) * w)                                  # (32, D)
    lex = {ft: off(rng.normal(size=D)) * 3.0 for ft in traits}
    base_state = {c: rng.normal(size=(32, D)) * 3.0 for c in notes}
    case_ids = sorted(notes)
    correct = {c: LETTERS[i % 4] for i, c in enumerate(case_ids)}
    wrong = {c: LETTERS[(i + 1) % 4] for i, c in enumerate(case_ids)}
    pe_case = {c: (0.85 if i % 4 != 3 else 0.30) for i, c in enumerate(case_ids)}             # three quarters of the cases yield
    clean_p = {c: np.zeros(4) for c in case_ids}
    for c in case_ids:
        p = np.full(4, 0.03)
        p[LETTERS.index(correct[c])] = 0.91
        p[LETTERS.index(wrong[c])] = 0.03
        clean_p[c] = p / p.sum()

    def pressured_probs(c, t_ses):
        p = clean_p[c].copy()
        s = LETTERS.index(wrong[c])
        k = LETTERS.index(correct[c])
        shift = pe_case[c] * (1 + rho * t_ses / 2.0)
        p[s] += shift
        p[k] = max(0.005, p[k] - 0.95 * shift)
        return p / p.sum()
    uniq, rec_map = {}, []
    VARS = ("clean", "user_ff", "user_fl", "assistant")
    for r in recs:
        c = r["case_id"]
        is_base = r["kind"] == "base"
        for v in VARS:
            if is_base and v == "user_fl":
                v_key = "user_ff"
            else:
                v_key = v
            key = hashlib.sha1(f"{v_key}|{r['text']}".encode()).hexdigest()
            if key not in uniq:
                t = traits[r["footer"]] if not is_base else {"race": 0, "sex": 0, "ses": 0}
                pressured = v_key != "clean"
                p = pressured_probs(c, t["ses"]) if pressured else clean_p[c].copy()
                p = np.clip(p + rng.normal(0, 0.004, 4), 0.001, None)
                p /= p.sum()
                st = base_state[c][LAYERS_I].copy()
                if not is_base:
                    for trait in ("race", "sex", "ses"):
                        st += 0.9 * t[trait] * gvec[trait][LAYERS_I]
                    st += lex[r["footer"]][None, :]
                if pressured:
                    st += 6.0 * u[LAYERS_I]
                st = st + rng.normal(size=st.shape) * sigma
                uniq[key] = (p, st)
            rec_map.append((r["id"], v, key))
    keys = sorted(uniq)
    acts = np.stack([uniq[k][1] for k in keys]).astype(np.float16)
    probs = np.stack([uniq[k][0] for k in keys]).astype(np.float32)
    meta = {"model": NAME, "layers": LAYERS_I, "variants": list(VARS), "base_only": False}
    ipath = os.path.join(tmp, f"interaction_acts_{NAME}.npz")
    np.savez(ipath, meta=np.array(json.dumps(meta)), keys=np.array(keys), acts=acts, probs=probs, done=np.ones(len(keys), bool), layers=np.array(LAYERS_I),
             class_letters=np.array(LETTERS), map_rec=np.array([m[0] for m in rec_map]), map_variant=np.array([m[1] for m in rec_map]), map_key=np.array([m[2] for m in rec_map]),
             case_ids=np.array(case_ids), case_correct=np.array([correct[c] for c in case_ids]), case_wrong=np.array([wrong[c] for c in case_ids]))
    # the confirmatory run's files for both channels
    runs = os.path.join(tmp, "runs")
    os.makedirs(runs, exist_ok=True)
    known = case_ids[:n_defer]
    folds = np.arange(n_defer) % 10
    for ch in ("user", "assistant"):
        clean = np.zeros((n_defer, 32, D), np.float32)
        pressed = np.zeros_like(clean)
        for i, c in enumerate(known):
            for l in range(32):
                b = base_state[c][l] + rng.normal(size=D) * sigma
                clean[i, l] = b
                pressed[i, l] = b + 6.0 * u[l] + rng.normal(size=D) * sigma
        key_of = {(r_, v_): k_ for r_, v_, k_ in rec_map}                                            # the run file stores EXACTLY what the base prompts give
        pv = "user_ff" if ch == "user" else "assistant"
        p_clean = [uniq[key_of[(f"{c}|base", "clean")]][0] for c in known]
        p_press = [uniq[key_of[(f"{c}|base", pv)]][0] for c in known]
        ps_clean = np.array([p[LETTERS.index(wrong[c])] for p, c in zip(p_clean, known)])
        pc_clean = np.array([p[LETTERS.index(correct[c])] for p, c in zip(p_clean, known)])
        ps_press = np.array([p[LETTERS.index(wrong[c])] for p, c in zip(p_press, known)])
        pc_press = np.array([p[LETTERS.index(correct[c])] for p, c in zip(p_press, known)])
        if perturb_run_case is not None:
            ps_press = ps_press.copy()
            ps_press[perturb_run_case] += 0.05
        np.savez(os.path.join(runs, f"crossfit_{NAME}_consensus_{ch}_acts.npz"), acts_pos=clean.astype(np.float16), acts_neg=pressed.astype(np.float16), case_ids=np.array(known))
        np.savez(os.path.join(runs, f"crossfit_{NAME}_consensus_{ch}.npz"), case_ids=np.array(known), folds=folds, p_stated_clean=ps_clean, p_stated_pressured=ps_press,
                 p_correct_clean=pc_clean, p_correct_pressured=pc_press)
    return {"pairs": pairs_path, "interaction": ipath, "runs": runs, "known": known, "n_cases": n_cases}


def quick_params():
    p = json.loads(json.dumps(AI.DEFAULTS))
    p["interaction"].update({"bootstrap": 500, "axis_bootstrap": 150})
    p["overlap"].update({"n_splits": 40, "bootstrap": 60})
    return p


def analyse(w, params=None):
    data = AI.IData(w["interaction"], w["pairs"])
    z = np.load(w["interaction"], allow_pickle=False)
    runs = {"correct": dict(zip([str(c) for c in z["case_ids"]], [str(x) for x in z["case_correct"]]))}
    defer = {ch: A.load_defer(os.path.join(w["runs"], f"crossfit_{NAME}_consensus_{ch}_acts.npz"), os.path.join(w["runs"], f"crossfit_{NAME}_consensus_{ch}.npz")) for ch in ("user", "assistant")}
    return AI.run_analysis(data, params or quick_params(), defer, runs, w["runs"], NAME)


def test_no_interaction_and_null_overlap():
    with tempfile.TemporaryDirectory() as t:
        w = make_world(t, rho=0.0, race_overlap=0.0, seed=1)
        res = analyse(w)
        assert all(r["reproduces"] for r in res["reproducibility"].values()), res["reproducibility"]
        for k in ("user|user_ff|known|ses|PRX", "user|user_ff|known|race|PRX", "user|user_ff|known|sex|PRX", "assistant|assistant|known|ses|PRX", "user|user_fl|known|ses|PRX"):
            c = res["interaction"][k]
            lo, hi = c["ci"]
            assert -0.05 <= lo and hi <= 0.05 and c["reading"].startswith("NO INTERACTION"), (k, c)
        moves = [k for k, c in res["axis"].items() if c["reading"].startswith("MOVES")]
        assert len(moves) <= 4, moves                          # 24 cells at 95%: a few false "movement" labels are expected by chance; coverage is tested directly below
        for k in ("user|ses", "user|race", "user|sex"):
            assert res["overlap"][k]["reading"] != "OVERLAP" and res["overlap"][k]["ratio"] < 0.25, (k, res["overlap"][k])
    print("PASS: with no interaction planted every primary interval lies inside the +-5% margin (user_ff, user_fl and assistant agree); no axis movement; orthogonal directions do not read as OVERLAP")


def test_axis_interval_coverage():
    """When the true shift along the deference axis is exactly 0, how often does the 95% interval contain 0? With the direction re-fitted on each resample, and without (the bias pilot's T1)."""
    D_, m, n, K = 48, 40, 135, 5
    nW, reps, nb = 3, 120, 120
    rng = np.random.default_rng(7)
    cover_refit = cover_naive = 0
    for _ in range(reps):
        u = np.zeros((nW, D_))
        u[:, :K] = rng.normal(size=(nW, K))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        cl = rng.normal(size=(m, nW, D_)) * 3.0
        pr = cl + 6.0 * u[None] + rng.normal(size=(m, nW, D_))
        cl = cl + rng.normal(size=(m, nW, D_))
        mu = np.zeros((nW, D_))
        mu[:, K:] = rng.normal(size=(nW, D_ - K))                                    # a real group difference, exactly orthogonal to the true axis
        mu *= 10.0 / np.linalg.norm(mu, axis=1, keepdims=True)
        diff = mu[None] + rng.normal(size=(n, nW, D_)) * 1.4
        lo, hi, _ = AI.axis_ratio_refit(diff, cl.astype(np.float32), pr.astype(np.float32), nb, int(rng.integers(1_000_000)), 0.95)
        cover_refit += int(lo <= 0.0 <= hi)
        # the NAIVE interval: one direction fitted on all cases, resample pairs and cases, gap measured in sample
        uh = A.dir_from(pr, cl)
        s = np.einsum("nld,ld->nl", diff, uh)
        g = np.einsum("mld,ld->ml", pr - cl, uh)
        rhos = []
        for _b in range(nb):
            rhos.append(float(np.mean(s[rng.integers(0, n, n)].mean(0) / g[rng.integers(0, m, m)].mean(0))))
        cover_naive += int(np.percentile(rhos, 2.5) <= 0.0 <= np.percentile(rhos, 97.5))
    cr, cn = cover_refit / reps, cover_naive / reps
    assert cr >= 0.90, cr                                                                # close to the nominal 95%
    assert cn < cr - 0.05, (cn, cr)                                                      # the naive interval covers visibly less often
    print(f"PASS: when the true shift along the axis is 0, the direction-refit 95% interval contains 0 in {100 * cr:.0f}% of {reps} simulated datasets; "
          f"the naive interval (direction not re-fitted) in only {100 * cn:.0f}%")


def test_planted_interaction_and_overlap():
    with tempfile.TemporaryDirectory() as t:
        w = make_world(t, rho=0.25, race_overlap=0.6, seed=2)
        res = analyse(w)
        for k in ("user|user_ff|known|ses|PRX", "user|user_fl|known|ses|PRX", "assistant|assistant|known|ses|PRX", "user|user_ff|all|ses|PRX"):
            c = res["interaction"][k]
            assert c["ci"][0] > 0 and 0.20 < c["rho"] < 0.28 and c["reading"].startswith("INTERACTION"), (k, c)       # recovered: renormalizing the planted probabilities makes the true value 0.24, not 0.25
        for k in ("user|user_ff|known|race|PRX", "user|user_ff|known|sex|PRX"):                                                     # no cue effect planted for these
            assert res["interaction"][k]["reading"].startswith("NO INTERACTION"), (k, res["interaction"][k])
        for k in ("user|race", "assistant|race"):
            c = res["overlap"][k]
            assert c["reading"].startswith("OVERLAP") and c["ratio_ci"][0] > 0.25 and c["abs_cos_mean"] > 4 * c["placebo_abs_cos_mean"], (k, c)
        assert res["overlap"]["user|ses"]["reading"] != "OVERLAP" and res["overlap"]["user|sex"]["reading"] != "OVERLAP"
    print("PASS: a planted 25% interaction is recovered (0.24 after the world's own renormalization) in every variant and channel, the nulls stay null, and a planted overlap reads OVERLAP while orthogonal directions do not")


def test_yields_and_the_gate():
    with tempfile.TemporaryDirectory() as t:
        w = make_world(t, seed=3)
        res = analyse(w)
        for ch in ("user", "assistant"):
            y = res["yields"][ch]
            assert y["n_cases"] == 40 and y["n_knew"] == 40 and abs(y["yield_rate"] - 0.75) < 1e-9 and y["n_yield"] == 30, y        # three quarters of the planted cases yield
            assert y["yield_ci"][0] < 0.75 < y["yield_ci"][1] and y["reproduces_run_file"] is True
            assert abs(y["pressure_effect"] - 0.684) < 0.02                                                 # by hand: 0.814 for 30 cases and 0.295 for 10, after the world's own renormalization
    with tempfile.TemporaryDirectory() as t:
        w = make_world(t, seed=3, perturb_run_case=5)                                                    # the run file's stored value differs by 0.05 for one case
        res = analyse(w)
        assert all(r["reproduces"] is False for r in res["reproducibility"].values())
        text = AI.render(res, "x", False, NAME)
        assert "REPRODUCIBILITY GATE FAILED" in text and "Do not interpret" in text
    print("PASS: the recomputed yield equals the planted 75% (30 of 40) with its interval, and a 0.05 mismatch with the confirmatory run trips the reproducibility gate and warns")


def test_cli_refuses_unfrozen_and_plan_is_consistent():
    with tempfile.TemporaryDirectory() as t:
        w = make_world(t, seed=4)
        plan = {"plan_version": "test", **json.loads(json.dumps(AI.DEFAULTS))}
        plan["interaction"].update({"bootstrap": 300, "axis_bootstrap": 60})
        plan["overlap"].update({"n_splits": 30, "bootstrap": 40})
        pp = os.path.join(t, "analysis_plan.json")
        json.dump(plan, open(pp, "w"))
        base = ["x", "--plan", pp, "--interaction", w["interaction"], "--pairs", w["pairs"], "--runs-dir", w["runs"], "--out-dir", os.path.join(t, "out")]
        old = sys.argv
        try:
            sys.argv = base
            with contextlib.redirect_stderr(io.StringIO()) as err:
                assert AI.main() == 2
            assert "must be frozen" in err.getvalue() and not os.path.exists(os.path.join(t, "out", "interaction_report.txt"))
            sys.argv = base + ["--allow-unfrozen"]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                assert AI.main() == 0
        finally:
            sys.argv = old
        text = buf.getvalue()
        assert "EXPLORATORY" in text and "INTERACTION: does the cue change" in text and "OVERLAP between the bias direction" in text and "BEHAVIORAL YIELDS" in text
        assert os.path.exists(os.path.join(t, "out", "interaction_results.json"))
    print("PASS: the command line refuses an unfrozen plan, and with --allow-unfrozen writes a report and results labelled EXPLORATORY")
    pp = os.path.join(HERE, "bias_stage1", "analysis_plan.json")
    if not os.path.exists(pp):
        print("SKIP: bias_stage1/analysis_plan.json not present next to this test")
        return
    plan = json.load(open(pp))
    for k, v in AI.DEFAULTS.items():
        assert plan[k] == v, f"plan and the script's defaults differ for {k}"
    missing = [f for f in plan["frozen_files"] if not os.path.exists(os.path.normpath(os.path.join(HERE, "bias_stage1", f)))]
    assert not missing, missing
    md = open(os.path.join(HERE, "bias_stage1", "ANALYSIS_PLAN.md")).read()
    for k in ("0.05", "0.25", "0.15", "user_ff", "user_fl", "ceiling", "reproducib"):
        assert k in md, k
    print("PASS: the interaction plan's numbers equal the script's defaults, every frozen file is listed and exists, and the rules appear in the written plan")


if __name__ == "__main__":
    print("(using a stand-in for the project's `shared` package)" if TPF.STUBBED else "(using the project's own `shared` package)")
    test_axis_interval_coverage()
    test_no_interaction_and_null_overlap()
    test_planted_interaction_and_overlap()
    test_yields_and_the_gate()
    test_cli_refuses_unfrozen_and_plan_is_consistent()
    test_collector_end_to_end()
    print("\nALL INTERACTION TESTS PASSED")
