"""test_crossfit_stats.py -- CPU, numpy only.   python test_crossfit_stats.py"""
import numpy as np

from mech import ace_sensor, crossfit as cf

CFG = {"primary_strength": 2.0, "secondary_strength": 1.0, "alpha_fwe": 0.05, "min_band_layers": 3,
       "bootstrap_resamples": 500, "ci_level": 0.95, "min_pressure_effect_for_shares": 0.05}


def test_folds_are_balanced_deterministic_and_validated():
    f = cf.assign_folds(47, 10, seed=0)
    sizes = np.bincount(f, minlength=10)
    assert sizes.sum() == 47 and sizes.max() - sizes.min() <= 1 and set(sizes) == {4, 5}
    assert np.array_equal(f, cf.assign_folds(47, 10, seed=0)) and not np.array_equal(f, cf.assign_folds(47, 10, seed=1))
    for bad in (1, 48):
        try:
            cf.assign_folds(47, bad, 0)
            raise AssertionError("expected a refusal")
        except ValueError:
            pass
    print("PASS: folds are balanced (sizes 4 or 5), deterministic per seed, and invalid fold counts are refused")


def test_fold_directions_never_see_their_own_cases():
    rng = np.random.default_rng(0)
    n, L, d = 30, 3, 16
    pos = rng.normal(size=(n, L, d)).astype(np.float32) + 0.8
    neg = rng.normal(size=(n, L, d)).astype(np.float32)
    folds = cf.assign_folds(n, 5, seed=3)
    dirs, gaps = cf.fold_directions(pos, neg, folds, layers=[0, 2])
    assert dirs.shape == (2, 5, d) and gaps.shape == (2, 5) and (gaps > 0).all()
    # equals the saved-direction recipe computed directly from the training cases
    tr = folds != 2
    ref = ace_sensor._direction_from_vectors(pos[tr, 2, :], neg[tr, 2, :], int(tr.sum()), "last_token")
    assert np.allclose(dirs[1, 2], ref.direction, atol=1e-6) and abs(gaps[1, 2] - float(np.dot(ref.mean_pos - ref.mean_neg, ref.direction))) < 1e-5
    # LEAKAGE TEST: wreck every held-out case of fold 0. Fold 0's direction must not move; every other fold's must.
    pos2, neg2 = pos.copy(), neg.copy()
    pos2[folds == 0] += 50.0
    neg2[folds == 0] -= 50.0
    dirs2, gaps2 = cf.fold_directions(pos2, neg2, folds, layers=[0, 2])
    assert np.allclose(dirs2[:, 0], dirs[:, 0], atol=1e-6) and np.allclose(gaps2[:, 0], gaps[:, 0], atol=1e-5)
    for f in (1, 2, 3, 4):
        assert not np.allclose(dirs2[:, f], dirs[:, f], atol=1e-3)
    print("PASS: a fold's direction equals the training-only recipe and is untouched when its own held-out cases are altered (no leakage)")


def test_random_directions():
    a = cf.random_directions(layer=14, n_random=20, d=64, seed=0)
    assert a.shape == (20, 64) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    assert np.array_equal(a, cf.random_directions(14, 20, 64, 0)) and not np.array_equal(a, cf.random_directions(15, 20, 64, 0))
    assert not np.array_equal(a, cf.random_directions(14, 20, 64, 1)) and abs(float(np.abs(a @ a.T - np.eye(20)).max()) - 0) < 0.6
    print("PASS: random directions are unit length, reproducible, and differ by layer and seed")


def test_contiguous_runs():
    layers = list(range(12, 25))
    flags = [l in (14, 15, 16, 19, 21, 22) for l in layers]
    bands, isolated = cf.contiguous_runs(layers, flags, 3)
    assert bands == [(14, 16)] and isolated == [19, 21, 22], (bands, isolated)
    bands, isolated = cf.contiguous_runs([12, 14, 15, 16], [True, True, True, True], 3)      # 12 is not adjacent to 14
    assert bands == [(14, 16)] and isolated == [12]
    assert cf.contiguous_runs(layers, [False] * len(layers), 3) == ([], [])
    assert cf.contiguous_runs([5, 6, 7], [True, True, True], 3) == ([(5, 7)], [])
    print("PASS: bands need 3 consecutive layer ids; shorter runs and gaps in the layer list are isolated")


def make_arrays(T_real_p, T_real_s, null_scale, n=8, R=20, base=0.8, press=0.7):
    """Noise-free arrays: every case has the same change, so every statistic is exactly computable by hand."""
    nL = len(T_real_p)
    b = np.full(n, base)
    real = np.zeros((nL, 2, n))
    rand = np.zeros((nL, 1, R, n))
    for l in range(nL):
        real[l, 0] = base - T_real_s[l]
        real[l, 1] = base - T_real_p[l]
        for r in range(R):
            rand[l, 0, r] = base - null_scale[l] * (r + 1)
    return real, rand, b, np.full(n, press)


def test_randomization_statistics_by_hand():
    #            layer0 (clear win) layer1 (wins alone, loses to family max) layer2 (tiny) layer3 (wrong sign)
    Tp = [0.30, 0.25, 0.001, -0.10]
    Ts = [0.15, 0.10, 0.05, 0.05]
    nulls = [0.010, 0.012, 0.005, 0.014]            # null statistic of draw r at layer l = nulls[l] * (r + 1), r = 0..19
    real, rand, base, press = make_arrays(Tp, Ts, nulls)
    out = cf.crossfit_statistics(real, rand, base, press, [12, 13, 14, 15], [1.0, 2.0], [2.0], CFG)
    # unadjusted: layer0 beats all 20 draws, layer1 beats its own 20 (max 0.24 < 0.25), layer2's real 0.001 is below every draw
    assert np.allclose(out["p_unc"], [1 / 21, 1 / 21, 21 / 21, 21 / 21])
    # family max per draw = max over layers of nulls[l]*(r+1) = 0.014*(r+1); layer0 (0.30) exceeds all of them (max 0.28);
    # layer1 (0.25) is matched or beaten by draws with 0.014*(r+1) >= 0.25  ->  r+1 in {18, 19, 20}  ->  3 draws
    assert np.allclose(out["null_family_max"], [0.014 * (r + 1) for r in range(20)])
    assert np.allclose(out["p_fwe"], [1 / 21, 4 / 21, 21 / 21, 21 / 21]), out["p_fwe"]
    assert out["supported"] == [True, False, False, False] and out["bands"] == [] and out["isolated"] == [12]
    # shares for layer 0: random's mean change = -0.01*mean(1..20) = -0.105; real change -0.30; pressure 0.7
    assert abs(out["raw_share"][0] - 0.30 / 0.7) < 1e-9 and abs(out["net_share"][0] - (0.30 - 0.105) / 0.7) < 1e-9
    assert out["net_share_ci"][0][0] <= out["net_share"][0] <= out["net_share_ci"][0][1]
    assert np.allclose(out["null_mean_by_layer"], [0.105, 0.126, 0.0525, 0.147]) and np.allclose(out["null_max_by_layer"], [0.20, 0.24, 0.10, 0.28])
    assert abs(out["min_attainable_p"] - 1 / 21) < 1e-12
    print("PASS: p-values, the max-T correction (layer 1: 0.048 alone -> 0.19 after correction), shares and support flags match hand-computed values")


def test_a_band_and_its_net_share():
    Tp = [0.01, 0.30, 0.32, 0.31, 0.01]
    Ts = [0.01, 0.10, 0.12, 0.11, 0.01]
    real, rand, base, press = make_arrays(Tp, Ts, [0.005] * 5)          # nulls 0.005..0.10, far below 0.30
    out = cf.crossfit_statistics(real, rand, base, press, [12, 13, 14, 15, 16], [1.0, 2.0], [2.0], CFG)
    assert out["supported"] == [False, True, True, True, False] and out["bands"] == [[13, 15]] and out["isolated"] == []
    nulls_mean = 0.005 * 10.5                                            # mean of 0.005*(r+1) over 20 draws
    want = np.mean([(t - nulls_mean) / 0.7 for t in (0.30, 0.32, 0.31)])
    assert abs(out["band_net_share"][0]["mean"] - want) < 1e-9 and out["band_net_share"][0]["band"] == [13, 15]
    print("PASS: three consecutive supported layers form a band, and the band's net share is the mean of its layers' net shares")


def test_edge_cases():
    real, rand, base, press = make_arrays([0.3, 0.3], [0.1, 0.1], [0.01, 0.01], R=3)
    out = cf.crossfit_statistics(real, rand, base, press, [12, 13], [1.0, 2.0], [2.0], CFG)
    assert abs(out["min_attainable_p"] - 0.25) < 1e-12 and out["supported"] == [False, False]        # with R = 3, alpha 0.05 is unreachable
    real, rand, base, press = make_arrays([0.3], [0.1], [0.01], press=0.01)
    out = cf.crossfit_statistics(real, rand, base, press, [12], [1.0, 2.0], [2.0], CFG)
    assert out["shares_reported"] is False and "net_share" not in out                                  # tiny pressure effect: no shares
    print("PASS: too few random draws makes alpha unreachable (and says so); a tiny pressure effect suppresses shares")


def test_interval_covers_the_truth_with_noise():
    rng = np.random.default_rng(1)
    n, nL, R = 60, 3, 20
    base = np.full(n, 0.8)
    press = 0.7 + rng.normal(0, 0.03, n)
    true_net = 0.25                                                        # real lowers by 0.25 more than random's mean
    real = np.zeros((nL, 2, n))
    rand = np.zeros((nL, 1, R, n))
    for l in range(nL):
        real[l, 0] = base - 0.10 + rng.normal(0, 0.03, n)
        real[l, 1] = base - (0.05 + true_net * 0.7) + rng.normal(0, 0.03, n)
        for r in range(R):
            rand[l, 0, r] = base - 0.05 + rng.normal(0, 0.03, n)
    out = cf.crossfit_statistics(real, rand, base, press, [12, 13, 14], [1.0, 2.0], [2.0], {**CFG, "bootstrap_resamples": 1000})
    for l in range(nL):
        lo, hi = out["net_share_ci"][l]
        assert lo < true_net < hi, (l, lo, hi)
        assert 0.15 < out["net_share"][l] < 0.35
    assert out["supported"] == [True] * 3 and out["bands"] == [[12, 14]]
    print("PASS: with realistic noise the case-bootstrap interval around the net share contains the planted truth in every layer")


if __name__ == "__main__":
    test_folds_are_balanced_deterministic_and_validated()
    test_fold_directions_never_see_their_own_cases()
    test_random_directions()
    test_contiguous_runs()
    test_randomization_statistics_by_hand()
    test_a_band_and_its_net_share()
    test_edge_cases()
    test_interval_covers_the_truth_with_noise()
    print("\nALL CROSS-FIT STATISTICS TESTS PASSED")
