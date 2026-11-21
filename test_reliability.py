"""test_reliability.py -- CPU, numpy only.   python test_reliability.py"""
import os
import tempfile

import numpy as np

from mech import ace_sensor, reliability as rel

L, D = 3, 64


def make(n, signal, seed=0):
    """pos/neg activations (n, L, D): shared noise + a planted +-signal*u along a fixed unit direction u."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    base = rng.normal(size=(n, L, D))
    pos = base + signal * u + rng.normal(scale=0.5, size=(n, L, D))
    neg = base - signal * u + rng.normal(scale=0.5, size=(n, L, D))
    return pos.astype(np.float32), neg.astype(np.float32), u


def test_directions_match_ace_exactly():
    pos, neg, _ = make(30, 1.0)
    mine = rel.directions_from_arrays(pos, neg)
    for l in range(L):
        ref = ace_sensor._direction_from_vectors(pos[:, l, :], neg[:, l, :], 30, "last_token").direction
        assert np.allclose(mine[l], ref), l
        assert abs(np.linalg.norm(mine[l]) - 1) < 1e-5
    print("PASS: directions recomputed here equal ace_sensor's, layer by layer, and are unit length")


def test_split_half_separates_signal_from_noise():
    pos, neg, _ = make(120, 1.0)
    strong = rel.split_half_cosines(pos, neg, n_splits=30, seed=1)
    assert strong.shape == (30, L) and strong.mean() > 0.8, strong.mean()
    pos0, neg0, _ = make(120, 0.0)
    pure_noise = rel.split_half_cosines(pos0, neg0, n_splits=60, seed=1)
    assert abs(pure_noise.mean()) < 0.08 and np.abs(pure_noise).mean() < 0.2, (pure_noise.mean(), np.abs(pure_noise).mean())
    try:
        rel.split_half_cosines(pos[:4], neg[:4])
        raise AssertionError("expected a refusal for fewer than 6 pairs")
    except ValueError:
        pass
    print("PASS: a planted contrast gives high split-half cosine, pure noise gives ~0, and tiny samples are refused")


def test_spearman_brown_and_ceiling_by_hand():
    assert abs(float(rel.spearman_brown(0.5)) - 2 * 0.5 / 1.5) < 1e-9          # 0.6667
    assert abs(float(rel.spearman_brown(0.8)) - 1.6 / 1.8) < 1e-9              # 0.8889
    assert float(rel.spearman_brown(-0.3)) == 0.0 and float(rel.spearman_brown(0.0)) == 0.0
    assert abs(float(rel.attenuation_ceiling(0.64, 0.25)) - 0.4) < 1e-9        # sqrt(0.16)
    assert abs(float(rel.attenuation_ceiling(1.0, 1.0)) - 1.0) < 1e-9
    print("PASS: Spearman-Brown and the attenuation ceiling match hand-computed values")


def test_signflip_null_is_much_smaller_than_real():
    pos, neg, u = make(120, 1.0, seed=3)
    real = rel.directions_from_arrays(pos, neg)
    null = rel.signflip_null_directions(pos, neg, n_null=10, seed=2)
    assert null.shape == (10, L, D)
    real_cos = np.abs(rel.layer_cosines(real, np.stack([u] * L)))
    null_cos = rel.null_abs_cosines(null, np.stack([u] * L))
    assert real_cos.min() > 0.9, real_cos
    assert null_cos.shape == (10, L) and null_cos.mean() < 0.3 and null_cos.max() < 0.6, (null_cos.mean(), null_cos.max())
    print("PASS: sign-flip null directions carry the noise but not the contrast (real cos ~1, null cos small)")


def test_save_and_load_roundtrip():
    pos, neg, _ = make(20, 1.0)
    sh = rel.split_half_cosines(pos, neg, n_splits=5)
    null = rel.signflip_null_directions(pos, neg, n_null=3)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x", "r.npz")
        rel.save_reliability(p, split_half=sh, null_dirs=null, n_pairs=20, matches_saved_mean_cos=0.9999)
        back = rel.load_reliability(p)
        assert back["n_pairs"] == 20 and back["split_half"].shape == sh.shape and back["null_dirs"].shape == null.shape
        assert abs(back["matches_saved_mean_cos"] - 0.9999) < 1e-6 and np.allclose(back["split_half"], sh, atol=1e-6)
        assert rel.load_reliability(os.path.join(tmp, "missing.npz")) is None
    assert str(rel.reliability_path("bias_esi", "M", "ses")).endswith("bias_esi__M__ses.npz")
    print("PASS: reliability files round-trip; a missing file returns None")


if __name__ == "__main__":
    test_directions_match_ace_exactly()
    test_split_half_separates_signal_from_noise()
    test_spearman_brown_and_ceiling_by_hand()
    test_signflip_null_is_much_smaller_than_real()
    test_save_and_load_roundtrip()
    print("\nALL RELIABILITY TESTS PASSED")
