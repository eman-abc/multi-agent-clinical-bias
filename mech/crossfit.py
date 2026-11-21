"""
mech/crossfit.py -- held-out validation of a steering direction. Pure numpy apart from fold_directions, which reuses
mech.ace_sensor's own direction function so a fold direction is the SAME recipe as the saved Phase 3 direction.

    assign_folds          balanced random folds
    fold_directions       per fold and layer: the direction and steering gap computed from the OTHER folds only
    random_directions     isotropic unit directions, deterministic per (seed, layer, draw)
    contiguous_runs       bands of consecutive supported layers
    crossfit_statistics   randomization p-values (max-T family-wise control), raw and net shares, case-bootstrap intervals

The fixed rules these implement are written in ANALYSIS_PLAN.md.
"""

from __future__ import annotations

import numpy as np


def assign_folds(n: int, k: int, seed: int) -> np.ndarray:
    """Fold index (0..k-1) per case; sizes differ by at most one; deterministic."""
    if not 2 <= k <= n:
        raise ValueError(f"need 2 <= folds <= n cases (got folds={k}, n={n})")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.empty(n, dtype=int)
    folds[perm] = np.arange(n) % k
    return folds


def fold_directions(acts_pos: np.ndarray, acts_neg: np.ndarray, folds: np.ndarray, layers) -> tuple[np.ndarray, np.ndarray]:
    """
    acts_pos / acts_neg: (n, n_layers_total, d) clean / pressured last-token activations.
    Returns directions (n_layers_sel, k, d) and gaps (n_layers_sel, k): for fold f, the direction and the clean-to-pressured
    distance along it, both computed from the cases NOT in fold f. A held-out case therefore never influences its own direction.
    """
    from mech import ace_sensor
    k = int(folds.max()) + 1
    d = acts_pos.shape[2]
    dirs = np.zeros((len(layers), k, d), dtype=np.float32)
    gaps = np.zeros((len(layers), k), dtype=np.float64)
    for li, layer in enumerate(layers):
        for f in range(k):
            tr = folds != f
            res = ace_sensor._direction_from_vectors(acts_pos[tr, layer, :], acts_neg[tr, layer, :], int(tr.sum()), "last_token")
            dirs[li, f] = res.direction
            gaps[li, f] = float(np.dot(res.mean_pos - res.mean_neg, res.direction))
    return dirs, gaps


def random_directions(layer: int, n_random: int, d: int, seed: int) -> np.ndarray:
    """(n_random, d) isotropic unit vectors; the same (seed, layer) always gives the same draws."""
    out = np.empty((n_random, d), dtype=np.float32)
    for r in range(n_random):
        v = np.random.default_rng([seed, 7919, layer, r]).normal(size=d)
        out[r] = v / np.linalg.norm(v)
    return out


def contiguous_runs(layers, flags, min_len: int):
    """([(first_layer, last_layer), ...] for runs of consecutive layer ids that are all flagged and at least min_len long, isolated layers)."""
    layers = list(layers)
    runs, cur = [], []
    for l, f in zip(layers, flags):
        if f and (not cur or l == cur[-1] + 1):
            cur.append(l)
        else:
            if cur:
                runs.append(cur)
            cur = [l] if f else []
    if cur:
        runs.append(cur)
    bands = [(r[0], r[-1]) for r in runs if len(r) >= min_len]
    inband = {l for r in runs if len(r) >= min_len for l in r}
    isolated = [l for r in runs if len(r) < min_len for l in r if l not in inband]
    return bands, isolated


def _pct(x, ci):
    lo, hi = np.percentile(x, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    return lo, hi


def crossfit_statistics(real, rand, base, press, layers, real_coefs, rand_coefs, cfg, boot_seed: int = 0) -> dict:
    """
    real : (nL, nC, n)   P(stated) after steering each held-out case with its fold's real direction
    rand : (nL, nRC, R, n) the same with R random directions
    base : (n,) unsteered pressured P(stated);   press : (n,) pressured minus clean P(stated)
    cfg  : the plan's "crossfit" section.
    """
    prim, sec = float(cfg["primary_strength"]), float(cfg["secondary_strength"])
    ci_p, ci_s = [float(c) for c in real_coefs].index(prim), [float(c) for c in real_coefs].index(sec)
    ri_p = [float(c) for c in rand_coefs].index(prim)
    alpha, min_band, B, level = cfg["alpha_fwe"], cfg["min_band_layers"], cfg["bootstrap_resamples"], cfg["ci_level"]
    nL, _, n = real.shape
    R = rand.shape[2]

    dR = real - base[None, None, :]                       # (nL, nC, n)
    dN = rand[:, ri_p] - base[None, None, :]              # (nL, R, n)
    T = -dR.mean(axis=-1)                                 # (nL, nC): positive = lowers P(stated)
    Tn = -dN.mean(axis=-1)                                # (nL, R)
    Tp = T[:, ci_p]
    p_unc = (1 + (Tn >= Tp[:, None]).sum(axis=1)) / (R + 1)
    M = Tn.max(axis=0)                                    # (R,) family maximum over layers, one value per random draw
    p_fwe = (1 + (M[None, :] >= Tp[:, None]).sum(axis=1)) / (R + 1)
    supported = (Tp > 0) & (p_fwe <= alpha) & (T[:, ci_s] > 0)
    bands, isolated = contiguous_runs(layers, supported, min_band)

    press_mean = float(press.mean())
    shares_ok = abs(press_mean) >= cfg["min_pressure_effect_for_shares"]
    out = {"layers": list(layers), "n_cases": int(n), "n_random": int(R), "min_attainable_p": 1.0 / (R + 1),
           "T_primary": Tp.tolist(), "T_secondary": T[:, ci_s].tolist(), "p_unc": p_unc.tolist(), "p_fwe": p_fwe.tolist(),
           "supported": [bool(x) for x in supported], "bands": [list(b) for b in bands], "isolated": isolated,
           "mean_pressure_effect": press_mean, "shares_reported": bool(shares_ok), "null_family_max": M.tolist(),
           "null_mean_by_layer": Tn.mean(axis=1).tolist(), "null_max_by_layer": Tn.max(axis=1).tolist()}
    if shares_ok:
        excess = dN.mean(axis=1) - dR[:, ci_p, :]         # (nL, n): what random does on average minus what the real direction does
        rawc = -dR[:, ci_p, :]                            # (nL, n)
        out["raw_share"] = (rawc.mean(axis=1) / press_mean).tolist()
        out["net_share"] = (excess.mean(axis=1) / press_mean).tolist()
        idx = np.random.default_rng(boot_seed).integers(0, n, size=(B, n))
        den = press[idx].mean(axis=1)                     # (B,)
        net_b = excess[:, idx].mean(axis=2) / den[None, :]   # (nL, B)
        raw_b = rawc[:, idx].mean(axis=2) / den[None, :]
        lo, hi = _pct(net_b.T, level)
        out["net_share_ci"] = [list(x) for x in zip(lo.tolist(), hi.tolist())]
        lo, hi = _pct(raw_b.T, level)
        out["raw_share_ci"] = [list(x) for x in zip(lo.tolist(), hi.tolist())]
        out["band_net_share"] = []
        for a, b in bands:
            sel = [i for i, l in enumerate(layers) if a <= l <= b]
            bl, bh = _pct(net_b[sel].mean(axis=0), level)
            out["band_net_share"].append({"band": [a, b], "mean": float(np.mean([out["net_share"][i] for i in sel])), "ci": [float(bl), float(bh)]})
    return out
