"""
mech/reliability.py

How repeatable is a direction, and what is the largest cosine two NOISY directions could show?

Without this, a cosine between the bias direction and the deference direction cannot be read: 0.2 is "high" if each
direction is only 0.25 similar to ITSELF on a fresh half of the data, and "low" if each is 0.95 similar to itself.

  * split-half cosine : build the direction from two disjoint random halves of the pairs, take the cosine between them.
  * Spearman-Brown    : r_full = 2 r / (1 + r) turns a half-sample reliability into a full-sample one.
  * attenuation ceiling for cos(A, B) = sqrt(r_full_A * r_full_B): the cosine two directions would show if the
    underlying constructs were IDENTICAL and only sampling noise separated them. Dividing the observed cosine by it
    gives a "disattenuated" cosine. (Classical test theory; it treats each direction as a noisy measurement of one
    true direction, which is an assumption, stated wherever the number is reported.)
  * sign-flip null    : randomly swap positive/negative within a random half of the pairs and rebuild the direction.
    That direction contains the same noise but no systematic contrast; its cosine with another direction is what
    noise alone produces.

Everything reuses mech.ace_sensor._direction_from_vectors, so a direction recomputed here is the SAME vector the
extraction scripts saved (checked in run_reliability.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mech import ace_sensor

RELIABILITY_DIR = "paper2_results/reliability"


def reliability_path(kind: str, model: str, key: str, root: str | None = None) -> Path:
    """kind in {bias_esi, bias_usmle, def_usmle, def_esi}; key = attribute, or '<pressure>_<channel>' for deference.
    root: the results folder (default paper2_results)."""
    base = Path(root) / "reliability" if root else Path(RELIABILITY_DIR)
    return base / f"{kind}__{model}__{key}.npz"


def collect_activations(model, tokens_list, n_layers: int, component: str = "resid_post") -> np.ndarray:
    """(n_prompts, n_layers, d_model) float32 last-token activations, one forward pass per prompt."""
    return np.stack([ace_sensor.extract_all_layers_last_token(model, t, n_layers, component=component) for t in tokens_list]).astype(np.float32)


def directions_from_arrays(pos_all: np.ndarray, neg_all: np.ndarray) -> np.ndarray:
    """(n_layers, d) unit directions, computed exactly as ace_sensor does (pooled-std whitened mean difference)."""
    n = pos_all.shape[0]
    return np.stack([
        ace_sensor._direction_from_vectors(pos_all[:, l, :], neg_all[:, l, :], n, "last_token").direction
        for l in range(pos_all.shape[1])
    ])


def layer_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-layer cosine between two (n_layers, d) arrays."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.where(den < 1e-12, 1e-12, den)


def split_half_cosines(pos_all: np.ndarray, neg_all: np.ndarray, n_splits: int = 50, seed: int = 0) -> np.ndarray:
    """(n_splits, n_layers): cosine between directions built from two disjoint random halves of the pairs."""
    n = pos_all.shape[0]
    if n < 6:
        raise ValueError(f"need at least 6 pairs for a split-half estimate, got {n}")
    rng = np.random.default_rng(seed)
    h = n // 2
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[:h], perm[h:2 * h]
        out.append(layer_cosines(directions_from_arrays(pos_all[a], neg_all[a]), directions_from_arrays(pos_all[b], neg_all[b])))
    return np.array(out)


def spearman_brown(r_half):
    """Full-sample reliability from a half-sample one. Negative or zero half-reliability gives 0 (no usable signal)."""
    r = np.clip(np.asarray(r_half, dtype=float), 0.0, 1.0)
    return 2 * r / (1 + r)


def attenuation_ceiling(r_full_a, r_full_b):
    """Largest cosine two directions could show if the constructs were identical: sqrt(r_a * r_b)."""
    return np.sqrt(np.asarray(r_full_a, dtype=float) * np.asarray(r_full_b, dtype=float))


def signflip_null_directions(pos_all: np.ndarray, neg_all: np.ndarray, n_null: int = 20, seed: int = 0) -> np.ndarray:
    """(n_null, n_layers, d): directions rebuilt after randomly swapping pos/neg inside a random half of the pairs."""
    n = pos_all.shape[0]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_null):
        flip = (rng.random(n) < 0.5)[:, None, None]
        out.append(directions_from_arrays(np.where(flip, neg_all, pos_all), np.where(flip, pos_all, neg_all)))
    return np.stack(out)


def null_abs_cosines(null_dirs: np.ndarray, other: np.ndarray) -> np.ndarray:
    """(n_null, n_layers): |cosine| between each noise-only direction and a fixed (n_layers, d) direction."""
    return np.abs(np.stack([layer_cosines(nd, other) for nd in null_dirs]))


def save_reliability(path, *, split_half: np.ndarray, null_dirs: np.ndarray, n_pairs: int, matches_saved_mean_cos: float | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, split_half=split_half.astype(np.float32), null_dirs=null_dirs.astype(np.float16), n_pairs=np.array(n_pairs),
        matches_saved_mean_cos=np.array(np.nan if matches_saved_mean_cos is None else matches_saved_mean_cos),
    )


def load_reliability(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path)
    return {"split_half": z["split_half"], "null_dirs": z["null_dirs"].astype(np.float32), "n_pairs": int(z["n_pairs"]),
            "matches_saved_mean_cos": float(z["matches_saved_mean_cos"])}
