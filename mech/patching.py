"""
mech/patching.py

Activation patching -- Kumarappan & Mujoo's actual localization method
(arXiv:2605.12991), not an adaptation of it. Distinct from mech/ace_sensor.py's
mean-difference direction extraction: patching swaps CACHED activations
from a clean run into a corrupted run and measures how much of the clean
behavior comes back, rather than computing a direction from group means.

THE ALIGNMENT PROBLEM, AND HOW THIS MODULE HANDLES IT (a design choice,
not confirmed to be Kumarappan & Mujoo's own exact protocol -- flagged
explicitly, same convention as mech/ace_sensor.py):
Standard activation patching (ROME, causal tracing) needs clean and
corrupted prompts to be the same length so a patched token POSITION means
the same thing in both. Our clean (N, no recommendation) and corrupted
(REC_wrong, a stated wrong recommendation appended) notes are NOT the same
length -- the corrupted note has extra sentences. Patching at a fixed
token index would therefore patch semantically different positions in the
two prompts. This module sidesteps that by patching ONLY the LAST TOKEN
POSITION, which is well-defined and aligned across both variants because
both get wrapped in the SAME final prompt template (shared/inference_engine.py's
USMLE_DIAGNOSTIC_PROMPT_TEMPLATE, ending in "...is: (") regardless of the
note's length. This is consistent with mech/ace_sensor.py's own
last-token extraction convention, but it is NOT a claim that this is
Kumarappan & Mujoo's own per-token patching scheme -- check their repo
before treating exact restoration percentages as comparable to theirs.

RESTORATION FRACTION uses the identical formula Paper 1's own
shared/metrics.py.compute_reversion_fraction already uses for its
redaction-causality check (Experiment 3) -- same idea, same formula,
reused for consistency rather than inventing a new one:
    restoration = (patched_ev - corrupted_ev) / (clean_ev - corrupted_ev)
1.0 = patching fully restores clean behavior at that layer/component.
0.0 = patching does nothing. Values outside [0,1] are possible (patching
can overshoot) and are not clamped -- report them as-is.

TOKENIZATION, FIXED: same bug and same fix as mech/ace_sensor.py -- every
function here now takes already-tokenized `tokens` (built via
mech/chat_prompting.py.tokenize_chat_prompt, which goes through the real
chat template with the suffix correctly placed after the generation-prompt
boundary) instead of tokenizing a raw string itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

COMPONENTS = ("resid_post", "attn_out", "mlp_out")


def capture_last_token_activations(model, tokens: torch.Tensor, n_layers: int, components: Sequence[str] = ("resid_post",)) -> dict[str, np.ndarray]:
    """
    Runs the already-tokenized `tokens` once, returns {component: array of
    shape (n_layers, d_model)} -- the clean-run cache that will later be
    patched into a corrupted run. One forward pass regardless of how many
    components/layers requested (same efficiency reasoning as
    ace_sensor.extract_all_layers_last_token).
    """
    hook_suffixes = {f"hook_{c}" for c in components}
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: any(n.endswith(s) for s in hook_suffixes))
    out = {}
    for c in components:
        suffix = f"hook_{c}"
        out[c] = np.stack([cache[f"blocks.{layer}.{suffix}"][0, -1, :].detach().float().cpu().numpy() for layer in range(n_layers)])
    return out


def _patch_hook_factory(clean_vec: np.ndarray):
    """Replaces the last-token position's activation with `clean_vec`, leaves every other position untouched."""
    clean_t = None

    def hook_fn(value, hook):
        nonlocal clean_t
        if clean_t is None:
            clean_t = torch.tensor(clean_vec, dtype=value.dtype, device=value.device)
        value = value.clone()
        value[0, -1, :] = clean_t
        return value

    return hook_fn


def patch_and_predict(model, corrupted_tokens: torch.Tensor, clean_activation: np.ndarray, *, layer: int, component: str, class_token_ids: dict[str, int]) -> tuple[str, dict[str, float]]:
    """
    Runs already-tokenized `corrupted_tokens` with the clean activation
    patched in at (layer, component)'s last-token position. Reads the
    forced-choice logit exactly like ace_sensor.steered_class_prediction
    (same suffixed-protocol convention), so results are directly comparable.

    `class_token_ids`: {class_label: token_id}, from
    mech/chat_prompting.py.resolve_class_token_ids -- see
    ace_sensor.steered_class_prediction's docstring for why this replaced
    a per-call isolated-string re-tokenization that would raise on every
    real call against a SentencePiece-tokenizer model (confirmed on Mistral).
    """
    hook_name = f"blocks.{layer}.hook_{component}"
    hook_fn = _patch_hook_factory(clean_activation)
    tokens = corrupted_tokens
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    next_token_logits = logits[0, -1, :]

    class_logits = {cls: float(next_token_logits[tid].item()) for cls, tid in class_token_ids.items()}
    max_logit = max(class_logits.values())
    exp_vals = {c: float(np.exp(v - max_logit)) for c, v in class_logits.items()}
    total = sum(exp_vals.values())
    probs = {c: v / total for c, v in exp_vals.items()}
    return max(probs, key=probs.get), probs


@dataclass(frozen=True)
class LocalizationResult:
    layer: int
    component: str
    clean_ev: float | None
    corrupted_ev: float | None
    patched_ev: float | None
    restoration_fraction: float | None
    n_pairs: int


def run_localization_sweep(
    model,
    *,
    clean_tokens: dict[str, torch.Tensor],   # case_id -> clean (e.g. Neutral) tokens
    corrupted_tokens: dict[str, torch.Tensor],  # case_id -> corrupted (e.g. REC_wrong) tokens, SAME case_ids
    class_token_ids: dict[str, int],   # from mech/chat_prompting.py.resolve_class_token_ids -- see patch_and_predict's docstring for why this replaced a raw `classes` list
    ev_fn,  # metric_fn(class_probs, case_id) -> float | None. Per-case on purpose: P(correct) depends on each case's correct letter.
    n_layers: int,
    components: Sequence[str] = ("resid_post",),
    layers: Sequence[int] | None = None,
) -> list[LocalizationResult]:
    """
    Coarse-then-fine staging lives in the CALLER (run_phase3_localization_patching.py),
    not here -- this function just runs whatever (layers x components) grid
    it's given, once, over the matched case_ids common to both prompt dicts.
    For an initial coarse pass: call with components=("resid_post",), layers=None
    (every layer), small n. For a fine pass on specific layers: call again with
    components=("attn_out","mlp_out") and `layers` set to just those layers --
    `n_layers` still needs to be the model's real layer count (used for the
    one-time clean-activation cache), but only `layers` gets iterated and
    scored, so testing a handful of layers costs a handful of forward passes,
    not one per layer up to the highest one requested.
    """
    target_layers = list(layers) if layers is not None else list(range(n_layers))
    case_ids = sorted(set(clean_tokens) & set(corrupted_tokens))
    if not case_ids:
        raise ValueError("No case_ids in common between clean_tokens and corrupted_tokens.")

    # Per-case clean cache (one clean forward pass each, reused across every layer/component tested).
    clean_caches = {cid: capture_last_token_activations(model, clean_tokens[cid], n_layers, components=components) for cid in case_ids}

    # Per-case clean and corrupted (unpatched) predictions -- the two endpoints the restoration fraction is measured between.
    clean_ev_by_case: dict[str, float | None] = {}
    corrupted_ev_by_case: dict[str, float | None] = {}
    for cid in case_ids:
        _, clean_probs = patch_and_predict(model, clean_tokens[cid], clean_caches[cid][components[0]][0], layer=0, component=components[0], class_token_ids=class_token_ids)
        # NOTE: the call above patches the clean prompt's OWN layer-0 activation into itself --
        # a no-op by construction, used only as a cheap way to reuse patch_and_predict's logit-reading
        # code path for the clean prompt's genuinely unpatched prediction.
        clean_ev_by_case[cid] = ev_fn(clean_probs, cid)
        with torch.no_grad():
            corrupted_logits = model(corrupted_tokens[cid])
        next_token_logits = corrupted_logits[0, -1, :]
        corrupted_class_logits = {cls: float(next_token_logits[tid].item()) for cls, tid in class_token_ids.items()}
        max_logit = max(corrupted_class_logits.values())
        exp_vals = {c: float(np.exp(v - max_logit)) for c, v in corrupted_class_logits.items()}
        total = sum(exp_vals.values())
        corrupted_probs = {c: v / total for c, v in exp_vals.items()}
        corrupted_ev_by_case[cid] = ev_fn(corrupted_probs, cid)

    results: list[LocalizationResult] = []
    for component in components:
        for layer in target_layers:
            patched_evs = []
            for cid in case_ids:
                _, patched_probs = patch_and_predict(model, corrupted_tokens[cid], clean_caches[cid][component][layer], layer=layer, component=component, class_token_ids=class_token_ids)
                ev = ev_fn(patched_probs, cid)
                if ev is not None:
                    patched_evs.append(ev)

            clean_evs = [clean_ev_by_case[c] for c in case_ids if clean_ev_by_case[c] is not None]
            corrupted_evs = [corrupted_ev_by_case[c] for c in case_ids if corrupted_ev_by_case[c] is not None]
            clean_mean = sum(clean_evs) / len(clean_evs) if clean_evs else None
            corrupted_mean = sum(corrupted_evs) / len(corrupted_evs) if corrupted_evs else None
            patched_mean = sum(patched_evs) / len(patched_evs) if patched_evs else None

            restoration = None
            if clean_mean is not None and corrupted_mean is not None and patched_mean is not None and clean_mean != corrupted_mean:
                restoration = (patched_mean - corrupted_mean) / (clean_mean - corrupted_mean)

            results.append(LocalizationResult(
                layer=layer, component=component, clean_ev=clean_mean, corrupted_ev=corrupted_mean,
                patched_ev=patched_mean, restoration_fraction=restoration, n_pairs=len(case_ids),
            ))
    return results


__all__ = ["COMPONENTS", "capture_last_token_activations", "patch_and_predict", "LocalizationResult", "run_localization_sweep"]