"""
mech/ace_sensor.py

Phase 2 -- the bias-direction sensor. Implements Karvonen & Marks' Affine
Concept Editing (ACE) recipe (arXiv:2506.10922) via TransformerLens, plus a
causal-validation step before the direction is trusted as a sensor.

WHAT IS A VERIFIED CITATION VS. A DESIGN CHOICE THIS PROJECT IS MAKING --
stated explicitly, not left implicit, per this project's own existing
convention (see shared/crew_pipeline.py's docstrings) of separating "this is
transcribed from a paper" from "this is a decision we made and why":

VERIFIED FROM THE PAPER (arXiv:2506.10922, read this session):
  - The ACE equation itself: d = (mu_pos - mu_neg) / sigma (per-dimension
    std, i.e. whitened mean-difference), then normalized; the edit
    v' = v - (<v,d> - b)*d, b = 0.5*(<mu_pos,d> + <mu_neg,d>), applied at
    every layer/token.
  - "Validate before trusting": the paper's own methodology is to steer
    the model with d and confirm behavior shifts in the expected direction
    BEFORE treating projection onto d as a measurement instrument.
  - Whitening is not cosmetic -- the paper's own ablations found unwhitened
    zero-ablation broke Gemma-3 (this project's research_plan_v2.md already
    carries this warning forward; it is not something I re-derived here).

DESIGN CHOICES THIS PROJECT IS MAKING, NOT FROM THE PAPER (the paper's own
public text, as retrieved this session, did not specify these at the level
of detail needed to write code -- check github.com/adamkarvonen/llm_bias
directly before treating any of the following as the paper's own choice):
  - Extraction point: last-token residual stream (resid_post) at each
    layer. This is a common convention in this style of work (see e.g. the
    causal-mediation tradition's post-attention last-token patching, Vig
    et al. 2020 onward) but is NOT confirmed as Karvonen & Marks' own exact
    choice from what I could retrieve.
  - The causal-validation step here uses ADDITIVE steering (v' = v +
    coefficient * d) rather than the paper's own neutralizing clamp --
    additive steering is the more direct test of "does pushing along d
    move behavior in a predictable direction," which is what "validate
    before trusting" is actually asking. The clamp equation is kept below
    (ace_edit_transform) for later use as an actual mitigation, once d is
    trusted.
  - No batching (one example at a time). Correct and simple; slower. Add
    batching once this runs correctly if throughput becomes the bottleneck.

TOKENIZATION, FIXED: this module used to call model.to_tokens(text) on a
flat string directly -- never going through the model's chat template.
Every function below now takes a `tokens` tensor instead of a text/prompt
string; building the RIGHT tokens (chat-templated, correct suffix
placement, correct channel framing) is mech/chat_prompting.py's job, not
this module's. See that module's docstring for why this matters -- it
isn't a style preference, it's a confound that existed in every Phase 2/3
run collected before this fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from mech.model_registry import MechModelSpec

logger = logging.getLogger(__name__)


def load_model(spec: MechModelSpec, *, device: str | None = None, dtype: "torch.dtype | None" = None):
    """
    Loads a HookedTransformer for the given spec. Requires HF_TOKEN to be
    set in the environment for gated models (see RUNPOD_SETUP.md §4).

    device/dtype auto-detect when not given: CUDA+float16 if a GPU is
    present (the RunPod path this was originally written for), else
    CPU+float32. This isn't just a convenience default -- float16 on CPU
    is poorly supported by PyTorch (several ops are either unimplemented
    or run far slower than float32), so hardcoding cuda/float16 the way
    this function originally did would fail outright, or silently
    misbehave, on a local CPU-only dry run (e.g. the gpt2 entry added to
    mech/model_registry.py specifically for that purpose). Pass either
    argument explicitly to override the auto-detection.
    """
    from transformer_lens import HookedTransformer  # local import: this module is imported by scripts that may not always need a GPU present (e.g. unit tests on direction math alone)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("[ace_sensor] Loading %s (tl_name=%r) on %s (dtype=%s)...", spec.display_name, spec.tl_name, device, dtype)
    model = HookedTransformer.from_pretrained(spec.tl_name, device=device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    direction: np.ndarray       # unit-normalized, shape (d_model,)
    mean_pos: np.ndarray        # raw activation mean, positive/privileged group
    mean_neg: np.ndarray        # raw activation mean, negative/unprivileged group
    std: np.ndarray             # pooled per-dimension std across both groups
    bias_scalar: float          # b in the ACE equation
    n_pairs: int
    extraction_point: str


def extract_all_layers_last_token(model, tokens: torch.Tensor, n_layers: int, component: str = "resid_post") -> np.ndarray:
    """
    Single forward pass per example, caching the given component at EVERY
    layer and reading the last-token position out of each. Returns shape
    (n_layers, d_model).

    `tokens`: already-tokenized input (shape (1, seq_len)), built via
    mech/chat_prompting.py's tokenize_chat_prompt -- this function no
    longer tokenizes raw text itself (see module docstring: that was the
    bug).

    `component`: "resid_post" (default -- the whole layer's combined
    output, what Phase 2's bias sensor uses), "attn_out" (that layer's
    attention output alone), or "mlp_out" (that layer's MLP output alone).
    The attn/mlp split exists specifically to test Kumarappan & Mujoo's
    own claim that their sycophancy effect is attention-driven with
    negligible MLP contribution -- resid_post can't distinguish the two,
    since it's their sum. Added for Phase 3's deference sensor; Phase 2's
    bias-sensor calls are unaffected (component defaults to resid_post,
    identical behavior to before).
    """
    hook_suffix = f"hook_{component}"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith(hook_suffix))
    out = np.stack([
        cache[f"blocks.{layer}.{hook_suffix}"][0, -1, :].detach().float().cpu().numpy()
        for layer in range(n_layers)
    ])
    return out  # (n_layers, d_model)


def extract_directions_all_layers(
    model,
    *,
    positive_tokens: Sequence[torch.Tensor],
    negative_tokens: Sequence[torch.Tensor],
    n_layers: int,
    extraction_point: str = "last_token",
    component: str = "resid_post",
) -> list["ExtractionResult"]:
    """
    ACE-style direction (whitened mean-difference), computed for EVERY
    layer from ONE pass per example. Used for BOTH Phase 2's bias
    direction (demographic E+/E- contrast) and Phase 3's deference
    direction (pressured/unpressured contrast) -- same recipe, different
    input tokens; a different `component` is only relevant if you want
    an attn-only or mlp-only direction rather than the combined
    resid_post one Phase 2 uses. positive_tokens/negative_tokens are
    already-tokenized (see mech/chat_prompting.py), matched-pair examples
    differing in whatever the contrast is. Returns one ExtractionResult
    per layer, index-aligned (result[i] is layer i).
    """
    if extraction_point != "last_token":
        raise NotImplementedError("Only last_token extraction is implemented; see module docstring for why this is a design choice, not a spec requirement.")

    # shape (n_examples, n_layers, d_model) for each group
    pos_all = np.stack([extract_all_layers_last_token(model, t, n_layers, component=component) for t in positive_tokens])
    neg_all = np.stack([extract_all_layers_last_token(model, t, n_layers, component=component) for t in negative_tokens])

    results = []
    for layer in range(n_layers):
        pos_vecs = pos_all[:, layer, :]
        neg_vecs = neg_all[:, layer, :]
        results.append(_direction_from_vectors(pos_vecs, neg_vecs, len(positive_tokens), extraction_point))
    return results


def _direction_from_vectors(pos_vecs: np.ndarray, neg_vecs: np.ndarray, n_pairs: int, extraction_point: str) -> "ExtractionResult":
    mean_pos = pos_vecs.mean(axis=0)
    mean_neg = neg_vecs.mean(axis=0)

    # Pooled per-dimension std across BOTH groups -- the whitening step
    # Karvonen & Marks' own ablations found load-bearing (see module
    # docstring). Pooling both groups (not just one) is the standard
    # whitened-mean-difference convention this project's docs already
    # describe as "whiten by per-dimension std."
    pooled = np.concatenate([pos_vecs, neg_vecs], axis=0)
    std = pooled.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)  # guard against a dead/constant dimension causing a divide-by-zero

    d_raw = (mean_pos - mean_neg) / std
    norm = np.linalg.norm(d_raw)
    if norm < 1e-8:
        raise ValueError("Degenerate direction (near-zero norm) -- positive/negative texts may not differ meaningfully at this layer/position.")
    direction = d_raw / norm

    bias_scalar = 0.5 * (float(np.dot(mean_pos, direction)) + float(np.dot(mean_neg, direction)))

    return ExtractionResult(
        direction=direction, mean_pos=mean_pos, mean_neg=mean_neg, std=std,
        bias_scalar=bias_scalar, n_pairs=n_pairs, extraction_point=extraction_point,
    )


# ---------------------------------------------------------------------------
# Edits: the paper's own clamp, and the additive-steering validation edit
# ---------------------------------------------------------------------------


def ace_edit_transform(v: torch.Tensor, direction: torch.Tensor, bias_scalar: float, coefficient: float = 1.0) -> torch.Tensor:
    """
    Karvonen & Marks' own equation, verbatim: v' = v - (<v,d> - b)*d.
    `coefficient` is NOT part of the paper's equation -- it's added here
    only so a partial-strength clamp can be tested (coefficient=1.0
    reproduces the paper's equation exactly; anything else is this
    project's own extrapolation for a dose-response sweep).
    """
    proj = torch.einsum("...d,d->...", v, direction) - bias_scalar
    return v - coefficient * proj.unsqueeze(-1) * direction


def additive_steering_transform(v: torch.Tensor, direction: torch.Tensor, coefficient: float) -> torch.Tensor:
    """
    v' = v + coefficient * d. This is the causal-validation edit used in
    validate_steering_causally below -- NOT from the paper's own equation
    (see module docstring). Positive coefficient pushes toward the
    "positive/privileged" group's mean; negative pushes toward
    "negative/unprivileged." A hook using this should show ESI/USMLE
    decisions shift accordingly if, and only if, the direction is causal.
    """
    return v + coefficient * direction


def make_hook(direction: np.ndarray, transform: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor], coefficient: float, bias_scalar: float | None = None):
    """
    Builds a TransformerLens hook function for a given layer's resid_post.
    `transform` is either ace_edit_transform (needs bias_scalar) or
    additive_steering_transform (ignores it) -- both take (v, direction,
    coefficient) as their last two positional args for a uniform call
    signature here.
    """
    direction_t = None  # lazily moved to the right device/dtype on first call

    def hook_fn(value, hook):
        nonlocal direction_t
        if direction_t is None:
            direction_t = torch.tensor(direction, dtype=value.dtype, device=value.device)
        if transform is ace_edit_transform:
            return ace_edit_transform(value, direction_t, bias_scalar, coefficient)
        return additive_steering_transform(value, direction_t, coefficient)

    return hook_fn


# ---------------------------------------------------------------------------
# Causal validation ("steer before trusting" -- Karvonen & Marks' own methodology)
# ---------------------------------------------------------------------------


def generate_with_steering(model, tokens: torch.Tensor, *, layer: int, direction: np.ndarray, coefficient: float, max_new_tokens: int = 5) -> str:
    """
    Runs the model on already-tokenized `tokens` (see
    mech/chat_prompting.py) with additive steering applied at resid_post
    of `layer`, for every token position (matching Karvonen & Marks'
    "applied at every layer/token" convention for the edit itself, even
    though the edit used HERE is the additive-steering validation variant,
    not their clamp -- see module docstring). Returns the generated
    continuation text only. Use this for eyeballing/debugging; for the
    actual ESI/USMLE validation metric, use steered_class_prediction below
    instead -- it reads the forced-choice logit directly rather than
    parsing free text.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = make_hook(direction, additive_steering_transform, coefficient)
    with torch.no_grad():
        out = model.generate(
            tokens, max_new_tokens=max_new_tokens, do_sample=False,
            fwd_hooks=[(hook_name, hook_fn)],
        )
    generated = model.to_string(out[0, tokens.shape[1]:])
    return generated.strip()


def steered_class_prediction(
    model, tokens: torch.Tensor, class_token_ids: dict[str, int], *, layer: int, direction: np.ndarray, coefficient: float, component: str = "resid_post",
) -> tuple[str, dict[str, float]]:
    """
    The validation metric run_phase2_bias_sensor.py and
    run_phase3_deference_sensor.py actually use. `tokens` must already be
    chat-templated with the forced-choice suffix appended AFTER the
    generation-prompt boundary (mech/chat_prompting.py.tokenize_chat_prompt
    does this correctly) -- the same "suffixed-protocol" convention
    Kumarappan & Mujoo use.

    `class_token_ids`: {class_label: token_id}, from
    mech/chat_prompting.py.resolve_class_token_ids -- NOT a raw list of
    class label strings re-tokenized per call. THE FIX for a real bug,
    confirmed on Mistral-7B-Instruct-v0.1: re-tokenizing a bare candidate
    like "1" in isolation gives 2 tokens on SentencePiece-style tokenizers
    (an implicit leading boundary marker + the digit), not the 1 token the
    model actually emits mid-generation -- this function used to do that
    isolated re-tokenization every call and would have raised on every
    single real invocation against Llama/Mistral (invisible during the
    gpt2 dry run, whose BPE tokenizer doesn't have this artifact). Resolving
    token IDs once, in the real context, and passing them in here instead
    fixes it -- see resolve_class_token_ids's own docstring for the
    verified before/after tokenization.
    """
    hook_name = f"blocks.{layer}.hook_{component}"
    hook_fn = make_hook(direction, additive_steering_transform, coefficient)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    next_token_logits = logits[0, -1, :]  # (vocab,)

    class_logits = {cls: float(next_token_logits[tid].item()) for cls, tid in class_token_ids.items()}
    max_logit = max(class_logits.values())
    exp_vals = {c: float(np.exp(v - max_logit)) for c, v in class_logits.items()}
    total = sum(exp_vals.values())
    probs = {c: v / total for c, v in exp_vals.items()}
    top_class = max(probs, key=probs.get)
    return top_class, probs


def plain_class_probs(model, tokens: torch.Tensor, class_token_ids: dict[str, int]) -> dict[str, float]:
    """
    One UNEDITED forward pass -> probabilities over the forced-choice class tokens (renormalised over just
    those classes, same convention as steered_class_prediction). Used for the know-the-answer filter, the
    clean/pressured behavioral baseline, and anywhere a steering coefficient of 0 is meant but a hook is not.
    """
    with torch.no_grad():
        logits = model(tokens)
    next_token_logits = logits[0, -1, :]
    class_logits = {c: float(next_token_logits[t].item()) for c, t in class_token_ids.items()}
    mx = max(class_logits.values())
    ex = {c: float(np.exp(v - mx)) for c, v in class_logits.items()}
    tot = sum(ex.values())
    return {c: v / tot for c, v in ex.items()}


__all__ = [
    "load_model", "plain_class_probs", "ExtractionResult", "extract_all_layers_last_token",
    "extract_directions_all_layers", "ace_edit_transform",
    "additive_steering_transform", "make_hook", "generate_with_steering",
    "steered_class_prediction",
]