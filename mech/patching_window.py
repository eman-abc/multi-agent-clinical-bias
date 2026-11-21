"""
mech/patching_window.py

Patch the clean run's last-token attention and/or MLP outputs into the pressured run at EVERY layer of a window at once.

Why a window and not one layer at a time (mech/patching.py): patching a single layer's attention output is largely
undone by the attention of the layers after it, which still read the pressured prompt's keys and values. Asking "does
the effect act through attention or through MLPs in this stretch of the network" needs the whole stretch patched
together. A single-layer sweep is still reported, as a descriptive complement.

EXACT CHECK (used in tests and in every GPU run): in these rotary-position models the residual stream at the last token is
    embedding(last token) + sum over layers of (attention output + MLP output).
The last token is the same token in both prompts, so patching attention AND MLP at EVERY layer reproduces the clean
residual, hence the clean answer distribution, exactly. So does patching attention ALONE at every layer: attention is the
only route by which earlier tokens reach the last position (MLPs and norms see that position alone), so once every
layer's attention output is the clean one, the last position's whole computation is the clean computation. If either
identity fails, the run is not valid.

CONSEQUENCE FOR READING A SPLIT: because attention is the only channel into the last token, "MLP restores less than
attention" is partly structural. MLPs can only transform what attention delivered. A window where attention-only
restores a lot and MLP-only restores little says the information enters through attention in that stretch; it does not by
itself show that MLPs are irrelevant to how that information is used. The plan states this limit.
"""

from __future__ import annotations

import numpy as np
import torch

from mech import patching


def patch_window_and_predict(model, corrupted_tokens: torch.Tensor, clean_acts: dict[str, np.ndarray], *, layers, components, class_token_ids: dict[str, int]) -> dict[str, float]:
    """
    clean_acts: {component: (n_layers, d_model)} from patching.capture_last_token_activations on the clean prompt.
    Returns the class probabilities (renormalised over class_token_ids) for the corrupted prompt with the clean
    last-token output of every (layer, component) in the window patched in.
    """
    hooks = [(f"blocks.{layer}.hook_{comp}", patching._patch_hook_factory(clean_acts[comp][layer])) for layer in layers for comp in components]
    with torch.no_grad():
        logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)
    nt = logits[0, -1, :]
    cl = {c: float(nt[t].item()) for c, t in class_token_ids.items()}
    mx = max(cl.values())
    ex = {c: float(np.exp(v - mx)) for c, v in cl.items()}
    tot = sum(ex.values())
    return {c: v / tot for c, v in ex.items()}


TYPES = {"attn_out": ("attn_out",), "mlp_out": ("mlp_out",), "both": ("attn_out", "mlp_out")}
