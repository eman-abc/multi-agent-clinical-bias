"""
mech/model_registry.py

Model registry for Paper 2's mechanistic phases (Phase 1 bridge replication,
Phase 2 bias-direction sensor). Parallel to shared/config_and_utils.py's
AGENT_B_MODELS registry from Paper 1 -- deliberately NOT merged into that
file, so Paper 1's working pipeline is never touched by Paper 2's code.

Model selection and priority order: master handoff §7 (VRAM-verified
priority list). The two models here are priority #1 and #2 from that list:

  1. Llama-3.1-8B-Instruct -- chosen because TWO of this project's most
     load-bearing mechanistic precedents were built on it: Kumarappan &
     Mujoo 2026 (arXiv:2605.12991, cross-agent sycophancy localization)
     and Pandey 2026 (arXiv:2604.19117, the sycophancy-lying circuit,
     restoration-ratio table includes Llama-3.1-70B but the *family* and
     methodology transfer to the 8B size this project can afford).
  2. Mistral-7B-Instruct-v0.3 -- no HF gate (start here while Llama's
     gated access request is pending), and Pandey 2026 also reports
     restoration ratios on Mistral-7B specifically (11x instruction,
     22x sycophancy -- see Table in that paper), giving this project a
     second model already characterized by the paper Phase 3 rebuilds.

VRAM figures are this project's own estimates (fp16 weights + inference
overhead), verified against RUNPOD_SETUP.md's GPU choice -- not claimed
to be from any paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MechModelSpec:
    display_name: str
    hf_repo_id: str
    # The identifier to pass to transformer_lens.HookedTransformer.from_pretrained.
    # NOTE: verify this string against your installed TransformerLens version's
    # transformer_lens.loading.OFFICIAL_MODEL_NAMES before a long run -- point
    # releases have renamed entries before. hf_repo_id is what actually gets
    # downloaded from Hugging Face; tl_name is what TransformerLens uses to look
    # up its own architecture config, and the two are not always identical.
    tl_name: str
    requires_hf_gate: bool
    approx_vram_gb_fp16: float
    primary_papers: tuple[str, ...]
    role: str
    n_layers: int
    d_model: int


# n_layers / d_model are the models' own published architecture configs, not
# derived from any paper in this project's reading list -- included here
# because Phase 2's per-layer direction extraction needs to know how many
# layers to loop over before the model is ever loaded.
MECH_MODELS: tuple[MechModelSpec, ...] = (
    MechModelSpec(
        display_name="Llama-3.1-8B-Instruct",
        hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        tl_name="meta-llama/Llama-3.1-8B-Instruct",
        requires_hf_gate=True,
        approx_vram_gb_fp16=20.1,
        primary_papers=(
            "Kumarappan & Mujoo 2026 (arXiv:2605.12991) -- primary bridge target",
            "Pandey 2026 (arXiv:2604.19117) -- deference-circuit family precedent",
        ),
        role="Full verified mechanistic pipeline; deference direction (Phase 3)",
        n_layers=32,
        d_model=4096,
    ),
    MechModelSpec(
        display_name="Mistral-7B-Instruct-v0.1",
        hf_repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        tl_name="mistralai/Mistral-7B-Instruct-v0.1",
        requires_hf_gate=False,
        approx_vram_gb_fp16=18.1,
        primary_papers=(
            "Kumarappan & Mujoo 2026 (arXiv:2605.12991) -- full replication target",
            "Pandey 2026 (arXiv:2604.19117) -- restoration ratios reported for 'Mistral-7B'",
        ),
        role="No-gate model to start on immediately; layer-for-layer replication target",
        n_layers=32,
        d_model=4096,
    ),
)

MECH_MODELS_BY_NAME: dict[str, MechModelSpec] = {m.display_name: m for m in MECH_MODELS}

__all__ = ["MechModelSpec", "MECH_MODELS", "MECH_MODELS_BY_NAME"]
