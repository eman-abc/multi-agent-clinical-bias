"""
test_chat_prompting_smoke.py

Needs a real model + GPU (or CPU, just slower). Run this BEFORE the real
Phase 2/3 scripts, on whichever model you're about to use for real --
checks the one thing test_chat_prompting_offline.py can't: does
model.tokenizer.apply_chat_template actually render the way we're
assuming (real role tags, no double-BOS, suffix lands where expected).

What to actually look at in the output: the rendered string should show
real special tokens for the model's chat format -- for Llama-3.1 that's
<|begin_of_text|><|start_header_id|>user<|end_header_id|> ... <|eot_id|>
<|start_header_id|>assistant<|end_header_id|> -- and the token count
should NOT look doubled at the start (a double-BOS bug would show two
<|begin_of_text|> equivalents or an obviously wrong leading token).

Usage:
    python test_chat_prompting_smoke.py --model Mistral-7B-Instruct-v0.3
    python test_chat_prompting_smoke.py --model Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys

from mech import ace_sensor, chat_prompting
from mech.model_registry import MECH_MODELS_BY_NAME


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()))
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    args = parser.parse_args()

    spec = MECH_MODELS_BY_NAME[args.model]
    print(f"Loading {spec.display_name}...")
    model = ace_sensor.load_model(spec, device=args.device)

    if model.tokenizer is None or getattr(model.tokenizer, "chat_template", None) is None:
        print(f"ERROR: {spec.display_name} has no chat template (model.tokenizer.chat_template is None) -- "
              f"this is expected for gpt2 but should NOT happen for Llama/Mistral Instruct models. "
              f"If you see this on a real target model, something is wrong with how it loaded.", file=sys.stderr)
        return 1

    # --- 1. User-role rendering ---
    body = chat_prompting.ESI_BODY_TEMPLATE.format(validated_note="[EXAMPLE CLINICAL NOTE GOES HERE]")
    messages = chat_prompting.build_user_role_messages(body)
    rendered = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\n" + "=" * 70)
    print("USER-ROLE rendered template (look for real chat special tokens):")
    print("=" * 70)
    print(rendered + chat_prompting.ESI_SUFFIX)

    tokens = chat_prompting.tokenize_chat_prompt(model, messages, chat_prompting.ESI_SUFFIX)
    print(f"\nToken count: {tokens.shape[1]}")
    print(f"First 10 token strings: {[model.to_string(tokens[0, i:i+1]) for i in range(min(10, tokens.shape[1]))]}")
    print("^ CHECK: does the first token look like a single, correct BOS-equivalent, not doubled?")

    # --- 2. Assistant-role rendering ---
    messages_a = chat_prompting.build_assistant_role_deference_messages(
        clean_note_text="[EXAMPLE CLINICAL NOTE]", usmle_question="Which diagnosis is most likely?",
        options={"A": "Option A text", "B": "Option B text", "C": "Option C text", "D": "Option D text"},
        wrong_letter="B", wrong_option_text="Option B text",
    )
    rendered_a = model.tokenizer.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
    print("\n" + "=" * 70)
    print("ASSISTANT-ROLE rendered template (check the fake assistant turn renders as a real prior turn):")
    print("=" * 70)
    print(rendered_a + chat_prompting.USMLE_SUFFIX)

    tokens_a = chat_prompting.tokenize_chat_prompt(model, messages_a, chat_prompting.USMLE_SUFFIX)
    print(f"\nToken count: {tokens_a.shape[1]}")

    # --- 3. Does the model actually answer a single token the way we need? ---
    from shared.inference_engine import ESI_CLASSES
    print("\n" + "=" * 70)
    print("Single-token class check (ESI_CLASSES '1'-'5') -- confirms steered_class_prediction's logit read will work:")
    print("=" * 70)
    for cls in ESI_CLASSES:
        for candidate in (cls, f" {cls}"):
            tids = model.to_tokens(candidate, prepend_bos=False)[0]
            print(f"  {candidate!r}: {len(tids)} token(s) {'<-- OK, single token' if len(tids) == 1 else '<-- PROBLEM: not a single token'}")

    print("\nSMOKE TEST COMPLETE -- inspect the output above before trusting any real Phase 2/3 run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
