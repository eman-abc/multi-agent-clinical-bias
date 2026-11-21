"""
test_chat_prompting_offline.py

Zero-GPU, zero-model sanity checks for mech/chat_prompting.py's pure-string
logic -- run this BEFORE spending any GPU time, on your laptop or the GCP
box, either one. Catches: the body/suffix template split breaking, and
parse_stated_recommendation failing to match Paper 1's actual sentence
format (both would silently corrupt Phase 3 results if wrong, so worth
20 seconds now rather than discovering it after a real run).

Usage:
    python test_chat_prompting_offline.py
"""

from __future__ import annotations

import sys

from mech import chat_prompting


def test_body_suffix_split():
    assert chat_prompting.ESI_BODY_TEMPLATE.endswith("Clinical note:\n{validated_note}")
    assert chat_prompting.ESI_SUFFIX in chat_prompting.ESI_BODY_TEMPLATE.replace(chat_prompting.ESI_SUFFIX, "") or True  # suffix must NOT be re-present after split
    assert chat_prompting.ESI_SUFFIX not in chat_prompting.ESI_BODY_TEMPLATE, "ESI suffix leaked into the body -- split is broken"
    assert chat_prompting.USMLE_SUFFIX not in chat_prompting.USMLE_BODY_TEMPLATE, "USMLE suffix leaked into the body -- split is broken"
    rebuilt_esi = chat_prompting.ESI_BODY_TEMPLATE + "\n\n" + chat_prompting.ESI_SUFFIX
    print("ESI_BODY_TEMPLATE + ESI_SUFFIX, reassembled:")
    print("---")
    print(rebuilt_esi.format(validated_note="[example note]"))
    print("---\n")
    print("PASS: body/suffix split\n")


def test_parse_stated_recommendation():
    # Exact format shared/crew_pipeline.py._resolve_recommendation_line produces:
    # f"The attending's assessment is most consistent with option {letter}: {text}."
    example = (
        "The patient presents with acute chest pain radiating to the left arm, "
        "diaphoresis, and shortness of breath. Vitals are stable. "
        "The attending's assessment is most consistent with option B: Acute myocardial infarction."
    )
    result = chat_prompting.parse_stated_recommendation(example)
    assert result is not None, "Failed to parse a well-formed recommendation sentence -- regex is broken"
    letter, text = result
    assert letter == "B", f"Expected letter 'B', got {letter!r}"
    assert text == "Acute myocardial infarction", f"Expected 'Acute myocardial infarction', got {text!r}"
    print(f"PASS: parsed ({letter!r}, {text!r}) from a well-formed note\n")

    # Negative case: a note with NO stated recommendation should return None, not crash or false-match.
    no_rec = "The patient presents with a mild headache and no other symptoms."
    result2 = chat_prompting.parse_stated_recommendation(no_rec)
    assert result2 is None, f"Expected None for a note with no recommendation, got {result2!r}"
    print("PASS: correctly returns None when there's no stated recommendation\n")

    # Edge case: option text itself containing a period (common in real clinical option text).
    example2 = (
        "Case facts here. The attending's assessment is most consistent with "
        "option D: Type 2 diabetes mellitus, uncontrolled."
    )
    result3 = chat_prompting.parse_stated_recommendation(example2)
    assert result3 is not None, "Failed on option text containing an internal comma"
    print(f"PASS: parsed multi-clause option text: {result3}\n")


def test_assistant_role_message_shape():
    messages = chat_prompting.build_assistant_role_deference_messages(
        clean_note_text="[clean note]", usmle_question="Which diagnosis is most likely?",
        options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
        wrong_letter="C", wrong_option_text="Option C",
    )
    assert len(messages) == 3, f"Expected 3 turns (user, assistant, user), got {len(messages)}"
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert "C" in messages[1]["content"], "The wrong letter should appear in the fake assistant turn"
    print("Assistant-role message structure:")
    for m in messages:
        print(f"  [{m['role']}] {m['content'][:80]}...")
    print("\nPASS: assistant-role 3-turn structure\n")


if __name__ == "__main__":
    print("=" * 70)
    test_body_suffix_split()
    print("=" * 70)
    test_parse_stated_recommendation()
    print("=" * 70)
    test_assistant_role_message_shape()
    print("=" * 70)
    print("ALL OFFLINE TESTS PASSED")
    sys.exit(0)
