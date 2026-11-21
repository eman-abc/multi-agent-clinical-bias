"""
mech/interaction_prompts.py -- builds the prompts of the interaction run: a note that carries a demographic footer line AND a false consensus.

It is deliberately a thin layer over the SAME primitives the deference prompts use (mech/chat_prompting.py and mech/deference_pairs.py), so that a prompt with
no footer is TOKEN-IDENTICAL to the existing clean and pressured prompts (a test checks this against run_phase3_deference_sensor.build_pressured_tokens).

Variants (the pressure is always the 'consensus' pressure of the deference runs, naming the case's fixed wrong option):

    clean       one user turn: the note, with the footer line if there is one; no pressure.
    user_ff     one user turn; the footer comes FIRST and the consensus sentence LAST, on its own line, just before the question, which is where the
                consensus sentence always sits in the deference prompts. PRIMARY user-channel variant.
    user_fl     one user turn; the consensus sentence is appended to the note exactly as in the deference prompts (same line, one space), and the footer
                follows on its own line. A POSITION CONTROL: does the footer's position relative to the consensus matter?
    assistant   three turns: the note (with the footer) and question in a user turn; the wrong answer asserted in a fake prior assistant turn; a closing user
                turn carrying the consensus claim.

For a prompt WITHOUT a footer, user_ff and user_fl are the same prompt, and it equals the deference pressured prompt exactly (the consensus sentence joined to the
note by one space). A prompt WITH a footer joins the consensus sentence by a newline in user_ff, so base prompts and footer prompts differ in that join as well; the
analysis therefore only compares footer prompts with footer prompts (side a against side b), never footer prompts with base prompts.
"""
from __future__ import annotations

from mech import chat_prompting, deference_pairs as dp

VARIANTS = ("clean", "user_ff", "user_fl", "assistant")


def note_with_footer(base: str, footer: str | None) -> str:
    """The note with its footer line. With no footer the note is returned UNCHANGED (not stripped), so a base prompt equals the deference clean prompt."""
    if not footer:
        return base
    return base.rstrip() + "\n" + footer


def _user_messages(case_record, note: str) -> list[dict]:
    opts = dp.options_dict(case_record)
    body = chat_prompting.USMLE_BODY_TEMPLATE.format(
        validated_note=note, usmle_question=case_record.neutralised_question,
        option_a=opts.get("A", ""), option_b=opts.get("B", ""), option_c=opts.get("C", ""), option_d=opts.get("D", ""))
    return chat_prompting.build_user_role_messages(body)


def consensus_sentence(wrong_letter: str, wrong_text: str) -> str:
    return dp.CONSENSUS_SENTENCE.format(letter=wrong_letter, text=wrong_text.strip().rstrip(".").strip())


def build_messages(variant: str, case_record, base: str, footer: str | None, wrong_letter: str, wrong_text: str) -> list[dict]:
    """The chat messages of one prompt variant. The caller tokenizes them with chat_prompting.tokenize_chat_prompt(model, messages, USMLE_SUFFIX)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if variant == "clean":
        return _user_messages(case_record, note_with_footer(base, footer))
    if variant == "user_ff":
        if footer:
            return _user_messages(case_record, base.rstrip() + "\n" + footer + "\n" + consensus_sentence(wrong_letter, wrong_text))
        return _user_messages(case_record, dp.append_consensus_recommendation(base, wrong_letter, wrong_text))
    if variant == "user_fl":
        note = dp.append_consensus_recommendation(base, wrong_letter, wrong_text)
        return _user_messages(case_record, note + ("\n" + footer if footer else ""))
    closing = dp.CONSENSUS_CLOSING.format(letter=wrong_letter)
    return chat_prompting.build_assistant_role_deference_messages(
        note_with_footer(base, footer), case_record.neutralised_question, dp.options_dict(case_record), wrong_letter, wrong_text.strip().rstrip(".").strip(), closing_text=closing)


def wrong_option_for(case_record):
    """(correct_letter, wrong_letter, wrong_text) with the SAME deterministic rule the deference pairs use (sha256 of the case id over the incorrect options), or None."""
    corr = dp.correct_letter(case_record)
    pick = dp.pick_wrong_option(str(case_record.question_id), dp.options_dict(case_record), corr)
    if pick is None:
        return None
    return corr, pick[0], pick[1].strip().rstrip(".").strip()
