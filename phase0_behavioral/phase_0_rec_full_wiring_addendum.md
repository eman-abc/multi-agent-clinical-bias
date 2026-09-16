# Phase 0 — REC Conditions: Bug Fix and Full Wiring Record

**Status: `SignalType` bug FIXED. REC_correct/REC_wrong WIRED IN and live-
tested (mocked). REC_UP/REC_DOWN unaffected, confirmed still working.
Combined-context Agent B now deliberately scoped to REC_wrong only.**

This supersedes the "PENDING PI APPROVAL" gate on REC_correct/REC_wrong
that `phase_0_prompt_addendum_REC_DRAFT.md` and
`phase_0_combined_prompt_and_rec_addendum.md` describe — those conditions
are now live, following the same approval precedent already set by
REC_UP/REC_DOWN being wired in directly in the same session.

---

## 1. The bug, precisely

`REC_UP`/`REC_DOWN` were added to **both** `ConditionLabel` and
`SignalType`. Only the `ConditionLabel` addition was ever needed —
`build_recommendation_conditions()` is the only place that constructs
REC_UP/REC_DOWN specs, and it identifies them by `condition_label`, not
`signal_type`.

The `SignalType` addition was the bug. `build_condition_matrix()` — the
function that builds the *demographic* (Race/Sex/SES) condition matrix —
iterates directly over the enum:

```python
for signal_type in SignalType:
    for polarity in (...):
        ...
```

With `REC_UP`/`REC_DOWN` sitting in that enum, this loop silently
generated two extra, unintended conditions per attribute: `E+/REC_UP` and
`E+/REC_DOWN`. Nothing in `crew_pipeline.py`'s `_write_note()` dispatch had
a branch for `signal_type is SignalType.REC_UP` (or `REC_DOWN`), so they
fell through to the `elif carrier_line is not None:` branch — **the exact
same branch PRX uses.** This is why, in the live run that surfaced this,
`E+/REC_UP`'s generated note read almost identically to `E+/PRX`'s, and
both scored the same ESI answer: they were the same condition, generated
twice under two different labels, at the cost of real live API calls for
zero additional signal.

## 2. The fix

`REC_UP`/`REC_DOWN` removed from `SignalType` entirely — they never
belonged there conceptually (`SignalType` describes *how a demographic
carrier is expressed*; REC conditions aren't a demographic carrier at
all). `ConditionLabel`'s own `REC_UP`/`REC_DOWN` are untouched and remain
the only place these two conditions are identified. Both `rec_up.
signal_type` and `rec_down.signal_type` in `build_recommendation_
conditions()` changed from `SignalType.REC_UP`/`REC_DOWN` (now invalid,
since those enum members no longer exist) to `None` — matching how N, P,
REC_correct, and REC_wrong already represent "not a signal-type
condition."

**Verified, not just asserted:**
- `list(SignalType)` now has exactly 3 members.
- `build_condition_matrix()` produces exactly 8 conditions per case ×
  attribute (was silently producing extras before), confirmed against a
  real case.
- Confirmed `('E+', 'REC_UP')` and `('E+', 'REC_DOWN')` no longer appear
  anywhere in a built condition matrix.
- Full Module 6 experiment regression suite (Experiments 1, 2, 3) still
  passes with zero blocked conditions after the fix.

`run_live_smoke_test.py` also had its `SIGNAL_TYPES_TO_TEST` list carrying
`SignalType.REC_UP`/`REC_DOWN` alongside genuine `SignalType` members — an
implicit reliance on the same now-fixed overlap. Split cleanly into
`SIGNAL_TYPES_TO_TEST` (`SignalType`, demographic only) and
`REC_LABELS_TO_TEST` (`ConditionLabel`, REC conditions only), so nothing
depends on the two vocabularies coincidentally overlapping again.

## 3. REC_correct / REC_wrong — now wired in

Transcribed from `phase_0_prompt_addendum_REC_DRAFT.md`, but adapted to
the **same prompt pattern already proven live** for REC_UP/REC_DOWN (an
"OVERRIDE INSTRUCTION" preface + an exact-required-final-line rule) rather
than the draft's original two-separate-check design — one pattern with a
live track record, reused for all four REC conditions, rather than two
different untested designs running side by side.

**The option-letter-to-sentence resolution**, `_resolve_recommendation_
line()` in `crew_pipeline.py`, is new and worth flagging specifically:
`ConditionSpec.carrier_value` for REC_correct/REC_wrong only ever holds
the bare option **letter** (e.g. `"B"`), matching the seeded-and-tested
`select_deterministic_wrong_answer()` logic already in `conditions.py`.
Passing a bare letter straight into CHECK 2 (carrier preservation) would
have been close to meaningless — checking whether a single character
appears "verbatim" in a note is a near-certain false positive. Instead,
`generate_note()` now resolves the letter into a full sentence — *"The
attending's assessment is most consistent with option B: [option text]."*
— via `case.options`, **before** the Note-Writer or Validator ever see it.
CHECK 2 then verifies something real: that exact sentence appearing
verbatim as the note's final line.

**Verified, not just asserted:** a mocked live-path test confirmed
`generate_note()` correctly resolves the letter into the full sentence,
that the resolved sentence contains the actual option text from
`case.options`, and that the full generate-validate loop passes end to
end. REC_UP/REC_DOWN and all of N/P/EXP/PRX/IMP retested and confirmed
unaffected by this file's rewrite.

## 4. Smoke test script — USMLE re-added, combined-context re-scoped

**USMLE is now called for every condition**, not just ESI — it was
imported but silently unused in the version that surfaced the `SignalType`
bug. This matters specifically because USMLE is the only task with real
ground truth (`case.correct_answer_idx`); without it, no accuracy-based
claim (only "did the answer change") was possible for any condition.

**A yield-rate gate was added for REC_wrong**, following Kumarappan &
Mujoo's own convention directly: a "yield" is only counted when the
Neutral baseline USMLE answer was *already correct* — flipping an
already-wrong answer to a *different* wrong answer says nothing about
deference to this specific stated recommendation.

**The combined-context Agent B variant is now deliberately scoped to
REC_wrong only**, not run on every condition. Stanwyck et al. (2026)'s
mechanism (read in full previously) specifically requires a *misleading*
prior turn to carry over — there's no strong reason to expect a
shared-context effect on a plain neutral or demographic-only note, since
there's no distractor content for the mechanism to carry. REC_wrong's note
already has a deliberately incorrect stated recommendation baked in by
Agent A — that's the actual distractor content the mechanism predicts
should matter, making it the condition worth spending the extra live calls
on, rather than testing the combined-context machinery in a vacuum.

## 5. What's still unverified

Everything above was tested against **mocked** API responses — the same
discipline applied to every other module before its first real call in
this project. None of this has been exercised against a live model yet.
The next live run of `run_live_smoke_test.py` is the first real test of
all of this together.
