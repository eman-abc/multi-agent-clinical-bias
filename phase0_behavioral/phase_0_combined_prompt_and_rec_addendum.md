# Phase 0 — Combined-Prompt Variant and Recommendation-Conformity Control
## Literature Review and Implementation Record

**Status: combined-prompt variant IMPLEMENTED and tested. REC_correct/
REC_wrong SCAFFOLDING implemented and tested; the actual prompts are
DRAFTED but PENDING PI APPROVAL** (`phase_0_prompt_addendum_REC_DRAFT.md`),
the same gating discipline already applied to EXP/IMP.

---

## 1. The question this resolves

Does Agent B answering ESI and USMLE as two fully independent, context-free
calls (the existing design) vs. one shared conversation (ESI first, then
USMLE with the ESI turn included as history) make the research stronger —
and is either design, or having both, actually grounded in real literature
rather than invented for this project?

## 2. Three full papers read to answer this

### Stanwyck, Adibi, Dozie-Nnamah, Alsentzer (2026), medRxiv — read in full

- **Venue/credibility**: medRxiv preprint (not yet peer-reviewed), but
  authors affiliated with Stanford Biomedical Data Science, UBC, and
  Meharry Medical College; rigorous factorial design (McNemar tests,
  Holm-Bonferroni correction, bootstrap CIs — the same statistical
  conventions this project's own Module 7 already uses).
- **Design, read directly from Methods**: does a message earlier in a
  shared conversation ("prior-chat distractor") bias a model's answer to a
  *later*, different question? Tested on MedQA (boards-style MCQ) and
  EHRNoteQA (real discharge-summary-grounded QA).
- **Finding, quoted directly**: a prior-chat distractor stating an
  incorrect answer caused accuracy to drop by an average of **-15.0
  percentage points across 13/14 models** on the boards-style MCQ task, but
  only **-0.8 to -1.9 percentage points, significant in a minority of
  models**, on the grounded documentation task.
- **Their own conclusion, quoted directly**: *"boards-style multiple-
  choice evaluations are imperfect proxies for how prior-chat bias will
  manifest in real clinical interactions... may overstate the impact of
  prior-chat bias."*
- **Why this maps onto this project almost exactly**: MedQA (boards-style
  MCQ) is the closest analog to this project's USMLE task; EHRNoteQA
  (grounded clinical documentation) is the closest analog to this
  project's ESI task (a note-grounded judgment, not an abstract exam
  question).

### Kumarappan & Mujoo (2026) — already read in full in this project's
earlier research

Their entire method is: pre-generated peer content shown to a subject
model in one forward pass ("pressured" vs. "clean"), measuring whether the
subject's answer shifts toward the peer's stated (possibly wrong) position
— their "yield rate." This is the direct precedent for the REC_wrong
condition, and for the *separate-calls* design being a real, standard
baseline, not merely "the simpler option."

### Sharma et al. (2023) — already read in full in this project's earlier
research

The foundational sycophancy paper (458 citations). Confirmed directly from
Methods: baseline (no stated opinion) vs. a separate prompt where a stated
opinion is added — two single-shot comparisons, not a conversation. This
is the field's own recognized baseline design, further confirming
separate-calls isn't an inferior stand-in for "real" multi-agent research.

## 3. Why building BOTH designs is the right call, not just "more thorough"

Stanwyck et al.'s own **Control** condition (no prior message) *is* this
project's existing separate-calls design. Their **distractor** condition
(prior turn included in the same conversation) *is* the combined-call
design. **Their paper's actual contribution is the difference between
those two conditions, measured separately per task type** — not either
condition alone. Building both lets this project run the identical
comparison on its own two tasks and check whether the same asymmetry
(large effect on the MCQ-style task, small/inconsistent effect on the
grounded task) replicates here. That is a specific, literature-grounded,
falsifiable prediction — not just "test more things to be safe."

**Concretely, once REC is approved and live data exists, the predicted
pattern (stated as a real hypothesis, not asserted as a finding) is**: a
REC_wrong recommendation, combined with the combined-prompt Agent B
variant, should shift the USMLE answer substantially more than it shifts
the ESI answer relative to the separate-calls baseline — mirroring
Stanwyck et al.'s own MedQA-vs-EHRNoteQA asymmetry. If it doesn't
replicate, that itself is a reportable finding about how this project's
specific setup differs from theirs.

## 4. What was implemented

### 4.1 Combined-prompt Agent B variant — implemented, no approval gate needed

`shared/inference_engine.py::run_esi_then_usmle_combined_chat_completions()`
— Agent B answers ESI first (no history), then answers USMLE with the ESI
question and Agent B's own prior answer included as conversation history
(two literal API calls sharing accumulated `messages`, since the
underlying Chat Completions API is itself stateless — this is exactly the
mechanism Stanwyck et al. use technically, confirmed from their own
Methods section). The existing separate-calls functions
(`run_esi_triage_chat_completions`, `run_usmle_diagnostic_chat_completions`)
are unchanged and remain the default path.

This required no PI-approval gate: it's a measurement-architecture choice
(how many calls, whether they share context), not new note content or new
experimental methodology touching what Agent A writes.

**Verified**: a mocked test confirmed the USMLE call's `messages` list
contains exactly `[ESI question, Agent B's own ESI answer as an
"assistant" turn, USMLE question]` — i.e., the conversation history
threading actually works, not just returns plausible-looking results.
Confirmed the existing single-turn functions are unaffected by the
refactor needed to support this (a shared `_run_chat_completion_messages_live`
helper now backs both).

### 4.2 REC_correct / REC_wrong — scaffolding implemented and tested; prompts drafted, pending approval

- `ConditionLabel.REC_CORRECT` / `REC_WRONG` added.
- `CarrierType.RECOMMENDATION` and `CarrierAuthoring.LLM_RECOMMENDATION`
  added (the latter is the deliberate *inverse* of every other condition's
  "no recommendation" rule).
- `ConditionSpec.attribute` relaxed to `Attribute | None`, since REC
  conditions aren't tied to a demographic attribute at all.
- `select_deterministic_wrong_answer()` — seeded by `case_id`, matching
  Kumarappan & Mujoo's "pre-committed wrong target" convention. **Tested**:
  same case always selects the same wrong answer; never selects the actual
  correct answer.
- `build_recommendation_conditions()` — returns `[N, REC_correct,
  REC_wrong]`. **Tested** against a real case with real options.
- `generate_note()` gated: raises `UnimplementedConditionError` for both
  REC labels, pointing to `phase_0_prompt_addendum_REC_DRAFT.md` — the
  actual prompts (§4-5 of that file) are drafted but not approved, so
  nothing generates from them yet. **Verified this gate fires correctly**
  and that the full experiment regression suite still passes with the new
  condition labels and the relaxed `attribute` typing in place.

## 5. What's still needed before either can produce real data

- REC_correct/REC_wrong: PI sign-off on the drafted prompts
  (`phase_0_prompt_addendum_REC_DRAFT.md`), then transcription into
  `crew_pipeline.py` and removal of the `UnimplementedConditionError`
  gate — the same process EXP/IMP already went through.
- The combined-prompt variant has no gate, but has never been run live —
  only tested against a mocked client so far, the same discipline applied
  to every other new function before its first real API call in this
  project.
