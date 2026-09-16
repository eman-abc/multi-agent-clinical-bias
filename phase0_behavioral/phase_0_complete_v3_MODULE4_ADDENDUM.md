# Phase 0 v3 — Module 4 Addendum
## cfprompt Integration Findings, and Consolidated TODOs

*This supplements `phase_0_complete_v3.md` (read-only in the project). It is
not a replacement — treat it as an appendix covering what came up while
building Module 4 (`shared/crew_pipeline.py`) that the original spec didn't
resolve.*

**Update: TODO #2 is RESOLVED (option (a), vendoring) — see section A.4
below for the implementation record.** TODO #4 (the token-edit-fraction
proxy) is reconciled as a consequence. Sections A.1–A.3 are left as
originally written, as the historical record of the investigation; A.4
documents what was actually built.

---

## A. cfprompt integration findings

**Context:** §7/§9 of `phase_0_complete_v3.md` require a token-edit-
magnitude-matched paraphrase for the P condition (from "Compared to What?",
arXiv:2605.01048), but §8 gives no agent/prompt for actually producing it.
The reading list flags `cfprompt` (pip package, companion to that paper) as
existing. Rather than guess at its API from the PyPI README, I cloned the
real repository (`github.com/redagavin/cfprompt`) and read the source
directly.

### A.1 What's real and confirmed

- The package exists, is MIT-licensed, and has a real `paraphrase.py` module
  containing exactly the function this project needs:

  ```python
  # cfprompt/paraphrase.py
  def generate_adjusted_paraphrase(
      text: str,
      target_edit_pct: float,
      paraphrase_model,        # anything with .generate(prompts, per_prompt_seeds=None) -> list[str]
      tokenizer,                # anything with .encode(text) -> list[int]
      tolerance: float = 0.5,
      max_retries: int = 50,
      seed: int | None = None,
  ) -> AdjustedParaphraseResult
  ```

  This is a standalone function — it does **not** require the heavier
  `Study`/`run_all` comparative-analysis workflow shown in the README's
  Quickstart. It iteratively re-prompts the paraphrase model until the
  *actual* token-edit percentage (real Levenshtein edit distance over
  tokenizer IDs, in `cfprompt/tokenization.py::token_edit_distance_pct`,
  **not** the `difflib.SequenceMatcher` ratio proxy currently used in
  `shared/conditions.py::compute_token_edit_fraction`) lands within
  `tolerance` of `target_edit_pct`, retrying up to `max_retries` times
  (default 50). It also has built-in LLM-refusal detection
  (`cfprompt.paraphrase.is_refusal`) that triggers a corrective re-prompt
  rather than silently accepting a refusal as the "paraphrase."

- `cfprompt.OpenAIModel(name="gpt-4o-mini-2024-07-18", ...)` (real class,
  read directly from `cfprompt/models/openai.py`) satisfies the
  `paraphrase_model` protocol via its `.generate()` method, and exposes a
  `.tokenizer` property (a `TiktokenWrapper`) satisfying the `tokenizer`
  protocol — so in principle, wiring this in is: construct one
  `OpenAIModel`, call `generate_adjusted_paraphrase(neutral_note,
  target_pct, model, model.tokenizer)`, done.

### A.2 What blocks using it right now

1. **Python version conflict.** `cfprompt`'s `pyproject.toml` pins
   `requires-python = ">=3.11,<3.12"`. This container runs Python 3.12.3.
   `pip install cfprompt` fails outright here. Whatever environment
   actually runs Module 4 live needs to either (a) use a separate Python
   3.11 virtualenv for the paraphrase-generation step only, or (b) vendor
   just `paraphrase.py` + `tokenization.py` + a minimal model wrapper
   satisfying the two-method protocol, sidestepping the version pin
   entirely (both files have no exotic dependencies beyond what's already
   needed elsewhere in this project).
2. **Internal-API risk.** `generate_adjusted_paraphrase` and
   `AdjustedParaphraseResult` are **not** in `cfprompt`'s public `__all__`
   (confirmed by reading `cfprompt/__init__.py` directly — only `Study`,
   `Report`, `TestResult`, `Model`, `Tokenizer`, `HFModel`, `OpenAIModel`,
   and a few exception classes are exported). Importing from
   `cfprompt.paraphrase` directly works today, but the maintainer has made
   no API-stability promise on that submodule specifically.
3. **Not yet decided against Module 3's existing proxy.** `shared/
   conditions.py::compute_token_edit_fraction` currently uses a
   `difflib.SequenceMatcher` ratio as a stand-in for edit percentage. If
   `cfprompt` (or a vendored equivalent) is adopted, that proxy should
   either be replaced with the real Levenshtein-based metric or explicitly
   kept as a cheap upstream estimate with the real metric used only inside
   the paraphrase-generation retry loop. Not resolved either way yet.

### A.3 Current code state (historical — see A.4 for what was actually built)
~~`shared/crew_pipeline.py::generate_paraphrase_note()` is a stub that raises
`NotImplementedError`~~ — superseded, see A.4.

### A.4 Resolution: option (a), vendoring (RESOLVED)

**Decision:** vendor `paraphrase.py` and `tokenization.py` directly into
`shared/vendor/cfprompt_paraphrase/`, rather than (b) a separate Python 3.11
venv or (c) an in-house paraphrase prompt.

**Why (a) over (b)/(c):** both vendored files are pure Python with no exotic
dependencies (stdlib only: `re`, `unicodedata`, `dataclasses`,
`collections.abc`) — confirmed by reading both files in full. They import
cleanly on this project's Python 3.12, with no code changes needed beyond an
attribution header. This avoids the operational overhead of maintaining a
second Python environment (option (b)) entirely, while still using the
paper's own real, tested algorithm rather than reimplementing something
similar from scratch (option (c)).

**What was NOT vendored, deliberately:** cfprompt's `Model` ABC
(`models/base.py`) and `OpenAIModel` (`models/openai.py`).
`generate_adjusted_paraphrase()` only needs an object with
`.generate(prompts, per_prompt_seeds=None) -> list[str]` and
`.encode(text) -> list[int]` — a much narrower duck-typed contract than
cfprompt's full `Model` ABC (which also requires `score_classes`, `close`,
`cache_id`, none of which paraphrase generation uses). This project supplies
its own minimal wrapper (`shared/crew_pipeline.py::_OpenAIParaphraseModel`)
satisfying just that contract, using Chat Completions (not the legacy
Completions API — confirmed during this project that GPT-4o-family models
don't support it) and `tiktoken` for encoding.

**License compliance:** cfprompt is MIT-licensed (Copyright (c) 2026 Zihao
Yang / redagavin). The original `LICENSE` file is included verbatim at
`shared/vendor/cfprompt_paraphrase/LICENSE`, and both vendored files carry a
header identifying their origin, per the MIT license's attribution
requirement.

**Design decision this resolution required — single reference carrier.**
`generate_adjusted_paraphrase()` needs one `target_edit_pct` to aim for. But
P is generated once per case and reused across all three attributes (same
caching pattern as N — see `shared/experiments.py`'s `paraphrase_cache`),
while the "matched carrier" concept in principle differs per attribute and
signal type (a PRX/SES carrier line is a different length than a PRX/Race
one). Rather than generate a separate P variant per (attribute, signal_type)
combination — which would multiply P's generation cost and abandon the
cross-attribute caching optimization entirely — this implementation targets
a **single reference carrier: the SES/PRX condition** (`DEFAULT_CARRIER_
REGISTRY[Attribute.SES].privileged_value`), since SES/PRX is the condition
Experiments 1, 2, 3, and 5 all use most heavily. This is a disclosed
simplification: P's edit magnitude is well-matched to SES/PRX comparisons,
and only approximately matched for Race/Sex comparisons. A more precise (but
far more expensive) design would generate per-attribute/signal-type-matched
P variants; this was not implemented.

**TODO #4 reconciled as a consequence.** `shared/conditions.py::
compute_token_edit_fraction` (the `SequenceMatcher`-ratio proxy) is now
superseded for actual generation — `generate_paraphrase_note()` uses the
real, vendored, Levenshtein-based `token_edit_distance_pct` (via a real
`tiktoken` tokenizer) for both setting the target and verifying the realized
paraphrase. `compute_token_edit_fraction` remains available as a cheap,
tokenizer-free estimate for contexts where that's more convenient, but is no
longer in the P-condition generation path.

**Refusal handling.** The vendored `is_refusal()` detector is used
as-is — if every retry attempt (up to `max_retries=50`) is refused, the
returned `ValidatedNote.validation_passed` is forced to `False` and a
warning is logged, rather than silently falling back to the unmodified
Neutral note text under the "P" label.

---

## B. Consolidated TODO list (all modules, as of Module 4)

| # | Where | What | Status |
|---|---|---|---|
| 1 | `shared/conditions.py` — `DEFAULT_CARRIER_REGISTRY[Attribute.RACE]` | Real, verified matched-pair institutions (HBCU-affiliated clinic vs. private-affluent-suburb clinic, same specialty, same region) are **not populated**. `validate_registry_complete()` will raise outside `DRY_RUN`. | **Blocking** for any live run touching the Race attribute. |
| 2 | ~~`shared/crew_pipeline.py` — `generate_paraphrase_note()`~~ | ~~P-condition generation is stubbed.~~ **RESOLVED** — vendored cfprompt's `paraphrase.py`/`tokenization.py` into `shared/vendor/cfprompt_paraphrase/` (option (a)). See section A.4 for the full implementation record, including the single-reference-carrier design decision this required. Verified working end-to-end in dry-run across Experiments 1 and 2. | **Resolved.** |
| 3 | `phase_0_prompt_addendum_DRAFT.md` | EXP and IMP Note-Writer prompts are drafted but marked **PENDING PI APPROVAL**; `generate_note()` raises `UnimplementedConditionError` for both signal types. | **Blocking** for Experiment 1's EXP/IMP arms only (PRX arm, N, P, and Exp 2/3/5 are unaffected — Experiment 2 is now fully unblocked). |
| 4 | ~~`shared/conditions.py::compute_token_edit_fraction`~~ | ~~Uses a `SequenceMatcher`-ratio proxy~~ **RECONCILED** as part of resolving #2 — actual P-condition generation now uses the real, vendored, Levenshtein-based `token_edit_distance_pct`. The `SequenceMatcher` proxy remains available as a cheap, tokenizer-free estimate elsewhere, but is no longer in the generation path. | **Resolved.** |
| 5 | `shared/crew_pipeline.py` — Agent 1 (Carrier-Generator) | Implemented per §8 verbatim and independently testable (`run_carrier_generator_agent()`), but **not called by the main pipeline** — every currently-active condition (PRX, IMP) uses a deterministic carrier value from the matched-pair registry rather than free LLM generation, since inventing an institution name would violate the matched-pair discipline this project depends on (TODO #1). Documented as a deliberate deviation from a literal reading of §8, not an oversight. | Informational — revisit only if a future condition needs genuine free-text carrier generation. |
| 6 | `shared/crew_pipeline.py::generate_paraphrase_note` — single-reference-carrier simplification | P's target edit % is matched against the SES/PRX carrier only, reused across Race/Sex comparisons rather than generating per-attribute-matched P variants (see A.4). Disclosed simplification, not a bug. | Informational — revisit only if per-attribute-matched P precision becomes important. |

**Only item 1 (Race carrier registry) and item 3 (EXP/IMP approval) remain
blocking.** Item 2 (the P condition) — previously blocking Experiments 1 and
2 — is resolved; **Experiment 2 is now fully unblocked end-to-end**, and
Experiment 1 is unblocked except for its EXP/IMP arms.
