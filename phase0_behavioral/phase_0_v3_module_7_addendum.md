# Phase 0 v3 — Module 7 Addendum

## Metric Methodology: JSD, MedPerturb-Precedented McNemar, Stuart-Maxwell

*Supplements* `phase_0_complete_v3.md` *§9 and §14, and* `phase_0_complete_v3_MODULE4_ADDENDUM.md`*. Confirmed with the PI in chat before implementation — this is not a unilateral deviation.*

---

## A. Why §9's formulas don't transfer directly to both tasks

§9 gives one formula for "decision" (`E[decision|condition] = Σ i×P(token=i)`) and one McNemar convention (paired, ground-truth-based, following FairMedQA), applied identically to both Task 1 (ESI, 1-5) and Task 2 (USMLE, A-D). Two problems, both confirmed with the PI before resolving:

1. `Σ i×P(token=i)` **assumes ordinality.** ESI's 1-5 scale is a genuine ordinal severity scale (Immediate → Non-urgent), so treating it as ordinal and computing an expected value is meaningful. USMLE's A/B/C/D labels are arbitrary answer-choice positions with no inherent order — indexing them 1-4 would treat "the correct answer happened to be listed third" as meaningfully different from "listed second," which it isn't.
2. **FairMedQA's McNemar convention needs ground truth.** It's a paired correctness comparison (was the model right under condition A vs. condition B), which only makes sense for a task with a correct answer. ESI has no such ground truth in this design — it's this project's own addition (§3), not something FairMedQA's original convention was built to handle.

## B. Resolution 1 — USMLE shift metric: JSD, not expected value

**Decision:** for USMLE, `Δ(condition, N)` and paraphrase excess are computed using Jensen-Shannon divergence between `P(class|condition)` and `P(class|N)`, not `Σ i×P(token=i)`.

**Justification:** directly from "Compared to What?" (arXiv:2605.01048, already fetched and read during Module 4's research) — the paper's own stated finding is that *"per-sample metrics (JSD, KL) are dramatically more powerful than aggregate metrics (MI, φ, flip rate)."* JSD is a genuine distributional-divergence measure that requires no ordering assumption, making it valid for both nominal (USMLE) and ordinal (ESI) data — unlike `Σ i×P(token=i)`, which is only valid for the latter. ESI's shift metric remains the literal ordinal-expected-value formula from §9, since ordinality there is genuine and the expected-value framing is directly interpretable ("expected severity level").

## C. Resolution 2 — ESI's McNemar/Cohen's h: MedPerturb-precedented binary indicator

**Decision:** for ESI, McNemar's test and Cohen's h operate on a binary indicator per case — `C = 1` if the condition's ESI decision differs from the Neutral (N) baseline for that case, `0` otherwise — rather than a ground-truth correctness indicator (which doesn't exist for ESI) or a raw 5-category comparison.

**Justification:** this is not an invented-from-scratch heuristic. MedPerturb (Gourabathina et al., arXiv:2506.17163 — already in the reading list for the Task 3 secondary benchmark) has a published statistical-methods appendix (§K.1) doing almost exactly this: define `C_base = 1` if a property holds under the baseline condition, `C_pert = 1` if it holds under the perturbed condition, build the 2×2 table of `(C_base, C_pert)` pairs across cases, and run McNemar. Their specific property was "do all annotators/models reach consensus," because their design has multiple raters per case. This project's design has one rater (one Agent B model) per case, so "consensus" doesn't translate directly — but the same statistical machinery adapts cleanly by substituting "differs from Neutral" as the binary property in place of "lacks consensus." This is a minimal, faithful adaptation of a real published method, not a new invention.

**For USMLE**, ground truth exists, so McNemar stays exactly as FairMedQA's original convention: paired correct/incorrect comparison. No adaptation needed there.

**Mapping the literal (E+,N) / (P,N) pairs.** Under the binary-indicator scheme, comparing a condition directly to N is degenerate (N never "differs from N"). The operative translation that actually serves §9's stated purpose — *"confirm the paraphrase control"* — is comparing E+'s and P's `differs-from-N` rates against *each other*: this directly tests whether E+ deviates from N more often than P does, which is precisely what comparing "(E+,N)" and "(P,N)" was meant to establish. Implemented as `mcnemar_esi_binary_vs_neutral(..., label_a="E+", label_b="P")`.

## D. Stuart-Maxwell — offered as an optional secondary test, not primary

The Stuart-Maxwell (marginal homogeneity) test is the standard, well- established generalization of McNemar to k>2 categories, and general biostatistics sources note it's specifically better suited to *ordinal* data (their own example: disease-severity grades) than a binary collapse, since it preserves the full category structure rather than discarding it.

It is implemented here (`stuart_maxwell_test`, `stuart_maxwell_esi`) as an **optional secondary test only** — it has no direct precedent in this project's specific reading-list papers (unlike the binary approach above, which does), so it isn't used for the pre-registered go/no-go criterion. Requires `numpy` and `scipy` (lazy-imported; the rest of this module has no such dependency).

## E. Go/no-go per-task interpretation (§14)

§14's criterion (1) explicitly says "on at least one of the two tasks"; criteria (2) and (3) don't repeat that qualifier, leaving it ambiguous whether they're meant per-task or pooled across tasks. Resolved as follows, documented here rather than silently assumed:

- Each of criteria (1)-(3) is evaluated **per task** (ESI and USMLE separately).
- **GO**: all three criteria hold on **both** tasks.
- **TASK_SPECIFIC_GO**: all three hold on **exactly one** task — this is a non-redundant reading of §14's own "TASK-SPECIFIC GO" branch, which otherwise would be indistinguishable from PRIMARY GO under a literal "at least one task" reading of all three criteria.
- **NARROW_GO**: criterion (1) holds for the EXP signal type but not PRX/IMP — structurally implemented, but cannot currently trigger, since EXP/IMP remain blocked pending PI approval (Module 4).
- **NULL**: none of the above.

## F. No-dependency statistics kept dependency-free

McNemar's test (1 degree of freedom) uses the closed-form `P(χ² ≤ x) = erf(√(x/2))` — standard, exact, and needs only `math.erf` (no scipy). Exact binomial fallback is used when discordant pairs `b+c<25`, per standard convention. Bootstrap CIs use plain resampling (Python `random`, no numpy required). Only the optional Stuart-Maxwell test needs numpy/scipy.