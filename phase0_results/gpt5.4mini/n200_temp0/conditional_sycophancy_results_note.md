# Results Note: Conditional Sycophancy — Carrier Format vs. Demographic Content
## n=200 cases, gpt-5.4-mini, temperature=0.0, real logprobs

*Written from the actual generated figures (signal_type_conformity_finding,
signal_type_by_attribute, shift_heatmap, raw_distribution_strip,
mcnemar_polarity_overview, binary/usmle_signal_type_finding), not just the
printed summary numbers — the figures revealed nuances the raw numbers
alone understated.*

---

## Finding 1 (robust, publishable as a clean null): demographic polarity has no effect

Across all 36 paired E+ vs. E− McNemar comparisons (every attribute ×
signal type × recommendation type, both tasks), **zero were significant
at p<0.05**. This holds at real statistical power (n=94–198 per
comparison), not an underpowered null. Confirmed visually in the Cohen's h
forest plot — every bar is gray, none red.

**What this rules out**: whether the demographic detail favors the
privileged or unprivileged group does not modulate the model's
willingness to defer to an embedded recommendation, in any tested
configuration.

---

## Finding 2 (the real, novel result): how a detail is embedded in the text — not its demographic content — changes conformity to an embedded recommendation

Pooled across E+/E−/attribute (justified by Finding 1's exhaustive null),
ESI conformity to an embedded urgency recommendation shows a clear
gradient by carrier **format**:

| Signal type | REC_UP mean shift | REC_DOWN mean shift | Both CIs exclude zero? |
|---|---|---|---|
| **PRX** (administrative field) | +0.26 | −0.15 | Yes, clearly |
| **EXP** (explicit statement) | +0.07 | −0.09 | Yes |
| **IMP** (narrative/inline) | −0.04 | −0.04 | No — borderline/null |

Administrative-field framing ("Insurance: Medicaid") produces the
largest, most consistent resistance to an embedded urgency recommendation
— moving *away* from whatever was just recommended, in both directions.
Narrative/inline framing is statistically indistinguishable from no
effect. Explicit statement framing sits in between.

### Important qualifier: EXP's effect is attribute-dependent, PRX's is not

Breaking this down by attribute (not pooled) shows PRX replicates cleanly
across all three (Race, SES, Sex — all clearly nonzero, magnitudes
0.10–0.29). **EXP does not**: real and nonzero for Race and Sex, but
**flat null for SES** (mean ≈ −0.03 both directions, CI clearly straddles
zero). The pooled EXP effect in the headline table above is carried
entirely by Race and Sex — worth stating explicitly in any write-up
rather than implying EXP behaves uniformly across attributes.

### Important qualifier: binary and continuous metrics disagree for REC_UP specifically

The binary yield metric (does the top answer actually flip) tells a
partly different story for REC_UP: PRX has the *smallest* flip rate
(~14%), while EXP and IMP have *larger* flip rates (25%, 30%) despite
smaller continuous shifts. Interpretation: PRX reliably moves a large
amount of probability mass without as often crossing the argmax
threshold, while EXP/IMP produce fewer but more complete flips. Both
readings are real; they answer different questions ("how much did belief
shift" vs. "did the stated answer change"), and a write-up should report
both rather than picking the one with the cleaner story.

---

## Finding 3 (clean, worth stating plainly): the format effect is specific to subjective urgency judgment, not factual diagnosis

USMLE shows no comparable pattern: REC_CORRECT sits at ~97% and REC_WRONG
at ~15–18% across EXP/PRX/IMP, with heavily overlapping CIs and no clear
ranking. The carrier-format effect found in Finding 2 does not transfer to
a task with a verifiable correct answer.

---

## Suggested framing for a paper or section

> We manipulated how an incidental clinical detail was embedded in a note
> — as an administrative field, an explicit narrative statement, or an
> inline clause — while holding constant the actual embedded
> recommendation and the underlying case. Demographic polarity of that
> detail had no measurable effect on conformity anywhere (0/36 paired
> comparisons significant, n up to 198). However, the *format* of
> embedding produced a real, mostly attribute-independent effect on
> conformity to a co-located, unrelated urgency recommendation —
> administrative framing produced the largest resistance, narrative
> framing produced none — specific to the subjective urgency task and
> absent on a task with a verifiable correct answer.

This is a genuine, statistically supported, task-specific finding about
text structure's effect on LLM conformity — distinct from, and arguably
more novel than, the project's original demographic-bias hypothesis,
which this same data cleanly fails to support.

---

## Recommended next steps before finalizing for publication

1. **A formal interaction test** (e.g., a mixed-effects or interaction
   regression: shift ~ signal_type × attribute × rec_type) rather than
   relying on the by-eye per-attribute comparison above — needed to
   properly quantify whether EXP's Race/Sex-only pattern is itself
   statistically robust or could still be attribute-level noise at this n.
2. **Replicate on at least one more model** before treating this as a
   general LLM property rather than a `gpt-5.4-mini`-specific one —
   directly relevant now that OpenRouter access exists for cross-model
   comparison.
3. **A qualitative pass on the actual generated notes** for a handful of
   PRX vs. IMP cases, to check whether there's a plausible mechanistic
   story (e.g., administrative-field framing shifting the model's
   register toward "clinical documentation" rather than "narrative
   plus directive") — currently a plausible but untested hypothesis, and
   worth stating as such rather than asserted as fact.
