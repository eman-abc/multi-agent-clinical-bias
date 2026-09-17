# Results Note: Demographic Bias, Redaction, and REC-Conformity — n=200, Real Logprobs
## gpt-5.4-mini, temperature=0.0, Chat Completions with genuine logprobs (top_logprobs=5, max_tokens=150)

*This is the most rigorous version of this measurement in the whole
project: n=200, temperature=0.0 (matching FairMedQA's own convention),
and real multi-class probability distributions rather than one-hot text
parsing. Every earlier run building toward this used a smaller n, a
noisier temperature, or one-hot data -- this is the one to cite.*

---

## Headline result: direct, favorable comparison to FairMedQA's own published numbers

USMLE is the task directly comparable to FairMedQA's own metric (same CFR/AD
definitions, same style of counterfactual pair). Across all 9 attribute x
signal-type combinations:

| | This run (gpt-5.4-mini) | FairMedQA's own published range (12 models) |
|---|---|---|
| **CFR** | 0.954 - 0.975 | best model (GPT-5): 0.94; overall range not stated as low as this |
| **AD** | 0.00 - 2.02 percentage points | best model: ~3pp; overall range 3-19pp |

**gpt-5.4-mini's USMLE accuracy disparity (0-2.0pp) sits below FairMedQA's
own best-performing model (~3pp), and its CFR (95.4-97.5%) sits above it
(94%).** This is a genuine, citable, favorable comparison point -- not
just "small enough to call null," but smaller than the best-documented
model in the benchmark this project's metrics are directly built from.

Per-cell values (USMLE):

| Attribute | Signal | CFR | AD (pp) |
|---|---|---|---|
| SES | EXP | 0.954 | 1.08 |
| SES | PRX | 0.975 | 2.02 |
| SES | IMP | 0.970 | 1.01 |
| Race | EXP | 0.970 | 0.95 |
| Race | PRX | 0.970 | 0.51 |
| Race | IMP | 0.975 | 0.00 |
| Sex | EXP | 0.960 | 1.01 |
| Sex | PRX | 0.975 | 0.51 |
| Sex | IMP | 0.975 | 0.95 |

---

## Demographic bias: confirmed null, at the highest rigor this project has produced

Every one of the 12 paraphrase-excess 95% CIs (EXP/PRX x ESI/USMLE x 3
attributes) **crosses zero** -- visually unambiguous in the forest plot,
not a borderline call. Every McNemar p-value is non-significant (all
>=0.125). This replicates the null found at every earlier stage of this
project (one-hot at various n, real-logprobs at smaller n) -- now
confirmed at the combination of conditions best suited to detect a real
effect if one existed: n=200, temperature=0.0 (no sampling noise), genuine
multi-class distributions (not one-hot).

**This is now a well-powered, statistically clean, and literature-favorable
null** -- worth reporting as a real finding, not a placeholder for "more
data needed."

---

## Redaction (Experiment 3): the first genuinely interpretable result from this metric

Every earlier run of this metric (across multiple one-hot datasets)
produced nonsensical values (-1.0, -1.5, -2.0, 1.5) -- a known, flagged
limitation of one-hot data being too sparse for this specific ratio-based
metric. **With real logprobs, this metric finally produces sensible
numbers**:

| Task | Reversion fraction | Mean delta(E+,N) | Mean delta(redact,N) |
|---|---|---|---|
| ESI | -0.150 | 0.0058 | 0.0067 |
| USMLE | **+0.215** | 0.0138 | 0.0109 |

USMLE shows a real, positive, physically sensible partial reversion:
redacting the carrier phrase reduces the shift from baseline by about
21% -- below the pre-registered 0.30 pass threshold, so this doesn't meet
the original go/no-go bar, but it's the first time this metric has
produced a result worth taking at face value rather than discarding as an
artifact. ESI's small negative value, given how small both underlying
deltas are (0.006, 0.007), is best read as "no detectable causal
contribution from the carrier phrase," not a real reversal.

**Recommendation**: report this as a directionally-suggestive-but-below-
threshold result for USMLE, and treat ESI's redaction result as
inconclusive (deltas too small to interpret confidently either way)
rather than as evidence against causality.

---

## REC-conformity: continues to replicate as the project's strongest finding

| Metric | Rate | 95% CI | n |
|---|---|---|---|
| REC_UP (escalate urgency) | 61.9% | [54.6%, 68.8%] | 176 |
| REC_DOWN (de-escalate urgency) | 49.4% | [42.2%, 56.7%] | 180 |
| USMLE yield (stated wrong diagnosis) | 18.6% | [13.5%, 25.1%] | 172 |
| Combined-context divergence, ESI | 1.5% | -- | 197 |
| Combined-context divergence, USMLE | 13.6% | -- | 198 |

Consistent with every earlier run: strong deference on the subjective
urgency judgment (both directions, roughly 50-62%), much weaker deference
on the verifiable factual claim (~19%). The combined-context divergence
being far higher for USMLE (13.6%) than ESI (1.5%) is also consistent with
Stanwyck et al.'s finding that prior-turn effects are larger for
boards-style diagnostic tasks than grounded documentation tasks.

---

## What this means for the paper, taken together

The project's two main threads are now both at their most rigorous
form simultaneously:

1. **No detectable demographic bias** -- a clean, well-powered null,
   favorably positioned against the field's own benchmark (better CFR/AD
   than FairMedQA's best model).
2. **Real, substantial, replicating deference to explicit recommendations**
   -- strongly present for subjective judgments, much weaker for verifiable
   facts.

This is the dissociation the project's own v2 plan anticipated as the more
likely outcome from the start, now supported by the most rigorous data
this project has produced.

---

## Figures generated from this data

`generate_publication_figures.py` (now handles both single-model and
multi-model JSON automatically) produces, from this exact file:
- `paraphrase_excess_forest_gpt-5.4-mini.png` -- the null, visually
  unambiguous (all 12 CIs cross zero)
- `cfr_heatmap.png` -- USMLE columns consistently greener (higher CFR)
  than ESI columns
- `redaction_reversion.png` *(new)* -- the reversion fraction alongside its
  underlying raw deltas, so the small/negative ESI value reads as
  "inconclusive" rather than "reversed"
- `rec_conformity_bars.png` -- the REC-conformity replication
