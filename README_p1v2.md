# p1v2 foundation, step 3b: base-note bank, stimulus file, Agent B runner, missing-answer diagnosis

    python test_p1v2.py && python test_p1v2_basenotes.py && python test_p1v2_runner.py && python test_export_cases.py        # 4 suites, all offline

Run in this order from the project root:

    python audit_cases.py --cases p1v2_data\cases.jsonl                                           # what the old exporter got wrong; which cases are excluded or ineligible
    python build_basenotes.py --cases p1v2_data\cases20.jsonl --run-name p1v2_smoke --client fake # the whole flow offline (no API)
    python build_basenotes.py --cases p1v2_data\cases20.jsonl --run-name p1v2_smoke20             # 20 real cases through GPT-6 Luna
    python build_basenotes.py --cases p1v2_data\cases.jsonl   --run-name p1v2_bank_main --workers 8

| file | role |
|---|---|
| `p1v2/facts.py` | splits a question into vignette and stem without cutting numbers; flags no-vignette and image-dependent cases; per-attribute eligibility (sex, race, SES) |
| `p1v2/basenotes.py` | writer prompt, deterministic validator, rewrite-with-feedback loop, parallel runner with a worker-side stop flag |
| `p1v2/llm.py` | OpenRouter client (guardrail errors stop the run; content filters are recorded once; transient errors retry) and a scriptable fake |
| `p1v2/stimuli.py` | 188 unique stimuli per eligible case (bias 54, conformity 30, interaction 104); ineligible attributes are dropped |
| `p1v2/provenance.py`, `parse.py`, `prompts.py` | step folders with hashes, hash-chained call log, strict answer reader, Agent B prompts |
| `inspect_bank.py` | reads the bank like a reviewer: first-attempt rate, rejection reasons, notes whose negations or content may have changed, reproducible random pairs |
| `audit_agent_b_parsing.py`, `check_models_route.py` | the parsing audit and the one-call-per-model routing check |

Removed as redundant: Experiment 3 (the redaction check IS the N condition: every carrier note is base note + one inserted string), Experiment 2 (the attribute loop of the bias arm), the standalone recommendation battery
(the conformity arm), per-model note regeneration (one frozen bank), the LLM validator (replaced by code), the exporter's `facts` field (truncated), and `audit_validation.py` (superseded).

## Step 3 order (priorities)

    python build_basenotes.py --cases p1v2_data\cases.jsonl --run-name p1v2_bank_main --workers 8                                  # P0: the frozen bank (a few dollars at most)
    python inspect_bank.py --bank-dir p1v2_data\banks\p1v2_bank_main --run-dir results\p1v2_bank_main --k 40 --out bank_inspection_main.md      # read it, esp. the flagged cases
    python build_stimuli.py --bank-dir p1v2_data\banks\p1v2_bank_main --bank luna --arms bias conformity --out p1v2_data\stimuli_luna_p0.jsonl
    python run_agent_b.py --stimuli p1v2_data\stimuli_luna_p0.jsonl --models GPT-6-Luna DeepSeek-V4-Flash Gemini-3.8-Flash --run-name p1v2_run1 --estimate
    python run_agent_b.py --stimuli p1v2_data\stimuli_luna_p0.jsonl --models GPT-6-Luna DeepSeek-V4-Flash Gemini-3.8-Flash --run-name p1v2_run1 --n-cases 20     # pilot, becomes part of the full run
    python run_agent_b.py --stimuli p1v2_data\stimuli_luna_p0.jsonl --models GPT-6-Luna DeepSeek-V4-Flash Gemini-3.8-Flash --run-name p1v2_run1 --workers 12    # P0 full run; rerun the same command to resume

Then P1: the interaction arm (`--arms interaction`), the writer-free `verbatim` bank (bias arm), and a 3-draw test-retest (`--n-cases 100 --draws 3`, its own run name).

## After the pilot (what the first real run taught)
* GPT-6 Luna: the provider's content filter refuses some cases (case 104). These are recorded as `content_filter` refusals, never retried, and reported by case; they remove both sides of that case's pairs.
* DeepSeek: 16% of calls failed after retries (empty text, no finish reason): a rate-limit/transport problem. The client now waits longer with jitter; use `--model-workers DeepSeek-V4-Flash=4`.
* `python diagnose_unparseable.py --run-dir results\p1v2_run1 --stimuli p1v2_data\stimuli_luna_p0.jsonl` separates refusals, content filters and transport failures and shows whether any design factor drives missing answers.

## Launching the full P0 run (pilot becomes part of it; rerun the same command to resume)

    python run_agent_b.py --stimuli p1v2_data\stimuli_luna_p0.jsonl --models GPT-6-Luna DeepSeek-V4-Flash Gemini-3.8-Flash --run-name p1v2_run1 --parallel-models --model-workers GPT-6-Luna=16 DeepSeek-V4-Flash=5 Gemini-3.8-Flash=12

The progress lines show calls per second and an ETA per model. Ctrl+C is safe: every finished call is already on disk.
