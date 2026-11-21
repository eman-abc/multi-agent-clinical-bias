"""
run_phase1_bridge_replication.py

Phase 1 (master handoff §7) -- behavioral bridge replication of Experiment 1
(signal gradient) and the REC-conformity battery on the two open-weight
models. n=40-60 (NOT 200) -- go/no-go replication, not publication scale.

TWO MODES, per this project's own explicit choice:

  --mode generate  ("generate and validate thru OpenRouter Luna API key")
      Calls Paper 1's own shared/experiments.py functions completely
      unchanged (run_experiment_1, run_experiment_rec) -- Agent A writes a
      FRESH note for each case/condition and the Validator checks it, both
      via GENERATOR_MODEL_ID/VALIDATOR_MODEL_ID (openai/gpt-5.6-luna,
      routed through OPENROUTER_API_KEY -- unchanged default, see
      shared/config_and_utils.py). Costs Agent A + Validator API calls on
      top of Agent B inference.

  --mode reuse  ("use the records from Paper 1")
      Zero Agent A calls, zero Validator calls, zero OPENROUTER_API_KEY
      usage. Reads note_text from extract_paper1_notes.py's OUTPUT file
      (paper2_results/extracted_paper1_notes.jsonl by default) -- notes
      Paper 1 already generated and validated when testing its API models
      -- and runs ONLY the local Agent B inference step
      (mech/local_inference.py) on them. Cheaper, and gives byte-identical
      stimuli to the ones DeepSeek/Gemini/Luna were already evaluated on,
      which is the methodologically cleaner comparison if that's what you
      want (same note, different Agent B, nothing else different).

NEITHER MODE TOUCHES phase0_results/ OR ANY OTHER PAPER 1 FILE:
  - generate mode writes to paper2_results/phase1/, not
    phase0_results/ -- a deliberate change from an earlier draft of this
    script, which wrote into Paper 1's own checkpoint files to reuse its
    reporting functions for free. That coupling is exactly what this
    project doesn't want, so this version pays a small cost instead
    (see _report_locally() below, which reimplements the reporting using
    shared/metrics.py directly rather than run_full_experiment.py's
    report_* functions, which are hardcoded to read from phase0_results/).
  - reuse mode's only READS are through extract_paper1_notes.py's output
    file, which is itself a read-only snapshot (see that script's own
    docstring) -- this script never opens anything under phase0_results/
    directly.

vLLM launch commands (generate mode, or reuse mode's local-inference step
-- both need Agent B served locally; run in a separate terminal/tmux pane):
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 --port 8000 --dtype float16
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --dtype float16

Usage:
    python run_phase1_bridge_replication.py --mode generate --model Mistral-7B-Instruct-v0.3 --n-cases 40
    python run_phase1_bridge_replication.py --mode reuse --model Mistral-7B-Instruct-v0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shared.config_and_utils import APIModelSpec
from shared import experiments as exp
from shared import metrics as met
from shared.data_loader import load_amqa_dataset, AMQARecord
from mech.model_registry import MECH_MODELS_BY_NAME
from mech.local_inference import run_esi_on_note, run_usmle_on_note

OUTPUT_DIR = Path("paper2_results/phase1")  # Paper-2-owned. Never phase0_results/.
DEFAULT_EXTRACTED_NOTES = Path("paper2_results/extracted_paper1_notes.jsonl")


def _local_vllm_spec(model_display_name: str, *, port: int = 8000) -> APIModelSpec:
    """Wraps a locally-served vLLM model as an APIModelSpec (see mech/local_inference.py's docstring for why this is the same trick used everywhere in this project for a new inference provider)."""
    spec = MECH_MODELS_BY_NAME[model_display_name]
    return APIModelSpec(
        display_name=spec.display_name,
        api_model_id=spec.hf_repo_id,
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
        role=f"Phase 1 bridge replication target ({', '.join(spec.primary_papers)})",
    )


# ---------------------------------------------------------------------------
# Mode 1: generate fresh via Luna (Paper 1's pipeline, unmodified, new output dir)
# ---------------------------------------------------------------------------


def run_generate_mode(local_spec: APIModelSpec, *, n_cases: int, skip_rec: bool) -> None:
    print("MODE: generate -- Agent A/Validator will make real API calls via OPENROUTER_API_KEY (GPT-5.6-Luna).")
    print(f"Output directory (Paper-2-owned, NOT phase0_results/): {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"EXPERIMENT 1 (signal gradient) on {local_spec.display_name}, n={n_cases}")
    print("=" * 78)
    r1 = exp.run_experiment_1(
        n_cases=n_cases, dry_run=False, use_chat_completions=True, use_logprobs=False,
        chat_completions_models=[local_spec],
        checkpoint_path=OUTPUT_DIR / "experiment_1.jsonl",
    )
    print(f"Records written (this file): {r1.records_written}")
    print(f"API calls (agent_a/agent_b/validation/total): "
          f"{r1.call_tracker.agent_a_calls}/{r1.call_tracker.agent_b_calls}/{r1.call_tracker.validation_calls}/{r1.call_tracker.total}")

    if not skip_rec:
        print("\n" + "=" * 78)
        print(f"REC-CONFORMITY BATTERY on {local_spec.display_name}, n={n_cases}")
        print("=" * 78)
        r_rec = exp.run_experiment_rec(
            n_cases=n_cases, dry_run=False, use_logprobs=False,
            chat_completions_models=[local_spec],
            checkpoint_path=OUTPUT_DIR / "experiment_rec.jsonl",
        )
        print(f"Records written (this file): {r_rec.records_written}")

    _report_locally(local_spec.display_name)


# ---------------------------------------------------------------------------
# Mode 2: reuse Paper 1's already-generated, already-validated notes
# ---------------------------------------------------------------------------


def _load_extracted_notes(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run extract_paper1_notes.py first:\n"
            f"    python extract_paper1_notes.py --source-dir phase0_results --out-dir paper2_results"
        )
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_reuse_mode(local_spec: APIModelSpec, *, notes_file: Path, temperature: float = 0.0) -> None:
    print("MODE: reuse -- zero Agent A calls, zero Validator calls, zero OPENROUTER_API_KEY usage.")
    print(f"Reading notes from: {notes_file}")
    print(f"Output directory (Paper-2-owned, NOT phase0_results/): {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notes = _load_extracted_notes(notes_file)
    print(f"Loaded {len(notes)} unique (case, condition) notes.")

    # Split by shape: Experiment-1-style (has an attribute) vs REC-battery-
    # style (attribute is None, condition_label is one of the REC labels).
    # Anything else (rec_type is set -> came from the conditional-
    # sycophancy matrix) is reported but not processed here -- that
    # experiment gets its own reuse-mode script next, not bolted on here
    # half-finished.
    exp1_notes = [n for n in notes if n.get("attribute") is not None and n.get("rec_type") is None]
    rec_notes = [n for n in notes if n.get("attribute") is None and n.get("rec_type") is None and n.get("condition_label") in ("REC_correct", "REC_wrong", "REC_UP", "REC_DOWN")]
    skipped = len(notes) - len(exp1_notes) - len(rec_notes)
    if skipped:
        print(f"NOTE: {skipped} extracted notes came from the conditional-sycophancy matrix (rec_type set) -- "
              f"not processed by this script yet. Flagging rather than silently dropping.")

    # AMQA/FairMedQA's own schema (shared/data_loader.py, verified this
    # session): AMQARecord carries neutralised_question, options (dict
    # A-D), and correct_answer_idx together, keyed by question_id -- which
    # is exactly what this project calls `case_id` everywhere downstream
    # (see shared/experiments.py: ObservationRecord.case_id is always set
    # from case.question_id). One dataset load gives us everything the
    # USMLE prompt (question text + options) AND ground truth (for later
    # AD/CFR reporting) need -- no separate ground-truth-only lookup call.
    amqa_records = load_amqa_dataset(dry_run=False)
    case_lookup: dict[str, AMQARecord] = {r.question_id: r for r in amqa_records}
    print(f"Loaded {len(case_lookup)} AMQA cases (question + options + ground truth) for USMLE prompt construction.")

    exp1_ckpt = exp.JSONLCheckpoint(OUTPUT_DIR / "experiment_1_reused.jsonl")
    rec_ckpt = exp.JSONLCheckpoint(OUTPUT_DIR / "experiment_rec_reused.jsonl")

    print(f"\nRunning local Agent B ({local_spec.display_name}) on {len(exp1_notes)} Experiment-1-shaped notes...")
    _run_local_batch(exp1_notes, local_spec, exp1_ckpt, experiment_name="experiment_1", case_lookup=case_lookup, temperature=temperature)

    print(f"\nRunning local Agent B ({local_spec.display_name}) on {len(rec_notes)} REC-battery-shaped notes...")
    _run_local_batch(rec_notes, local_spec, rec_ckpt, experiment_name="experiment_rec", case_lookup=case_lookup, temperature=temperature)

    _report_locally(local_spec.display_name, exp1_path=OUTPUT_DIR / "experiment_1_reused.jsonl", rec_path=OUTPUT_DIR / "experiment_rec_reused.jsonl")


def _run_local_batch(notes: list[dict], local_spec: APIModelSpec, checkpoint: "exp.JSONLCheckpoint", *, experiment_name: str, case_lookup: dict[str, AMQARecord], temperature: float) -> None:
    n_missing_case = 0
    for n_idx, note in enumerate(notes, start=1):
        case_id = note["case_id"]
        note_text = note["note_text"]

        for task in ("esi", "usmle"):
            record = exp.ObservationRecord(
                experiment=experiment_name, case_id=case_id, attribute=note.get("attribute"),
                condition_label=note["condition_label"], signal_type=note.get("signal_type"),
                model=local_spec.display_name, task=task, temperature=temperature,
                top_class=None, class_probabilities=None, validation_passed=True,  # reused notes were already validated by Paper 1
                blocked_reason=None, logprobs_available=False, note_text=note_text,
            )
            if checkpoint.has(record):
                continue

            if task == "esi":
                pred = run_esi_on_note(note_text, local_spec, temperature=temperature)
            else:
                case_record = case_lookup.get(case_id)
                if case_record is None:
                    # Genuinely possible, not just defensive: the extracted
                    # notes file can outlive the specific AMQA sample Paper 1
                    # drew from (different n_cases -> different
                    # sample_cases() draw, though the seed=42 convention
                    # means this should be rare in practice for cases drawn
                    # from the same underlying pool). Recorded as a blocked
                    # observation, not silently dropped, so a systematic
                    # mismatch is visible in the data rather than just
                    # shrinking your USMLE n unexplained.
                    record.blocked_reason = "case_id_not_found_in_amqa_dataset"
                    checkpoint.append(record)
                    n_missing_case += 1
                    continue
                pred = run_usmle_on_note(
                    note_text, case_record.neutralised_question, case_record.options,
                    local_spec, temperature=temperature,
                )

            record.top_class = pred.top_class
            record.class_probabilities = pred.class_probabilities
            record.agent_b_raw_text = pred.raw_text
            checkpoint.append(record)

        if n_idx % 10 == 0 or n_idx == len(notes):
            print(f"  {n_idx}/{len(notes)} notes processed.")

    if n_missing_case:
        print(f"  WARNING: {n_missing_case} USMLE observations blocked -- case_id not found in the loaded AMQA dataset (see blocked_reason='case_id_not_found_in_amqa_dataset').")


# ---------------------------------------------------------------------------
# Local reporting (reimplements run_full_experiment.py's report functions
# against Paper-2-owned files, since those functions hardcode
# OUTPUT_DIR = Path("phase0_results") at module level in run_full_experiment.py
# and are not parameterized -- rather than monkeypatch that module-level
# constant, this computes the same shared/metrics.py functions directly.)
# ---------------------------------------------------------------------------


def _report_locally(model: str, *, exp1_path: Path = OUTPUT_DIR / "experiment_1.jsonl", rec_path: Path = OUTPUT_DIR / "experiment_rec.jsonl") -> None:
    print("\n" + "=" * 78)
    print(f"METRICS (from {exp1_path.parent}, computed via shared/metrics.py directly)")
    print("=" * 78)
    if not exp1_path.exists():
        print(f"{exp1_path} not found -- nothing to report yet.")
        return
    records = exp.JSONLCheckpoint(exp1_path).load_all()
    # Same dataset load as run_reuse_mode's case_lookup, done again here --
    # this function can also be called standalone (e.g. after --mode
    # generate, which never builds a case_lookup), so it can't assume one
    # already exists in scope. AMQA is 801 rows; loading it twice in one
    # script invocation is a real but small inefficiency, not a
    # correctness issue -- worth collapsing into a single shared load
    # later if this script's runtime ever makes that worth doing.
    ground_truth = met.build_ground_truth_lookup(dry_run=False)
    for attribute in ("ses", "race", "sex"):
        for signal_type in ("EXP", "PRX", "IMP"):
            cfr_esi = met.compute_cfr(records, task="esi", attribute=attribute, model=model, temperature=0.0, logprobs_available=False, signal_type=signal_type)
            cfr_usmle = met.compute_cfr(records, task="usmle", attribute=attribute, model=model, temperature=0.0, logprobs_available=False, signal_type=signal_type)
            ad_usmle = met.compute_ad_usmle(records, ground_truth, attribute=attribute, model=model, temperature=0.0, logprobs_available=False, signal_type=signal_type)
            print(
                f"  [{attribute}/{signal_type}] CFR esi={cfr_esi.get('cfr')!r} (n={cfr_esi.get('n_total')})  "
                f"usmle={cfr_usmle.get('cfr')!r} (n={cfr_usmle.get('n_total')})  "
                f"AD(usmle)={ad_usmle.get('ad')!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["generate", "reuse"])
    parser.add_argument("--model", required=True, choices=list(MECH_MODELS_BY_NAME.keys()), help="Which model is CURRENTLY being served by vLLM on this pod.")
    parser.add_argument("--n-cases", type=int, default=40, help="generate mode only. Default matches Phase 0's own go/no-go n (not the n=200 publication scale).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-rec", action="store_true", help="generate mode only.")
    parser.add_argument("--notes-file", type=Path, default=DEFAULT_EXTRACTED_NOTES, help="reuse mode only. Output of extract_paper1_notes.py.")
    args = parser.parse_args()

    local_spec = _local_vllm_spec(args.model, port=args.port)
    print(f"Targeting local vLLM endpoint: {local_spec.base_url} (model={local_spec.api_model_id})")

    if args.mode == "generate":
        run_generate_mode(local_spec, n_cases=args.n_cases, skip_rec=args.skip_rec)
    else:
        run_reuse_mode(local_spec, notes_file=args.notes_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())