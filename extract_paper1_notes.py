"""
extract_paper1_notes.py

Read-only extraction: scans Paper 1's own checkpoint files and pulls out
ONLY the records that captured note_text, into TWO complementary outputs:

  1. paper2_results/extracted_notes/<model>__<experiment>.jsonl
     One file per (model, experiment) pair -- e.g.
     "DeepSeek-V4-Flash__experiment_1.jsonl",
     "Gemini-3.8-Flash__experiment_rec.jsonl". This is the organized,
     auditable form: open one file, know exactly which model and which
     experiment every row in it came from, without needing to filter
     anything. Within each file, the two task rows (esi/usmle) for the
     same (case, condition) are still collapsed to one row -- they're
     duplicates of the SAME note, not two different notes -- but nothing
     is collapsed ACROSS models or experiments here, unlike output #2.

  2. paper2_results/extracted_paper1_notes.jsonl
     The combined, deduplicated, MODEL-AGNOSTIC pool -- the same output
     this script always produced, kept because run_phase1_bridge_replication.py
     (--mode reuse) and run_phase2_bias_sensor.py both consume it directly,
     and both explicitly don't care which original model read a note (the
     note itself is Agent-A-generated, independent of Agent B). Now tagged
     with `seen_by_models`: the list of every model whose row contributed
     this exact note_text under this (case, condition) key -- so "model" as
     a concept isn't lost here, just not used as a partition key the way
     output #1 does.

READ-ONLY GUARANTEE: every Paper 1 file this script touches is opened with
"r" (read) only. Nothing under phase0_results/ is ever written, appended,
or modified. Every open() call that WRITES targets a path under --out-dir,
never --source-dir.

NOT EVERY PAPER 1 FILE HAS note_text -- this is expected, not a bug:
per the master handoff §5, Gemini's original n=200 conditional-sycophancy
collection predates the note_text capture feature. This script will find
0 usable records there and SAY SO explicitly, rather than silently
skipping it or failing.

Usage:
    python extract_paper1_notes.py
    python extract_paper1_notes.py --source-dir phase0_results --out-dir paper2_results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Every Paper 1 output file pattern this project's codebase produces.
# experiment_1/2/3/rec.jsonl use shared/experiments.py's ObservationRecord
# schema; experiment_conditional_sycophancy_full_<model>.jsonl uses
# run_full_conditional_sycophancy.py's own, slightly different schema
# (extra rec_type/target_letter/validation_attempts/validation_details
# fields). Both schemas are handled below via .get() with no assumption
# either has every field the other does.
DEFAULT_SOURCE_PATTERNS = (
    "experiment_1.jsonl",
    "experiment_2.jsonl",
    "experiment_3.jsonl",
    "experiment_rec.jsonl",
    "experiment_conditional_sycophancy_full_*.jsonl",
)

# The identity of "one note" within a single (model, experiment) group, and
# also the cross-model dedup key for the combined pool. Deliberately does
# NOT include `task`: the same note is read by both esi/usmle tasks, so
# those two rows are expected duplicates of the SAME note, not independent
# notes -- collapsing them is the point, not a data-loss risk. `model` and
# `experiment` are handled separately (see extract() below) rather than
# folded into this key, since output #1 partitions BY them and output #2
# deliberately ignores them.
_KEY_FIELDS = ("case_id", "attribute", "condition_label", "signal_type", "rec_type")


def _dedupe_key(r: dict) -> str:
    return "|".join(str(r.get(k)) for k in _KEY_FIELDS)


def _sanitize(name: str) -> str:
    """Same sanitization convention run_full_conditional_sycophancy.py already uses for filenames (re.sub non-alphanumerics to '_') -- kept identical so filenames look familiar."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))


def extract(source_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    per_model_dir = out_dir / "extracted_notes"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "extracted_paper1_notes.jsonl"
    summary_path = out_dir / "extraction_summary.json"

    files: list[Path] = []
    for pattern in DEFAULT_SOURCE_PATTERNS:
        files.extend(sorted(source_dir.glob(pattern)))

    if not files:
        print(f"No Paper 1 checkpoint files found under {source_dir} matching {DEFAULT_SOURCE_PATTERNS}. Nothing to extract.")
        return combined_path, summary_path

    per_file_summary: dict[str, dict] = defaultdict(lambda: {"total_records": 0, "with_note_text": 0})

    # Output #1: grouped by (model, experiment), deduped only WITHIN each
    # group (across the esi/usmle task-duplicate rows).
    grouped: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)

    # Output #2: combined, deduped ACROSS models/experiments, tagged with
    # which models contributed to each kept note.
    kept: dict[str, dict] = {}
    n_duplicates_same_text = 0
    n_duplicates_conflicting_text = 0  # same key, DIFFERENT note_text -- see note below

    for path in files:
        with path.open("r", encoding="utf-8") as f:  # READ-ONLY -- see module docstring
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from an interrupted Paper 1 run -- same tolerant handling Paper 1's own code uses
                per_file_summary[path.name]["total_records"] += 1
                note_text = r.get("note_text")
                if not note_text or r.get("blocked_reason"):
                    continue
                per_file_summary[path.name]["with_note_text"] += 1

                model = r.get("model") or "unknown_model"
                experiment = r.get("experiment") or "unknown_experiment"
                key = _dedupe_key(r)

                tagged_record = {
                    "case_id": r.get("case_id"),
                    "attribute": r.get("attribute"),
                    "condition_label": r.get("condition_label"),
                    "signal_type": r.get("signal_type"),
                    "rec_type": r.get("rec_type"),
                    "model": model,
                    "experiment": experiment,
                    "note_text": note_text,
                    "source_file": path.name,
                }

                # --- Output #1: per-(model, experiment) group ---
                group = grouped[(model, experiment)]
                if key not in group:
                    group[key] = tagged_record
                # else: same task-duplicate row within this group, already have it -- skip.

                # --- Output #2: combined, cross-model dedup ---
                if key in kept:
                    if kept[key]["note_text"] == note_text:
                        n_duplicates_same_text += 1
                        kept[key]["seen_by_models"].add(model)
                    else:
                        # Real possibility, not a bug: Agent A regenerates a note
                        # the FIRST time a new model is added to an existing Paper 1
                        # run (see shared/experiments.py's _generation_needed --
                        # regeneration happens per NEW model, not per case). Two
                        # different note_texts under the same (case, condition) key
                        # are both legitimate; this script keeps the first one seen
                        # in the COMBINED pool and COUNTS the conflict rather than
                        # silently overwriting or discarding -- the per-(model,
                        # experiment) files above are NOT affected by this at all,
                        # since each model's own row is preserved there regardless.
                        n_duplicates_conflicting_text += 1
                    continue

                kept[key] = {
                    "case_id": r.get("case_id"),
                    "attribute": r.get("attribute"),
                    "condition_label": r.get("condition_label"),
                    "signal_type": r.get("signal_type"),
                    "rec_type": r.get("rec_type"),
                    "note_text": note_text,
                    "source_file": path.name,
                    "source_experiment": experiment,
                    "seen_by_models": {model},
                }

    # --- Write output #1: one file per (model, experiment) ---
    per_group_counts: dict[str, int] = {}
    for (model, experiment), group in grouped.items():
        fname = f"{_sanitize(model)}__{_sanitize(experiment)}.jsonl"
        fpath = per_model_dir / fname  # writes ONLY under --out-dir
        with fpath.open("w", encoding="utf-8") as f:
            for rec in group.values():
                f.write(json.dumps(rec) + "\n")
        per_group_counts[fname] = len(group)

    # --- Write output #2: combined, deduped, model-agnostic ---
    with combined_path.open("w", encoding="utf-8") as f:  # writes ONLY under --out-dir
        for rec in kept.values():
            out_rec = dict(rec)
            out_rec["seen_by_models"] = sorted(out_rec["seen_by_models"])
            f.write(json.dumps(out_rec) + "\n")

    summary = {
        "source_dir": str(source_dir),
        "files_scanned": [str(p) for p in files],
        "per_source_file": dict(per_file_summary),
        "per_model_experiment_files": per_group_counts,
        "combined_pool": {
            "output_file": str(combined_path),
            "n_unique_notes_written": len(kept),
            "n_duplicates_same_text_collapsed": n_duplicates_same_text,
            "n_duplicates_conflicting_text": n_duplicates_conflicting_text,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:  # writes ONLY under --out-dir
        json.dump(summary, f, indent=2)

    print(f"Scanned {len(files)} Paper 1 file(s), read-only:")
    for name, s in per_file_summary.items():
        pct = 100 * s["with_note_text"] / s["total_records"] if s["total_records"] else 0
        flag = "" if s["with_note_text"] else "  <-- ZERO notes captured here (predates note_text feature -- see master handoff §5, expected for Gemini's original n=200 conditional-sycophancy run)"
        print(f"  {name}: {s['with_note_text']}/{s['total_records']} records had usable note_text ({pct:.0f}%){flag}")

    print(f"\nWrote {len(per_group_counts)} per-(model, experiment) file(s) to {per_model_dir}/:")
    for fname, n in sorted(per_group_counts.items()):
        print(f"  {fname}: {n} notes")

    print(f"\nWrote {len(kept)} unique (case, condition) notes to the combined pool: {combined_path}")
    if n_duplicates_conflicting_text:
        print(f"WARNING: {n_duplicates_conflicting_text} (case, condition) keys had CONFLICTING note_text across models in the COMBINED pool "
              f"(different Agent A generations for the same nominal condition) -- first occurrence kept there. This does NOT affect the "
              f"per-(model, experiment) files above, which keep every model's own row intact regardless. "
              f"See {summary_path} for the full count.")
    print(f"Summary: {summary_path}")
    return combined_path, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="phase0_results", help="Paper 1's output directory (read-only).")
    parser.add_argument("--out-dir", default="paper2_results", help="Where to write extracted notes (Paper 2-owned, never read by Paper 1).")
    args = parser.parse_args()
    extract(args.source_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())