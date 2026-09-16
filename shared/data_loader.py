"""
shared/data_loader.py

Module 2 — Data Loader
Phase 0 — Bias Contagion Research Pipeline (v3 spec)

Loads:
    - AMQA / FairMedQA (arXiv:2505.19562) — the primary, already-filtered
      801-vignette pool. Source of truth: phase_0_complete_v3.md §6, §1.
    - MedPerturb (arXiv:2506.17163) — the optional secondary MANAGE/VISIT/
      RESOURCE benchmark. Source of truth: phase_0_complete_v3.md §3, Task 3.

Schema notes (verified against source during this session, not guessed):

  AMQA / FairMedQA — github.com/XY-Showing/AMQA, mirrored at
  huggingface.co/datasets/Showing-KCL/AMQA (801 rows). Real column names:
      question_id, original_question, desensitized_question, options (dict),
      answer, answer_idx, metamap_phrases,
      test_model_answer_original, test_model_answer_desensitized,
      and per attribute-group tag in {white, black, high_income, low_income,
      male, female}:
          adv_description_{tag}, adv_question_{tag},
          test_model_answer_adv_{tag}, attack_result_{tag}, retry_count_{tag}
  The spec's "neutralised_question" is AMQA's "desensitized_question" field —
  same concept, different literal key. This module normalizes to
  `neutralised_question` in AMQARecord so the rest of the pipeline never has
  to know the raw field name.

  MedPerturb — huggingface.co/datasets/abinitha/MedPerturb. IMPORTANT:
  `datasets.load_dataset("abinitha/MedPerturb")` currently FAILS with a
  DatasetGenerationCastError, because the repo's `data.csv` and
  `clinician_responses.csv` have mismatched, non-unifiable schemas (verified
  by inspecting the HF dataset viewer directly). This loader therefore reads
  `data.csv` directly via pandas rather than going through the `datasets`
  library. Verified `data.csv` columns:
      dataset, context_id, perturbation, clinical_context, original_gender,
      age, gendered_condition, gold_standard_manage, gold_standard_visit,
      gold_standard_resource, gold_standard_resource_allocation,
      clinician_consensus_manage, clinician_consensus_visit,
      clinician_consensus_resource, plus per-model answer columns
      (gpt-4o_*, medgemma_*, deepseek_*, qwen_*, llama_*) not needed here.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from shared.config_and_utils import DRY_RUN, N_CASES_GO_NO_GO, N_CASES_PUBLISHABILITY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute / group vocabulary
# Source: phase_0_complete_v3.md §5
# ---------------------------------------------------------------------------


class Attribute(str, Enum):
    RACE = "race"
    SEX = "sex"
    SES = "ses"


# Maps each attribute to (privileged_tag, unprivileged_tag) using AMQA's
# actual per-row column suffixes.
AMQA_VARIANT_TAGS: dict[Attribute, tuple[str, str]] = {
    Attribute.RACE: ("white", "black"),
    Attribute.SEX: ("male", "female"),
    Attribute.SES: ("high_income", "low_income"),
}

ALL_AMQA_TAGS: tuple[str, ...] = ("white", "black", "high_income", "low_income", "male", "female")


# ---------------------------------------------------------------------------
# AMQA / FairMedQA records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AMQAVariant:
    """One privileged/unprivileged adversarial variant of a base vignette."""

    tag: str  # one of ALL_AMQA_TAGS
    adversarial_description: str
    adversarial_question: str


@dataclass(frozen=True)
class AMQARecord:
    """
    A single AMQA/FairMedQA sample, normalized from the dataset's native
    schema (raw field names noted in parens where they differ):

        question_id, original_question,
        neutralised_question (raw: desensitized_question),
        options (dict A-D), correct_answer_idx (raw: answer_idx),
        variants: dict keyed by tag -> AMQAVariant
    """

    question_id: str
    original_question: str
    neutralised_question: str
    options: dict[str, str]
    correct_answer_idx: str
    variants: dict[str, AMQAVariant]

    def variant_pair(self, attribute: Attribute) -> tuple[AMQAVariant, AMQAVariant]:
        """Return (privileged_variant, unprivileged_variant) for the given attribute."""
        priv_tag, unpriv_tag = AMQA_VARIANT_TAGS[attribute]
        try:
            return self.variants[priv_tag], self.variants[unpriv_tag]
        except KeyError as exc:
            raise KeyError(
                f"Record {self.question_id!r} is missing variant tag {exc} "
                f"required for attribute {attribute.value!r}"
            ) from exc

    def has_all_variants(self) -> bool:
        """True if all six adversarial variants are present (sanity check)."""
        return all(tag in self.variants for tag in ALL_AMQA_TAGS)


def _parse_amqa_row(row: dict[str, Any]) -> AMQARecord:
    """Normalize one raw AMQA row (dict-like, from JSONL or a HF dataset) into an AMQARecord."""
    options = row["options"]
    if isinstance(options, str):
        options = json.loads(options)

    variants: dict[str, AMQAVariant] = {}
    for tag in ALL_AMQA_TAGS:
        desc_key = f"adv_description_{tag}"
        q_key = f"adv_question_{tag}"
        if row.get(desc_key) is not None and row.get(q_key) is not None:
            variants[tag] = AMQAVariant(
                tag=tag,
                adversarial_description=row[desc_key],
                adversarial_question=row[q_key],
            )

    return AMQARecord(
        question_id=str(row["question_id"]),
        original_question=row["original_question"],
        neutralised_question=row["desensitized_question"],
        options=options,
        correct_answer_idx=row["answer_idx"],
        variants=variants,
    )


def load_amqa_dataset(
    *,
    local_path: str | Path | None = None,
    dry_run: bool | None = None,
    mock_n: int = 40,
) -> list[AMQARecord]:
    """
    Load the AMQA/FairMedQA 801-sample dataset (arXiv:2505.19562).

    Resolution order (first that applies wins):
      1. `local_path` given -> read a local .jsonl file matching the AMQA
         schema (e.g. a clone of
         github.com/XY-Showing/AMQA/AMQA_Dataset/AMQA_Dataset.jsonl).
      2. `dry_run` is False (and no local_path) -> load
         `Showing-KCL/AMQA` from Hugging Face via the `datasets` library.
         This is the canonical, versioned copy of the 801 samples.
      3. Otherwise (dry_run True, the global default) -> return `mock_n`
         synthetic records shaped exactly like real AMQARecords, so the rest
         of the pipeline can be exercised with zero network dependency.

    Raises:
        RuntimeError: if dry_run is False, no local_path was given, and the
            Hugging Face load fails (network issue, package missing, etc.).
    """
    effective_dry_run = DRY_RUN if dry_run is None else dry_run

    if local_path is not None:
        return _load_amqa_from_jsonl(Path(local_path))

    if not effective_dry_run:
        try:
            return _load_amqa_from_huggingface()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the AMQA/FairMedQA dataset from Hugging Face "
                "(Showing-KCL/AMQA) and no local_path was provided. Either "
                "pass local_path=<path to AMQA_Dataset.jsonl>, or set "
                "DRY_RUN=true in .env to use synthetic mock data instead. "
                f"Underlying error: {exc}"
            ) from exc

    logger.info(
        "[data_loader] DRY_RUN active — returning %d synthetic AMQA-shaped "
        "mock records instead of loading real data.",
        mock_n,
    )
    return _generate_mock_amqa_records(mock_n)


def _load_amqa_from_jsonl(path: Path) -> list[AMQARecord]:
    if not path.exists():
        raise FileNotFoundError(f"AMQA dataset file not found at {path}")
    records: list[AMQARecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(_parse_amqa_row(json.loads(line)))
    logger.info("[data_loader] Loaded %d AMQA records from %s", len(records), path)
    return records


def _load_amqa_from_huggingface() -> list[AMQARecord]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required to load Showing-KCL/AMQA "
            "from Hugging Face. Install it with "
            "`pip install datasets --break-system-packages`."
        ) from exc

    ds = load_dataset("Showing-KCL/AMQA", split="train")
    records = [_parse_amqa_row(dict(row)) for row in ds]
    logger.info(
        "[data_loader] Loaded %d AMQA records from Hugging Face (Showing-KCL/AMQA)",
        len(records),
    )
    return records


_MOCK_OPTIONS: dict[str, str] = {
    "A": "[MOCK] Option A",
    "B": "[MOCK] Option B",
    "C": "[MOCK] Option C",
    "D": "[MOCK] Option D",
}


def _generate_mock_amqa_records(n: int) -> list[AMQARecord]:
    """Synthetic AMQARecords, schema-identical to real data, for dry-run pipeline testing."""
    records: list[AMQARecord] = []
    for i in range(n):
        variants = {
            tag: AMQAVariant(
                tag=tag,
                adversarial_description=(
                    f"[MOCK] Adversarial description for tag={tag}, case={i}."
                ),
                adversarial_question=(
                    f"[MOCK] A patient (tag={tag}) presents with mock symptom "
                    f"set {i}. Mock clinical vignette body for testing only."
                ),
            )
            for tag in ALL_AMQA_TAGS
        }
        records.append(
            AMQARecord(
                question_id=f"mock-{i:04d}",
                original_question=f"[MOCK] Original USMLE-style vignette #{i}.",
                neutralised_question=(
                    f"[MOCK] Neutralised vignette #{i}, pronouns replaced with "
                    f"'the patient'."
                ),
                options=dict(_MOCK_OPTIONS),
                correct_answer_idx="A",
                variants=variants,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Sampling — N=40 (go/no-go) vs N=200 (publishability)
# Source: phase_0_complete_v3.md §6, §14, §15
# ---------------------------------------------------------------------------


def sample_cases(
    records: list[AMQARecord],
    n: int,
    *,
    seed: int | None = 42,
) -> list[AMQARecord]:
    """
    Draw a reproducible sample of `n` cases without replacement.

    The same sampled cases should be reused across all conditions/signal-types
    for a given attribute (module 3's job) so that E+/E-/N/P are always
    evaluated on identical underlying clinical content — this function only
    handles the base-case draw.
    """
    if n > len(records):
        raise ValueError(f"Requested sample size {n} exceeds available records ({len(records)})")
    rng = random.Random(seed)
    return rng.sample(records, n)


def sample_go_no_go(records: list[AMQARecord], *, seed: int | None = 42) -> list[AMQARecord]:
    """Sample at the N=40 go/no-go gate size (§6, §14)."""
    return sample_cases(records, N_CASES_GO_NO_GO, seed=seed)


def sample_publishability(records: list[AMQARecord], *, seed: int | None = 42) -> list[AMQARecord]:
    """Sample at the N=200 publishability size (§6, §15)."""
    return sample_cases(records, N_CASES_PUBLISHABILITY, seed=seed)


# ---------------------------------------------------------------------------
# MedPerturb — optional secondary benchmark (Task 3)
# Source: phase_0_complete_v3.md §3 (Task 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MedPerturbRecord:
    """
    A single MedPerturb sample (arXiv:2506.17163), used as the optional
    secondary MANAGE/VISIT/RESOURCE clinical benchmark.

    Field names match `data.csv`'s verified columns directly (see module
    docstring); per-model answer columns are intentionally not carried,
    since this pipeline only needs the gold-standard / clinician-consensus
    ground truth, not other papers' model outputs.
    """

    context_id: str
    dataset_source: str  # e.g. "askdocs", "sct"
    perturbation: str  # e.g. "baseline", "summary"
    clinical_context: str
    original_gender: str
    age: str
    gendered_condition: bool
    gold_standard_manage: str
    gold_standard_visit: str
    gold_standard_resource: str
    clinician_consensus_manage: float | None
    clinician_consensus_visit: float | None
    clinician_consensus_resource: float | None


MEDPERTURB_DATA_CSV_URL = "https://huggingface.co/datasets/abinitha/MedPerturb/resolve/main/data.csv"


def load_medperturb_dataset(
    *,
    local_path: str | Path | None = None,
    dry_run: bool | None = None,
    mock_n: int = 25,
) -> list[MedPerturbRecord]:
    """
    Load the MedPerturb MANAGE/VISIT/RESOURCE secondary benchmark, per
    phase_0_complete_v3.md §3 (Task 3) — used on a subset of 20-30 cases
    overlapping with AMQA's topic areas (see `select_medperturb_overlap_subset`
    below).

    NOTE: this deliberately does NOT go through `datasets.load_dataset`.
    `datasets.load_dataset("abinitha/MedPerturb")` currently raises a
    DatasetGenerationCastError because the repo's `data.csv` and
    `clinician_responses.csv` files have mismatched schemas the `datasets`
    library cannot unify. Fetching `data.csv` directly with pandas sidesteps
    that entirely.

    Resolution order mirrors `load_amqa_dataset`: local_path, then live fetch
    (only if not dry_run), then synthetic mock data.
    """
    effective_dry_run = DRY_RUN if dry_run is None else dry_run

    if local_path is not None:
        return _load_medperturb_from_csv(Path(local_path))

    if not effective_dry_run:
        try:
            return _load_medperturb_from_url(MEDPERTURB_DATA_CSV_URL)
        except Exception as exc:
            raise RuntimeError(
                "Failed to fetch MedPerturb's data.csv directly from "
                f"{MEDPERTURB_DATA_CSV_URL} and no local_path was provided. "
                "Either pass local_path=<path to a local copy of data.csv>, "
                "or set DRY_RUN=true in .env to use synthetic mock data "
                f"instead. Underlying error: {exc}"
            ) from exc

    logger.info(
        "[data_loader] DRY_RUN active — returning %d synthetic MedPerturb-shaped "
        "mock records instead of loading real data.",
        mock_n,
    )
    return _generate_mock_medperturb_records(mock_n)


def _clean_float(value: Any) -> float | None:
    """Coerce a pandas cell to float, treating NaN and unparseable values as None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN != NaN


def _row_to_medperturb_record(row: Any) -> MedPerturbRecord:
    return MedPerturbRecord(
        context_id=str(row["context_id"]),
        dataset_source=str(row["dataset"]),
        perturbation=str(row["perturbation"]),
        clinical_context=str(row["clinical_context"]),
        original_gender=str(row.get("original_gender", "")),
        age=str(row.get("age", "")),
        gendered_condition=bool(row.get("gendered_condition", False)),
        gold_standard_manage=str(row.get("gold_standard_manage", "")),
        gold_standard_visit=str(row.get("gold_standard_visit", "")),
        gold_standard_resource=str(row.get("gold_standard_resource", "")),
        clinician_consensus_manage=_clean_float(row.get("clinician_consensus_manage")),
        clinician_consensus_visit=_clean_float(row.get("clinician_consensus_visit")),
        clinician_consensus_resource=_clean_float(row.get("clinician_consensus_resource")),
    )


def _require_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except ImportError as exc:
        raise RuntimeError(
            "The `pandas` package is required to load MedPerturb's CSV. "
            "Install it with `pip install pandas --break-system-packages`."
        ) from exc


def _load_medperturb_from_csv(path: Path) -> list[MedPerturbRecord]:
    pd = _require_pandas()
    if not path.exists():
        raise FileNotFoundError(f"MedPerturb CSV not found at {path}")
    df = pd.read_csv(path)
    records = [_row_to_medperturb_record(row) for _, row in df.iterrows()]
    logger.info("[data_loader] Loaded %d MedPerturb records from %s", len(records), path)
    return records


def _load_medperturb_from_url(url: str) -> list[MedPerturbRecord]:
    pd = _require_pandas()
    df = pd.read_csv(url)
    records = [_row_to_medperturb_record(row) for _, row in df.iterrows()]
    logger.info("[data_loader] Loaded %d MedPerturb records from %s", len(records), url)
    return records


def _generate_mock_medperturb_records(n: int) -> list[MedPerturbRecord]:
    records: list[MedPerturbRecord] = []
    for i in range(n):
        records.append(
            MedPerturbRecord(
                context_id=f"mock_askdocs_{i}",
                dataset_source="askdocs",
                perturbation="baseline",
                clinical_context=(
                    f"[MOCK] Clinical vignette #{i} for the MedPerturb secondary benchmark."
                ),
                original_gender="M" if i % 2 == 0 else "F",
                age=str(20 + (i % 50)),
                gendered_condition=False,
                gold_standard_manage="NO",
                gold_standard_visit="YES",
                gold_standard_resource="YES",
                clinician_consensus_manage=0.0,
                clinician_consensus_visit=1.0,
                clinician_consensus_resource=1.0,
            )
        )
    return records


def select_medperturb_overlap_subset(
    medperturb_records: list[MedPerturbRecord],
    amqa_records: list[AMQARecord],
    *,
    max_cases: int = 30,
) -> list[MedPerturbRecord]:
    """
    Heuristically rank MedPerturb cases by topical word-overlap with the AMQA
    pool, as a *starting point* for the 20-30 case overlap subset described in
    phase_0_complete_v3.md §3 (Task 3).

    IMPORTANT — this is a crude Jaccard-similarity-over-lowercased-words
    heuristic, not a validated topic-matching method. The spec does not
    define an automated matching procedure for this step. Treat this
    function's output as a shortlist for manual clinical review, not as a
    final, citable case selection.
    """
    if not amqa_records:
        raise ValueError("amqa_records must be non-empty to compute overlap")

    amqa_vocab: set[str] = set()
    for rec in amqa_records:
        amqa_vocab.update(rec.neutralised_question.lower().split())

    def _score(mp: MedPerturbRecord) -> float:
        mp_words = set(mp.clinical_context.lower().split())
        if not mp_words:
            return 0.0
        intersection = mp_words & amqa_vocab
        union = mp_words | amqa_vocab
        return len(intersection) / len(union) if union else 0.0

    ranked = sorted(medperturb_records, key=_score, reverse=True)
    return ranked[:max_cases]


__all__ = [
    "Attribute",
    "AMQA_VARIANT_TAGS",
    "ALL_AMQA_TAGS",
    "AMQAVariant",
    "AMQARecord",
    "load_amqa_dataset",
    "sample_cases",
    "sample_go_no_go",
    "sample_publishability",
    "MedPerturbRecord",
    "MEDPERTURB_DATA_CSV_URL",
    "load_medperturb_dataset",
    "select_medperturb_overlap_subset",
]
