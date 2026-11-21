"""
mech/checkpoint.py

Results/metrics persistence for Paper 2's mechanistic phases.

Design directly mirrors shared/experiments.py's JSONLCheckpoint (Paper 1) --
same append-only-JSONL, resumable-by-key pattern -- for two reasons: (1)
consistency, so anyone who already understands Paper 1's checkpoint files
can read these without learning a new format; (2) the same resumability
argument applies here even more, since a single direction-extraction pass
over many contrastive pairs on a rented GPU is exactly the kind of run
where losing progress to a pod interruption is expensive.

Two stores, because directions and steering-validation observations have
genuinely different shapes and different re-run semantics:

  - DirectionStore: one directory of .npz files (numpy arrays don't belong
    in JSON), one file per (model, attribute, layer) direction, plus a
    JSON sidecar per file recording HOW it was computed (n_pairs used,
    extraction method, timestamp) -- so a direction file is never opaque
    later. Layers are cheap to recompute individually, so this store
    is keyed for per-layer resumability, not all-or-nothing.

  - SteeringValidationCheckpoint: JSONL, one row per (model, attribute,
    layer, coefficient, case_id) steering-validation observation --
    structurally identical in spirit to Paper 1's ObservationRecord.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Direction storage
# ---------------------------------------------------------------------------


class DirectionStore:
    """
    Persists one extracted direction vector per (model, attribute, layer),
    as a .npz file (the vector itself) + a .json sidecar (how it was made).
    Resumable per-layer: re-running Phase 2 on a model that already has
    some layers extracted only computes the missing ones.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, model: str, attribute: str, layer: int) -> tuple[Path, Path]:
        safe_model = model.replace("/", "_")
        stem = f"{safe_model}__{attribute}__layer{layer:02d}"
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def has(self, model: str, attribute: str, layer: int) -> bool:
        npz_path, json_path = self._paths(model, attribute, layer)
        return npz_path.exists() and json_path.exists()

    def save(
        self,
        *,
        model: str,
        attribute: str,
        layer: int,
        direction: np.ndarray,
        mean_pos: np.ndarray,
        mean_neg: np.ndarray,
        std: np.ndarray,
        bias_scalar: float,
        n_pairs: int,
        extraction_point: str,
        method: str = "ACE_whitened_mean_difference",
    ) -> None:
        """
        `extraction_point` documents WHERE in the sequence the activation was
        read from (e.g. "last_token") -- see mech/ace_sensor.py's module
        docstring for why this is a project design choice, not a verified
        transcription from Karvonen & Marks, and must therefore always be
        recorded alongside the vector it produced.
        """
        npz_path, json_path = self._paths(model, attribute, layer)
        np.savez(npz_path, direction=direction, mean_pos=mean_pos, mean_neg=mean_neg, std=std)
        json_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "attribute": attribute,
                    "layer": layer,
                    "bias_scalar": bias_scalar,
                    "n_pairs": n_pairs,
                    "extraction_point": extraction_point,
                    "method": method,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, model: str, attribute: str, layer: int) -> dict:
        npz_path, json_path = self._paths(model, attribute, layer)
        if not (npz_path.exists() and json_path.exists()):
            raise FileNotFoundError(f"No stored direction for model={model!r} attribute={attribute!r} layer={layer}")
        arrays = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return {
            "direction": arrays["direction"],
            "mean_pos": arrays["mean_pos"],
            "mean_neg": arrays["mean_neg"],
            "std": arrays["std"],
            **meta,
        }

    def list_layers(self, model: str, attribute: str) -> list[int]:
        safe_model = model.replace("/", "_")
        prefix = f"{safe_model}__{attribute}__layer"
        layers = []
        for p in self.root.glob(f"{prefix}*.json"):
            layer_str = p.stem.split("layer")[-1]
            layers.append(int(layer_str))
        return sorted(layers)


# ---------------------------------------------------------------------------
# Steering-validation checkpoint (JSONL, mirrors shared/experiments.py)
# ---------------------------------------------------------------------------


@dataclass
class SteeringObservation:
    """One (model, attribute, layer, coefficient, case_id) steering-validation row."""

    model: str
    attribute: str
    layer: int
    coefficient: float
    case_id: str
    task: str  # "esi" | "usmle"
    condition_source: str  # which note this came from, e.g. "N" (neutral, steered)
    top_class: str | None
    class_probabilities: dict[str, float] | None
    raw_text: str | None

    def key(self) -> str:
        return "|".join([self.model, self.attribute, str(self.layer), str(self.coefficient), self.case_id, self.task, self.condition_source])


class SteeringCheckpoint:
    """Append-only JSONL store for steering-validation observations. Same resumability contract as shared/experiments.py's JSONLCheckpoint."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_keys: set[str] = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line from an interrupted run -- same handling as Paper 1's checkpoint
                    key = record.get("_key")
                    if key:
                        self._seen_keys.add(key)

    def has(self, obs: SteeringObservation) -> bool:
        return obs.key() in self._seen_keys

    def append(self, obs: SteeringObservation) -> None:
        payload = asdict(obs)
        payload["_key"] = obs.key()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        self._seen_keys.add(obs.key())

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def export_metrics_json(metrics: dict, path: str | Path) -> None:
    """Single structured metrics file per run, matching Paper 1's publication_metrics_*.json convention (run_full_experiment.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


__all__ = ["DirectionStore", "SteeringObservation", "SteeringCheckpoint", "export_metrics_json"]