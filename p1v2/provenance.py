"""
p1v2/provenance.py -- record everything: what went in, what came out, what was decided, at every step.

    run = Run("results", "paper1_v2_run1", config={...}, code_files=["p1v2/stimuli.py", ...])     # results/paper1_v2_run1/run_manifest.json: config + its hash, code hashes, python and package versions
    with run.step("build_stimuli", purpose="...", inputs=cases_path) as s:                        # results/<run>/steps/01_build_stimuli/{input.jsonl or a hashed pointer, output.jsonl, notes.md, manifest.json}
        s.append_output({...}); s.note("decided to ...")
    call_id = run.calls.log(role="agent_b", model=..., request=..., response_raw=..., ...)        # results/<run>/calls.jsonl, append-only, each line carries the hash of the line before it
    verify_run("results/paper1_v2_run1")  -> list of problems (empty if every hash and the call chain verify)

Nothing is ever overwritten: a step folder that exists is a refusal, not a silent reuse (use a new step name or a new run), and the call log is append-only with a hash chain, so edits and deletions show.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
from pathlib import Path


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _versions() -> dict:
    out = {}
    try:
        from importlib import metadata
        for p in ("openai", "numpy", "scipy", "pandas", "statsmodels", "python-dotenv"):
            try:
                out[p] = metadata.version(p)
            except metadata.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


class CallLog:
    """Append-only JSONL of every API call. Each record carries `prev`, the SHA-256 of the previous line, so deleting or editing a line breaks the chain."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._prev = "0" * 64
        self._n = 0
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    self._prev = sha256_text(line.rstrip("\n"))
                    self._n += 1

    def log(self, **fields) -> str:
        with self._lock:
            self._n += 1
            rec = {"call_index": self._n, "ts": utcnow(), **fields, "prev": self._prev}
            rec["call_id"] = sha256_text(canonical({k: v for k, v in rec.items() if k != "call_id"}))[:16]
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._prev = sha256_text(line)
            return rec["call_id"]


class Step:
    def __init__(self, run: "Run", name: str, purpose: str, inputs):
        self.run, self.name, self.purpose, self.inputs = run, name, purpose, inputs
        idx = len([d for d in (run.root / "steps").iterdir() if d.is_dir()]) + 1 if (run.root / "steps").exists() else 1
        slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
        self.dir = run.root / "steps" / f"{idx:02d}_{slug}"
        if self.dir.exists():
            raise FileExistsError(f"{self.dir} exists: a step folder is never reused")
        self.dir.mkdir(parents=True)
        self._notes: list[str] = []
        self._out_path = self.dir / "output.jsonl"
        self._n_out = 0
        self._t0 = time.time()
        self._started = utcnow()
        self._input_info: dict = {}

    def __enter__(self):
        if self.inputs is not None:
            if isinstance(self.inputs, (str, Path)):
                p = Path(self.inputs)
                self._input_info = {"kind": "file", "path": str(p), "sha256": sha256_file(p), "n_lines": sum(1 for _ in open(p, "rb"))}
            else:
                p = self.dir / "input.jsonl"
                with open(p, "w", encoding="utf-8") as f:
                    for rec in self.inputs:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                self._input_info = {"kind": "inline", "path": "input.jsonl", "sha256": sha256_file(p), "n_lines": sum(1 for _ in open(p, "rb"))}
        return self

    def append_output(self, record: dict) -> None:
        with open(self._out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._n_out += 1

    def note(self, text: str) -> None:
        self._notes.append(text)

    def __exit__(self, exc_type, exc, tb):
        status = "ok" if exc_type is None else f"failed: {exc_type.__name__}: {exc}"
        out = {"path": "output.jsonl", "sha256": sha256_file(self._out_path), "n_lines": self._n_out} if self._out_path.exists() else {}
        manifest = {"step": self.name, "purpose": self.purpose, "started_utc": self._started, "finished_utc": utcnow(), "seconds": round(time.time() - self._t0, 2), "status": status, "input": self._input_info, "output": out, "notes": self._notes}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        md = [f"# Step {self.dir.name}", "", f"**Purpose.** {self.purpose}", "", "## Input", f"- {self._input_info or 'none'}", "", "## Output", f"- {out or 'none'}", "", "## Notes and decisions"] + [f"- {n}" for n in self._notes] + ["", f"**Status.** {status} ({manifest['seconds']} s, started {self._started})"]
        (self.dir / "notes.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        return False


class Run:
    def __init__(self, root, name: str, config: dict, code_files=()):
        self.root = Path(root) / name
        (self.root / "steps").mkdir(parents=True, exist_ok=True)
        cfg_hash = sha256_text(canonical(config))
        mpath = self.root / "run_manifest.json"
        if mpath.exists():
            old = json.loads(mpath.read_text(encoding="utf-8"))
            if old["config_sha256"] != cfg_hash:
                raise ValueError("this run folder was created with a different configuration; use a new run name (a changed design is a new run)")
            self.manifest = old
        else:
            self.manifest = {"name": name, "created_utc": utcnow(), "python": sys.version.split()[0], "platform": platform.platform(), "packages": _versions(), "config": config, "config_sha256": cfg_hash,
                             "code": {str(p): sha256_file(p) for p in code_files if Path(p).exists()}}
            mpath.write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        self.calls = CallLog(self.root / "calls.jsonl")
        self._step_lock = threading.Lock()

    def step(self, name: str, purpose: str, inputs=None) -> Step:
        with self._step_lock:                                  # several models may start their steps at the same moment
            return Step(self, name, purpose, inputs)


def verify_run(root) -> list[str]:
    """Recompute every hash recorded in the step manifests and walk the call-log chain. Returns the problems found (empty = all verify)."""
    root, problems = Path(root), []
    for mf in sorted((root / "steps").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        for part in ("input", "output"):
            info = m.get(part) or {}
            if info.get("sha256"):
                p = Path(info["path"]) if info.get("kind") == "file" else mf.parent / info["path"]
                if not p.exists():
                    problems.append(f"{mf.parent.name}: {part} file missing ({p})")
                elif sha256_file(p) != info["sha256"]:
                    problems.append(f"{mf.parent.name}: {part} file changed since the step finished ({p})")
    cl = root / "calls.jsonl"
    if cl.exists():
        prev = "0" * 64
        for i, line in enumerate(open(cl, encoding="utf-8"), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prev") != prev:
                problems.append(f"calls.jsonl line {i}: the hash chain is broken (a line before it was edited or removed)")
                break
            prev = sha256_text(line.rstrip("\n"))
    return problems
