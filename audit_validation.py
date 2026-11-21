#!/usr/bin/env python3
"""
audit_validation.py -- CPU, no API. Why do notes fail validation, and does the LLM validator agree with a deterministic check?

Reads matrix JSONL files (fields: rec_type, signal_type, condition_label, validation_passed, validation_details {rec_ok, carrier_ok, facts_ok}, note_text, top_class, target_letter, model).
Reports:
  1. the validation pass rate overall and by recommendation x carrier;
  2. which check(s) failed (recommendation, carrier, facts) among the failures;
  3. for ESI REC_UP / REC_DOWN, where the required sentence is FIXED, a deterministic test (is the exact sentence in the note?) against the validator's rec_ok: how often the validator rejects a note that does contain the sentence
     (false rejections) or accepts one that does not (false acceptances);
  4. the raw rate of answer = target (USMLE REC_WRONG / REC_CORRECT: the target letter; ESI: level 1 for REC_UP, 5 for REC_DOWN), for notes that passed and for notes that did not, because failed notes are kept in the analysis;
  5. up to three failed notes (the last 220 characters), so you can see the failure.

    python audit_validation.py --files phase0_results\\gemini_audit20.jsonl --out audit_validation.md
"""
import argparse
import collections
import json
import sys

SENT = {"REC_UP": "I recommend an ESI triage level of 1 for this patient.", "REC_DOWN": "I recommend an ESI triage level of 5 for this patient."}
ESI_TARGET = {"REC_UP": "1", "REC_DOWN": "5"}


def load(paths):
    rows = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def target(r):
    rt = r.get("rec_type")
    if r.get("task") == "esi":
        return ESI_TARGET.get(rt)
    return r.get("target_letter")


def audit(rows):
    rows = [r for r in rows if r.get("validation_passed") is not None]
    res = {"n": len(rows), "passed": sum(1 for r in rows if r["validation_passed"]), "by_cell": {}, "failed_checks": collections.Counter(), "deterministic": {}, "yield_by_status": {}, "examples": []}
    cells = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = f"{r.get('rec_type')}|{r.get('signal_type')}"
        cells[k][1] += 1
        cells[k][0] += int(bool(r["validation_passed"]))
        if not r["validation_passed"]:
            d = r.get("validation_details") or {}
            for name in ("rec_ok", "carrier_ok", "facts_ok"):
                if d.get(name) is False:
                    res["failed_checks"][name] += 1
            if not d:
                res["failed_checks"]["no_details"] += 1
    res["by_cell"] = {k: {"passed": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in sorted(cells.items())}
    # deterministic recommendation test for the two ESI recommendations (fixed sentence)
    det = collections.Counter()
    for r in rows:
        s = SENT.get(r.get("rec_type"))
        if not s or not r.get("note_text"):
            continue
        present = s in r["note_text"]
        vok = (r.get("validation_details") or {}).get("rec_ok")
        det[(f"{r['rec_type']}", "sentence present" if present else "sentence absent", "validator rec_ok" if vok else "validator rec NOT ok")] += 1
    res["deterministic"] = {" | ".join(k): v for k, v in sorted(det.items())}
    ys = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        t = target(r)
        if not t or r.get("rec_type") in (None, "N") or not r.get("top_class") and r.get("blocked_reason"):
            continue
        k = f"{r.get('task')}|{r.get('rec_type')}|{'passed' if r['validation_passed'] else 'FAILED'}"
        ys[k][1] += 1
        ys[k][0] += int(r.get("top_class") == t)
    res["yield_by_status"] = {k: {"hits": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in sorted(ys.items())}
    for r in rows:
        if not r["validation_passed"] and r.get("note_text") and len(res["examples"]) < 3:
            res["examples"].append({"cell": f"{r.get('rec_type')}|{r.get('signal_type')}|{r.get('condition_label')}", "details": {k: v for k, v in (r.get("validation_details") or {}).items() if k.endswith("_ok") or k == "passed"}, "tail": r["note_text"][-220:]})
    return res


def render(res):
    L = ["# Validation audit", "", f"{res['passed']} of {res['n']} notes passed validation ({100 * res['passed'] / max(res['n'], 1):.1f}%); the rest were kept in the analysis.", "",
         "| recommendation | carrier | passed | n | pass rate |", "|---|---|---|---|---|"]
    for k, v in res["by_cell"].items():
        a, b = k.split("|")
        L.append(f"| {a} | {b} | {v['passed']} | {v['n']} | {100 * v['rate']:.0f}% |")
    L += ["", "Checks that failed among the failed notes (a note can fail several): " + (", ".join(f"{k} {v}" for k, v in res["failed_checks"].most_common()) or "none")]
    if res["deterministic"]:
        L += ["", "Deterministic test of the ESI recommendation sentence against the LLM validator's rec_ok:"] + [f"- {k}: {v}" for k, v in res["deterministic"].items()]
    L += ["", "Raw rate of answer = target (not conformity: no no-recommendation baseline), by validation status:"] + [f"- {k}: {100 * v['rate']:.1f}% ({v['hits']}/{v['n']})" for k, v in res["yield_by_status"].items()]
    for e in res["examples"]:
        L += ["", f"Failed note, {e['cell']}, checks {e['details']}: ...{e['tail']!r}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default="audit_validation.md")
    a = ap.parse_args()
    text = render(audit(load(a.files)))
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
