"""check_paraphrases.py -- CPU. Early quality check of the paraphrase controls.   python check_paraphrases.py [stage3/paraphrases.jsonl]"""
import json, re, statistics, collections, sys
POOL = "paper2_results/extracted_paper1_notes.jsonl"
FILE = sys.argv[1] if len(sys.argv) > 1 else "stage3/paraphrases.jsonl"
orig = {}
for line in open(POOL, encoding="utf-8"):
    r = json.loads(line)
    if r.get("condition_label") == "N" and r.get("rec_type") is None and r.get("note_text"):
        orig.setdefault(str(r["case_id"]), r["note_text"])
rows = [json.loads(l) for l in open(FILE, encoding="utf-8") if l.strip()]
STOP = set("about after again against because before being between could during other patient patients should their there these those under which while would having where there's".split())
word = lambda t: [w.lower() for w in re.findall(r"[A-Za-z]+", t)]
content = lambda t: {w for w in word(t) if len(w) >= 5 and w not in STOP}
NUM = re.compile(r"\d+(?:\.\d+)?")
REC = {"diagnosis", "diagnose", "diagnosed", "recommend", "recommended", "recommends", "should", "likely", "consistent", "treatment", "plan", "impression", "assessment", "suggests", "suspect", "triage", "admit", "discharge"}
DEMO = {"male", "female", "man", "woman", "he", "she", "his", "her", "him", "black", "white", "african", "caucasian", "hispanic", "asian", "race", "ethnic", "medicaid", "medicare", "insurance", "insured", "uninsured", "income", "wealthy", "poor"}
PRE = re.compile(r"^\s*(here is|here's|sure|certainly|of course|paraphrase|the paraphrase|below is)", re.I)
print(f"{len(rows)} rows; valid {sum(r['valid'] for r in rows)}; refused {sum(r['refused'] for r in rows)}; numbers lost {sum(not r['facts_preserved'] for r in rows)}")
act = [r["actual_edit_pct"] for r in rows if not r["refused"]]
if act:
    print(f"edit % (target {rows[0]['target_pct']}): mean {statistics.mean(act):.1f}, min {min(act):.1f}, max {max(act):.1f}; retries used: {dict(collections.Counter(r['retries_used'] for r in rows))}")
print(f"\n{'case':>6s} {'edit%':>6s} {'try':>4s} {'valid':>5s} {'len x':>6s} {'recall':>6s}  flags")
bad = 0
detail = []
for r in rows:
    o, p = orig.get(str(r["case_id"])), r["paraphrase"]
    if o is None:
        print(f"{r['case_id']:>6s}  (no original note found)"); continue
    lo, lp = len(word(o)), len(word(p))
    co, cp = content(o), content(p)
    recall = len(co & cp) / max(1, len(co))
    new = sorted(cp - co)
    flags = []
    if PRE.search(p): flags.append("PREAMBLE")
    if lp / max(1, lo) < 0.75 or lp / max(1, lo) > 1.35: flags.append("LENGTH")
    if recall < 0.55: flags.append(f"LOW-RECALL")
    nr = (REC & set(word(p))) - set(word(o))
    if nr: flags.append("NEW-REC-WORDS:" + ",".join(sorted(nr)))
    nd = (DEMO & set(word(p))) - set(word(o))
    if nd: flags.append("NEW-DEMOGRAPHIC:" + ",".join(sorted(nd)))
    if len(new) > 8: flags.append(f"{len(new)}-NEW-LONG-WORDS")
    if not r["facts_preserved"]: flags.append("NUMBERS-LOST")
    if r["refused"]: flags.append("REFUSED")
    bad += bool(flags)
    print(f"{r['case_id']:>6s} {r['actual_edit_pct']:6.1f} {r['retries_used']:4d} {str(r['valid']):>5s} {lp / max(1, lo):6.2f} {recall:6.2f}  {' '.join(flags) or '-'}")
    detail.append((r, o, p, new))
print(f"\n{bad} of {len(rows)} rows carry at least one flag. recall = share of the original's content words (5+ letters) kept; len x = paraphrase words / original words.")
for r, o, p, new in detail[:3]:
    print(f"\n=== case {r['case_id']}  edit {r['actual_edit_pct']:.1f}%  valid {r['valid']}")
    print("ORIGINAL  :", o[:700].replace("\n", " "))
    print("PARAPHRASE:", p[:700].replace("\n", " "))
    print("new long words:", new[:12])
