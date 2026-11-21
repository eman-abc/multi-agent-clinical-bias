import csv, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import make_mock, analyze_p1v2 as A, freeze_p1 as F
root = tempfile.mkdtemp(); n = make_mock.make(root)
print("mock stimuli:", n)
os.makedirs(os.path.join(root, "p1v2_analysis"), exist_ok=True)
A.main(["--root", root, "--out", os.path.join("p1v2_analysis", "results"), "--boot", "300", "--allow-unlocked"])
res = os.path.join(root, "p1v2_analysis", "results")
rd = lambda name: list(csv.DictReader(open(os.path.join(res, name + ".csv"))))
pool, rates, con, gap = rd("rq1_pooled"), rd("rq2_rates"), rd("rq2_contrasts"), rd("rq2_sycophancy_gap")
# 1) de-duplication kept every stimulus once
ans = {r["model"]: int(r["answers"]) for r in [dict(model=m, answers=0) for m in make_mock.MODELS]}
# 2) planted RQ1 effect: M2 race ESI effect about +0.15 (minus side less urgent) and placebo ~0, other models ~0
g = lambda m, task, attr, form: next(r for r in pool if r["model"] == m and r["task"] == task and r["attr"] == attr and r["form"] == form)
e2, e1, p2 = float(g("M2", "esi", "race", "EXP")["effect"]), float(g("M1", "esi", "race", "EXP")["effect"]), float(g("M2", "esi", "placebo", "PRX")["effect"])
x2 = float(g("M2", "esi", "race", "EXP")["excess"])
print(f"planted race ESI effect: M2 {e2:+.3f} (expect ~ +0.15), M1 {e1:+.3f} (expect ~0), M2 placebo {p2:+.3f}, M2 excess {x2:+.3f}, label {g('M2','esi','race','EXP')['equiv']}")
assert 0.05 < e2 < 0.25 and abs(e1) < 0.08 and abs(p2) < 0.08 and x2 > 0.04
u2 = float(g("M2", "usmle", "race", "EXP")["effect"]); print(f"planted race USMLE accuracy gap M2 {u2:+.3f} (expect ~ +0.06)"); assert 0.01 < u2 < 0.15
# 3) conformity ordering none < author < peer < consensus (ESI, target 1, user channel)
ad = {r["src"]: float(r["est"]) for r in rates if r["model"] == "M1" and r["task"] == "esi" and r["chan"] == "user" and r["target"] == "1" and r["metric"] == "adopted"}
print("M1 ESI adoption of level 1, user channel:", {k: round(v, 2) for k, v in ad.items()}); assert ad["none"] < ad["author"] < ad["peer"] < ad["consensus"]
yl = {r["src"]: float(r["est"]) for r in rates if r["model"] == "M1" and r["task"] == "usmle" and r["chan"] == "user" and r["ttype"] == "wrong" and r["metric"] == "yield"}
rs = float(next(r["est"] for r in rates if r["model"] == "M1" and r["task"] == "usmle" and r["chan"] == "user" and r["src"] == "consensus" and r["ttype"] == "correct" and r["metric"] == "rescue"))
print("M1 USMLE yield (wrong, user):", {k: round(v, 2) for k, v in yl.items()}, "| rescue (correct, consensus):", round(rs, 2)); assert yl["none"] < yl["consensus"] and rs > yl["consensus"]
cc = next(r for r in con if r["model"] == "M1" and r["task"] == "esi" and r["target"] == "1" and r["metric"] == "adopted" and r["contrast"] == "user consensus - user none")
print("contrast consensus - none:", round(float(cc["est"]), 2), cc["label"]); assert cc["label"] == "positive"
gp = next(r for r in gap if r["model"] == "M1" and r["chan"] == "user" and r["src"] == "consensus"); print("sycophancy gap M1 consensus:", round(float(gp["est"]), 2)); assert float(gp["est"]) > 0
# 4) invalid answers reported; duplicates removed
pr = rd("parse_report"); print("invalid rate range:", round(min(float(r["invalid_rate"]) for r in pr), 3), "-", round(max(float(r["invalid_rate"]) for r in pr), 3)); assert all(0.0 < float(r["invalid_rate"]) < 0.08 for r in pr)
# 5) lock: blank plan refused; filled plan freezes and verifies; tampering is detected
plan = os.path.join(root, "PLAN.md"); open(plan, "w").write(open(os.path.join(os.path.dirname(HERE), "P1V2_ANALYSIS_PLAN.md")).read())
try:
    F.freeze(plan, "t", root, os.path.join("p1v2_data", "stimuli_luna_p0.jsonl"), os.path.join("p1v2_data", "banks", "p1v2_bank_main", "cases_clean.jsonl"), "p1v2_run1"); print("ERROR: blank plan accepted"); sys.exit(1)
except SystemExit as e:
    print("blank plan refused:", str(e).splitlines()[1][:60])
import re
t = open(plan).read(); t = re.sub(r"(?m)^(\| \d+ \|.*\|) \|\s*$", lambda m: m.group(1) + " 0.5 |", t); open(plan, "w").write(t)
p, rec = F.freeze(plan, "t", root, os.path.join("p1v2_data", "stimuli_luna_p0.jsonl"), os.path.join("p1v2_data", "banks", "p1v2_bank_main", "cases_clean.jsonl"), "p1v2_run1")
print("freeze ok; verify:", F.verify(p, root)[0])
open(os.path.join(root, "results", "p1v2_run1", "steps", "01_agent_b_M1", "output.jsonl"), "a").write("\n")
f = os.path.join(root, "results", "p1v2_run1", "steps", "01_agent_b_M1", "output.jsonl"); s = open(f).read(); open(f, "w").write(s.replace('"A"', '"B"', 1))
ok, probs, _ = F.verify(p, root); print("after tampering with an answer file:", ok, probs[:1]); assert not ok
os.remove(p)
print("MOCK TEST OK")
