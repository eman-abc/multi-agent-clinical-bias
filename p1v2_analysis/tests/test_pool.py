import csv, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import make_mock, analyze_p1v2 as A, pool_p1v2 as P
root = tempfile.mkdtemp(); make_mock.make(root)
A.main(["--root", root, "--boot", "300", "--allow-unlocked"])
P.main(["--root", root, "--boot", "300"])
res = os.path.join(root, "p1v2_analysis", "results")
rd = lambda n: list(csv.DictReader(open(os.path.join(res, n + ".csv"))))
rates, pooled = rd("rq2_rates"), rd("pooled_rates")
worst = 0.0
for m in ("M1", "M2", "M3"):
    for chan, src in (("user", "consensus"), ("user", "none"), ("assistant", "own")):
        for tt, metric in (("wrong", "yield"), ("wrong", "degradation"), ("correct", "rescue")):
            rs = [r for r in rates if r["model"] == m and r["task"] == "usmle" and r["chan"] == chan and r["src"] == src and r["ttype"] == tt and r["metric"] == metric]
            n = sum(int(r["n"]) for r in rs); w = sum(float(r["est"]) * int(r["n"]) for r in rs) / n
            pr = next(r for r in pooled if r["model"] == m and r["task"] == "usmle" and r["chan"] == chan and r["src"] == src and r["ttype"] == tt and r["target"] == "all" and r["metric"] == metric)
            assert int(pr["n_cases"]) == n, (m, chan, src, metric, pr["n_cases"], n)
            worst = max(worst, abs(w - float(pr["est"])))
print("largest difference between the pooled point estimate and the case-weighted per-letter combination:", f"{worst:.2e}"); assert worst < 1e-9
e1 = next(r for r in rates if r["model"] == "M1" and r["task"] == "esi" and r["chan"] == "user" and r["src"] == "consensus" and r["target"] == "1" and r["metric"] == "adopted")
e2 = next(r for r in pooled if r["model"] == "M1" and r["task"] == "esi" and r["chan"] == "user" and r["src"] == "consensus" and r["target"] == "1" and r["metric"] == "adopted")
assert abs(float(e1["est"]) - float(e2["est"])) < 1e-9; print("ESI single-target cell identical to the registered output")
ep = next(r for r in pooled if r["model"] == "M1" and r["task"] == "esi" and r["chan"] == "user" and r["src"] == "consensus" and r["target"] == "1|5" and r["metric"] == "adopted")
print("ESI pooled over levels 1 and 5 (M1, user consensus):", round(float(ep["est"]), 3), "from", ep["n_cases"], "cases and", ep["n_obs"], "observations"); assert int(ep["n_obs"]) > int(ep["n_cases"])
print("POOL TEST OK")
