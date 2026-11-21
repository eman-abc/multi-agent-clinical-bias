"""test_plan.py -- CPU only.   python test_plan.py"""
import json
import os
import subprocess
import sys
import tempfile

from mech import plan as P


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def make_dir(tmp):
    write(os.path.join(tmp, "analysis_plan.json"), json.dumps({"crossfit": {"folds": 10}, "frozen_files": ["code.py"], "frozen_data": ["data.jsonl"]}))
    write(os.path.join(tmp, "ANALYSIS_PLAN.md"), "# plan\nrules\n")
    write(os.path.join(tmp, "code.py"), "print('analysis')\n")
    write(os.path.join(tmp, "data.jsonl"), '{"a": 1}\n')
    return os.path.join(tmp, "analysis_plan.json")


def test_status_transitions():
    with tempfile.TemporaryDirectory() as tmp:
        plan = make_dir(tmp)
        st = P.plan_status(plan, tmp)
        assert st["status"] == "NOT-FROZEN" and "EXPLORATORY" in P.banner(st)
        lock = P.write_lock(plan, tmp)
        assert set(lock["files"]) == {"code.py"} and set(lock["data"]) == {"data.jsonl"} and lock["plan_sha256"] and lock["frozen_at_utc"].endswith("+00:00")
        st = P.plan_status(plan, tmp)
        assert st["status"] == "FROZEN-OK" and st["differences"] == [] and "FROZEN-OK" in P.banner(st)

        # reformatting the plan JSON (key order, whitespace) is NOT a change of content
        write(plan, json.dumps({"frozen_data": ["data.jsonl"], "frozen_files": ["code.py"], "crossfit": {"folds": 10}}, indent=4))
        assert P.plan_status(plan, tmp)["status"] == "FROZEN-OK"
        # changing a number IS
        write(plan, json.dumps({"crossfit": {"folds": 5}, "frozen_files": ["code.py"], "frozen_data": ["data.jsonl"]}))
        st = P.plan_status(plan, tmp)
        assert st["status"] == "MODIFIED" and "analysis_plan.json content changed" in st["differences"] and "NO LONGER CONFIRMATORY" in P.banner(st)
        write(plan, json.dumps({"crossfit": {"folds": 10}, "frozen_files": ["code.py"], "frozen_data": ["data.jsonl"]}))
        assert P.plan_status(plan, tmp)["status"] == "FROZEN-OK"
        # narrative, code and data each trip it, and the message says which
        for fname, text, expect in (("ANALYSIS_PLAN.md", "# plan\nrules changed\n", "ANALYSIS_PLAN.md changed"),
                                    ("code.py", "print('different analysis')\n", "file: code.py changed"),
                                    ("data.jsonl", '{"a": 2}\n', "data: data.jsonl changed")):
            original = open(os.path.join(tmp, fname)).read()
            write(os.path.join(tmp, fname), text)
            st = P.plan_status(plan, tmp)
            assert st["status"] == "MODIFIED" and expect in st["differences"], (fname, st)
            write(os.path.join(tmp, fname), original)
            assert P.plan_status(plan, tmp)["status"] == "FROZEN-OK"
        os.remove(os.path.join(tmp, "code.py"))
        st = P.plan_status(plan, tmp)
        assert st["status"] == "MODIFIED" and "file: code.py missing" in st["differences"]
        write(os.path.join(tmp, "code.py"), "print('analysis')\n")
        # adding a file to the plan after freezing is also visible
        write(plan, json.dumps({"crossfit": {"folds": 10}, "frozen_files": ["code.py", "extra.py"], "frozen_data": ["data.jsonl"]}))
        write(os.path.join(tmp, "extra.py"), "x = 1\n")
        st = P.plan_status(plan, tmp)
        assert st["status"] == "MODIFIED" and any("extra.py was added" in d for d in st["differences"])
    print("PASS: NOT-FROZEN -> FROZEN-OK -> MODIFIED; reformatting is harmless; plan, narrative, code, data, a deleted file and an added file are each detected and named")


def test_resolve_refuses_silent_overrides():
    assert P.resolve("folds", None, None, 10, plan_given=False) == 10 and P.resolve("folds", 5, None, 10, plan_given=False) == 5       # no plan: CLI beats built-in
    assert P.resolve("folds", None, 10, 7, plan_given=True) == 10 and P.resolve("folds", 10, 10, 7, plan_given=True) == 10               # plan wins; agreeing flag is fine
    try:
        P.resolve("n_random", 5, 20, 20, plan_given=True)
        raise AssertionError("a conflicting flag must be refused")
    except ValueError as e:
        assert "--n-random" in str(e) and "frozen plan" in str(e)
    print("PASS: with a plan, its values are used and a conflicting command-line value is an error naming the flag")


def test_cli():
    with tempfile.TemporaryDirectory() as tmp:
        plan = make_dir(tmp)
        env_cwd = os.getcwd()
        script = os.path.join(env_cwd, "freeze_plan.py")
        run = lambda *a: subprocess.run([sys.executable, script, "--plan", plan, "--root", tmp, *a], capture_output=True, text=True, cwd=env_cwd)
        r = run("--verify")
        assert r.returncode == 1 and "NOT FROZEN" in r.stdout
        r = run("--freeze")
        assert r.returncode == 0 and "frozen at" in r.stdout and "git commit" in r.stdout and os.path.exists(os.path.join(tmp, P.LOCK_NAME))
        assert run("--verify").returncode == 0
        r = run("--freeze")                                                                    # a second freeze is refused without --force
        assert r.returncode == 1 and "already exists" in r.stdout
        write(os.path.join(tmp, "code.py"), "changed\n")
        r = run("--verify")
        assert r.returncode == 1 and "MODIFIED" in r.stdout and "code.py" in r.stdout
    print("PASS: freeze_plan.py freezes once, verifies, refuses a silent re-freeze, and names the file that changed")


if __name__ == "__main__":
    test_status_transitions()
    test_resolve_refuses_silent_overrides()
    test_cli()
    print("\nALL PLAN TESTS PASSED")
