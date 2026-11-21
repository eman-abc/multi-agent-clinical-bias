"""test_export_cases.py -- runs export_cases.py against a FAKE shared package."""
import json, os, subprocess, sys, tempfile, textwrap


def test_export():
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "shared")); open(os.path.join(t, "shared", "__init__.py"), "w").write("")
        open(os.path.join(t, "shared", "crew_pipeline.py"), "w").write("def _strip_trailing_question_stem(x):\n    return x.rsplit('. ', 1)[0] + '.'\n")
        open(os.path.join(t, "shared", "data_loader.py"), "w").write(textwrap.dedent('''
            import random
            from types import SimpleNamespace as S
            def load_amqa_dataset(dry_run=None):
                return [S(question_id=100 + i, neutralised_question=f"Patient {i} has findings. Which is the diagnosis?", options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_answer_idx="ABCD"[i % 4]) for i in range(801)]
            def sample_cases(records, n, seed=42):
                return random.Random(seed).sample(records, n)
            '''))
        env = dict(os.environ, PYTHONPATH=t); script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_cases.py")
        out200, out20 = os.path.join(t, "o", "c200.jsonl"), os.path.join(t, "o", "c20.jsonl")
        for n, o in ((200, out200), (20, out20)):
            r = subprocess.run([sys.executable, script, "--out", o, "--n", str(n)], cwd=t, capture_output=True, text=True, env=env); assert r.returncode == 0, r.stderr
        a = [json.loads(l) for l in open(out200)]; b = [json.loads(l) for l in open(out20)]
        assert len(a) == 200 and len(b) == 20 and [x["case_id"] for x in b] == [x["case_id"] for x in a[:20]], "samples must be nested"
        assert a[0].keys() == {"case_id", "facts", "question", "options", "correct"} and a[0]["facts"].endswith("findings.") and a[0]["question"].endswith("diagnosis?")
    print("PASS: the export writes the five fields, strips the question stem for the facts, and the n=20 sample is the first 20 of the n=200 sample")


if __name__ == "__main__":
    test_export(); print("\nALL EXPORT TESTS PASSED")
