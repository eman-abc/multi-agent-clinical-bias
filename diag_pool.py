"""diag_pool.py -- read-only look at what is really inside paper2_results/extracted_paper1_notes.jsonl.
Run from the project folder:  python3 diag_pool.py
"""
import collections
import glob
import json
import os

PATH = "paper2_results/extracted_paper1_notes.jsonl"
rows = [json.loads(line) for line in open(PATH) if line.strip()]
print("total notes in pool:", len(rows))
print("keys in a record:", sorted(rows[0].keys()))


def sig(r):
    return (r.get("condition_label"), r.get("attribute"), r.get("signal_type"), r.get("rec_type"))


def is_rec(r):
    return r.get("rec_type") is not None or "REC" in str(r.get("condition_label"))


print("\n(condition_label, attribute, signal_type, rec_type) -> count   [top 40]")
for k, v in collections.Counter(sig(r) for r in rows).most_common(40):
    print(f"  {k}  ->  {v}")

print("\nsource files contributing:", dict(collections.Counter(r.get("source_file") for r in rows)))

# Which recommendation-bearing notes could be paired with a plain N note of the SAME case?
n_cases = set(r["case_id"] for r in rows if r.get("condition_label") == "N" and not is_rec(r))
print("\ncases that have a plain N note (no recommendation):", len(n_cases))
combo, combo_with_n = collections.Counter(), collections.Counter()
for r in rows:
    if is_rec(r):
        combo[sig(r)] += 1
        if r["case_id"] in n_cases:
            combo_with_n[sig(r)] += 1
if not combo:
    print("\n(NO note in the pool carries a recommendation marker at all)")
else:
    print("\nrecommendation-bearing notes:  total / of which the same case also has a plain N note")
    for k, v in combo.most_common(30):
        print(f"  {k}: {v} / {combo_with_n[k]}")

# how do recommendation-bearing notes actually end? (one example per distinct combo, max 4)
seen = set()
for r in rows:
    if is_rec(r) and sig(r) not in seen:
        seen.add(sig(r))
        print("\n--- example ending:", sig(r))
        print(repr(r["note_text"][-260:]))
        if len(seen) == 4:
            break

print("\nextracted_notes files:")
for p in sorted(glob.glob("paper2_results/extracted_notes/*.jsonl")):
    print("  ", os.path.basename(p))
