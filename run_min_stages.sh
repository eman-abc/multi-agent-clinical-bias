#!/bin/bash
# run_min_stages.sh -- the quick C1 pass (behavior + direction + reliability) on the MINIMAL-PAIR notes, for every model
# and attribute. Outputs carry the tag _min, so nothing from the original C1 runs is overwritten.
#
#   python build_minimal_pairs.py                              # first: writes paper2_results/minimal_pairs.jsonl
#   tmux new -d -s minpairs "bash $(pwd)/run_min_stages.sh"   # ~20-30 min on a 4090
#   DRY=1 bash run_min_stages.sh                               # print what it WOULD run
#
# Resume-safe (skips steps whose metrics file exists). Logs: run_logs/min_<name>.log.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [ -z "$DRY" ]; then
  [ -f paper2_results/minimal_pairs.jsonl ] || { echo "missing paper2_results/minimal_pairs.jsonl -- run: python build_minimal_pairs.py"; exit 1; }
  source .venv_mech/bin/activate || { echo "no .venv_mech here"; exit 1; }
fi
export PYTHONUNBUFFERED=1
mkdir -p run_logs
declare -A THR=(["Mistral-7B-Instruct-v0.1"]=0.5 ["Llama-3.1-8B-Instruct"]=0.8)
for M in Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct; do
  for A in ses race sex; do
    out="paper2_results/phase2usmle/metrics/phase2usmle_metrics_${M}_${A}_min.json"
    name="${M}_${A}"
    if [ -f "$out" ]; then echo "[skip] $name"; continue; fi
    cmd=(python run_bias_usmle.py --model "$M" --attribute "$A" --source-jsonl paper2_results/minimal_pairs.jsonl
         --paraphrase-jsonl paper2_results/extracted_paper1_notes.jsonl --signal-type MIN --tag _min
         --min-clean-p-correct "${THR[$M]}" --skip-validation)
    echo "[run ] $name   $(date +%T)"
    if [ -n "$DRY" ]; then echo "         ${cmd[*]}"; continue; fi
    { echo; echo "===== $(date) :: ${cmd[*]} ====="; } >> "run_logs/min_${name}.log"
    "${cmd[@]}" >> "run_logs/min_${name}.log" 2>&1 || { echo "[FAIL] $name -- see run_logs/min_${name}.log"; echo "$name" >> run_logs/_MIN_FAILED.txt; }
  done
done
[ -z "$DRY" ] && echo "finished $(date)" > run_logs/_MIN_DONE.txt
echo "all steps processed. Failures: $(cat run_logs/_MIN_FAILED.txt 2>/dev/null | tr '\n' ' ')"
