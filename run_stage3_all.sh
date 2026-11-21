#!/usr/bin/env bash
# run_stage3_all.sh -- every stage-3 collection, both models, in order, with a log per step. Run from the project folder on the VM:  nohup bash run_stage3_all.sh > run_logs/stage3_all.log 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p run_logs stage3/acts
NAT_PATH=$(python -c 'import json,os; p="stage3/natural_config.json"; print(json.load(open(p)).get("path","") if os.path.exists(p) else "")')
MODELS="${MODELS:-Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct}"          # to split the work between two machines: MODELS=Mistral-7B-Instruct-v0.1 bash run_stage3_all.sh
for M in $MODELS; do
  for T in usmle esi; do
    echo "== $(date '+%H:%M:%S')  collect_stage3 $M $T"
    python collect_stage3.py --model "$M" --task "$T" > "run_logs/stage3_${T}_${M}.log" 2>&1 || echo "!! FAILED: collect_stage3 $M $T (see run_logs/stage3_${T}_${M}.log)"
  done
  if [ -n "$NAT_PATH" ] && [ -f "$NAT_PATH" ]; then
    echo "== $(date '+%H:%M:%S')  natural_notes collect $M"
    python natural_notes.py collect --model "$M" > "run_logs/stage3_natural_${M}.log" 2>&1 || echo "!! FAILED: natural_notes collect $M"
  else
    echo "== natural notes skipped: set \"path\" in stage3/natural_config.json to the Luna file and copy the file to the VM"
  fi
done
echo "ALL DONE $(date)"
