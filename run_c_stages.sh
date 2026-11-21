#!/bin/bash
# run_c_stages.sh -- stage C4 (reliability of every existing direction, with the integrity check) and the QUICK pass of
# stage C1 (bias direction on USMLE prompts: behavior check + direction + reliability, no steering sweep).
#
#   tmux new -d -s stages "bash $(pwd)/run_c_stages.sh"     # ~45-60 min on a 4090
#   DRY=1 bash run_c_stages.sh                               # print what it WOULD run, run nothing
#
# Resume-safe: a step whose output file already exists is skipped. Every step logs to run_logs/c_<name>.log.
# Run it only when the Phase 3 queue has finished (one model on the GPU at a time).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [ -z "$DRY" ]; then source .venv_mech/bin/activate || { echo "no .venv_mech here -- run: bash pipeline.sh setup"; exit 1; }; fi
export PYTHONUNBUFFERED=1
mkdir -p run_logs paper2_results/reliability
declare -A THR=(["Mistral-7B-Instruct-v0.1"]=0.5 ["Llama-3.1-8B-Instruct"]=0.8)   # same know-the-answer thresholds the Phase 3 queue used

run() {   # run <skip-if-this-file-exists> <log-name> <command...>
  local skip="$1" name="$2"; shift 2
  if [ -n "$skip" ] && [ -f "$skip" ]; then echo "[skip] $name"; return; fi
  echo "[run ] $name   $(date +%T)"
  if [ -n "$DRY" ]; then echo "         $*"; return; fi
  { echo; echo "===== $(date) :: $* ====="; } >> "run_logs/c_${name}.log"
  "$@" >> "run_logs/c_${name}.log" 2>&1 || { echo "[FAIL] $name -- see run_logs/c_${name}.log"; echo "$name" >> run_logs/_C_FAILED.txt; }
}

for M in Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct; do
  T=${THR[$M]}
  # C4: Phase 2 used 30 cases for SES and 100 for race and sex; Phase 3 used the queue's flags. The integrity check needs the SAME ones.
  run "paper2_results/reliability/bias_esi__${M}__ses.npz" "rel_biasesi_${M}_ses" python run_reliability.py --model "$M" --kind bias_esi --attribute ses --n-pairs 30
  for A in race sex; do
    run "paper2_results/reliability/bias_esi__${M}__${A}.npz" "rel_biasesi_${M}_${A}" python run_reliability.py --model "$M" --kind bias_esi --attribute "$A" --n-pairs 100
  done
  for C in user assistant; do
    run "paper2_results/reliability/def_usmle__${M}__consensus_${C}.npz" "rel_defusmle_${M}_${C}" python run_reliability.py --model "$M" --kind def_usmle --channel "$C" --n-pairs 100 --min-clean-p-correct "$T"
  done
  # C1, quick pass: does the demographic swap move USMLE answers more than rewording, and what is the direction's reliability?
  for A in ses race sex; do
    run "paper2_results/phase2usmle/metrics/phase2usmle_metrics_${M}_${A}.json" "biasusmle_${M}_${A}" python run_bias_usmle.py --model "$M" --attribute "$A" --min-clean-p-correct "$T" --skip-validation
  done
done
[ -z "$DRY" ] && echo "finished $(date)" > run_logs/_C_DONE.txt
echo "all steps processed. Failures (if any): $(cat run_logs/_C_FAILED.txt 2>/dev/null | tr '\n' ' ' || true)"
