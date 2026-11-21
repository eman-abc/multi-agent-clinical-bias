#!/bin/bash
# run_final_stages.sh -- the confirmatory runs, in order, from the FROZEN plan.
#
#   python freeze_plan.py --freeze                                  # once, first (see FINAL_STAGES.md)
#   tmux new -d -s final "bash $(pwd)/run_final_stages.sh"          # about 2.5 hours on a 4090
#   DRY=1 bash run_final_stages.sh                                  # print what it WOULD run
#
# 1. attention-vs-MLP window patching: Mistral and Llama x user and assistant channel  (a few minutes each)
# 2. cross-fitting:                    the same four                                    (about 30 minutes each)
# 3. the two analyses (CPU), written to paper2_results/*/REPORT.txt
# Resume-safe: finished runs are skipped, and an interrupted cross-fit run resumes at its first unfinished layer.
# Refuses to start if the plan is not frozen or has been modified (ALLOW_UNFROZEN=1 overrides, and every output is then labelled).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
PLAN=analysis_plan.json
if [ -z "$DRY" ]; then
  source .venv_mech/bin/activate || { echo "no .venv_mech here"; exit 1; }
fi
export PYTHONUNBUFFERED=1
mkdir -p run_logs paper2_results/crossfit paper2_results/attn_mlp

if [ -z "$DRY" ]; then
  python freeze_plan.py --verify
  if [ $? -ne 0 ] && [ -z "$ALLOW_UNFROZEN" ]; then
    echo "The plan is not frozen or was modified. Freeze it (python freeze_plan.py --freeze) or, knowing the results will be labelled exploratory, run with ALLOW_UNFROZEN=1."
    exit 1
  fi
  EXTRA=""; [ -n "$ALLOW_UNFROZEN" ] && EXTRA="--allow-modified-plan"
else
  EXTRA=""
fi

MODELS=(Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct)
CHANNELS=(user assistant)
fail() { echo "[FAIL] $1 -- see run_logs/final_$1.log"; echo "$1" >> run_logs/_FINAL_FAILED.txt; }

complete_crossfit() { python - "$1" << 'EOF'
import sys, numpy as np
try:
    z = np.load(sys.argv[1]); sys.exit(0 if bool(z["done"].all()) else 1)
except Exception:
    sys.exit(1)
EOF
}

for stage in attn_mlp crossfit; do
  for M in "${MODELS[@]}"; do
    for C in "${CHANNELS[@]}"; do
      name="${stage}_${M}_${C}"
      if [ "$stage" = "attn_mlp" ]; then
        out="paper2_results/attn_mlp/attn_mlp_${M}_consensus_${C}.npz"
        cmd=(python run_attn_mlp_split.py --model "$M" --channel "$C" --plan $PLAN $EXTRA)
        if [ -f "$out" ]; then echo "[skip] $name"; continue; fi
      else
        out="paper2_results/crossfit/crossfit_${M}_consensus_${C}.npz"
        cmd=(python run_crossfit.py --model "$M" --channel "$C" --plan $PLAN $EXTRA)
        if [ -z "$DRY" ] && [ -f "$out" ] && complete_crossfit "$out"; then echo "[skip] $name"; continue; fi
      fi
      echo "[run ] $name   $(date +%T)"
      if [ -n "$DRY" ]; then echo "         ${cmd[*]}"; continue; fi
      { echo; echo "===== $(date) :: ${cmd[*]} ====="; } >> "run_logs/final_${name}.log"
      "${cmd[@]}" >> "run_logs/final_${name}.log" 2>&1 || fail "$name"
    done
  done
done

if [ -z "$DRY" ]; then
  python analyze_attn_mlp.py > paper2_results/attn_mlp/REPORT.txt 2>&1
  python analyze_crossfit.py > paper2_results/crossfit/REPORT.txt 2>&1
  echo "finished $(date)" > run_logs/_FINAL_DONE.txt
  echo "reports: paper2_results/attn_mlp/REPORT.txt and paper2_results/crossfit/REPORT.txt"
fi
echo "all steps processed. Failures: $(cat run_logs/_FINAL_FAILED.txt 2>/dev/null | tr '\n' ' ')"
