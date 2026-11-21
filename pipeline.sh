#!/bin/bash
# pipeline.sh -- check on, pause, persist, and resume the Paper 2 run.
#
#   bash pipeline.sh status      progress dashboard (read-only, safe any time)
#   bash pipeline.sh pause       let the CURRENT step finish, then halt the queue
#   bash pipeline.sh pause-now   halt immediately (current step restarts from scratch later)
#   bash pipeline.sh persist     copy project (no venv) into /workspace and verify the copy
#   bash pipeline.sh setup       (re)build .venv_mech and sanity-check GPU + imports
#   bash pipeline.sh run         run every step that is not already finished (resume-safe)
#
# A step counts as FINISHED when its metrics JSON exists. Each run_phase*.py writes that
# file as its very last action, so a half-finished step can never look finished.
#
# Works from wherever it lives (home dir or /workspace).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WORKSPACE_DIR:-/workspace}"
cd "$ROOT" || exit 1
mkdir -p run_logs

MODELS=(Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct)
ATTRS=(race sex)              # ses was already run at n=30 on both models
CHANNELS=(user assistant)
PRESSURES=(consensus)         # add "single" to also run the one-sentence pressure (cheap; useful as a robustness check)
# Know-the-answer threshold per model. Llama: 0.8 as in Kumarappan & Mujoo (48 of 140 cases pass). Mistral rarely reaches
# 0.8 on these questions (18 of 140), so it gets 0.5 -- still "top answer is the correct one with majority probability".
declare -A MINP=(["Mistral-7B-Instruct-v0.1"]=0.5 ["Llama-3.1-8B-Instruct"]=0.8)

STEP_LABELS=(); STEP_DONE=(); STEP_CMDS=()
add_step() { STEP_LABELS+=("$1"); STEP_DONE+=("$2"); STEP_CMDS+=("$3"); }

for m in "${MODELS[@]}"; do
  for a in "${ATTRS[@]}"; do
    add_step "p2_${m}_${a}" "paper2_results/phase2/metrics/phase2_metrics_${m}_${a}.json" \
      "python run_phase2_bias_sensor.py --model $m --attribute $a --n-pairs 100 --test-clamp"
  done
done
for m in "${MODELS[@]}"; do
  for pr in "${PRESSURES[@]}"; do
    for c in "${CHANNELS[@]}"; do
      add_step "p3_${m}_${pr}_${c}" "paper2_results/phase3/metrics/phase3_metrics_${m}_${pr}_${c}.json" \
        "python run_phase3_deference_sensor.py --model $m --channel $c --pressure $pr --n-pairs 100 --min-clean-p-correct ${MINP[$m]} --test-clamp"
    done
  done
done
for m in "${MODELS[@]}"; do
  for pr in "${PRESSURES[@]}"; do
    for c in "${CHANNELS[@]}"; do
      add_step "p3loc_${m}_${pr}_${c}" "paper2_results/phase3/metrics/localization_${m}_${pr}_${c}.json" \
        "python run_phase3_localization_patching.py --model $m --channel $c --pressure $pr --n-pairs 60 --min-clean-p-correct ${MINP[$m]}"
    done
  done
done
for m in "${MODELS[@]}"; do
  for a in ses race sex; do
    for pr in "${PRESSURES[@]}"; do
      for c in "${CHANNELS[@]}"; do
        add_step "p4_${m}_${a}_${pr}_${c}" "paper2_results/phase4/entanglement_${m}_${a}_${pr}_${c}.json" \
          "python run_phase4_entanglement.py --model $m --attribute $a --pressure $pr --channel $c"
      done
    done
  done
done

# ---------------------------------------------------------------- helpers

run_active() { pgrep -f '^python run_phase' >/dev/null 2>&1; }

tree_stats() {   # "N files, B bytes" for a directory
  [ -d "$1" ] || { echo "(missing)"; return; }
  find "$1" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%d files, %d bytes\n", NR, s}'
}

halt_all() {
  tmux kill-session -t mech 2>/dev/null
  pkill -f '^bash .*run_everything.sh' 2>/dev/null
  pkill -f '^bash .*pipeline.sh run' 2>/dev/null
  pkill -f '^python run_phase' 2>/dev/null
  sleep 2
  if run_active; then pkill -9 -f '^python run_phase' 2>/dev/null; sleep 1; fi
}

# ---------------------------------------------------------------- status

do_status() {
  echo "=== $(date) | project: $ROOT ==="
  case "$ROOT" in
    "$WS"/*) echo "Location: PERSISTENT ($WS) -- survives a pod Stop." ;;
    *)       echo "Location: CONTAINER DISK -- a pod Stop would ERASE this. Run 'persist' + back up first." ;;
  esac

  echo; echo "--- running now ---"
  if run_active; then
    for p in $(pgrep -f '^python run_phase'); do
      printf "  pid %s  elapsed %s  %s\n" "$p" "$(ps -o etime= -p "$p" | tr -d ' ')" "$(ps -ww -o args= -p "$p")"
    done
  else
    echo "(no step running)"
  fi

  echo; echo "--- steps ---"
  n_done=0; total=${#STEP_LABELS[@]}
  for i in "${!STEP_LABELS[@]}"; do
    if [ -f "${STEP_DONE[$i]}" ]; then st="done   "; n_done=$((n_done+1))
    elif ps -eww -o args= | grep -Fxq -- "${STEP_CMDS[$i]}"; then st="RUNNING"
    else st="pending"; fi
    printf "  [%s] %s\n" "$st" "${STEP_LABELS[$i]}"
  done
  echo "  => $n_done / $total steps finished"

  echo; echo "--- newest log (last lines) ---"
  newest=$(ls -t run_logs/*.log 2>/dev/null | head -1)
  if [ -n "$newest" ]; then echo "$newest"; tail -n 2 "$newest" | cut -c1-200; else echo "(no logs yet)"; fi

  echo; echo "--- logs containing a Python traceback (= a step crashed) ---"
  grep -l "Traceback" run_logs/*.log 2>/dev/null || echo "(none)"

  echo; echo "--- markers ---"
  for f in _DONE_MARKER.txt _PAUSED.txt _FAILED.txt; do
    [ -f "run_logs/$f" ] && { echo "run_logs/$f:"; sed 's/^/    /' "run_logs/$f"; }
  done
  ls run_logs/_*.txt >/dev/null 2>&1 || echo "(none)"

  echo; echo "--- GPU ---"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

  echo; echo "--- numbers to compare against your laptop copy ---"
  echo "  paper2_results: $(tree_stats paper2_results)"
  echo "  run_logs:       $(tree_stats run_logs)"

  echo; echo "--- disk ---"
  df -h / "$WS" 2>/dev/null
}

# ---------------------------------------------------------------- pause

do_pause() {
  pid=$(pgrep -f '^python run_phase' | head -1)
  frozen=""
  if [ -z "$pid" ]; then
    echo "[pause] no step is running right now."
  else
    echo "[pause] current step: $(ps -ww -o args= -p "$pid")"
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    parent_cmd=$(ps -ww -o args= -p "$ppid" 2>/dev/null)
    case "$parent_cmd" in
      *run_everything.sh*|*pipeline.sh*run*)
        # Freeze the QUEUE (parent shell) so it cannot launch another step, while the
        # current step keeps running to completion. No race, nothing half-started.
        kill -STOP "$ppid"
        frozen="$ppid"
        echo "[pause] queue frozen (pid $ppid): it cannot start another step."
        ;;
      *)
        echo "[pause] WARNING: could not identify the queue process (parent: ${parent_cmd:-unknown})."
        echo "[pause] will wait for this step, then halt everything."
        ;;
    esac
    echo "[pause] waiting for the current step to finish (can be up to ~50 min)..."
    # A finished child of a FROZEN parent cannot be reaped, so it lingers as a zombie ("Z")
    # and `kill -0` keeps succeeding on it forever. Treat a zombie as finished.
    while kill -0 "$pid" 2>/dev/null; do
      state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ')
      case "$state" in Z*|"") break ;; esac
      sleep 2
    done
    echo "[pause] step finished at $(date +%T)."
    [ -n "$frozen" ] && kill -KILL "$frozen" 2>/dev/null
  fi
  halt_all
  echo "PAUSED $(date)" > run_logs/_PAUSED.txt
  echo "[pause] queue halted. Safe to persist / back up / stop the pod."
}

do_pause_now() {
  echo "[pause-now] halting immediately -- the interrupted step will restart from scratch on resume."
  halt_all
  echo "PAUSED-NOW $(date)" > run_logs/_PAUSED.txt
  echo "[pause-now] halted."
}

# ---------------------------------------------------------------- persist

do_persist() {
  if run_active && [ "$1" != "--force" ]; then
    echo "A step is still running -- run 'pause' first (or pass --force)."; exit 1
  fi
  [ -d "$WS" ] || { echo "No $WS on this pod -- nothing to persist to. Back up to your laptop instead."; exit 1; }
  DEST="$WS/$(basename "$ROOT")"
  if [ "$ROOT" = "$DEST" ]; then
    echo "Project already lives in $WS (persistent)."
  else
    echo "Copying $ROOT -> $DEST (excluding venv / caches)..."
    mkdir -p "$DEST"
    ( set -o pipefail
      tar --exclude='.venv_mech' --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
          -cf - -C "$(dirname "$ROOT")" "$(basename "$ROOT")" | tar -xf - -C "$WS" ) \
      || { echo "COPY FAILED"; exit 1; }
  fi
  echo; echo "--- verification: source vs copy ---"
  ok=1
  for d in paper2_results run_logs; do
    a=$(tree_stats "$ROOT/$d"); b=$(tree_stats "$DEST/$d")
    echo "  $d: $a   |   copy: $b"
    [ "$a" = "$b" ] || ok=0
  done
  if [ "$ok" = 1 ]; then echo "COPY VERIFIED"; else echo "MISMATCH -- do NOT trust the copy"; exit 1; fi
  echo; echo "--- space on $WS ---"; df -h "$WS" | tail -1
  du -sh "$ROOT/.venv_mech" 2>/dev/null | sed 's/^/  venv size: /'
}

# ---------------------------------------------------------------- setup

do_setup() {
  [ -d .venv_mech ] || python3 -m venv .venv_mech
  # shellcheck disable=SC1091
  source .venv_mech/bin/activate || exit 1
  pip install --upgrade pip -q
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "GPU: ${gpu:-none detected}"
  if echo "$gpu" | grep -qiE "blackwell|RTX 50|B100|B200|RTX PRO"; then
    echo "Blackwell-class GPU -> plain torch install"; pip install torch
  else
    echo "Standard GPU -> cu121 torch build"; pip install torch --index-url https://download.pytorch.org/whl/cu121
  fi
  pip install "transformer_lens==2.9.0" "transformers==4.46.1" accelerate einops jaxtyping \
      huggingface_hub typeguard python-dotenv openai datasets pandas hf_transfer
  python -c "import torch; print('cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
  python -c "from transformer_lens import HookedTransformer; print('transformer_lens import OK')"
  python test_chat_prompting_offline.py | tail -2
}

# ---------------------------------------------------------------- run (resume-safe)

do_run() {
  if run_active; then echo "A step is already running -- use 'status', or 'pause' first."; exit 1; fi
  [ -n "$HF_TOKEN" ] || { echo "export HF_TOKEN=<your token> first (Llama is gated)."; exit 1; }
  # shellcheck disable=SC1091
  source .venv_mech/bin/activate || { echo "no .venv_mech here -- run 'setup' first."; exit 1; }
  rm -f run_logs/_FAILED.txt run_logs/_PAUSED.txt run_logs/_DONE_MARKER.txt
  export PYTHONUNBUFFERED=1     # live log lines (otherwise Python buffers ~8KB and progress looks frozen)
  total=${#STEP_LABELS[@]}
  for i in "${!STEP_LABELS[@]}"; do
    label="${STEP_LABELS[$i]}"; donef="${STEP_DONE[$i]}"; cmd="${STEP_CMDS[$i]}"
    if [ -f "$donef" ]; then echo "[skip] ($((i+1))/$total) $label"; continue; fi
    echo "[run ] ($((i+1))/$total) $label   $(date +%T)"
    { echo; echo "===== $(date) :: $cmd ====="; } >> "run_logs/${label}.log"
    $cmd >> "run_logs/${label}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$donef" ]; then
      echo "[FAIL] $label (exit $rc) -- see run_logs/${label}.log"
      echo "$(date +%T) $label exit=$rc" >> run_logs/_FAILED.txt
    fi
  done
  if [ -f run_logs/_FAILED.txt ]; then
    echo "FINISHED WITH FAILURES $(date) -- see _FAILED.txt" > run_logs/_DONE_MARKER.txt
  else
    echo "ALL STEPS COMPLETE $(date)" > run_logs/_DONE_MARKER.txt
  fi
  echo "[run ] finished: $(cat run_logs/_DONE_MARKER.txt)"
}

case "$1" in
  status)     do_status ;;
  pause)      do_pause ;;
  pause-now)  do_pause_now ;;
  persist)    do_persist "$2" ;;
  setup)      do_setup ;;
  run)        do_run ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" ;;
esac