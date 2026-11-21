#!/bin/bash
# progress.sh -- how far along is the run, and roughly how long is left. Read-only; safe to run any time.
# Timing is read from the HEADER LINE the queue writes at the start of every step in that step's own log
# ("===== <date> :: <command> ====="), so pauses, restarts and old killed runs cannot corrupt it.
cd "${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}" || { echo "project dir not found"; exit 1; }
MODELS=(Mistral-7B-Instruct-v0.1 Llama-3.1-8B-Instruct)
PRESSURES=(consensus)   # keep in sync with pipeline.sh
labels=(); files=(); est=(); grp=(); logs=()
add() { grp+=("$1"); labels+=("$2"); files+=("$3"); est+=("$4"); logs+=("$5"); }   # est = rough default minutes; logs = log name without .log
for m in "${MODELS[@]}"; do for a in race sex; do
  add P2 "Phase2 $m $a" "paper2_results/phase2/metrics/phase2_metrics_${m}_${a}.json" 26 "p2_${m}_${a}"; done; done
for m in "${MODELS[@]}"; do for pr in "${PRESSURES[@]}"; do for c in user assistant; do
  add P3 "Phase3 $m $pr $c" "paper2_results/phase3/metrics/phase3_metrics_${m}_${pr}_${c}.json" 20 "p3_${m}_${pr}_${c}"; done; done; done
for m in "${MODELS[@]}"; do for pr in "${PRESSURES[@]}"; do for c in user assistant; do
  add LOC "Localize $m $pr $c" "paper2_results/phase3/metrics/localization_${m}_${pr}_${c}.json" 4 "p3loc_${m}_${pr}_${c}"; done; done; done
for m in "${MODELS[@]}"; do for a in ses race sex; do for pr in "${PRESSURES[@]}"; do for c in user assistant; do
  add P4 "Phase4 $m $a $pr $c" "paper2_results/phase4/entanglement_${m}_${a}_${pr}_${c}.json" 0.2 "p4_${m}_${a}_${pr}_${c}"; done; done; done; done

step_start() {   # $1 = log name -> epoch seconds of the LAST header in that log, or nothing
  local f="run_logs/$1.log" h
  [ -f "$f" ] || return 0
  h=$(grep -a '^=====' "$f" | tail -1 | sed -E 's/^===== (.*) :: .*/\1/')
  [ -n "$h" ] && date -d "$h" +%s 2>/dev/null
  return 0
}

now=$(date +%s)
declare -A gdone gtotal gsum gcnt
done_n=0; cur=-1; total=${#labels[@]}
for i in "${!labels[@]}"; do
  g="${grp[$i]}"; gtotal[$g]=$(( ${gtotal[$g]:-0} + 1 ))
  if [ -f "${files[$i]}" ]; then
    done_n=$((done_n+1)); gdone[$g]=$(( ${gdone[$g]:-0} + 1 ))
    st=$(step_start "${logs[$i]}"); en=$(stat -c %Y "${files[$i]}")
    # a finished step's real duration = its output file's time minus its log header (ignore > 12 h: a paused/resumed step)
    if [ -n "$st" ] && [ "$en" -gt "$st" ] && [ $((en-st)) -lt 43200 ]; then
      gsum[$g]=$(( ${gsum[$g]:-0} + en - st )); gcnt[$g]=$(( ${gcnt[$g]:-0} + 1 ))
    fi
  elif [ "$cur" -lt 0 ]; then cur=$i; fi
done
gavg() { [ "${gcnt[$1]:-0}" -gt 0 ] && awk -v s="${gsum[$1]}" -v c="${gcnt[$1]}" 'BEGIN{print s/c/60}'; }

# the CURRENT step: elapsed from its own header; progress counted only AFTER its last header
elapsed=0; f=0; v=0; c=0; have_log=0
if [ "$cur" -ge 0 ]; then
  st=$(step_start "${logs[$cur]}")
  [ -n "$st" ] && elapsed=$(( (now-st)/60 ))
  logf="run_logs/${logs[$cur]}.log"
  if [ -f "$logf" ]; then
    have_log=1
    txt=$(awk '/^=====/{buf=""; next} {buf = buf $0 "\n"} END{printf "%s", buf}' "$logf")
    if [ "${grp[$cur]}" = P2 ] || [ "${grp[$cur]}" = P3 ]; then
      v=$(printf '%s' "$txt" | grep -cE "shift by coefficient|dP\(stated\) by coef"); c=$(printf '%s' "$txt" | grep -c "unclamped")
      f=$(awk -v v="$v" -v c="$c" 'BEGIN{print v/32*0.667 + c/32*0.333}')   # validation ~2/3 of the work, clamp test ~1/3
    fi
  fi
fi

rem=0
for i in "${!labels[@]}"; do
  [ -f "${files[$i]}" ] && continue
  g="${grp[$i]}"; avg=$(gavg "$g")
  e=${avg:-${est[$i]}}
  if [ "$i" -eq "$cur" ] && [ "$elapsed" -ge 0 ]; then
    if awk -v f="$f" 'BEGIN{exit !(f>=0.05)}'; then
      e=$(awk -v el="$elapsed" -v f="$f" 'BEGIN{r=el*(1-f)/f; if(r<1)r=1; print r}')
    else
      e=$(awk -v e="$e" -v el="$elapsed" 'BEGIN{r=e-el; if(r<1)r=1; print r}')
    fi
  fi
  rem=$(awk -v a="$rem" -v b="$e" 'BEGIN{print a+b}')
done

echo "=== $(date '+%a %H:%M') ==="
printf "Finished %d/%d steps   |   Phase2 %d/%d  Phase3 %d/%d  Localize %d/%d  Phase4 %d/%d\n" "$done_n" "$total" \
  "${gdone[P2]:-0}" "${gtotal[P2]}" "${gdone[P3]:-0}" "${gtotal[P3]}" "${gdone[LOC]:-0}" "${gtotal[LOC]}" "${gdone[P4]:-0}" "${gtotal[P4]}"
if [ "$cur" -ge 0 ]; then
  if pgrep -f '^python run_phase' >/dev/null; then stt="RUNNING"; else stt="NOT RUNNING (stopped or between steps)"; fi
  echo "Current step: ${labels[$cur]}  [$stt]"
  if [ "$have_log" = 1 ] && [ -n "$(step_start "${logs[$cur]}")" ]; then
    echo "  elapsed in this step: $elapsed min"
    echo "  inside the step: validation layers $v/32, clamp-test layers $c/32"
  else
    echo "  (this step has not written its log header yet -- not started, or still loading the model)"
  fi
  printf "Rough time left: ~%dh %02dm   (likely finish ~%s)\n" "$(awk -v r="$rem" 'BEGIN{print int(r/60)}')" \
    "$(awk -v r="$rem" 'BEGIN{print int(r%60)}')" "$(date -d "@$(awk -v n="$now" -v r="$rem" 'BEGIN{printf "%d", n+r*60}')" +%H:%M)"
  basis=""
  for g in P2 P3 LOC P4; do a=$(gavg "$g"); [ -n "$a" ] && basis="$basis $g=$(printf '%.0f' "$a")min(n=${gcnt[$g]})"; done
  echo "  (estimate basis -- measured average per step:${basis:- none yet; using my rough defaults for every group})"
  if [ "$elapsed" -gt 90 ] && awk -v f="$f" 'BEGIN{exit !(f<0.05)}'; then
    echo "!! this step has run $elapsed min with no visible log progress -- check the GPU line below and that it is RUNNING"
  fi
else
  echo "All steps have output files. $(cat run_logs/_DONE_MARKER.txt 2>/dev/null)"
fi
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo 'n/a')"
bad=$(grep -l "Traceback" run_logs/*.log 2>/dev/null)
[ -n "$bad" ] && echo "!! CRASHED logs (a step died and the queue moved on): $bad" || echo "Crashes: none"