#!/bin/bash
# Instance-side half of scripts/vast_status.sh (run there through `ssh ... bash -s`; also runnable locally for tests).
# Env: REMOTE (repo dir), STALL_MIN, DISK_MIN_PCT; for tests, PROC_RE overrides the pattern for "our work" processes and
# GPU_UTIL the sampled GPU utilisation.
STALL_MIN=${STALL_MIN:-15}; DISK_MIN_PCT=${DISK_MIN_PCT:-10}; REMOTE=${REMOTE:-/workspace/SPAR-2026}
PROC_RE=${PROC_RE:-'scripts/.*\.(py|sh)|\.venv/bin/python|rclone|hf download'}
cd "$REMOTE" 2>/dev/null || { echo "VERDICT WARN repo $REMOTE missing"; exit; }
now=$(date +%s)
echo "time        $(date -u '+%F %T UTC')   tmux: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
# GPU_UTIL overrides the sample (tests); otherwise the max of 5 one-second samples
gpu=${GPU_UTIL:-$(for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 1; done | sort -n | tail -1)}
echo "gpu         max util over 5 s ${gpu}%, mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
free_pct=$(df --output=pcent . | tail -1 | tr -dc 0-9); free_pct=$((100 - free_pct))
echo "disk        $(df -h . | tail -1 | awk '{print $4" free of "$2}') (${free_pct}% free)"
# our work only: repo scripts and python, rclone, hf downloads (not Vast's own /opt/supervisor-scripts services)
procs=$(pgrep -af "$PROC_RE" | grep -v -E "pgrep|bash -s|/opt/supervisor-scripts|/opt/instance-tools" )
echo "processes   $(echo "$procs" | grep -c .) relevant: $(echo "$procs" | grep -oE "scripts/[a-z_0-9]+\.(py|sh)|rclone (copy|check|copyto|lsf)|hf download" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
for L in runs/chain_*.log runs/sequence.log; do [ -f "$L" ] && echo "$(basename $L | cut -c1-28)  $(grep -E '^20' "$L" | tail -1 | cut -c1-140)"; done
cap=$(ls -t runs/*_*.log 2>/dev/null | grep -v -E "analysis|chain_|reproduce_all|sequence|upload|verify|tests|hf_download|setup" | head -1)
if [ -n "$cap" ]; then age=$(( (now - $(stat -c %Y "$cap")) / 60 )); echo "capture     $(basename $cap): $(tail -1 "$cap" | cut -c1-90)  [log written ${age} min ago]"; fi
# failures since the latest (re)start of each chain / the sequence; earlier, already-handled failures are ignored
since_start() { awk -v pat="$2" '$0 ~ pat {buf=""} {buf = buf $0 "\n"} END {printf "%s", buf}' "$1"; }
fails=$( { for L in runs/chain_*.log; do [ -f "$L" ] && since_start "$L" "CHAIN START"; done
           [ -f runs/sequence.log ] && since_start runs/sequence.log "SEQUENCE START"; } | grep "FAILED" | tail -2)
if [ -n "$fails" ]; then echo "VERDICT WARN failure: $(echo "$fails" | tail -1 | cut -c1-160)"
elif grep -qs "SEQUENCE DONE" runs/sequence.log; then echo "VERDICT DONE sequence finished"
elif [ "$free_pct" -lt "$DISK_MIN_PCT" ]; then echo "VERDICT WARN disk ${free_pct}% free"
elif [ -z "$procs" ] && [ "${gpu:-0}" -lt 5 ]; then
  if grep -qs "CHAIN DONE" runs/chain_*.log && [ ! -f runs/sequence.log ]; then echo "VERDICT DONE chain finished, nothing running (no sequence)"
  else echo "VERDICT WARN idle: GPU ${gpu}% and no capture/analysis/upload process running"; fi
elif [ -n "$cap" ] && echo "$procs" | grep -q "capture.py" && [ "$age" -gt "$STALL_MIN" ]; then echo "VERDICT WARN stalled: capture log not written for ${age} min"
else echo "VERDICT OK"; fi
