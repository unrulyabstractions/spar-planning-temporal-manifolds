#!/bin/bash
# One-shot status of a rented Vast.ai instance, for the 10-minute budget check (CLAUDE.md, Compute). Prints a compact
# summary and ends with exactly one verdict line:
#   OK       work is running (GPU busy, or a capture/analysis/upload/download process alive and its log moving)
#   WARN ... something needs attention: FAILED in a chain/sequence log; nothing running (idle GPU we are paying for);
#            a capture log not written for > STALL_MIN minutes while its process is alive; disk < DISK_MIN_PCT % free
#   DONE     runs/sequence.log says SEQUENCE DONE (or a single chain says CHAIN DONE with nothing else running):
#            fetch/verify, delete the bucket key on the instance, then destroy the instance
# Exit status 0 for OK, 1 for WARN, 2 for DONE, 3 if the instance cannot be reached.
# Usage: scripts/vast_status.sh HOST PORT [REMOTE_REPO=/workspace/SPAR-2026]
set -u
HOST=${1:?HOST}; PORT=${2:?PORT}; REMOTE=${3:-/workspace/SPAR-2026}
STALL_MIN=${STALL_MIN:-15}; DISK_MIN_PCT=${DISK_MIN_PCT:-10}
out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 -p "$PORT" "root@$HOST" "STALL_MIN=$STALL_MIN DISK_MIN_PCT=$DISK_MIN_PCT REMOTE=$REMOTE bash -s" \
      < "$(dirname "$0")/vast_status_remote.sh" 2>/dev/null)
[ -z "$out" ] && { echo "VERDICT WARN unreachable: ssh to $HOST:$PORT failed"; exit 3; }
echo "$out"
case "$(echo "$out" | grep '^VERDICT' | awk '{print $2}')" in OK) exit 0;; DONE) exit 2;; *) exit 1;; esac
