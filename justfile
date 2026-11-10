# spar-planning-temporal-manifolds -- box verbs for this project.
# Run `just` with no argument to see this list. Start every session with
#
#   just new ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080
#
# pasting the ssh line from the vast.ai console (quotes optional). Every other
# verb then uses that box. They are the SPAR root verbs (../justfile), run
# through the same ../infra/vast.sh, except `push`, which ships this project.

vast := "bash " + quote(justfile_directory() / ".." / "infra" / "vast.sh")
spar := justfile_directory() / ".."
project := file_name(justfile_directory())

default:
    @just --list

# --- connection ----------------------------------------------------------

# Use the box from a pasted vast.ai ssh line; also tracks its instance for status/stop/down.
new +LINK:
    @{{vast}} connect {{LINK}}

# Shell on the box, or run a command there: just ssh "nvidia-smi"
ssh *ARGS:
    @{{vast}} ssh {{quote(ARGS)}}

# Forward the box's Jupyter to a local port. Foreground; ctrl-C to close.
tunnel:
    @{{vast}} tunnel

# Ship this project to /workspace/SPAR/ on the box. Local outputs (out/) stay home.
push:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{quote(spar)}}
    read -r host port <<<"$({{vast}} hostport)"
    rsync -avzR --exclude '.venv' --exclude '__pycache__' --exclude 'results' --exclude 'jared/out/' \
      --exclude '.env' --exclude '.git' --exclude '.state' --exclude '.claude' \
      --exclude '.ipynb_checkpoints' \
      -e "ssh -p $port" {{quote(project)}} "root@$host:/workspace/SPAR/"

# Preview what `just push` would send, without sending it.
push-dry:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{quote(spar)}}
    read -r host port <<<"$({{vast}} hostport)"
    rsync -avzRn --stats --exclude '.venv' --exclude '__pycache__' --exclude 'results' --exclude 'jared/out/' \
      --exclude '.env' --exclude '.git' --exclude '.state' --exclude '.claude' \
      --exclude '.ipynb_checkpoints' \
      -e "ssh -p $port" {{quote(project)}} "root@$host:/workspace/SPAR/" | tail -n 15

# Pull the box's /workspace/results/ into SPAR/results/.
sync:
    @{{vast}} sync

# Set up the box: base packages, Jupyter and the idle watchdog (../infra/provision.sh).
provision:
    @{{vast}} provision

# --- money ---------------------------------------------------------------

# Instance state and dollars spent.
status:
    @{{vast}} status

# Sync results/, then stop without destroying (GPU released, disk kept).
stop:
    @{{vast}} stop

# Start a stopped instance again. Its port may change: paste the new line with `just new` if ssh fails.
resume:
    @{{vast}} resume

# Sync results/, then destroy the instance. `just down --force` skips the sync.
down *ARGS:
    @{{vast}} down {{ARGS}}

# Total spend against the budget in ../.env (local estimate).
burn:
    @{{vast}} burn

# Real vast.ai account balance.
balance:
    @{{vast}} balance

# Keep the idle watchdog from stopping the box during a long break.
hold:
    @{{vast}} ssh "touch /workspace/.nokill" && echo "watchdog held -- 'just unhold' to re-arm"

# Re-arm the idle watchdog.
unhold:
    @{{vast}} ssh "rm -f /workspace/.nokill" && echo "watchdog re-armed"
