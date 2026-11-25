# spar-planning-temporal-manifolds -- box verbs for this project.
# Run `just` with no argument to see this list. Get a box with
#
#   just box=probe up 2h   # rent + provision; 2h is the expected use, shown in its label
#
# or, for a box rented in the vast.ai console, paste its ssh line:
#
#   just box=probe new ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080
#
# Then name the box on every verb: `just box=probe push`, `just box=probe ssh
# "..."`. With no box= a verb uses the only box there is (up/new default to
# "main") and refuses when several are up. `just status` lists every box.
#
# The verbs are the SPAR root verbs (SPAR/justfile), run through the same
# SPAR/infra/vast.sh, except `push`, which ships this checkout.
#
# Verbs not defined here (up, label, search, doctor, vastai, adopt, services,
# test) fall through to SPAR/justfile and run from SPAR/, so they work from
# jared/ and from worktrees too. `just --list` shows only this file's verbs.

set fallback := true

# What `push` leaves home: every dotfile and dot-directory (.git, .env, .claude, .venv, shell rc files),
# caches and results, local smoke outputs, and the sibling projects (alan/, relevance_aware_decoder/,
# PromptDefinitions/), which jared/ does not import at run time.
push_excludes := "--exclude '.*' --exclude '__pycache__' --exclude 'results' --exclude '/jared/out/' --exclude '/alan/' --exclude '/relevance_aware_decoder/' --exclude '/PromptDefinitions/'"

# SPAR/ is the nearest ancestor holding infra/vast.sh: the parent of the main
# checkout, or several levels up from a worktree under .claude/worktrees/.
# The box a verb acts on; passed through to SPAR/justfile on fallback too.
box := ""
spar := shell('d="$1"; while [ "$d" != / ] && [ ! -f "$d/infra/vast.sh" ]; do d="$(dirname "$d")"; done; echo "$d"', justfile_directory())
vast := "BOX=" + quote(box) + " bash " + quote(spar / "infra" / "vast.sh")
# Where this project lives on the box, whichever checkout or worktree ships it.
box_project := "/workspace/SPAR/spar-planning-temporal-manifolds"

default:
    @just --list

# --- connection ----------------------------------------------------------

# Use the box from a pasted vast.ai ssh line; also tracks its instance for status/stop/down.
new +LINK:
    @{{vast}} connect {{LINK}}

# Shell on the box, or run a command there: just ssh "nvidia-smi"
ssh *ARGS:
    @{{vast}} ssh {{quote(ARGS)}}

# Forward the box's Jupyter to a local port (PORT, default 8888). Foreground; ctrl-C to close.
tunnel PORT="":
    @{{vast}} tunnel {{PORT}}

# Ship this checkout to /workspace/SPAR/spar-planning-temporal-manifolds/ on the
# box, then install jared/requirements.txt into the image's python. Local
# outputs (out/) stay home.
push:
    #!/usr/bin/env bash
    set -euo pipefail
    read -r host port <<<"$({{vast}} hostport)"
    rsh="$({{vast}} rsh)"
    rsync -avz --mkpath {{push_excludes}} \
      -e "$rsh" {{quote(justfile_directory() + "/")}} "root@$host:{{box_project}}/"
    echo ">> installing jared/requirements.txt on the box"
    # Pin the image's torch so nothing in requirements can upgrade it and break CUDA
    # (as provision.sh does). A no-op in seconds when everything is already installed.
    {{vast}} ssh 'cd {{box_project}}/jared && \
      /venv/main/bin/python -m pip install -q -r requirements.txt \
        "torch==$(/venv/main/bin/python -c "import torch; print(torch.__version__)")"'

# Preview what `just push` would send, without sending it.
push-dry:
    #!/usr/bin/env bash
    set -euo pipefail
    read -r host port <<<"$({{vast}} hostport)"
    rsh="$({{vast}} rsh)"
    rsync -avzn --stats --mkpath {{push_excludes}} \
      -e "$rsh" {{quote(justfile_directory() + "/")}} "root@$host:{{box_project}}/" | tail -n 15

# Pull the box's /workspace/results/ into SPAR/results/<box>/. `just box=all sync` does every box.
sync:
    @{{vast}} sync

# Set up the box: base packages, Jupyter and the idle watchdog (../infra/provision.sh).
provision:
    @{{vast}} provision

# --- money ---------------------------------------------------------------

# Every box's state and dollars spent (or only the one named with box=).
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
