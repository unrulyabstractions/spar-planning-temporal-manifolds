#!/usr/bin/env python
"""Replication manifest for a pipeline run: everything needed to reproduce or audit it later.

Usage: manifest.py MANIFEST.json --stage start|captured|done [--prompts P.parquet] [--model M] [--status ok|"failed at ..."]
  start    : code (git commit, dirty files), prompt file sha256 + dataset hash, software (pip freeze, python, torch,
             CUDA), hardware (GPUs, driver, hostname), pipeline knobs (env: MODEL BATCH SEED LAYERS ...), start time
  captured : model revision actually used (the HF cache snapshot sha, resolved offline) + capture end time
  done     : end time and status (ok, or the failed step)
Stages merge into the same file, so a crashed run still keeps what was recorded before the crash.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rad  # noqa: E402,F401  (puts ../alan on sys.path)

KNOBS = ["PY", "MODEL", "BATCH", "SEED", "LAYERS", "SUBSET_LAYERS", "SMOKE", "OVERWRITE", "EXTRA_CAPTURE_ARGS",
         "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "CUDA_VISIBLE_DEVICES"]


def sh(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def start(a) -> dict:
    import pandas as pd
    import torch
    from rad.build import dataset_hash
    repo = Path(__file__).resolve().parents[1]
    dirty = sh("git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no")
    return dict(
        started=dt.datetime.now().isoformat(timespec="seconds"),
        git=dict(commit=sh("git", "-C", str(repo), "rev-parse", "HEAD"), branch=sh("git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"),
                 dirty=bool(dirty), dirty_files=dirty.splitlines() if dirty else []),
        prompts=dict(file=a.prompts, sha256=sha256(Path(a.prompts)), dataset_hash=dataset_hash(pd.read_parquet(a.prompts)),
                     n=len(pd.read_parquet(a.prompts, columns=["sample_uid"]))),
        model=dict(name=a.model),
        software=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), pip_freeze=(sh(sys.executable, "-m", "pip", "freeze") or "").splitlines()),
        hardware=dict(hostname=socket.gethostname(), cpu_count=os.cpu_count(),
                      gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                      gpu_mem_gib=[round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(torch.cuda.device_count())],
                      nvidia_smi=sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")),
        env={k: os.environ.get(k) for k in KNOBS},
    )


def captured(a, m: dict) -> dict:
    from huggingface_hub import snapshot_download
    name = a.model or m.get("model", {}).get("name")
    try:
        path = snapshot_download(name, local_files_only=True)       # the cached snapshot transformers loaded
        rev = dict(name=name, revision=Path(path).name, snapshot_path=path)
    except Exception as e:
        rev = dict(name=name, revision=None, error=repr(e))
    return dict(model=rev, captured=dt.datetime.now().isoformat(timespec="seconds"))


ap = argparse.ArgumentParser()
ap.add_argument("manifest", type=Path); ap.add_argument("--stage", required=True, choices=["start", "captured", "done"])
ap.add_argument("--prompts"); ap.add_argument("--model"); ap.add_argument("--status")
a = ap.parse_args()
m = json.loads(a.manifest.read_text()) if a.manifest.exists() and a.stage != "start" else {}
if a.stage == "start":
    m.update(start(a))
elif a.stage == "captured":
    m.update(captured(a, m))
else:
    m["finished"] = dt.datetime.now().isoformat(timespec="seconds"); m["status"] = a.status
a.manifest.parent.mkdir(parents=True, exist_ok=True)
a.manifest.write_text(json.dumps(m, indent=2))
print(f"manifest {a.manifest}: stage {a.stage}")
