"""Build the items and run the protocols on one model: choice -> state -> multiturn, all into RUN_DIR.

  python scripts/capture.py RUN_DIR --model Qwen/Qwen3-14B [--protocols choice,state,multiturn] [--smoke]
         [--batch 16] [--gen-batch 32] [--gpu-gib 6]
--smoke: a small slice of every protocol (local plumbing test).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import transformers  # noqa: E402

from phr import multiturn  # noqa: E402
from phr.items import VARIANTS_YAML, build_choice, build_state, check_config, load_config  # noqa: E402
from phr.model import default_layers, load  # noqa: E402
from phr.single import CHOICE_POSITIONS, run_choice, run_state, write_meta  # noqa: E402


def smoke_choice(df):
    return df[(df.config == "c3") & (df.domain == "investment")].reset_index(drop=True)


def smoke_specs(specs):
    keep = [s for s in specs if s.scenario == "fitness" and (s.level in (None, "6 months", "baby_8wk", "lease")) and s.sample < 2]
    ids = {s.conv_id for s in keep}
    return [s for s in keep if s.parent is None or s.parent in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--protocols", default="choice,state,multiturn")
    ap.add_argument("--batch", type=int, default=16, help="choice forward batch")
    ap.add_argument("--gen-batch", type=int, default=16, help="generation batch (state, multiturn, gencheck)")
    ap.add_argument("--gpu-gib", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = a.run_dir
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)   # noqa: E731

    cfg = load_config()
    errs = check_config(cfg)
    if errs:
        sys.exit("variants.yaml problems:\n  " + "\n  ".join(errs))
    (out / "variants.yaml").write_text(VARIANTS_YAML.read_text())          # the exact text used
    choice, state = build_choice(cfg), build_state(cfg)
    specs = multiturn.build_specs(cfg)
    if a.smoke:
        choice, state, specs = smoke_choice(choice), state.iloc[::8].reset_index(drop=True), smoke_specs(specs)
    choice.to_parquet(out / "items_choice.parquet")
    state.to_parquet(out / "items_state.parquet")
    log(f"items: choice {len(choice)}, state {len(state)}, multiturn {len(specs)} conversations")

    tok, model = load(a.model, gpu_gib=a.gpu_gib)
    n_layers = model.config.num_hidden_layers
    layers = default_layers(n_layers)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    write_meta(out, model=a.model, model_revision=getattr(model.config, "_commit_hash", None), git_commit=commit,
               torch=torch.__version__, transformers=transformers.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, n_layers=n_layers,
               d_model=model.config.hidden_size, layers=layers, choice_positions=CHOICE_POSITIONS,
               multiturn_positions=multiturn.POSITIONS, smoke=a.smoke, batch=a.batch, gen_batch=a.gen_batch)
    log(f"model {a.model}: {n_layers} layers, keeping {layers}")
    protos = a.protocols.split(",")
    t0 = time.time()
    if "choice" in protos:
        run_choice(choice, out, tok, model, layers, batch=a.batch, log=log)
        write_meta(out, choice_seconds=round(time.time() - t0))
    if "state" in protos:
        t = time.time()
        run_state(state, out, tok, model, batch=a.gen_batch, log=log)
        write_meta(out, state_seconds=round(time.time() - t))
    if "multiturn" in protos:
        t = time.time()
        multiturn.run(specs, cfg, out, tok, model, layers, batch=a.gen_batch, log=log)
        write_meta(out, multiturn_seconds=round(time.time() - t), multiturn_system=multiturn.SYSTEM,
                   multiturn_sampling=multiturn.SAMPLING, multiturn_max_new=multiturn.MAX_NEW)
    log(f"CAPTURE DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
