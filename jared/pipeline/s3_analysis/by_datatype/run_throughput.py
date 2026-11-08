"""Measured throughput of earlier box runs, and prompt token counts (big_experiment_plan.md, Steps 1-2).

Reads saved results only, runs no model. The tokenizer part needs transformers:
    ~/miniconda3/envs/mp/bin/python analysis/run_throughput.py ../../results/horizon
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spar_horizon.banks import build_records  # noqa: E402
from spar_horizon.config import Settings  # noqa: E402
from spar_horizon.model_io import chat_text  # noqa: E402

STEERING_INPUTS = Path(__file__).resolve().parents[1] / "steering"

# GPU type as recorded (queue logs print the device from 2026-10-25 on; earlier runs are named in the docs)
EXP8_RUNS = {  # run: (GPU, model)
    "exp8_Qwen3-8B_full": ("4090 (08_results)", "8B"),
    "exp8_Qwen3-8B_answer_slot": ("4090 (08_results)", "8B"),
    "exp8_Qwen3-8B_decision_L24": ("4090 (08_results)", "8B"),
    "exp8_Qwen3-8B_decision_L22": ("2-GPU box, not recorded", "8B"),
    "exp8_Qwen3-14B_answer_slot_off": ("2-GPU box, not recorded", "14B"),
    "exp8_Qwen3-14B_answer_slot_on": ("2-GPU box, not recorded", "14B"),
    "exp8_Qwen3-14B_line_off": ("2-GPU box, not recorded", "14B"),
    "exp8_Qwen3-14B_line_on": ("2-GPU box, not recorded", "14B"),
    "exp8_Qwen3-8B_pristine": ("RTX 5090 (queue log)", "8B"),
}
# runs that used the prefix-cache path (steering.score_last_edits) for last-position cells
PREFIX_CACHE_RUNS = {"exp8_Qwen3-14B_line_off", "exp8_Qwen3-14B_line_on", "exp8_Qwen3-8B_pristine"}


def stats(x):
    x = np.asarray(x)
    return f"mean {x.mean():.1f}  median {np.median(x):.0f}  min {x.min()}  max {x.max()}"


def token_counts():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    for bank in ("full", "fill", "units"):
        on = Settings(model="Qwen/Qwen3-8B", thinking="on", bank=bank)
        off = Settings(model="Qwen/Qwen3-8B", thinking="off", bank=bank)   # prefill "I choose:" appended
        records = build_records(on)
        pre_think = [len(tok.encode(chat_text(tok, r["prompt"], on), add_special_tokens=False)) for r in records]
        readout = [len(tok.encode(chat_text(tok, r["prompt"], off), add_special_tokens=False)) for r in records]
        print(f"{bank}: {len(records)} prompts; up to <think>: {stats(pre_think)}; "
              f"thinking-off readout sequence: {stats(readout)}")
        if bank == "full":
            off_len = float(np.mean(readout))
    on_8b = np.diff(np.load(STEERING_INPUTS / "inputs_Qwen3-8B_answer_slot.npz")["post|offsets"])
    on_14b = np.diff(np.load(STEERING_INPUTS / "inputs_Qwen3-14B_line_on.npz")["post|offsets"])
    print(f"cached-reasoning sequences scored by exp8: 8B {stats(on_8b)}; 14B {stats(on_14b)}")
    return off_len, float(on_8b.mean()), float(on_14b.mean())


def generation(root):
    for exp in ("exp10_Qwen3-8B_reasoning", "exp13_Qwen3-8B_pristine_suffix", "exp14_Qwen3-8B_phrase"):
        n = seconds = generated = 0
        for f in sorted(glob.glob(f"{root}/{exp}/conditions/*.json")):
            z = np.load(f[:-5] + ".npz")
            gen = np.diff(z["offsets"]) - z["prompt_len"]
            n, seconds, generated = n + len(gen), seconds + json.load(open(f))["seconds"], generated + gen.sum()
        print(f"{exp}: {n} generations, {seconds:.0f} s, {generated} tokens -> {generated / seconds:.0f} tok/s, "
              f"{seconds / n:.2f} s/generation, mean {generated / n:.0f} tokens")
    for run in ("exp0_Qwen3-8B_think-on_full", "exp0_Qwen3-8B_think-on_fill", "exp0_Qwen3-14B_think-on_full"):
        m = json.load(open(f"{root}/{run}/manifest.json"))
        z = np.load(f"{root}/{run}/activations.npz")
        # E0 writes the think block, then the lead and an answer window
        generated = (z["think_len"] + z["lead"] + m["settings"]["answer_window"]).sum()
        seconds = m["phases"]["generate + extract"]
        print(f"{run}: batch {m['settings']['batch_size']}, think {stats(z['think_len'])}, forced "
              f"{int(z['forced_mask'].sum())}; {seconds:.0f} s -> {generated / seconds:.0f} tok/s, "
              f"{seconds / len(z['think_len']):.2f} s/prompt")


def forward(root, off_len, on_len_8b, on_len_14b):
    for run, (gpu, model) in EXP8_RUNS.items():
        r = json.load(open(f"{root}/{run}/results.json"))
        for cell, c in r["cells"].items():
            mode, label, _ = cell.split("|")
            n = len(r["base"][mode]) if mode in r.get("base", {}) else r["n_records"]
            sequences = n * (2 if mode == "off" else 1)   # thinking off: each prompt and its swapped twin
            length = off_len if mode == "off" else (on_len_14b if model == "14B" else on_len_8b)
            cached = run in PREFIX_CACHE_RUNS and label == "pre-label"
            for test in ("steer", "ablate"):
                if f"{cell} {test}" not in r["phases"] or test not in c:
                    continue
                edits = sum(len(v) for v in c[test].values())
                seconds = r["phases"][f"{cell} {test}"]
                line = f"{run} [{gpu}] {cell} {test}: {edits * sequences} passes in {seconds:.1f} s = " \
                       f"{edits * sequences / seconds:.1f}/s"
                if cached:
                    base = r["phases"][f"{mode} baseline"]
                    line += f" (prefix cache; {(seconds - base) / ((edits - 1) * sequences) * 1e3:.1f} ms per extra edit)"
                else:
                    line += f", {edits * sequences * length / seconds:.0f} tok/s"
                print(line)
        if "patch" in r:   # T1: per batch a donor and a recipient pass, then one pass per (layer >= 1, group)
            passes = r["patch"]["n_pairs"] * (2 + (len(r["patch"]["transfer"]) - 1) * len(r["patch"]["groups"]))
            print(f"{run} [{gpu}] T1 patch: {passes} passes in {r['phases']['T1 patch']:.0f} s = "
                  f"{passes / r['phases']['T1 patch']:.1f}/s")


def main():
    root = sys.argv[1]
    off_len, on_8b, on_14b = token_counts()
    generation(root)
    forward(root, off_len, on_8b, on_14b)


if __name__ == "__main__":
    main()
