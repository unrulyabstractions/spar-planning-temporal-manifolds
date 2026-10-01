"""E6: PCA geometry at every token of the think block (overnight, resumable).

    python experiments/exp6_think_pca.py --model Qwen/Qwen3.5-4B --hours 11
    python experiments/exp6_think_pca.py --model Qwen/Qwen3.5-0.8B --limit 4 \
        --max-think-tokens 64 --stride 4 --hours 0.2            # smoke

Phases, each resumable from what is already on disk:

1. generate   greedy think + answer for every prompt (batched, gen_cache).
              Prompts are ordered round-robin over horizons so a partial run
              still spans the whole axis.
2. fit        one forward pass per prompt keeping hidden states at `--layers`
              for every `--stride`-th think token plus the suffix positions.
              Per layer: PCA(6) on the think subsample ("think basis"), PCA(6)
              on the suffix probe position across prompts ("suffix basis"),
              and the ridge horizon probe at the suffix probe position.
              The fp16 subsample is kept in fit/ so bases can be refit later.
3. project    one forward pass per prompt projecting EVERY think token (and
              the suffix, pre and answer positions) onto both bases and the
              probe direction. Even-ranked prompts store 3 components, odd-
              ranked store 6 (Jared's split); the probe reading is always
              stored. One npz per prompt under tokens/.
4. summarize  assembles tokens/*.npz into think_pca.npz, writes
              summary.json (explained variance, rho of each PC with log
              horizon by think-fraction bin, probe track stats) and figures.

`--hours` is a wall-clock budget checked between prompts; when it runs out
the run writes what it has, summarizes, and exits 0. Rerunning continues.
Layers are hidden_states indices (0 = embeddings), given as fractions of the
model depth or as integers.
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm

from common import parse_args
from spar_horizon.geometry import horizon_probe
from spar_horizon.model_io import generate_batch, load_model, prefill_ids, suffix_length
from spar_horizon.plots import horizon_spec
from spar_horizon.prompts import build_prompts
from spar_horizon.runs import Manifest, resolve_out_dir, stamp

warnings.filterwarnings("ignore", category=ConstantInputWarning)

EXP = "exp6"
RESERVE = 0.3   # share of the budget kept back from generation for the fit and project passes
N_COMP = 6
DEFAULT_LAYERS = "0.25,0.375,0.5,0.625,0.75,0.875"
N_BINS = 10


# ---------------------------------------------------------------- pure helpers

def interleave_by_horizon(records):
    """Round-robin over horizon groups (None last) so any prefix of the order
    spans the axis. Returns the permutation as original indices."""
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r["horizon"], []).append(i)
    keys = sorted((k for k in groups if k is not None)) + ([None] if None in groups else [])
    order, depth = [], max(len(v) for v in groups.values())
    for d in range(depth):
        for k in keys:
            if d < len(groups[k]):
                order.append(groups[k][d])
    return order


def components_for(rank):
    """3 components for even ranks in the run order, 6 for odd."""
    return 3 if rank % 2 == 0 else N_COMP


def resolve_layers(spec, n_hidden_states):
    """'0.25,0.5' or '8,16' -> sorted unique hidden_states indices in
    [1, n_hidden_states - 1]."""
    out = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        idx = int(round(v * (n_hidden_states - 1))) if v < 1 else int(v)
        idx = min(max(idx, 1), n_hidden_states - 1)
        out.add(idx)
    return sorted(out)


def probe_position(n_suffix):
    """Kept-layout index of the newline after <|im_end|>: suffix position 1."""
    return 1 if n_suffix > 1 else 0


def bin_means(values, n_bins=N_BINS):
    """Mean of `values` in n_bins equal slices along its length (nan for empty)."""
    edges = np.linspace(0, len(values), n_bins + 1).round().astype(int)
    return np.array([values[a:b].mean() if b > a else np.nan for a, b in zip(edges[:-1], edges[1:])])


def rho_by_bin(binned, log_h):
    """Spearman rho between each bin column of `binned` [n, n_bins] and log_h."""
    out = []
    for j in range(binned.shape[1]):
        col = binned[:, j]
        ok = ~np.isnan(col)
        out.append(spearmanr(col[ok], log_h[ok]).statistic if ok.sum() > 3 else np.nan)
    return np.array(out)


# ---------------------------------------------------------------- model access

def decoder_layers(model):
    n = model.config.get_text_config().num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            return mod
    raise RuntimeError("could not find the decoder layer list")


class HiddenTap:
    """Forward hooks that keep the output of decoder layers (hidden_states
    index l is the output of decoder layer l - 1), without materialising
    full-sequence logits."""

    def __init__(self, model, layers):
        self.model, self.layers, self.store, self.handles = model, layers, {}, []
        mods = decoder_layers(model)
        for l in layers:
            self.handles.append(mods[l - 1].register_forward_hook(self._hook(l)))

    def _hook(self, l):
        def fn(_m, _i, out):
            self.store[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def run(self, ids):
        """ids [T] -> dict layer -> [T, d] float32 on CPU."""
        self.store.clear()
        with torch.no_grad():
            self.model(input_ids=ids.unsqueeze(0), logits_to_keep=1)
        return {l: self.store[l][0].float().cpu().numpy() for l in self.layers}

    def close(self):
        for h in self.handles:
            h.remove()


class Budget:
    def __init__(self, hours):
        self.deadline = time.time() + hours * 3600

    def left(self):
        return self.deadline - time.time()

    def exhausted(self):
        return self.left() <= 0


# ---------------------------------------------------------------- phases

def generate_within_budget(tokenizer, model, device, prompts, cfg, budget, hours, reserve=RESERVE):
    """generate_batch one batch at a time, stopping once less than `reserve`
    of the budget is left. Cached prompts cost nothing, so a rerun always gets
    further. Returns the Generations for a prefix of `prompts`."""
    out = []
    for b in range(0, len(prompts), cfg.batch_size):
        if budget.left() < reserve * hours * 3600 and out:
            break
        out += generate_batch(tokenizer, model, device, prompts[b:b + cfg.batch_size], cfg)
        stamp(f"generated {len(out)}/{len(prompts)}  ({budget.left() / 3600:.1f} h left)")
    return out

def regions_for(g, n_suffix, n_pre, n_response):
    """Absolute index ranges in the sequence: suffix, think, pre, answer."""
    close = g.answer_start - g.lead - n_pre
    return {"suffix": (g.prompt_len - n_suffix, g.prompt_len),
            "think": (g.prompt_len, g.prompt_len + g.think_len),
            "pre": (close, g.answer_start),
            "answer": (g.answer_start, min(g.answer_start + n_response, int(g.ids.shape[0])))}


def phase_fit(tap, gens, order, layers, cfg, n_suffix, n_pre, fit_dir, stride, budget):
    """Subsampled states per prompt -> fit/<rank>.npz. Returns True if every
    prompt is on disk."""
    fit_dir.mkdir(exist_ok=True)
    for rank, i in enumerate(tqdm(order, desc="fit pass", unit="prompt", mininterval=2, dynamic_ncols=True)):
        p = fit_dir / f"{rank:03d}.npz"
        if p.exists():
            continue
        if budget.exhausted():
            stamp("budget exhausted during fit")
            return False
        g = gens[i]
        reg = regions_for(g, n_suffix, n_pre, cfg.n_response)
        hs = tap.run(g.ids[: reg["answer"][1]])
        a, b = reg["think"]
        sub = np.arange(a, b, stride)
        s0, s1 = reg["suffix"]
        np.savez(p, sub_idx=sub, **{f"think_L{l}": hs[l][sub].astype(np.float16) for l in layers},
                 **{f"suffix_L{l}": hs[l][s0:s1].astype(np.float16) for l in layers})
    return True


def fit_bases(fit_dir, order, layers, n_suffix, log_h_by_rank, has_h_by_rank):
    """PCA(6) on the think subsample and on the suffix probe position, plus the
    horizon probe, per layer. Returns dict layer -> dict of arrays."""
    bases = {}
    files = [np.load(fit_dir / f"{rank:03d}.npz") for rank in range(len(order))]
    ppos = probe_position(n_suffix)
    for l in tqdm(layers, desc="fit PCA", unit="layer", dynamic_ncols=True):
        think = np.concatenate([f[f"think_L{l}"].astype(np.float32) for f in files])
        suffix = np.stack([f[f"suffix_L{l}"][ppos].astype(np.float32) for f in files])
        k = min(N_COMP, think.shape[0] - 1, think.shape[1]) if think.shape[0] > 1 else 0
        pt = PCA(n_components=k, random_state=0).fit(think) if k else None
        ks = min(N_COMP, suffix.shape[0] - 1, suffix.shape[1])
        ps = PCA(n_components=ks, random_state=0).fit(suffix)
        hp = horizon_probe(suffix[has_h_by_rank], log_h_by_rank[has_h_by_rank])[2] \
            if has_h_by_rank.sum() >= 5 else None
        d = think.shape[1]

        def pad(pca):
            comp = np.zeros((N_COMP, d), np.float32)
            mean = np.zeros(d, np.float32)
            evr = np.full(N_COMP, np.nan)
            if pca is not None:
                comp[: pca.n_components_] = pca.components_
                mean[:] = pca.mean_
                evr[: pca.n_components_] = pca.explained_variance_ratio_
            return comp, mean, evr

        tc, tm, te = pad(pt)
        sc, sm, se = pad(ps)
        bases[l] = {"think_comp": tc, "think_mean": tm, "think_evr": te,
                    "suffix_comp": sc, "suffix_mean": sm, "suffix_evr": se,
                    "probe_coef": (hp.coef_ if hp is not None else np.zeros(d)).astype(np.float32),
                    "probe_intercept": float(hp.intercept_) if hp is not None else 0.0,
                    "n_think_fit": int(think.shape[0])}
    return bases


def project(h, base, n_comp):
    """h [T, d] -> think PCs [T, n_comp], suffix PCs [T, n_comp], probe [T]."""
    zt = (h - base["think_mean"]) @ base["think_comp"][:n_comp].T
    zs = (h - base["suffix_mean"]) @ base["suffix_comp"][:n_comp].T
    pr = h @ base["probe_coef"] + base["probe_intercept"]
    return zt.astype(np.float32), zs.astype(np.float32), pr.astype(np.float32)


def phase_project(tap, gens, order, layers, cfg, n_suffix, n_pre, bases, tok_dir, budget):
    tok_dir.mkdir(exist_ok=True)
    for rank, i in enumerate(tqdm(order, desc="project", unit="prompt", mininterval=2, dynamic_ncols=True)):
        p = tok_dir / f"{rank:03d}.npz"
        if p.exists():
            continue
        if budget.exhausted():
            stamp("budget exhausted during project")
            return False
        g = gens[i]
        reg = regions_for(g, n_suffix, n_pre, cfg.n_response)
        lo, hi = reg["suffix"][0], reg["answer"][1]
        hs = tap.run(g.ids[:hi])
        n_comp = components_for(rank)
        arrays = {"n_comp": n_comp, "offset": lo, "forced": g.forced, "think_len": g.think_len,
                  "ids": g.ids[lo:hi].cpu().numpy(),
                  **{f"region_{k}": np.array(v) - lo for k, v in reg.items()}}
        for l in layers:
            zt, zs, pr = project(hs[l][lo:hi], bases[l], n_comp)
            arrays[f"think_pc_L{l}"], arrays[f"suffix_pc_L{l}"], arrays[f"probe_L{l}"] = zt, zs, pr
        np.savez(p, **arrays)
    return True


def summarize(tok_dir, order, layers, records, bases, out_dir):
    files = sorted(tok_dir.glob("*.npz"))
    if not files:
        return {"n_prompts": 0}
    ranks = [int(f.stem) for f in files]
    zs = [np.load(f) for f in files]
    horizons = np.array([records[order[r]]["horizon"] if records[order[r]]["horizon"] is not None
                         else np.nan for r in ranks])
    has_h = ~np.isnan(horizons)
    log_h = np.log10(np.where(has_h, horizons, 1.0))
    forced = np.array([bool(z["forced"]) for z in zs])
    think_len = np.array([int(z["think_len"]) for z in zs])
    summary = {"n_prompts": len(zs), "n_forced": int(forced.sum()),
               "think_len_median": float(np.median(think_len)), "layers": layers, "per_layer": {}}
    spec = horizon_spec(log_h[has_h])
    for l in layers:
        entry = {"think_evr": bases[l]["think_evr"].tolist(), "suffix_evr": bases[l]["suffix_evr"].tolist(),
                 "n_think_fit": bases[l]["n_think_fit"]}
        for basis in ("think_pc", "suffix_pc"):
            for c in range(3):   # the first three exist for every prompt
                binned = np.stack([bin_means(z[f"{basis}_L{l}"][slice(*z["region_think"]), c])
                                   if z["region_think"][1] > z["region_think"][0] else np.full(N_BINS, np.nan)
                                   for z in zs])
                entry[f"rho_{basis}{c + 1}_by_bin"] = rho_by_bin(binned[has_h], log_h[has_h]).tolist()
        probe = np.stack([bin_means(z[f"probe_L{l}"][slice(*z["region_think"])])
                          if z["region_think"][1] > z["region_think"][0] else np.full(N_BINS, np.nan)
                          for z in zs])
        entry["rho_probe_by_bin"] = rho_by_bin(probe[has_h], log_h[has_h]).tolist()
        entry["probe_slope_by_bin"] = [
            float(np.polyfit(log_h[has_h][~np.isnan(probe[has_h][:, j])],
                             probe[has_h][:, j][~np.isnan(probe[has_h][:, j])], 1)[0])
            if (~np.isnan(probe[has_h][:, j])).sum() > 3 else float("nan") for j in range(N_BINS)]
        summary["per_layer"][str(l)] = entry
        _trajectory_figure(zs, l, has_h, log_h, forced, spec, out_dir / f"think_pc_L{l}.png")
        _rho_figure(entry, l, out_dir / f"rho_by_bin_L{l}.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'summary.json'}")
    return summary


def _trajectory_figure(zs, l, has_h, log_h, forced, spec, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, basis, name in zip(axes, ("think_pc", "suffix_pc"), ("think basis", "suffix basis")):
        for z, h, ok, f in zip(zs, log_h, has_h, forced):
            a, b = z["region_think"]
            if b <= a:
                continue
            Z = z[f"{basis}_L{l}"][a:b, :2]
            w = max(1, min(32, (b - a) // 8))
            Zs = np.stack([np.convolve(Z[:, k], np.ones(w) / w, mode="valid") for k in range(2)], 1)
            color = plt.get_cmap("turbo")(spec.norm(h)) if ok else (0.5, 0.5, 0.5, 0.6)
            ax.plot(Zs[:, 0], Zs[:, 1], lw=0.8, alpha=0.7, color=color, ls="--" if f else "-")
            ax.plot(Zs[0, 0], Zs[0, 1], "o", ms=3, color=color)
        ax.set_title(f"L{l} {name}: PC1 vs PC2 along the think block (o = start)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=spec.norm)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.12)
    cb.set_ticks([t for t, _ in spec.ticks])
    cb.set_ticklabels([lab for _, lab in spec.ticks])
    cb.set_label("time horizon")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _rho_figure(entry, l, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    for key, lab in (("rho_think_pc1_by_bin", "think PC1"), ("rho_think_pc2_by_bin", "think PC2"),
                     ("rho_think_pc3_by_bin", "think PC3"), ("rho_suffix_pc1_by_bin", "suffix PC1"),
                     ("rho_probe_by_bin", "probe")):
        ax.plot(x, np.abs(entry[key]), marker="o", label=lab)
    ax.set_ylim(0, 1)
    ax.set_xlabel("fraction of the think block")
    ax.set_ylabel("|Spearman rho| with log horizon")
    ax.set_title(f"L{l}: horizon readability along the think block")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def add_args(p):
    p.add_argument("--layers", default=DEFAULT_LAYERS,
                   help="hidden_states indices, as fractions of depth or integers")
    p.add_argument("--stride", type=int, default=16, help="think-token subsample stride for the PCA fit")
    p.add_argument("--hours", type=float, default=11.0, help="wall-clock budget")


def run(cfg, layers_spec, stride, hours, out=None):
    budget = Budget(hours)
    out_dir = resolve_out_dir(EXP, cfg, out)
    manifest = Manifest(EXP, cfg, out_dir)
    records = build_prompts(seed=cfg.seed, limit=cfg.limit)
    order = interleave_by_horizon(records)
    print(f"{len(records)} prompts, model {cfg.model}, thinking {cfg.thinking}, budget {hours} h")
    tokenizer, model, device = load_model(cfg)
    n_hidden = model.config.get_text_config().num_hidden_layers + 1
    layers = resolve_layers(layers_spec, n_hidden)
    print(f"layers (hidden_states indices of {n_hidden}): {layers}")
    manifest.add(layers=layers, stride=stride, hours=hours, order=order,
                 n_comp_by_rank=[components_for(r) for r in range(len(order))])

    prompts = [records[i]["prompt"] for i in order]
    with manifest.phase("generate"):
        gens_ordered = generate_within_budget(tokenizer, model, device, prompts, cfg, budget, hours)
    if len(gens_ordered) < len(order):
        stamp(f"generation stopped at {len(gens_ordered)}/{len(order)} prompts to leave time for the passes")
        order = order[: len(gens_ordered)]
        manifest.add(n_generated=len(gens_ordered))
    gens = [None] * len(records)
    for i, g in zip(order, gens_ordered):
        gens[i] = g
    forced = sum(g.forced for g in gens_ordered)
    print(f"think length: median {int(np.median([g.think_len for g in gens_ordered]))}  "
          f"forced: {forced}/{len(gens_ordered)}")
    manifest.add(n_forced=forced)
    (out_dir / "answers.json").write_text(json.dumps(
        [{"rank": r, "index": i, "horizon": records[i]["horizon"], "forced": gens[i].forced,
          "think_len": gens[i].think_len, "answer": gens[i].answer_text(tokenizer),
          "think": gens[i].think_text(tokenizer)} for r, i in enumerate(order)], indent=1))

    n_suffix = suffix_length(tokenizer, prompts[0], cfg)
    n_pre = 2 + len(prefill_ids(tokenizer, cfg)) if cfg.thinking == "on" else 0
    tap = HiddenTap(model, layers)
    fit_dir, tok_dir = out_dir / "fit", out_dir / "tokens"
    done = True
    with manifest.phase("fit pass"):
        done = phase_fit(tap, gens, order, layers, cfg, n_suffix, n_pre, fit_dir, stride, budget)
    if not done:
        manifest.add(status="incomplete: fit pass")
        manifest.write()
        return
    log_h = np.array([np.log10(records[i]["horizon"]) if records[i]["horizon"] is not None else 0.0
                      for i in order])
    has_h = np.array([records[i]["horizon"] is not None for i in order])
    with manifest.phase("fit bases"):
        bases = fit_bases(fit_dir, order, layers, n_suffix, log_h, has_h)
    np.savez(out_dir / "bases.npz", layers=np.array(layers),
             **{f"{k}_L{l}": np.asarray(v) for l in layers for k, v in bases[l].items()})
    with manifest.phase("project"):
        done = phase_project(tap, gens, order, layers, cfg, n_suffix, n_pre, bases, tok_dir, budget)
    tap.close()
    with manifest.phase("summarize"):
        summary = summarize(tok_dir, order, layers, records, bases, out_dir)
    manifest.add(status="complete" if done else "incomplete: project", summary=summary)
    manifest.write()


if __name__ == "__main__":
    args, cfg = parse_args(__doc__, add_args)
    run(cfg, args.layers, args.stride, args.hours, out=args.out)
