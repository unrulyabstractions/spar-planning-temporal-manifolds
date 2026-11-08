"""Cost model for the "first big experiment" (big_experiment_plan.md).

Pure arithmetic, no model. Every rate below was measured on earlier box runs by
analysis/run_throughput.py; prices are market assumptions. Change a constant and
rerun:  python analysis/big_experiment_costs.py
"""

# --- measured (Qwen3-8B, HF transformers + forward hooks, bf16) ------------------------
FULL_PASS_PER_S = 77.0        # thinking-off sequence (~142 tok), batch 32. 4090: 75.2-84.0, T1 patch 75.9;
                              # 5090: 76.8 (exp8 pristine suffix-newline steer). Same on both cards.
EDIT_FIXED_MS = 1.6           # cached-prefix extra edit, 1-token suffix, per sequence (5090, exp8 pristine)
SEQ_LEN = 141.7               # thinking-off readout sequence: template + empty think + "I choose:" (mean)
PRE_THINK_POSITIONS = 134.7   # tokens up to <think> with the chat template (mean; median 136, max 141)
POST_HORIZON_POSITIONS = 41.7 # of those, from the first horizon token on (41-43; earlier tokens cannot see it)
# per-token cost of a recomputed suffix, from the two measured points (1 token and the full sequence)
SUFFIX_MS_PER_TOKEN = (1000 / FULL_PASS_PER_S - EDIT_FIXED_MS) / (SEQ_LEN - 1)
SECONDS_PER_GENERATION = 11.66  # thinking on, 3,072 think budget, steering hooks, batch 12 (E14, 5090);
                                # range 10.66 (E0 4090, no hooks) to 16.57 (E10, hooks, 2-GPU box)
GEN_TOKENS_PER_GENERATION = 1760  # E14 mean generated tokens (E0 full bank: think mean 1,799)
CAPTURE_GPU_HOURS = 0.25      # activation capture of one bank, all layers x positions, incl. disk writes
SETUP_GPU_HOURS = 0.3         # per rental per GPU: model download (0.4-5 min logged), push, warm-up

# --- assumed ($/GPU-hour, Vast.ai on-demand, Nov 2026 snapshot; nothing paid is recorded) --
PRICE = {"RTX 4090": 0.40, "RTX 5090": 0.65, "A100 80GB": 1.10, "H100 SXM": 1.93}
# speed relative to the measured 4090/5090 rates. The 5090 measured the same as the 4090,
# so this stack is software-bound; H100/A100 speedups are unmeasured (pilot decides)
SPEED = {"RTX 4090": 1.0, "RTX 5090": 1.0, "A100 80GB": 1.0, "H100 SXM": 1.0}
SPEED_OPTIMISTIC = {"RTX 4090": 1.0, "RTX 5090": 1.0, "A100 80GB": 1.5, "H100 SXM": 2.5}

N_PROMPTS = 500
N_LAYERS = 36


def full_pass_hours(runs):
    return runs / FULL_PASS_PER_S / 3600


def cached_ms_per_run(mean_suffix_tokens):
    return EDIT_FIXED_MS + SUFFIX_MS_PER_TOKEN * mean_suffix_tokens


def mean_suffix(first, last):
    """Mean recomputed length when steering at a position t uniform over [first, last]
    (0-based): positions t..end of the readout sequence are recomputed."""
    return SEQ_LEN - (first + last) / 2


ALL_POSITIONS = mean_suffix(0, PRE_THINK_POSITIONS - 1)
POST_HORIZON = mean_suffix(PRE_THINK_POSITIONS - POST_HORIZON_POSITIONS, PRE_THINK_POSITIONS - 1)
PRE_HORIZON = mean_suffix(0, PRE_THINK_POSITIONS - POST_HORIZON_POSITIONS - 1)


def cached_hours(runs, suffix=ALL_POSITIONS):
    return runs * cached_ms_per_run(suffix) / 1000 / 3600


def generation_hours(generations):
    return generations * SECONDS_PER_GENERATION / 3600


def cost_row(name, runs, tokens, gpu_hours, note=""):
    prices = " | ".join(f"${gpu_hours / SPEED[g] * p:,.2f}" for g, p in PRICE.items())
    return f"| {name} | {runs:,.0f} | {tokens / 1e6:,.1f}M | {gpu_hours:,.2f} | {prices} | {note} |"


def designs():
    rows = []
    grid = lambda layers: layers * PRE_THINK_POSITIONS * N_PROMPTS * 2   # x (+alpha, -alpha)

    for name, layers in (("A  36 layers, full passes", 36), ("B  18 layers, full passes", 18)):
        runs = grid(layers) + N_PROMPTS
        rows.append(cost_row(name, runs, runs * SEQ_LEN, full_pass_hours(runs)))
    for name, layers in (("C  = A with prefix KV cache", 36), ("C  = B with prefix KV cache", 18)):
        runs = grid(layers) + N_PROMPTS
        rows.append(cost_row(name, runs, runs * ALL_POSITIONS, cached_hours(runs),
                             f"mean recomputed suffix {ALL_POSITIONS:.1f} tok"))
    runs = 36 * POST_HORIZON_POSITIONS * N_PROMPTS * 2 + N_PROMPTS
    rows.append(cost_row("C* = C, 36 layers, positions from the horizon on", runs, runs * POST_HORIZON,
                         cached_hours(runs, POST_HORIZON), f"{POST_HORIZON_POSITIONS} positions, suffix {POST_HORIZON:.1f} tok"))
    rows.append("| D  edit at t only vs t..end | same as A/B/C | | | | both recompute positions t..end; cost identical")
    for name, layers in (("E  thinking-on gen, every cell of A, 3 samples", 36), ("E  same, 18 layers", 18)):
        gens = grid(layers) * 3 + N_PROMPTS * 3
        rows.append(cost_row(name, gens, gens * GEN_TOKENS_PER_GENERATION, generation_hours(gens)))
    gens = 5 * N_PROMPTS * 3 * 3
    rows.append(cost_row("F  top 5 cells x 500 x 3 coef x 3 samples", gens, gens * GEN_TOKENS_PER_GENERATION,
                         generation_hours(gens), "as specified"))
    gens = (5 * 2 + 1) * N_PROMPTS * 3
    rows.append(cost_row("F' same, alpha=0 shared across cells", gens, gens * GEN_TOKENS_PER_GENERATION,
                         generation_hours(gens), "unsteered baseline run once"))
    runs = N_PROMPTS
    rows.append(cost_row("G  activation cache, every layer x position", runs, runs * SEQ_LEN,
                         full_pass_hours(runs) + CAPTURE_GPU_HOURS, "dominated by capture + disk"))
    runs = N_PROMPTS * (2 + 5 * 2 * 4)   # clean donor+recipient; 5 layers x 2 position groups x 4 patch kinds
    rows.append(cost_row("H  patching, logit readout", runs, runs * SEQ_LEN, full_pass_hours(runs),
                         "full / proj-1d / proj-8d / random-1d"))
    gens = N_PROMPTS * 5 * 2 * 4
    rows.append(cost_row("H' patching, thinking-on readout", gens, gens * GEN_TOKENS_PER_GENERATION,
                         generation_hours(gens), "1 sample each"))
    return rows


def bundle():
    """The recommended first big experiment (big_experiment_plan.md, Step 4)."""
    V = 2                         # each prompt and its option-swapped twin (project convention)
    full, units = 500, 515        # full bank; unit-overlap bank (same durations in several units)
    post = POST_HORIZON_POSITIONS
    pre_sampled = (PRE_THINK_POSITIONS - post) / 4   # every 4th horizon-blind position
    items = []
    items.append(("1 activation cache, full + units, every layer x position, V=1", "pass",
                  full + units, full_pass_hours(full + units) + 2 * CAPTURE_GPU_HOURS))
    runs = N_LAYERS * post * (full + units) * V * 2
    items.append(("2a screen, horizon direction: 36 L x post-horizon pos x (full+units) x V x +-a", "cached",
                  runs, cached_hours(runs, POST_HORIZON)))
    runs = 3 * N_LAYERS * post * full * V * 2
    items.append(("2b screen, random / reward-magnitude / written-number dirs, same grid, full bank", "cached",
                  runs, cached_hours(runs, POST_HORIZON)))
    runs = 18 * pre_sampled * full * V * 2
    items.append(("2c null: transported horizon direction, 18 L x every 4th pre-horizon pos", "cached",
                  runs, cached_hours(runs, PRE_HORIZON)))
    runs = 8 * 5 * (full + units) * V * 7
    items.append(("3 dose-response at top 5 cells: 8 alphas x 7 dirs x (full+units) x V", "cached",
                  runs, cached_hours(runs, POST_HORIZON)))
    runs = (full + units) * V * (2 + 5 * 2 * 4)
    items.append(("4 patching full vs projection, top 5 layers, full+units, V", "pass", runs, full_pass_hours(runs)))
    gens = (5 * 2 + 1) * full * 3
    items.append(("5 thinking-on check (F'): top 5 cells x +-a (+shared 0) x 500 x 3 samples", "gen",
                  gens, generation_hours(gens)))
    return items


def main():
    print(f"suffix cost model: {EDIT_FIXED_MS} ms + {SUFFIX_MS_PER_TOKEN:.4f} ms/token per sequence "
          f"(full {SEQ_LEN} tok -> {cached_ms_per_run(SEQ_LEN):.2f} ms = 1/{1000 / cached_ms_per_run(SEQ_LEN):.1f} s)")
    head = " | ".join(PRICE)
    print(f"\n| design | runs | tokens | GPU-h (4090/5090 rate) | {head} | note\n|" + "---|" * (5 + len(PRICE)))
    for row in designs():
        print(row)

    print("\n| bundle item | kind | runs | GPU-h |\n|---|---|---|---|")
    total = 0.0
    for name, kind, runs, hours in bundle():
        print(f"| {name} | {kind} | {runs:,.0f} | {hours:.2f} |")
        total += hours
    print(f"| total GPU work | | | {total:.1f} |")
    for n_gpus in (1, 4):
        hours = total + SETUP_GPU_HOURS * n_gpus
        print(f"\n{n_gpus} GPU(s): wall {hours / n_gpus:.1f} h, GPU-hours {hours:.1f}")
        for g, p in PRICE.items():
            print(f"  {g:10} pessimistic (x{SPEED[g]}) ${hours / SPEED[g] * p:7.2f}"
                  f"   optimistic (x{SPEED_OPTIMISTIC[g]}) ${hours / SPEED_OPTIMISTIC[g] * p:7.2f}")
    print("\nbreak-even speedup over a 4090 at its price:",
          ", ".join(f"{g} {p / PRICE['RTX 4090']:.2f}x" for g, p in PRICE.items()))


if __name__ == "__main__":
    main()
