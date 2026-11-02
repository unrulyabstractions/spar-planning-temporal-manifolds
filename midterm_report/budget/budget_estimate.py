"""Cost estimate for the midterm report's budget tiers (2026-11-02).

Every number below is an assumption; edit and rerun. Measured anchors:
- Gemma 4 31B-it, thinking on, HF generate, batch 4 on 1x A100 80 GB: 384 prompts in 1 h 23 min
  (median 716 generated tokens, p90 1,014, none truncated at 8,192) -> 13.0 s per generation.
  alan/progress-20261017.md
- Qwen3.8-27B (stand-in for Qwen3-32B), thinking on, batch 20: 384 prompts in 3 h 11 min (median 1,354, p90 5,816,
  22 hit the 8,192 cap) -> 29.8 s per generation. Same file.
- A100 80 GB rental observed at $0.81-1.36/h (Alan's rentals, ours at $1.19/h); 2x A100 once at $1.07/h total.
"""

PRICE = {"low": 0.9, "mid": 1.2, "high": 1.4}          # $/GPU-hour, 1x A100 80 GB
S_PER_GEN = {"gemma": 13.0, "qwen32": 29.8,          # measured, structured prompts, unsteered
             "qwen14": 41.6,                          # measured: Qwen3-14B, 192 prompts per GPU in 2 h 13 min (batch 24)
             "qwen27": 29.8,                          # assumed = Qwen3.8-27B
             "qwen8": 15.0}                           # assumed, not measured
LEN = {"T0": 1.0, "T12": 1.5, "T3": 2.5}              # trace length relative to structured prompts (assumed)
STEER_SLOWDOWN = {"low": 1.0, "mid": 1.3, "high": 1.8}  # steered traces run longer / hit the cap more (assumed)
OVERHEAD = {"low": 1.2, "mid": 1.4, "high": 1.7}      # setup, weight download, idle, reruns, failed runs (assumed)


def gpu_cost(n_gen, model, tier_len, case):
    hours = n_gen * S_PER_GEN[model] * LEN[tier_len] * STEER_SLOWDOWN[case] / 3600
    return hours * OVERHEAD[case] * PRICE[case], hours * OVERHEAD[case]


# Conditions = steering settings, each run on every prompt of the set.
RANDOM_DIRS = 20
T0_PROMPTS = 200                                       # verification grid, both option orders
t0_conditions = 1 + 12 + 12 + 4 + RANDOM_DIRS * 2      # none, CAA x12 strengths, manifold x12 shifts, reward x4, random x2
tier1 = [  # (label, generations, model, length class)
    ("extraction bank (500 prompts)", 500, "gemma", "T0"),
    ("gate + calibration sweep", T0_PROMPTS * t0_conditions, "gemma", "T0"),
    ("checks: generated-only steering, reward indifference, prompt baseline", 800 + 500 + 400, "gemma", "T0"),
    ("H2: held-out horizons, monotonicity, distance to curve", 2000, "gemma", "T0"),
]
T12_PROMPTS = 1200                                     # ~400 items (T1 200, T2 200 incl. 4/8-option versions) x 2 orders
t12_conditions = 1 + 1 + 1 + 4 + 4 + 1 + 10            # none, prompt baseline, rewrite R, CAA x4, manifold x4, reward, 10 random
tier2 = [
    ("T0 template variants (transfer gate)", 1000 * 5, "gemma", "T0"),
    ("T1-T2 steering", T12_PROMPTS * t12_conditions, "gemma", "T12"),
]
tier3 = [
    ("T3 open-ended (200 items)", 200 * 15, "gemma", "T3"),
    ("replication Qwen3-32B: gate + calibration", T0_PROMPTS * t0_conditions + 500, "qwen32", "T0"),
    ("replication Qwen3-32B: T1-T2, reduced (5 random)", T12_PROMPTS * 16, "qwen32", "T12"),
]
T0_GATE_SMALL = T0_PROMPTS * (1 + 12 + 12 + 4 + 10)   # gate only, 10 random directions
tier4 = [  # stretch: rough
    ("extra sizes: Qwen3-8B gate", T0_GATE_SMALL, "qwen8", "T0"),
    ("extra sizes: Qwen3-14B gate", T0_GATE_SMALL, "qwen14", "T0"),
    ("extra sizes: Qwen3.5-27B gate", T0_GATE_SMALL, "qwen27", "T0"),
    ("multi-turn 'are you sure?' (400 items x 2 turns x 10 settings)", 8000, "gemma", "T12"),
    ("durations outside planning", 2000, "gemma", "T0"),
]
API = {  # $ (low, mid, high): item generation + checking, coherence judge on every steered output
    1: (5, 15, 30),
    2: (10, 30, 60),
    3: (10, 30, 60),
    4: (5, 20, 40),
}

cum = {c: 0.0 for c in PRICE}
for i, tier in enumerate([tier1, tier2, tier3, tier4], 1):
    print(f"\nTier {i}")
    tot = {c: 0.0 for c in PRICE}
    for label, n, model, tl in tier:
        row = {c: gpu_cost(n, model, tl, c) for c in PRICE}
        for c in PRICE:
            tot[c] += row[c][0]
        print(f"  {label:<70} {n:>7,} gens  ${row['low'][0]:>4.0f}-{row['high'][0]:<4.0f} (mid ${row['mid'][0]:.0f}, "
              f"{row['mid'][1]:.0f} GPU-h)")
    for j, c in enumerate(PRICE):
        tot[c] += API[i][j]
    print(f"  API: ${API[i][0]}-{API[i][2]}")
    print(f"  TOTAL tier {i}: ${tot['low']:.0f}-{tot['high']:.0f} (mid ${tot['mid']:.0f})")
    for c in PRICE:
        cum[c] += tot[c]
    print(f"  CUMULATIVE through tier {i}: ${cum['low']:.0f}-{cum['high']:.0f} (mid ${cum['mid']:.0f})")
