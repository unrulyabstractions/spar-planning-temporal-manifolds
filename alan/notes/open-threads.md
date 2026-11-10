# Open threads

Questions parked for later, with the evidence they rest on. Each entry says where the data and numbers live so the thread
can be picked up without re-deriving anything. Add new threads at the bottom; mark a thread closed rather than deleting it.

## 1. Why do horizons rendered in years have lower-dimensional local scatter? (opened 2026-10-03)

**Status:** parked by Alan pending a decision on whether to pull on it (this project or a follow-on).

**What is established** (Qwen3-14B only, one fixed scenario, cells L22 R0, L29 R0, L37 T3, L22 T3; Levina–Bickel MLE at
k = 10..20; values are orderings at matched n, not absolute dimensions):
- Years renderings give a lower estimate than months at matched durations in every design tried: tier B, n = 100
  (progress-20261002.md, matched-duration entry); same digit strings / different range (digit-count control, same file);
  decimal years vs months at n = 1200 (same file, last entry); and with the option delays in months instead of years
  (progress-20261003.md) — gap 4.2–6.7 with year delays, 4.8–9.3 with month delays.
- At n = 1200 decimal years are also below days (by 3.6–4.5) and weeks (by 4.3–5.7) on the same base durations; days,
  weeks and months do not differ among themselves (spread 0.3–2.3).
- The gap is in the local scatter, not the global spread: with month delays the Gaussian-matched values of years and
  months differ by 0.2–5.7 while the estimates differ by 4.8–9.3, and the scatter left after the smooth horizon curve is
  lower for years by 4.1–11.5.
- It is not explained by matching the options' unit (that drives the choice instead: short-choice rate at 3–10 y is 0.99
  when horizon and delays share a unit, 0.73–0.78 when not — geometry and behaviour dissociate).
- "Years is more curve-like" did not survive (+0.002 to +0.077 at n = 1200; the +0.21 to +0.29 at n = 100 was not robust).

**Open questions:**
1. *What* lowers the years scatter? Candidates: the unit token itself; how often years co-occur with long horizons in
   training text; conversion (the model restates years directly but converts months/days in its reasoning); the decimal
   format used for matched durations (integer years were also lower in tier B and the digit-count control, so not only
   decimals).
2. How does the scatter *differ*, beyond its dimension?
   - its eigen-spectrum (local PCA in the k-NN neighbourhoods);
   - its alignment with the curve: how much lies along the curve's tangent vs in the normal space;
   - how it varies along the curve (local estimate vs log horizon; already computed as `local_by_decade` columns).
3. Does it replicate on other models (Qwen3.8-27B, Gemma 4 31B) and a second scenario?
4. Where between k = 300 and the whole-set scale does the smooth curve take over from the scatter?

**Data already captured** (no new capture needed for 1–2 and 4): runs/qwen3-14b_matched_months_dwm_n1200 (days/weeks/
months, year delays), runs/qwen3-14b_matched_months_years_dec2_n1200, runs/qwen3-14b_matched_months_months_dm_n1200,
runs/qwen3-14b_matched_months_years_dec2_dm_n1200 (month delays), runs/qwen3-14b_matched_years_dwmy_n100,
runs/qwen3-14b_years_render_n1200. Generators: scripts/gen_dense_horizons.py; analysis: scripts/intrinsic_dim.py,
scripts/run_matched_units.sh, scripts/run_years_render.sh, scripts/run_delay_unit.sh.

## 2. Peak-cell reruns for Qwen3.8-27B and Gemma 4 31B-it (opened 2026-10-04)

**Status:** shelved by Alan on 2026-10-04 because of egress cost; he wants to come back and run them.

**What:** `scripts/peak_cells.sh <tag>` for both models (curve model, Guttman, relevance, scaling, LEACE, PLS, within-bin,
late readout at each model's own sweep-peak layers). The fixed fractional cells (0.55/0.65 L) sit before Gemma's
ordinality peak (≈ 0.62 L), so the fixed-cell comparison may understate Gemma (progress-20261004.md, review entry).
Peak cells: Qwen3.8-27B T3 → L32, R0 → L21 (overall L21 R0); Gemma T3 → L33, R0 → L37, overall L38 R6 — the last is
after the answer token; use the best pre-answer cell instead (L37 R0) unless Alan wants the post-answer one.

**Why it is not trivial:** needs the FULL shards (only Gemma L33 is among the exported subset layers) of six runs per model
(investment, matrix, extended, families, matrix_long, families_long): 138.5 GiB (Qwen3.8) + 136.4 GiB (Gemma) ≈ 295 GB
in gs://alan-captures-20261003. Downloading that to any non-Google machine (Vast, local) is billed egress, ≈ $35 at
~$0.12/GB list (recollection; check). CPU-only, ≈ 25 min per model (27B peak run on 09-30).
**Options:** a CPU VM in us-east4 reading the bucket (no egress; a few dollars; console setup in project
spar2026-temp-man), or Vast (≥ 200 GB disk processing one model at a time, ≥ 8 cores of real cgroup quota, ≥ 32 GB RAM,
≥ 1 Gbps down, cheapest GPU) plus the egress charge.
