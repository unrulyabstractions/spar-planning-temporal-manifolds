# Spec: unit-overlap bank, horizon vs unit word (2026-10-24)

Status: **approved 2026-10-25** (8B only). Decisions, 2026-10-24:
- paired renderings;
- Qwen3-8B, thinking on and off (14B dropped for now, 2026-10-25);
- E0 plus local analyses only;
- integer renderings in neighbouring units.

## 1. Why

On the 48-point grid, every duration is said in its natural unit (`natural_horizon`), so the
unit only grows with the duration. Unit and magnitude are confounded, and the unit word alone
scores R² 0.96 (`01_plan.md` §3). The tests so far each leave a gap:
- **E9** (`08` §10) took the unit apart only at the suffix newline, on 8B thinking on, with no
  behavior. It found a duration code with ~0.3-decade unit shifts.
- **Check 1** (`12` §1) found that at the answer slot the low-k code is unit-only on 14B
  (pre-label off, k = 1–2, within-unit R² ≈ 0). On 8B, per prompt and on horizon averages
  disagree.
- **E10** (`08` §11) found that the last-token patch swaps the unit, not the horizon.

A bank where the same duration appears in two or three units tests directly whether each stage
holds the duration or the unit word, at every position and in behavior.

## 2. The bank (`bank = "units"`)

**Durations.** The 48-point grid, keeping each duration that has a second integer rendering:
- in the natural unit or an adjacent one in `NATURAL_UNITS`;
- within 10% of the natural rendering's duration;
- with a value from 1 to 1000.

That keeps 33 durations, from 3 minutes to 59 years:
- 31 durations with 2 renderings and 2 with 3 (`8 weeks`: 2 months / 8 weeks / 56 days;
  `23 months`: 2 years / 23 months / 100 weeks);
- every unit boundary covered (seconds↔minutes … months↔years).

The units then overlap widely. For example:
- months run from "2 months" to "708 months" (0.17–59 years);
- weeks run from "2 weeks" to "100 weeks".

Dropped, because they have no second rendering under these rules:
- 30–108 s (no smaller unit);
- 23–84 min and 18–43 h (the integer neighbour is more than 10% off, or the value is over
  1000);
- 91–500 years (months would be over 1000).

**Pairing.** Each base prompt draws one duration (uniform over the 33), then its rewards,
delays, order and labels, as `generate` does. It is emitted once per rendering, so a pair
shares everything but the horizon phrase. 250 base prompts give about 520 prompts, close to
the 500-prompt full bank in cost, with about 7–8 bases per duration.

**Labels per record:**
- `pair_id`;
- `duration_id` (the grid duration, for grouping);
- `rendered_unit`;
- `natural` (is this the natural rendering);
- `horizon_years` of the rendered value, as now.

Regimes (near / both / none) come from the duration and the delays, so they are the same
within a pair. There are no null prompts: they don't pair.

**Code:**
- `generate` already has `unit_rewrite` and `paired` (Alan's, keyed by the 17-point grid).
- Add a `rewrites` table to `DatasetConfig` (default: the existing `UNIT_REWRITES`, so the
  full bank and its RNG stream are unchanged).
- Build the neighbour table from the grid with the rule above.
- Add `build_units_prompts` and `bank = "units"` in `Settings` and `banks.py`. The run folder
  becomes `..._think-{mode}_units`.

## 3. Runs (box, `queue_units.sh`)

E0 on the units bank for **Qwen3-8B only**, thinking off and on (Jared, 2026-10-25; 14B
later, with no code changes). Each run saves activations at all layers and behavior, as the
full-bank E0 does.

Box: 2× RTX 5090 (32 GB, Blackwell, needs PyTorch ≥ 2.7 / CUDA 12.8). Lanes, with no
sharding:
- GPU 0 runs 8B on, `--batch-size 20`, about 50–70 min;
- GPU 1 runs 8B off, about 5 min.

Same `step`/`.done` resume as `queue_14b_rest.sh`, with a download step first. The full-bank
8B on run took 124 min on a 4090 at batch 12.

Local, before the box (handed to you via `local_checks.py`): a 0.6B smoke on 8 units-bank
prompts, which checks the rendering, the pair labels and that the label slot is found.

## 4. Analyses (local, saved activations; `pipeline/s3_analysis/by_datatype/unit_overlap.py`)

These are readings, not pass/fail (as for `10`). For each of the two 8B mode cells, at
suffix newline, think close (thinking on) and pre-label, at every layer:

1. **Cross-unit transfer.** Fit the leave-one-duration-out ridge on natural renderings only,
   then score the other renderings of the held-out duration.
   - A duration code: R² on alternates ≈ R² on naturals.
   - A unit code: the alternates are read at the size of their unit, and R² collapses.
2. **Pair shift in decades.** The readout of the alternate minus that of the natural rendering,
   averaged per duration, in decades (E9's measure, now at every position and on the units bank).
   For scale, one grid step is 0.15 decades.
3. **Unit vs duration share.** In the top-k horizon-average subspace (k = 1, 2, 4, 8), the
   variance share from rendered unit vs duration, in a two-way decomposition. That is possible
   now because unit and duration are crossed on the 33 durations.
   - Changed in implementation (2026-10-25): the crossing is partial (weeks only cover weeks
     to months), so most variance is shared. Each factor's unique (type II) part is small for
     any code: planted duration code 0.00 unit / 0.10 duration; planted unit code 0.37 / 0.00.
     The reading is their contrast, with the shared part reported alongside.
4. **Within-unit R² over a wide range.** Months span 2.5 decades and minutes 2.5 decades. This
   reruns Check 1's within-unit R², at both levels, on the units bank.
5. **Behavior.** Over the pairs:
   - the share of pairs that give the same label;
   - the mean |Δ near log-odds|, against the order-swap baseline (`11` §4);
   - the other rendering's near log-odds minus the natural one's (bootstrap CI), split by
     whether the other rendering uses the smaller or the larger unit. This replaced the planned
     regression (2026-10-25): within a pair everything else is fixed, so the difference is the
     unit effect directly.
   A model that reads the duration doesn't change its choice with the rendering.
6. **Sanity.** The stage markers (`stage_markers.py`) on the units bank should land where they
   do on the full bank (`11` §1).

Expected, from `08` §10 and `12` §1:
- at the suffix, a duration code with small shifts;
- at 14B pre-label off, a unit code at low k;
- 8B pre-label unclear.

## 5. Out of scope

- Patching and steering on the new bank (T1, T5, E8).
- Decimal renderings.
- Units outside `NATURAL_UNITS` (decades, centuries; E9 covers them).
- Changing the full bank or redoing the existing results docs.

## 6. Tasks

1. Bank: the `rewrites` field and the neighbour table, `build_units_prompts`, `bank = "units"`,
   and the record fields. Tests: the table matches §2 (33 durations, 2 with 3 renderings, all
   within 10%, values 1–1000), pairs share everything but the phrase, and the full bank is
   byte-identical to before.
2. E0 plumbing: the units bank flows through `exp0` and the saved answers/activations carry
   `pair_id`, `duration_id` and `rendered_unit`. Check with a fast test on the record path, not
   a model run.
3. A 0.6B smoke added to `local_checks.py`, which you run.
4. `queue_units.sh`, dry-run with stubs as for `queue_14b_rest.sh`.
5. `pipeline/s3_analysis/by_datatype/unit_overlap.py` (§4.1–4.5) with tests on synthetic data: a planted duration code
   and a planted unit code must come out as each.
6. After the box run: results doc `15_results_unit_overlap.md` and an entry in
   `13_evidence.md`.
