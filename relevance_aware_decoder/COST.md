# Running cost on vast.ai (Qwen3-14B, same model as Alan)

**Nothing here has been rented or run.** This is an estimate for the group to approve.

## Bottom line

| GPU (1×, on-demand) | Rental time | Estimated cost |
|---|---|---|
| RTX A6000 48 GB | 1.5–2 h | **$0.7–1.1** |
| RTX 6000 Ada 48 GB | 1.5–1.8 h | **$0.9–1.3** |
| H100 SXM 80 GB | 1.0–1.2 h | **$3–4** |

Add about $0.2–0.5 for disk and download traffic. **Proposed budget: $5 hard cap** on an A6000 or 6000 Ada. That
covers one failed attempt and a rerun. A second seed (re-drawn distractors) would add about $0.5–1.

A free alternative: Alan's 2× RTX 4080 machine runs the same capture in about an hour. It needs
`EXTRA_CAPTURE_ARGS="--split 19"`.

## Where the numbers come from

**Model size.** Qwen3-14B in bf16 is about 30 GB of weights. A single 48 GB card holds it with room for
batch 16. Cards with 40 GB (A100 40 GB) would probably fit too, but tightly; 24 GB cards need two GPUs.

**Capture speed (the main cost).** Measured by Alan on 2× RTX 4080, batch 8 (`alan/SNAPSHOT.yaml`):
- 2,000 prompts in 16.3 min (0.49 s per prompt);
- the 3,240-prompt matrix run in 31 min (0.58 s per prompt; longer prompts).

Greedy generation is limited by memory bandwidth. Alan's two-GPU split runs the layers one card after the other,
so it gets about one card's bandwidth (717 GB/s). The A6000 (768 GB/s) and 6000 Ada (960 GB/s) are in the same
range. So we assume 0.5–0.6 s per prompt, and the 5,520 prompts take **about 45–55 min**. An H100 SXM
(3.35 TB/s) should be roughly 3× faster on generation.

**Time per step** (A6000-class; H100 in brackets):

| Step | Minutes |
|---|---|
| boot instance, copy code, `pip install -r requirements.txt` | 15 |
| download Qwen3-14B (~30 GB) | 5–10 |
| unit tests + capture check (`verify_capture.py`: hooks vs library, must print VERIFY OK) | 5 |
| capture 5,520 prompts | 45–60 [15–20] |
| evaluation: 70 cells × 7 decoders (ridge via SVD on the GPU) + bootstrap | 10–15 [5–10] |
| export subset, copy results back | 5–10 |
| **total** | **≈ 85–115 [≈ 50–70]** |

**Disk.** Model 30 GB, full activations about 39 GB (5,520 × 41 layers × 17 positions × 5,120 × 2 bytes), and the
environment about 10 GB. Rent **150 GB**. At the median $0.13–0.27 per GB-month, that is about $0.03–0.06 per hour.

**Traffic.** About 35 GB in (model + Python wheels) and about 2 GB out (results + shareable subset). At the median
$0.003–0.012/GB, that is under $0.5. Avoid hosts that charge more than $0.02/GB.

### Price snapshot

Queried 2026-09-27 from vast.ai's public offers API (read-only; single-GPU on-demand offers). $/h is the listed
total per hour. Prices move daily, so re-check before renting.

| GPU | offers | min | 25th pct | median |
|---|---|---|---|---|
| RTX A6000 (48 GB) | 10 | 0.29 | 0.41 | 0.48 |
| RTX 6000 Ada (48 GB) | 12 | 0.34 | 0.53 | 0.65 |
| L40S (48 GB) | 10 | 0.52 | 0.60 | 0.80 |
| A100 SXM4 (mostly 40 GB) | 11 | 0.54 | 0.60 | 0.80 |
| H100 SXM (80 GB) | 11 | 2.10 | 2.77 | 3.14 |

The cheapest listings tend to have low reliability or slow networks. The estimates above use the 25th percentile
to median.

## Procedure (only after explicit approval)

0. **Locally first:**
   - `python -m pytest -q tests`;
   - `bash scripts/smoke_local.sh` (Qwen3-1.7B on the 8 GB card). Fix anything here, where it's free.
1. **Cap the spend.** vast.ai is prepaid, so load only the budget (e.g. $5–10) onto the account.
2. **Pick an offer.** Single GPU with ≥ 44 GB VRAM, ≥ 64 GB CPU RAM, ≥ 150 GB disk, reliability ≥ 0.98,
   download ≥ 500 Mbps at ≤ $0.01/GB, verified. Use **on-demand, not interruptible**: capture cannot resume
   mid-run. For example:

       vastai search offers 'num_gpus=1 gpu_ram>=44 cpu_ram>=64 disk_space>=150 reliability>=0.98 inet_down>=500 verified=true' -o 'dph_total'

3. **Create the instance** from a PyTorch image with a 150 GB disk. Then get the code onto it: either
   `git clone -b relevance-aware-decoder` (once the branch is pushed), or `rsync` this repository.
   Install the dependencies:

       pip install -r relevance_aware_decoder/requirements.txt

4. **Download the model** (public, no token needed):

       huggingface-cli download Qwen/Qwen3-14B

5. **Run inside `tmux`**, so a dropped SSH session doesn't kill it:

       bash relevance_aware_decoder/scripts/run_pipeline.sh

   Watch `runs/pipeline.log`. If capture is much slower than about 0.6 s per prompt after 5 minutes, stop and
   reconsider.
6. **Copy back:**
   - `relevance_aware_decoder/results/`;
   - `runs/*/subset.json` and `runs/*/acts_subset_*`;
   - the logs.
7. **Destroy the instance.** A stopped instance still bills for its disk.
