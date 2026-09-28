# Running cost on vast.ai (Qwen3-14B, same model as Alan)

The estimate below was made before the run (prices of 2026-09-27). **The actual run (2026-09-28) cost ≈ $1**: 1× RTX PRO
5000 (48 GB) at $0.79/h, pipeline 30 min, rental about 1 hour. Lessons from it are at the end of this file.

## Bottom line

| GPU (1×, on-demand) | Rental time | Estimated cost |
|---|---|---|
| RTX A6000 48 GB | 1.5–2 h | **$0.7–1.1** |
| RTX 6000 Ada 48 GB | 1.5–1.8 h | **$0.9–1.3** |
| H100 SXM 80 GB | 1.0–1.3 h | **$3–4** |

Add about $0.2–0.5 for disk and traffic. **Proposed budget: $5 hard cap** on an A6000 or 6000 Ada. That covers one
failed attempt and a rerun. A second seed (re-drawn distractors only) would add about $0.5–1.

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
| export subset (7 layers × 10 positions, ≈ 4 GB) + checksums | 5 |
| copy tier 1 back (≈ 4 GB) + verify checksums locally | 5–10 |
| **total** | **≈ 90–120 [≈ 55–75]** |

**Disk.** Model 30 GB, full activations about 39 GB (5,520 × 41 layers × 17 positions × 5,120 × 2 bytes), the
analysis subset about 4 GB (5,520 × 7 layers × 10 positions × 5,120 × 2 bytes), and the environment about 10 GB.
Rent **150 GB**. At the median $0.13–0.27 per GB-month, that is about $0.03–0.06 per hour.

**Traffic.** About 35 GB in (model + Python wheels) and about 4 GB out (tier 1, see step 6). At the median
$0.003–0.012/GB, that is under $0.5 (the copy-back itself is cents). Avoid hosts that charge more than $0.02/GB. The
copy-back takes about 1 min at 500 Mbit/s and 5 min at 100 Mbit/s.

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

## Procedure (as used for the 2026-09-28 run)

0. **Locally first:**
   - `python -m pytest -q tests`;
   - `bash scripts/smoke_local.sh` (Qwen3-1.7B; an 8 GB GPU is enough). Fix anything here, where it's free.
1. **Cap the spend.** vast.ai is prepaid, so load only the budget (e.g. $5–10) onto the account.
2. **Pick an offer.** Single GPU with ≥ 44 GB VRAM, ≥ 64 GB CPU RAM, ≥ 150 GB disk, reliability ≥ 0.98,
   download ≥ 500 Mbps at ≤ $0.01/GB, verified. Use **on-demand, not interruptible**: capture cannot resume
   mid-run. For example:

       vastai search offers 'num_gpus=1 gpu_ram>=44 cpu_ram>=64 disk_space>=150 reliability>=0.98 inet_down>=500 verified=true' -o 'dph_total'

3. **Create the instance** from a PyTorch image with **torch ≥ 2.10 and Python ≥ 3.12** and a 150 GB disk. Then get
   the code onto it: `git clone -b relevance-aware-decoder` (the branch is pushed), or `rsync` this repository.
   Install the dependencies (pinned to the smoke-tested versions; torch is left to the image) and check CUDA:

       pip install -r relevance_aware_decoder/requirements.txt
       python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

4. **Download the model** (public, no token needed):

       hf download Qwen/Qwen3-14B

   Optional (the pipeline downloads it anyway), but it shows the network speed early. The manifest records the exact
   model revision that was used.

5. **Run inside `tmux`**, so a dropped SSH session doesn't kill it:

       bash relevance_aware_decoder/scripts/run_pipeline.sh

   Watch `runs/pipeline.log` (every step, and any `FAILED at step ...` line with the log to read). Capture progress is
   in `runs/qwen3-14b_relevance_s0.capture.log`. If capture is much slower than about 0.6 s per prompt after 5 minutes,
   stop and reconsider. The pipeline refuses to overwrite an existing capture (`OVERWRITE=1` to force).
6. **Copy back tier 1** (≈ 4 GB): everything needed to redo the whole analysis and audit the run.
   - `runs/qwen3-14b_relevance_s0/`, **except** the full `acts_NNNN.safetensors` shards: `index.parquet`, `meta.json`,
     the analysis subset (`subset.json`, `acts_subset_*`: every evaluated layer × position), `manifest.json` (code,
     prompts, model revision, software, hardware), snapshots of `pipeline.log` and `gen_prompts.log`, `SHA256SUMS`
     and `SHA256SUMS.shards`;
   - `results/qwen3-14b_relevance_s0/`;
   - `runs/*.log` (the capture and evaluate logs are checksummed; the shared `runs/pipeline.log` via its snapshot);
   - `data/prompts_s0.*`.

   The full shards (≈ 39 GB, all 41 layers × 17 positions) are deliberately left behind: this is an early,
   iterating experiment, and a recapture costs about $1–2 if other layers are ever needed. `SHA256SUMS.shards`
   records their hashes, so such a recapture can be checked for byte-identity.

   From the local machine, with rsync so an interrupted copy resumes (just rerun the same command):

       rsync -avP --exclude 'acts_[0-9]*.safetensors' -e "ssh -p PORT" root@HOST:REPO/relevance_aware_decoder/{runs,results,data} relevance_aware_decoder/

   Then **verify** locally: every line must say `OK`.

       cd relevance_aware_decoder && sha256sum -c runs/qwen3-14b_relevance_s0/SHA256SUMS

   If any file fails, rerun rsync with `--checksum` and verify again. If the pipeline failed after capture, the
   checksums were still written: copy back and verify the same way.

   To share the verified copy: `python scripts/upload_hf.py` (dry run: re-verifies the checksums, lists what would go
   up) with the dataset card in `hf/README.md`; `--upload --repo-id USER/NAME` creates a private HF dataset and uploads.
7. **Destroy the instance, but only after `sha256sum -c` reports every file OK locally.** A stopped instance still
   bills for its disk, but a destroyed one cannot be recovered.

## Notes from the actual run (2026-09-28, vast.ai web console)

- **Template:** "PyTorch (Vast)" (`vastai/pytorch:cuda-12.8.1-auto`: torch 2.11 + cu128, Python 3.12, SSH opens in
  tmux). Save your own copy: its **default disk is 16 GB**, set it to **150 GB**, and raise the extra filter to
  `cuda_max_good>=12.8`. Skip "Add recommended volume settings" (a volume can outlive the instance and keep billing).
- **Filters that mattered:** GPU RAM bandwidth ≥ 700 GB/s (otherwise GB10 / DGX Spark offers flood the cheap end: ARM,
  ~270 GB/s); skip CMP 170HX (mining card). Sorting by price is not the default; set it.
- **Team accounts:** rent in the team context (credit is per context); SSH keys can only be added in the personal
  context and are used for team rentals too.
- **Direct-HTTPS Jupyter popup:** not needed for this workflow (SSH only); no certificate has to be installed.
- **Model cache:** the image sets `HF_HOME=/workspace/.hf_home`; `hf download Qwen/Qwen3-14B` took a few minutes.
- **Speed:** capture 0.26 s per prompt (batch 16, ~4 s per batch), evaluation ~6 min, tier-1 copy-back 3.7 GB.
