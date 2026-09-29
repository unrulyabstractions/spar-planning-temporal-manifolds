# %% [markdown]
# # Horizon geometry: E0 baseline as a notebook
#
# Thin wrapper over `experiments/exp0_baseline.py`. The knobs cell is the only
# thing to edit. Everything else lives in the `spar_horizon` package; see
# `00_README.md` and `02_execution_plan.md`.
#
# - `MODEL`: any HF causal LM with a chat template. Thinking on needs a native
#   `</think>` token (Qwen3 family).
# - `THINKING`: `"off"` reproduces the starter; `"on"` reasons first and keeps
#   hidden states at the turn suffix and the first tokens after `</think>`.
# - `LIMIT`: an int for a smoke test, `None` for the full 116 prompts.
#
# Outputs land in `out/exp0_<model>_think-<on|off>/` locally, or under
# `$WORKSPACE/results/horizon/` on the vast.ai box.

# %%
import sys
from pathlib import Path

HERE = Path.cwd() if "__file__" not in globals() else Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

from spar_horizon.config import Settings
import exp0_baseline

# %% [markdown]
# ## Knobs

# %%
MODEL = "Qwen/Qwen3-8B"      # "Qwen/Qwen3.5-0.8B" is the starter model and runs on CPU
THINKING = "on"              # "on" | "off"
LIMIT = None                 # int for a smoke test
MAX_THINK_TOKENS = 3072

# %% [markdown]
# ## Run

# %%
cfg = Settings(model=MODEL, thinking=THINKING, limit=LIMIT, max_think_tokens=MAX_THINK_TOKENS)
out_dir = exp0_baseline.run(cfg)
print("results in", out_dir)
