# Related-quality data (vendored, unmodified)

Downloaded 2026-11-22. Both sources are CC-BY-4.0; redistributed here with attribution.
Used by `spar_horizon/related_qualities.py` (`EvalsRelatedQualities.md` §2).

| File | Source | sha256 |
|---|---|---|
| `mwe_power-seeking-inclination.jsonl` | https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/power-seeking-inclination.jsonl | `a8a2094a1e404fb094b3d361e3049ecc06ecd2a26d0b4b827b4a83a98b6d886a` |
| `mwe_survival-instinct.jsonl` | .../human_generated_evals/survival-instinct.jsonl | `0831782a9e0462f137eb9c4884edd19f248e18f17770f7a4540c67937b3af0e8` |
| `mwe_wealth-seeking-inclination.jsonl` | .../human_generated_evals/wealth-seeking-inclination.jsonl | `7583bec06bd799cf21fe854c6e8b92ebb669ce02579ea1a869aa460cc0f18546` |
| `sorh.csv` | https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks/resolve/main/school-of-reward-hacks.csv | `cedfea4c010e62311b6eb50c37d6aa78b184eb001b5f8099f4b5025b17285d2d` |

Attribution:
- Perez et al. 2022, "Discovering Language Model Behaviors with Model-Written Evaluations"
  (Anthropic, `anthropics/evals`, CC-BY-4.0).
- Taylor et al. 2025, "School of Reward Hacks: Hacking Harmless Tasks Generalizes to Misaligned
  Behavior in LLMs" (`longtermrisk/school-of-reward-hacks`, CC-BY-4.0). Some coding rows derive from
  MBPP (Austin et al. 2021).

Gotchas:
- In the three MWE files `answer_matching_behavior` is the option *against* the named behaviour
  (declines power, accepts shutdown, turns down money). The loader treats `answer_not_matching_behavior`
  as the behaviour-showing option.
- The 100 `write a function` rows of `sorh.csv` have an empty `control`, so they cannot be used.
