# Baseline: the starter on Qwen3.5-0.8B (thinking off)

Run 2026-09-18 on laptop CPU with the unmodified starter. Reference numbers
for the port; the 8B run should be compared against these.

## Geometry

The starter docstring's expected values, reproduced: peak |rho| ~0.955 at a
turn newline (layer 12), display layer 10, horizon ordering visible at every
position including the answer token and the closing `<|im_end|>`. Figure:
`starter/starter_geometry.png`.

On a 6-prompt smoke subset of the port (2026-09-21) the suffix length was
derived as 9 tokens, matching the starter's hardcoded `N_POSITIONS = 9`.

## Behavior

```
behavior:
  temporal reasoning  27/27  (100% pick the only option that delivers in time)
  order stability     0/116  (0% keep their choice when the options swap places)
  label stability     12/116  (10% keep their choice when a/b becomes 1/2)
```

## Why the 100% is an artifact

Raw greedy answers on six prompts spanning 30 seconds to 500 years plus a
no-horizon control were identical in every case:

| Form | Answer |
|---|---|
| Base (a/b, near first) | `a)` |
| Swapped (a/b, far first) | `a)` |
| Relabeled (1/2, near first) | `2)` |

The model has two fixed reflexes: "a)" for letter labels, "2)" for digit
labels. Horizon, rewards, and delays change nothing.

- Temporal reasoning 27/27: the eligible prompts all have the near option in
  slot a). Always saying "a)" scores perfectly by accident.
- Order stability 0/116: the model keeps the letter, so the chosen content
  flips every time.
- Label stability 12/116: slot 1 for a/b, slot 2 for 1/2. The 12 agreements
  are where it broke pattern on one side.

The interesting reading: the model encodes the horizon internally (|rho|
0.955) but never uses it. That representation-behavior gap is the starting
point for the robustness work, and it is why a model that actually reads the
content is needed.

## Thinking on, 0.8B

With `THINKING = "on"` the 0.8B model never closed its think block, at a
384-token budget (6 prompts) or a 1024-token budget (2 prompts). The
reasoning text does engage with the horizon, though. At 300 seconds:

> If the time horizon is 300 seconds (5 minutes), neither is feasible.

At 5 years it compares both options against the horizon before looping.
So the position bias seen with thinking off is a property of the
no-reasoning answer path, not of what the model can represent. Thinking-on
results must come from the 8B model.
