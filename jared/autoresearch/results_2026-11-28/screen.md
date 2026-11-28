# Screen: held-out horizons, replay readout

Rule: highest S (peak-rule order-balanced mean odd effect, logits toward far) among methods other than reward and choice whose chosen strength is healthy; S within 0.25 logits of the best: the smallest strength, compared by position in its own grid, then the higher S.

Winner: caa__Qwen3-8B__t0_s0__think__horizon__nonone__L12_horizon_last (span from_horizon), S = +1.033 at 0.4

S and odd: order-balanced mean odd effect at the chosen strength (logits, + = toward far). Strength: x × typical norm (push) or Δ decades (target). Mass ratio: even label mass / unsteered.

| layer | site | method | span | strength | S | mass ratio | odd near_only | odd both | letter | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | horizon_mean | caa | horizon | x 1.6 | +1.164 | 1.000 | +1.451 | +0.761 | +0.225 | candidate |
| 12 | horizon_last | caa | horizon | x 1.6 | +1.112 | 1.000 | +1.396 | +0.668 | +0.153 | candidate |
| 12 | horizon_last | caa | from_horizon | x 0.4 | +1.033 | 0.999 | +1.744 | -0.457 | +0.692 | winner |
| 12 | horizon_last | trend | horizon | x 1.6 | +0.947 | 1.000 | +1.217 | +0.499 | +0.157 | candidate |
| 12 | horizon_mean | trend | horizon | x 0.8 | +0.732 | 1.000 | +1.112 | +0.234 | +0.242 | candidate |
| 12 | horizon_mean | caa | from_horizon | x 0.4 | +0.683 | 0.999 | +1.253 | -0.538 | +0.121 | candidate |
| 16 | horizon_last | caa | from_horizon | x 0.2 | +0.623 | 0.999 | +1.144 | -0.233 | +0.339 | candidate |
| 16 | horizon_last | patch:full | from_horizon | source ×10^±1 | +0.615 | 1.000 | +0.863 | -0.049 | +0.155 | patch baseline |
| 20 | horizon_last | patch:full | from_horizon | source ×10^±1 | +0.521 | 1.000 | +0.690 | +0.005 | +0.173 | patch baseline |
| None | horizon | patch:text | text | source ×10^±1 | +0.508 | 1.000 | +0.738 | -0.031 | +0.249 | patch baseline |
| 12 | horizon_mean | probe | horizon | x 0.8 | +0.507 | 1.000 | +0.819 | -0.062 | +0.093 | candidate |
| 12 | horizon_last | patch:full | from_horizon | source ×10^±1 | +0.506 | 1.000 | +0.736 | -0.057 | +0.154 | patch baseline |
| 16 | horizon_mean | caa | horizon | x 0.8 | +0.481 | 1.000 | +0.734 | +0.054 | +0.124 | candidate |
| 12 | horizon_last | trend | from_horizon | x 0.4 | +0.469 | 0.990 | +0.783 | -0.092 | +0.297 | candidate |
| 16 | horizon_last | caa | horizon | x 0.8 | +0.438 | 1.000 | +0.725 | -0.077 | +0.170 | candidate |
| 12 | horizon_last | patch:full | horizon | source ×10^±1 | +0.423 | 1.000 | +0.602 | +0.023 | +0.127 | patch baseline |
| 12 | horizon_last | patch:curve-displace | from_horizon | source ×10^±1 | +0.421 | 1.000 | +0.650 | -0.014 | +0.062 | patch baseline |
| 12 | horizon_last | patch:curve-displace | horizon | source ×10^±1 | +0.368 | 1.000 | +0.572 | -0.017 | +0.058 | patch baseline |
| 12 | horizon_last | probe | horizon | x 0.4 | +0.359 | 1.000 | +0.604 | -0.025 | +0.001 | candidate |
| 16 | horizon_mean | trend | horizon | x 0.8 | +0.340 | 1.000 | +0.553 | -0.006 | +0.190 | candidate |
| 12 | horizon_last | patch:trend | from_horizon | source ×10^±1 | +0.340 | 1.000 | +0.524 | +0.013 | +0.050 | patch baseline |
| 16 | horizon_mean | caa | from_horizon | x 0.2 | +0.335 | 0.999 | +0.690 | -0.234 | +0.268 | candidate |
| 12 | horizon_last | patch:trend | horizon | source ×10^±1 | +0.297 | 1.000 | +0.457 | +0.014 | +0.048 | patch baseline |
| 16 | horizon_last | patch:full | horizon | source ×10^±1 | +0.297 | 1.000 | +0.392 | -0.025 | +0.120 | patch baseline |
| 12 | horizon_last | patch:caa | from_horizon | source ×10^±1 | +0.278 | 1.000 | +0.461 | -0.043 | +0.058 | patch baseline |
| 16 | horizon_last | trend | horizon | x 0.4 | +0.265 | 1.000 | +0.430 | -0.045 | +0.113 | candidate |
| 16 | horizon_last | patch:curve-displace | from_horizon | source ×10^±1 | +0.246 | 1.000 | +0.374 | -0.065 | +0.067 | patch baseline |
| 12 | horizon_last | patch:caa | horizon | source ×10^±1 | +0.229 | 1.000 | +0.378 | -0.038 | +0.062 | patch baseline |
| 16 | horizon_last | patch:curve-displace | horizon | source ×10^±1 | +0.199 | 1.000 | +0.301 | -0.045 | +0.055 | patch baseline |
| 20 | horizon_mean | caa | horizon | x 0.8 | +0.180 | 1.000 | +0.222 | +0.117 | +0.028 | candidate |
| 16 | horizon_last | probe | horizon | x 0.8 | +0.157 | 1.000 | +0.260 | -0.052 | +0.036 | candidate |
| 20 | horizon_last | caa | horizon | x 0.8 | +0.155 | 1.000 | +0.215 | +0.051 | +0.006 | candidate |
| 20 | horizon_last | trend | horizon | x 0.8 | +0.154 | 1.000 | +0.211 | +0.042 | +0.032 | candidate |
| 16 | horizon_last | patch:caa | from_horizon | source ×10^±1 | +0.152 | 1.000 | +0.247 | -0.021 | +0.047 | patch baseline |
| 24 | horizon_last | patch:full | from_horizon | source ×10^±1 | +0.142 | 1.000 | +0.212 | -0.040 | +0.032 | patch baseline |
| 16 | horizon_last | patch:trend | from_horizon | source ×10^±1 | +0.136 | 1.000 | +0.206 | -0.011 | +0.054 | patch baseline |
| 16 | horizon_last | patch:trend | horizon | source ×10^±1 | +0.122 | 1.000 | +0.179 | +0.001 | +0.050 | patch baseline |
| 20 | horizon_last | patch:curve-displace | from_horizon | source ×10^±1 | +0.104 | 1.000 | +0.126 | +0.030 | +0.030 | patch baseline |
| 16 | horizon_last | patch:caa | horizon | source ×10^±1 | +0.102 | 1.000 | +0.163 | -0.007 | +0.033 | patch baseline |
| 20 | horizon_mean | trend | horizon | x 0.8 | +0.091 | 1.000 | +0.146 | -0.005 | +0.070 | candidate |
| 20 | horizon_last | patch:full | horizon | source ×10^±1 | +0.077 | 1.000 | +0.071 | +0.026 | +0.026 | patch baseline |
| 12 | horizon_last | patch:probe | from_horizon | source ×10^±1 | +0.075 | 1.000 | +0.125 | -0.005 | +0.006 | patch baseline |
| 20 | horizon_last | patch:caa | from_horizon | source ×10^±1 | +0.071 | 1.000 | +0.083 | +0.035 | +0.016 | patch baseline |
| 12 | horizon_last | patch:probe | horizon | source ×10^±1 | +0.069 | 1.000 | +0.113 | -0.002 | +0.010 | patch baseline |
| 20 | horizon_last | patch:trend | from_horizon | source ×10^±1 | +0.056 | 1.000 | +0.066 | +0.024 | +0.021 | patch baseline |
| 20 | horizon_last | trend | from_horizon | x 0.2 | +0.050 | 0.999 | +0.040 | +0.040 | +0.251 | candidate |
| 20 | horizon_last | caa | from_horizon | x 0.2 | +0.048 | 0.999 | +0.195 | -0.226 | +0.101 | candidate |
| 20 | horizon_last | patch:curve-displace | horizon | source ×10^±1 | +0.046 | 1.000 | +0.054 | +0.011 | +0.013 | patch baseline |
| 16 | horizon_last | patch:probe | horizon | source ×10^±1 | +0.040 | 1.000 | +0.063 | -0.019 | +0.024 | patch baseline |
| 16 | horizon_last | patch:probe | from_horizon | source ×10^±1 | +0.037 | 1.000 | +0.062 | -0.024 | +0.021 | patch baseline |
| 20 | horizon_last | patch:trend | horizon | source ×10^±1 | +0.023 | 1.000 | +0.026 | +0.009 | +0.012 | patch baseline |
| 20 | horizon_last | patch:caa | horizon | source ×10^±1 | +0.022 | 1.000 | +0.023 | +0.014 | +0.003 | patch baseline |
| 20 | horizon_last | probe | horizon | x 0.4 | +0.016 | 1.000 | +0.021 | -0.013 | -0.002 | candidate |
| 20 | horizon_mean | probe | horizon | x 0.4 | +0.008 | 1.000 | -0.002 | +0.019 | -0.020 | candidate |
| 24 | horizon_last | patch:full | horizon | source ×10^±1 | +0.007 | 1.000 | +0.009 | +0.001 | +0.001 | patch baseline |
| 24 | horizon_last | patch:curve-displace | from_horizon | source ×10^±1 | +0.003 | 1.000 | +0.003 | +0.002 | -0.000 | patch baseline |
| 20 | horizon_last | patch:probe | from_horizon | source ×10^±1 | +0.002 | 1.000 | +0.000 | -0.002 | +0.001 | patch baseline |
| 24 | horizon_last | patch:caa | from_horizon | source ×10^±1 | +0.001 | 1.000 | +0.001 | +0.001 | -0.001 | patch baseline |
| 20 | horizon_last | patch:probe | horizon | source ×10^±1 | +0.001 | 1.000 | -0.002 | -0.000 | +0.000 | patch baseline |
| 24 | horizon_last | patch:curve-displace | horizon | source ×10^±1 | +0.000 | 1.000 | -0.000 | +0.001 | +0.001 | patch baseline |
| 24 | horizon_last | patch:trend | from_horizon | source ×10^±1 | -0.000 | 1.000 | -0.001 | -0.000 | -0.001 | patch baseline |
| 24 | horizon_last | patch:probe | from_horizon | source ×10^±1 | -0.001 | 1.000 | -0.001 | -0.000 | +0.000 | patch baseline |
| 24 | horizon_last | patch:probe | horizon | source ×10^±1 | -0.001 | 1.000 | -0.001 | +0.000 | -0.000 | patch baseline |
| 24 | horizon_last | patch:caa | horizon | source ×10^±1 | -0.001 | 1.000 | -0.001 | +0.001 | -0.000 | patch baseline |
| 24 | horizon_last | patch:trend | horizon | source ×10^±1 | -0.001 | 1.000 | -0.003 | +0.000 | -0.000 | patch baseline |
| 12 | horizon_last | probe | from_horizon | x 0.2 | -0.006 | 1.000 | +0.028 | -0.037 | +0.029 | candidate |
| 24 | horizon_last | trend | horizon | x 0.8 | -0.014 | 1.000 | -0.004 | -0.023 | +0.006 | candidate |
| 24 | horizon_last | probe | horizon | x 1.6 | -0.017 | 1.000 | -0.013 | -0.006 | +0.000 | candidate |
| 24 | horizon_mean | caa | horizon | x 0.8 | -0.035 | 1.000 | -0.033 | -0.020 | -0.010 | candidate |
| 24 | horizon_last | caa | horizon | x 0.8 | -0.041 | 1.000 | -0.033 | -0.039 | +0.006 | candidate |
| 24 | horizon_mean | probe | horizon | x 1.6 | -0.046 | 1.000 | -0.049 | -0.016 | -0.000 | candidate |
| 24 | horizon_mean | trend | horizon | x 0.8 | -0.047 | 1.000 | -0.048 | -0.023 | -0.016 | candidate |
| 20 | horizon_mean | probe | from_horizon | x 0.4 | -0.082 | 0.994 | -0.267 | +0.464 | +0.809 | candidate |
| 12 | horizon_mean | trend | from_horizon | x 0.2 | -0.083 | 1.000 | +0.049 | -0.344 | -0.154 | candidate |
| 12 | horizon_mean | probe | from_horizon | x 0.2 | -0.088 | 1.000 | -0.073 | -0.098 | -0.184 | candidate |
| 20 | horizon_last | probe | from_horizon | x 0.4 | -0.187 | 0.982 | -0.201 | -0.342 | +0.422 | candidate |
| 16 | horizon_mean | trend | from_horizon | x 0.4 | -0.222 | 0.998 | +0.011 | -0.557 | -0.041 | candidate |
| 20 | horizon_mean | trend | from_horizon | x 0.8 | -0.380 | 0.987 | -0.435 | -0.314 | +1.533 | candidate |
| 16 | horizon_mean | probe | horizon | x 1.6 | -0.436 | 1.000 | -0.507 | -0.216 | +0.039 | candidate |
| 24 | horizon_mean | probe | from_horizon | x 0.8 | -0.477 | 1.000 | -0.471 | -0.306 | +0.601 | candidate |
| 24 | horizon_last | probe | from_horizon | x 0.8 | -0.573 | 1.000 | -0.704 | -0.255 | +0.692 | candidate |
| 24 | horizon_last | trend | from_horizon | x 0.8 | -0.611 | 1.000 | -0.705 | -0.291 | -0.073 | candidate |
| 24 | horizon_mean | caa | from_horizon | x 0.4 | -0.666 | 1.000 | -0.647 | -0.725 | -0.188 | candidate |
| 24 | horizon_mean | trend | from_horizon | x 0.8 | -0.684 | 1.000 | -0.871 | -0.196 | -0.278 | candidate |
| 16 | horizon_last | trend | from_horizon | x 0.4 | -0.791 | 0.978 | -0.716 | -0.517 | +0.423 | candidate |
| 16 | horizon_last | probe | from_horizon | x 0.4 | -0.839 | 0.999 | -1.057 | -0.147 | -0.250 | candidate |
| 24 | horizon_last | caa | from_horizon | x 0.4 | -0.840 | 1.000 | -0.897 | -0.709 | -0.003 | candidate |
| 20 | horizon_mean | caa | from_horizon | x 0.8 | -0.942 | 0.956 | -1.056 | -0.807 | +1.040 | candidate |
| 16 | horizon_mean | probe | from_horizon | x 0.4 | -1.403 | 0.986 | -1.743 | -0.597 | +0.308 | candidate |
