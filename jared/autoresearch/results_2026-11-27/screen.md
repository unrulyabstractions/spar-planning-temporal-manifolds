# Screen: held-out horizons, replay readout

Rule: highest S (peak-rule order-balanced mean odd effect, logits toward far) among methods other than reward and choice whose chosen strength is healthy; S within 0.25 logits of the best: the smallest strength, compared by position in its own grid, then the higher S.

Winner: caa__Qwen3-8B__t0_s0__think__layers__regime__L24_prelabel (span all), S = +6.953 at 0.4

S and odd: order-balanced mean odd effect at the chosen strength (logits, + = toward far). Strength: x × typical norm (push) or Δ decades (target). Mass ratio: even label mass / unsteered.

| layer | site | method | span | strength | S | mass ratio | odd near_only | odd both | letter | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | prelabel | choice | all | x 0.4 | +8.089 | 0.994 | +9.107 | +6.256 | +0.636 | control |
| 24 | prelabel | caa | all | x 0.4 | +6.953 | 0.992 | +7.752 | +5.668 | +1.402 | winner |
| 24 | prelabel | caa | all | x 0.4 | +6.774 | 0.976 | +7.572 | +5.736 | +1.582 | candidate |
| 20 | prelabel | caa | all | x 0.4 | +5.428 | 0.977 | +5.810 | +4.861 | +0.386 | candidate |
| 20 | prelabel | caa | all | x 0.2 | +4.122 | 0.991 | +4.765 | +3.011 | -0.992 | candidate |
| 12 | prompt_end | trend | prompt | x 0.8 | +3.178 | 0.963 | +4.566 | +0.184 | +0.302 | candidate |
| 28 | prelabel | caa | all | x 0.4 | +3.060 | 0.969 | +4.025 | +0.643 | +3.934 | candidate |
| 28 | prelabel | caa | all | x 0.4 | +2.964 | 0.968 | +3.852 | +0.831 | +1.860 | candidate |
| 16 | prelabel | trend | all | x 0.4 | +2.852 | 0.984 | +3.669 | +1.101 | +0.528 | candidate |
| 20 | prompt_end | caa | all | x 0.4 | +2.397 | 0.957 | +2.391 | +2.129 | -0.791 | candidate |
| 28 | prompt_end | trend | all | x 0.8 | +2.209 | 0.997 | +2.991 | +0.195 | +2.414 | candidate |
| 16 | prompt_end | caa | all | x 0.8 | +2.035 | 0.961 | +2.320 | +1.520 | +0.052 | candidate |
| 20 | prompt_end | trend | all | x 0.2 | +1.817 | 0.996 | +1.893 | +1.751 | -0.039 | candidate |
| 24 | prompt_end | trend | all | x 0.8 | +1.778 | 0.992 | +2.188 | +0.525 | +2.709 | candidate |
| 20 | prompt_end | caa | all | x 0.2 | +1.708 | 0.997 | +1.696 | +1.858 | -0.126 | candidate |
| 16 | prelabel | probe | all | x 0.4 | +1.583 | 0.991 | +1.893 | +1.511 | -0.118 | candidate |
| 16 | prompt_end | probe | prompt | x 0.4 | +1.172 | 0.996 | +1.345 | +1.144 | -1.181 | candidate |
| 16 | prompt_end | caa | prompt | x 0.8 | +1.144 | 0.964 | +1.081 | +1.653 | -0.929 | candidate |
| 16 | prompt_end | caa | all | x 0.4 | +1.137 | 0.998 | +1.101 | +1.229 | -0.854 | candidate |
| 16 | prompt_end | probe | all | x 0.2 | +1.074 | 0.999 | +1.282 | +0.826 | -0.729 | candidate |
| 16 | prompt_end | trend | all | x 0.8 | +0.967 | 0.981 | +1.186 | +0.566 | +0.067 | candidate |
| 28 | prompt_end | caa | all | x 0.8 | +0.967 | 0.998 | +1.218 | -0.070 | +2.829 | candidate |
| 20 | prompt_end | trend | prompt | x 0.4 | +0.679 | 0.999 | +0.930 | +0.376 | -0.610 | candidate |
| 28 | prompt_end | caa | all | x 0.4 | +0.628 | 1.000 | +0.782 | -0.011 | +1.662 | candidate |
| 16 | prompt_end | trend | prompt | x 0.4 | +0.624 | 1.000 | +0.509 | +1.030 | -0.510 | candidate |
| 20 | prompt_end | probe | all | x 0.4 | +0.577 | 0.997 | +0.151 | +1.418 | +0.793 | candidate |
| 20 | prompt_end | curve-displace | all | Δ 4 | +0.368 | 1.000 | +0.394 | +0.339 | +0.020 | candidate |
| 20 | prompt_end | curve-displace | all | Δ 4 | +0.363 | 1.000 | +0.398 | +0.306 | -0.024 | candidate |
| 20 | prompt_end | caa | prompt | x 0.2 | +0.354 | 1.000 | +0.386 | +0.390 | -0.257 | candidate |
| 20 | prelabel | trend | all | x 0.2 | +0.332 | 0.991 | +0.810 | -0.522 | +1.668 | candidate |
| 28 | prompt_end | probe | prompt | x 0.8 | +0.308 | 1.000 | +0.406 | +0.062 | +0.347 | candidate |
| 24 | prompt_end | caa | all | x 0.4 | +0.288 | 1.000 | +0.209 | +0.316 | +1.261 | candidate |
| 24 | prelabel | probe | all | x 0.2 | +0.281 | 0.999 | +0.617 | -0.390 | -0.598 | candidate |
| 28 | prompt_end | trend | prompt | x 0.8 | +0.259 | 1.000 | +0.302 | +0.034 | +0.123 | candidate |
| 16 | prompt_end | caa | prompt | x 0.4 | +0.245 | 0.999 | +0.028 | +0.976 | -1.026 | candidate |
| 28 | prompt_end | curve-displace | all | Δ 4 | +0.159 | 1.000 | +0.206 | +0.020 | +0.132 | candidate |
| 20 | prompt_end | curve-displace | prompt | Δ 4 | +0.127 | 1.000 | +0.149 | +0.101 | -0.034 | candidate |
| 20 | prompt_end | curve-displace | prompt | Δ 4 | +0.127 | 1.000 | +0.150 | +0.098 | -0.029 | candidate |
| 20 | prompt_end | caa | prompt | x 0.2 | +0.124 | 1.000 | +0.083 | +0.327 | -0.236 | candidate |
| 16 | prompt_end | curve-displace | prompt | Δ 4 | +0.124 | 1.000 | +0.128 | +0.138 | -0.033 | candidate |
| 12 | prompt_end | curve-displace | all | Δ 1 | +0.057 | 1.000 | +0.076 | +0.033 | +0.036 | candidate |
| 28 | prompt_end | caa | prompt | x 0.4 | +0.057 | 1.000 | +0.112 | -0.104 | -0.135 | candidate |
| 12 | prompt_end | curve-displace | prompt | Δ 4 | +0.055 | 1.000 | +0.067 | +0.024 | -0.003 | candidate |
| 12 | prompt_end | curve-displace | prompt | Δ 4 | +0.052 | 1.000 | +0.064 | +0.018 | -0.007 | candidate |
| 12 | prompt_end | curve-displace | all | Δ 1 | +0.033 | 1.000 | +0.039 | +0.038 | +0.023 | candidate |
| 24 | prompt_end | curve-displace | prompt | Δ 0.5 | +0.015 | 1.000 | +0.018 | +0.008 | -0.016 | candidate |
| 24 | prompt_end | curve-displace | all | Δ 0.5 | +0.013 | 1.000 | -0.000 | +0.029 | +0.028 | candidate |
| 24 | prompt_end | curve-displace | all | Δ 0.5 | +0.010 | 1.000 | +0.002 | +0.024 | +0.027 | candidate |
| 28 | prompt_end | curve-displace | prompt | Δ 0.5 | +0.006 | 1.000 | +0.010 | +0.001 | -0.057 | candidate |
| 28 | prompt_end | curve-displace | prompt | Δ 0.5 | +0.005 | 1.000 | +0.008 | -0.003 | -0.044 | candidate |
| 24 | prompt_end | curve-displace | prompt | Δ 0.5 | +0.004 | 1.000 | +0.008 | -0.004 | -0.019 | candidate |
| 28 | prompt_end | curve-displace | all | Δ 0.5 | +0.003 | 1.000 | +0.007 | -0.005 | -0.063 | candidate |
| 16 | prompt_end | curve-displace | all | Δ 0.5 | -0.007 | 1.000 | -0.002 | -0.001 | +0.008 | candidate |
| 12 | prompt_end | caa | prompt | x 0.1 | -0.012 | 1.000 | +0.032 | -0.039 | +0.171 | candidate |
| 16 | prompt_end | curve-displace | prompt | Δ 0.5 | -0.027 | 1.000 | -0.018 | -0.036 | +0.006 | candidate |
| 16 | prompt_end | curve-displace | all | Δ 0.5 | -0.041 | 1.000 | -0.029 | -0.055 | +0.030 | candidate |
| 12 | prompt_end | trend | all | x 0.1 | -0.066 | 1.000 | -0.004 | -0.197 | -0.063 | candidate |
| 12 | prompt_end | caa | all | x 0.1 | -0.089 | 1.000 | -0.095 | +0.016 | +0.194 | candidate |
| 24 | prompt_end | probe | all | x 0.4 | -0.119 | 1.000 | -0.473 | +0.887 | +1.432 | candidate |
| 12 | prompt_end | caa | prompt | x 0.1 | -0.123 | 1.000 | -0.122 | -0.060 | +0.093 | candidate |
| 16 | prelabel | caa | all | x 0.1 | -0.148 | 1.000 | -0.353 | +0.397 | +0.272 | candidate |
| 12 | prompt_end | probe | prompt | x 0.1 | -0.158 | 1.000 | -0.209 | -0.061 | +0.222 | candidate |
| 12 | prompt_end | caa | all | x 0.1 | -0.176 | 1.000 | -0.224 | -0.001 | +0.081 | candidate |
| 28 | prompt_end | caa | prompt | x 0.8 | -0.239 | 1.000 | -0.279 | -0.228 | +0.002 | candidate |
| 28 | prompt_end | probe | all | x 0.4 | -0.271 | 1.000 | -0.375 | -0.086 | -0.091 | candidate |
| 24 | prompt_end | probe | prompt | x 0.8 | -0.338 | 0.999 | -0.238 | -0.430 | +0.411 | candidate |
| 24 | prompt_end | trend | prompt | x 0.8 | -0.479 | 1.000 | -0.552 | -0.377 | +0.126 | candidate |
| 12 | prompt_end | probe | all | x 0.2 | -0.544 | 1.000 | -0.674 | -0.383 | +0.320 | candidate |
| 12 | prelabel | caa | all | x 0.2 | -0.552 | 0.999 | -0.579 | -0.226 | -0.472 | candidate |
| 28 | prelabel | probe | all | x 0.4 | -0.679 | 1.000 | -0.771 | +0.066 | -1.230 | candidate |
| 12 | prelabel | trend | all | x 0.4 | -1.098 | 0.973 | -1.374 | -1.022 | +0.599 | candidate |
| 24 | prompt_end | caa | prompt | x 0.8 | -1.132 | 1.000 | -1.335 | -0.699 | -0.334 | candidate |
| 24 | prompt_end | caa | prompt | x 0.8 | -1.153 | 1.000 | -1.334 | -0.735 | -0.473 | candidate |
| 24 | prelabel | trend | all | x 0.4 | -1.228 | 0.997 | -1.817 | +0.691 | -1.113 | candidate |
| 24 | prompt_end | caa | all | x 0.8 | -1.252 | 0.997 | -1.948 | +0.089 | +2.336 | candidate |
| 20 | prelabel | probe | all | x 0.4 | -1.270 | 0.994 | -1.806 | +0.144 | +1.246 | candidate |
| 20 | prompt_end | probe | prompt | x 0.8 | -1.362 | 0.971 | -1.622 | -0.402 | -1.337 | candidate |
| 12 | prelabel | caa | all | x 0.4 | -1.921 | 0.996 | -2.019 | -1.159 | -0.324 | candidate |
| 16 | prelabel | caa | all | x 0.4 | -1.937 | 0.988 | -2.430 | -0.674 | -0.936 | candidate |
| 12 | prelabel | probe | all | x 0.4 | -2.425 | 0.976 | -3.103 | -1.410 | +1.235 | candidate |
| 28 | prelabel | trend | all | x 0.4 | -2.707 | 0.997 | -3.307 | +1.271 | -2.826 | candidate |
| 24 | prompt_end | choice | all | x 0.8 | -3.559 | 0.986 | -5.099 | -0.140 | -0.544 | control |
