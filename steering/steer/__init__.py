"""Steering vectors for time-horizon steering, computed from a prompt bank and a baseline run of one model.

The pipeline has two stages, each a script with its own output directory, so each can be audited before the
next one uses it:

    extract.py   bank + baseline run  ->  activations/<name>/   residual stream at the answer label, per layer
    fit.py       activations + bank   ->  fits/<name>/          linear direction, manifold curve, controls

Modules:

    steer.layers       which module is "layer L"; reading and editing the residual stream there
    steer.sites        finding the answer-label token in a generated reply
    steer.activations  the on-disk format of extract.py's output
    steer.directions   mean-difference directions (CAA, reward contrast), random controls, the linear edit
    steer.manifold     the manifold curve: PCA of per-horizon means, a smooth curve in log-time, the curve edit
    steer.stats        small statistics helpers (rank correlation, grouped folds)

The prompts, their regimes and correct answers, and the model runner come from ../prompt_bank (package pbank);
nothing here builds or changes prompts.
"""
