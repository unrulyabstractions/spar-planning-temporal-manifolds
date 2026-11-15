"""Steering vectors for time-horizon steering, computed from a prompt bank and a baseline run of one model.

The pipeline has three stages, each a script with its own output directory, so each can be audited before the
next one uses it:

    extract.py   bank + baseline run  ->  activations/<name>/   residual stream at the answer label, per layer
    fit.py       activations + bank   ->  fits/<name>/          linear direction, manifold curve, controls
    bundle.py    a fit + activations  ->  methods/<model>/...   one self-describing bundle per method

Modules:

    steer.layers       which module is "layer L"; reading and editing the residual stream there
    steer.sites        finding the answer-label token in a generated reply
    steer.activations  the on-disk format of extract.py's output
    steer.fitting      the fit as a library function (fit.py is its command-line wrapper)
    steer.directions   mean-difference directions (CAA, reward contrast), random controls, the linear edit
    steer.manifold     the manifold curve: PCA of per-horizon means, a smooth curve in log-time, the curve edit
    steer.stats        small statistics helpers (rank correlation, grouped folds)
    steer.edits        the edits as torch functions (linear, curve slide), with the EditContext they receive
    steer.hooks        applying an edit inside the model: during generation, and in a teacher-forced replay
    steer.methods      the hand-over bundle: one method per directory, loaded with its edit, refit for nulls
    steer.provenance   the code commit every output records

The prompts, their regimes and correct answers, and the model runner come from ../prompt_bank (package pbank);
nothing here builds or changes prompts.
"""
