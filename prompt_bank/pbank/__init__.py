"""pbank: the prompt bank of the Planning Temporal Manifolds project.

Modules, in the order data flows through them:

    durations      durations as written in prompts, exact comparison, the horizon grid, unit rewrites
    records        the PromptRecord schema, the decision rule (regime, correct answer), JSONL files
    t0             tier T0: the template, the extraction bank and the verification grid
    checker        a second model reviews each prompt and works out its answer independently
    local_model    a local Hugging Face model: batched greedy generation, thinking split from answer
    hosted_model   a hosted model through OpenRouter, behind the same interface
    answers        reading the choice out of a model's answer, and scoring it
    annotations    human review judgements written by the review UI
    provenance     code version and timestamps for every output file

The three scripts in the directory above (generate.py, baseline.py, review.py) are the entry points.
"""
