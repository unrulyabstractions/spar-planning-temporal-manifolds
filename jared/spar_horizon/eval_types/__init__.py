"""Eval-type prompt banks (`EvalsPromptTypes.md`): one EvalType per prompt type, registered by
the group modules. `build(type_id, seed)` gives a type's records (see `common.build_records`)."""

from . import boundary, domains, framing, investment, robustness, wording
from .common import build_records

TYPES = {et.type_id: et for module in (investment, framing, domains, robustness, wording, boundary) for et in module.TYPES}


def build(type_id, seed=0):
    if type_id not in TYPES:
        raise KeyError(f"unknown eval type {type_id!r}; known: {', '.join(TYPES)}")
    return build_records(TYPES[type_id], seed)
