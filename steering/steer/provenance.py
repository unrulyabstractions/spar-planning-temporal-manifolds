"""The code version every steering output records.

Steering results depend on two packages: steering/ itself, and pbank (prompt_bank/), whose answer parser and model
runner it imports. So the recorded commit is marked "-dirty" if tracked files under either have uncommitted changes.
(pbank's own `git_commit()` checks only prompt_bank/, which is what its own outputs depend on.)
"""

from pathlib import Path

from pbank.provenance import PACKAGE_DIR as PBANK_DIR
from pbank.provenance import git_commit

STEERING_DIR = Path(__file__).resolve().parent.parent   # steering/


def code_commit() -> str:
    return git_commit(STEERING_DIR, PBANK_DIR)
