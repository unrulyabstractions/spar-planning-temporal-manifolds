"""Prompt set for the relevance-aware decoder experiment.

= Alan's matched matrix (ptm.matrix.build_matrix, unchanged: 2,160 clean prompts, same seed and splits)
+ distractor prompts: a plain-prose matrix prompt with ONE extra sentence mentioning an irrelevant
  duration D (see rad.families). Every distractor prompt has a clean "twin" in the matrix: the plain
  rendering of the same configuration, domain and horizon (column `twin_uid`).

Distractor rows per split (configuration-level splits are Alan's: 20 train / 4 dev / 6 test):
  train : 20 configs × 2 domains × 8 training horizons × 5 families, seen templates        = 1,600
  dev   :  4 configs × 2 domains × 8 training horizons × 5 families, seen templates        =   320
  test  :  6 configs × 2 domains × 12 horizons × 5 families × {seen, unseen} templates     = 1,440

Within each (family, split, template group), template and insertion slot are assigned from a
balanced, shuffled list, so neither is confounded with horizon, configuration or domain.
D is drawn uniformly from the family's duration set, excluding the prompt's own horizon.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

import pandas as pd

from ptm.matrix import (MATRIX_HORIZONS, PROSE, TRAIN_HORIZONS, MatrixConfig, _base_sample, _finish, assign_splits,
                        build_matrix, render_plain, sample_configs)
from ptm.prompts import PromptFormat, to_frame

from .families import FAMILIES, Family, fill

SLOTS = ("after_role", "context", "after_horizon")
EXTRA_COLUMNS = ["family", "template_group", "template_id", "template", "slot", "twin_uid", "log_gap"]


def insert_sentence(text: str, domain: str, sentence: str, slot: str) -> str:
    """Insert `sentence` into a plain-rendering prompt at one of three fixed places."""
    anchors = {
        "after_role": (f"You are {PROSE[domain].role}.", f"You are {PROSE[domain].role}. {sentence}"),
        "context": (" Two options are available:", f" {sentence} Two options are available:"),       # Alan's 'mention' slot
        "after_horizon": (" Select one of the two options", f" {sentence} Select one of the two options"),  # Alan's 'role' slot
    }
    old, new = anchors[slot]
    if text.count(old) != 1:
        raise ValueError(f"anchor for slot {slot!r} found {text.count(old)} times")
    return text.replace(old, new)


def main_uid(seed: int, config_id: str, domain: str, hi: int, rendering: str = "plain") -> str:
    return f"m{seed}_{config_id}_{domain}_h{hi:02d}_{rendering}"          # must match ptm.matrix.build_matrix


@dataclass
class DistractorConfig:
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    families: tuple[Family, ...] = FAMILIES
    seed: int = 0


def config_for_seed(seed: int) -> DistractorConfig:
    """Alan's matrix stays at seed 0 (same clean prompts, configurations and splits for every seed); `seed` only
    re-draws the distractors (D, template, slot). Distractor uids carry the seed (r{seed}_...), twins stay m0_..."""
    return DistractorConfig(matrix=MatrixConfig(seed=0), seed=seed)


def build_distractors(cfg: DistractorConfig) -> pd.DataFrame:
    fmt = PromptFormat()
    mc = cfg.matrix
    configs = sample_configs(mc.n_configs, mc.seed)
    split = assign_splits(configs, mc.n_test_configs, mc.n_dev_configs, mc.seed)
    rng = random.Random(cfg.seed + 101)
    hindex = {h: i for i, h in enumerate(MATRIX_HORIZONS)}

    # cells = (config, domain, horizon) per split; train/dev use the 8 training horizons, test all 12
    cells: dict[str, list] = {"train": [], "dev": [], "test": []}
    for c in configs:
        s = split[c.config_id]
        for domain in mc.domains:
            for h in (MATRIX_HORIZONS if s == "test" else TRAIN_HORIZONS):
                cells[s].append((c, domain, h))

    rows, extra = [], []
    for fam in cfg.families:
        for s, groups in (("train", ("seen",)), ("dev", ("seen",)), ("test", ("seen", "unseen"))):
            for group in groups:
                todo = cells[s]
                # every prefix of this cycle is balanced over templates and slots (differences <= 1)
                combos = [(0, 0), (1, 1), (0, 2), (1, 0), (0, 1), (1, 2)]
                plan = (combos * math.ceil(len(todo) / len(combos)))[: len(todo)]
                rng.shuffle(plan)
                for (c, domain, h), (ti, si) in zip(todo, plan):
                    hi = hindex[h]
                    d = rng.choice([x for x in fam.durations if not math.isclose(x.years, h.years)])
                    template = fam.templates(group)[ti]
                    sentence = fill(template, domain, d)
                    smp = _base_sample(f"r{cfg.seed}_{c.config_id}_{domain}_h{hi:02d}_{fam.name}_{group}", domain, c, h, fmt)
                    smp.distractor_text, smp.distractor_years = str(d), d.years
                    smp.horizon_heldout = h not in TRAIN_HORIZONS
                    text = insert_sentence(render_plain(smp), domain, sentence, SLOTS[si])
                    rows.append(_finish(smp, text, "plain", "distractor", s))
                    extra.append(dict(family=fam.name, template_group=group, template_id=f"{fam.name}/{group}{ti}",
                                      template=template, slot=SLOTS[si], twin_uid=main_uid(mc.seed, c.config_id, domain, hi),
                                      log_gap=math.log10(d.years / h.years)))
    return pd.concat([to_frame(rows), pd.DataFrame(extra)], axis=1)


def build_all(cfg: DistractorConfig) -> pd.DataFrame:
    main = to_frame(build_matrix(cfg.matrix))
    for col in EXTRA_COLUMNS:
        main[col] = None
    main["log_gap"] = float("nan")
    return pd.concat([main, build_distractors(cfg)], ignore_index=True)


def smoke_subset(df: pd.DataFrame, n_configs_per_split: int = 1) -> pd.DataFrame:
    """A small stratified slice for a local plumbing test: a few configurations per split, every
    family and template group, with the clean twins of every distractor row kept."""
    keep_cfg = (df[df.condition == "main"].drop_duplicates("config_id").groupby("split").head(n_configs_per_split).config_id)
    sub = df[df.config_id.isin(set(keep_cfg))]
    return sub.reset_index(drop=True)


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha1("\n".join(df.sample_uid + "\t" + df.text).encode()).hexdigest()[:12]
