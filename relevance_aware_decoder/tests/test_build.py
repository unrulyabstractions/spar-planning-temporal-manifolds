import math
from pathlib import Path

import pandas as pd
import pytest

from ptm.matrix import MATRIX_HORIZONS, TRAIN_HORIZONS, MatrixConfig, build_matrix
from ptm.prompts import to_frame
from rad.build import SLOTS, DistractorConfig, build_all, smoke_subset
from rad.families import FAMILIES, FAMILY_BY_NAME, fill

ALAN_MATRIX = Path(__file__).resolve().parents[2] / "alan" / "data" / "prompts" / "matrix" / "matrix_s0.parquet"


@pytest.fixture(scope="module")
def df():
    return build_all(DistractorConfig())


def test_counts(df):
    assert (df.condition == "main").sum() == 2160
    d = df[df.condition == "distractor"]
    got = d.groupby(["split", "template_group"]).size().to_dict()
    assert got == {("train", "seen"): 1600, ("dev", "seen"): 320, ("test", "seen"): 720, ("test", "unseen"): 720}
    assert df.sample_uid.is_unique and df.prompt_id.is_unique


def test_main_matrix_is_alans(df):
    ours = df[df.condition == "main"].reset_index(drop=True)
    ref = to_frame(build_matrix(MatrixConfig(seed=0)))
    assert ours.sample_uid.tolist() == ref.sample_uid.tolist() and ours.text.tolist() == ref.text.tolist()
    if ALAN_MATRIX.exists():                     # the file Alan captured, if present in the snapshot
        assert set(pd.read_parquet(ALAN_MATRIX).prompt_id) == set(ours.prompt_id)


def test_distractor_is_twin_plus_one_sentence(df):
    by_uid = df.set_index("sample_uid")
    for r in df[df.condition == "distractor"].itertuples():
        t = by_uid.loc[r.twin_uid]
        assert t.condition == "main" and t.rendering == "plain"
        assert (t.config_id, t.domain, t.horizon_text) == (r.config_id, r.domain, r.horizon_text)
        sentence = fill(r.template, r.domain, _h(r.distractor_text))
        assert r.text.count(sentence) == 1
        assert r.text.replace(" " + sentence, "", 1) == t.text or r.text.replace(sentence + " ", "", 1) == t.text


def _h(text):
    from ptm.horizons import Horizon
    return Horizon.parse(text)


def test_durations_and_gap(df):
    d = df[df.condition == "distractor"]
    for r in d.itertuples():
        fam = FAMILY_BY_NAME[r.family]
        assert r.distractor_text in {str(h) for h in fam.durations}
        assert not math.isclose(r.distractor_years, r.horizon_years)
        assert math.isclose(r.log_gap, math.log10(r.distractor_years / r.horizon_years))


def test_splits_and_template_groups(df):
    d = df[df.condition == "distractor"]
    assert set(d[d.split != "test"].template_group) == {"seen"}               # unseen wording never reaches training or dev
    assert set(d[d.split != "test"].horizon_text) == {str(h) for h in TRAIN_HORIZONS}
    assert set(d[d.split == "test"].horizon_text) == {str(h) for h in MATRIX_HORIZONS}
    main = df[df.condition == "main"].drop_duplicates("config_id").set_index("config_id").split
    assert (d.split.to_numpy() == main.loc[d.config_id].to_numpy()).all()    # Alan's configuration-level splits


def test_balance(df):
    d = df[df.condition == "distractor"]
    for _, g in d.groupby(["family", "split", "template_group"]):
        assert g.slot.value_counts().reindex(list(SLOTS)).max() - g.slot.value_counts().min() <= 1
        assert g.template_id.value_counts().max() - g.template_id.value_counts().min() <= 1


def test_every_template_fills_in_both_domains():
    for f in FAMILIES:
        for t in f.templates("all"):
            for dom in ("investment", "climate"):
                s = fill(t, dom, f.durations[0])
                assert "{" not in s and s.endswith(".") and s[0].isupper()


def test_smoke_subset_keeps_twins(df):
    sm = smoke_subset(df)
    assert set(sm.split) == {"train", "dev", "test"}
    assert set(sm[sm.condition == "distractor"].twin_uid) <= set(sm.sample_uid)
    assert set(sm.family.dropna()) == {f.name for f in FAMILIES}
