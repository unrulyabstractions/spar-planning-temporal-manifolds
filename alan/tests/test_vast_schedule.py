"""scripts/vast_full_chain.sh schedule (DRY_RUN=1: captures and analyses replaced by sleeps; no GPU, no model).

With a CPU quota >= 12 cores (or none), the later analyses must overlap the remaining captures: group A starts right
after families_s0 is captured (before the long captures finish), analyze_long starts once matrix_s0_long is captured,
and late_readout/depth_profile start only after all captures. With a low quota everything runs after the captures.
The script is copied into a temporary tree, so the dry run never writes into the repo's runs/ or figures/.
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUP_A = ["curve_model", "relevance_decoder", "relevance_families", "scaling_test", "guttman_canonical",
           "guttman_extended", "leace_check", "pls_check", "stakes_pilot"]


def dry_chain(tmp_path: Path, quota: str) -> list[str]:
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "vast_full_chain.sh", tmp_path / "scripts")
    env = dict(os.environ, DRY_RUN="1", CPU_QUOTA=quota, MODEL="dry/model", TAG="t", DRY_CAPTURE_S="2", DRY_ANALYSIS_S="1")
    subprocess.run(["bash", str(tmp_path / "scripts" / "vast_full_chain.sh")], cwd=tmp_path, env=env,
                   capture_output=True, text=True, timeout=120, check=True)
    lines = [l.split(" ", 2)[2] for l in (tmp_path / "runs" / "chain_t.log").read_text().splitlines()]
    print("\n  " + "\n  ".join(lines))
    return lines


def at(lines, text):
    return next(i for i, l in enumerate(lines) if text in l)


def test_overlap_schedule(tmp_path):
    L = dry_chain(tmp_path, "23")
    assert "schedule: overlap" in L[at(L, "schedule:")]
    fam_done, ml_done = at(L, "capture t_families_s0 done"), at(L, "capture t_matrix_s0_long done")
    fl_done, caps_done = at(L, "capture t_families_s0_long done"), at(L, "captures done")
    assert fam_done < at(L, "analysis curve_model start") < ml_done                 # group A during the long captures
    assert ml_done < at(L, "analysis analyze_long start") < fl_done                 # analyze_long beside the last capture
    assert caps_done < at(L, "analysis late_readout start") < at(L, "analysis depth_profile start")
    done = at(L, "later analyses done")
    assert all(at(L, f"analysis {n} done") < done for n in GROUP_A + ["analyze_long", "late_readout", "depth_profile"])
    assert L[-1] == "CHAIN DONE"


def test_serial_schedule_on_low_quota(tmp_path):
    L = dry_chain(tmp_path, "7")
    assert "schedule: serial" in L[at(L, "schedule:")]
    caps_done = at(L, "captures done")
    assert all(i > caps_done for i, l in enumerate(L) if l.startswith("analysis"))
    assert L[-1] == "CHAIN DONE"
