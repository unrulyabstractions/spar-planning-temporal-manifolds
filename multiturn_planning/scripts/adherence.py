"""Format adherence of a capture run (printed; also used by the pipeline log).

    python scripts/adherence.py runs/<name>
"""

import sys

import numpy as np
import pandas as pd

df = pd.read_parquet(f"{sys.argv[1]}/index.parquet")
o, st, dn = df[df.kind == "outline"], df[df.kind == "step"], df[df.kind == "done"]
print(f"conversations {df.conv_id.nunique()}  turns {len(df)}")
print(f"outline: only titles {1 - o.outline_full.mean():.0%}   mentions a duration {o.outline_mentions_time.mean():.0%}"
      f"   truncated {o.truncated.mean():.0%}")
print(f"steps:   horizon parsed {st.h_step_years.notna().mean():.0%}   step number = turn-1 "
      f"{(st.step_no == st.turn - 1).mean():.0%}   cut {st.cut.mean():.0%}   truncated {st.truncated.mean():.0%}")
print(f"done:    'Plan Completed' {dn.reply.str.contains('Plan Completed').mean():.0%}")
ok = st.groupby("conv_id").apply(lambda g: g.h_step_years.notna().all() and (g.step_no == g.turn - 1).all())
print(f"conversations with all 5 steps parsed and numbered right: {ok.mean():.0%} ({ok.sum()}/{len(ok)})")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from mtp.analysis import clean_conversations  # noqa: E402
usable = clean_conversations(df)
print(f"USABLE (titles-only outline + all 5 steps right): {len(usable) / df.conv_id.nunique():.0%} "
      f"({len(usable)}/{df.conv_id.nunique()})")
h = st.dropna(subset=["h_step_years"])
mono = h.groupby("conv_id").h_step_years.apply(lambda v: bool(np.all(np.diff(v.values) >= 0)))
print(f"non-decreasing step horizons: {mono.mean():.0%}")
