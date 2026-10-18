"""Thinking-conclusion vs final answer for the 10-17 thinking pilots (audit aid, not a metric on its own).

Rule A: last strong commitment to a label inside the thinking, hypotheticals dropped. Rule B: last option label
mentioned in the final 400 characters of the thinking. Rows where rule A differs from the answer or finds nothing
must be read by hand; see progress-20261018.md for the audit and the contradiction count.
Usage: python scripts/think_conclusion_audit.py OUT.csv   (run from the repo root; paths are the 10-17 fetch folders)
"""
import re, sys
import pandas as pd

RUNS = {
    "qwen3-14b": "runs/vast_fetch_20261017/runs/qwen3-14b_think_pilot/index.parquet",
    "qwen3.8-27b": "runs/vast_fetch_20261017b/runs/qwen3.8-27b_think_pilot/index.parquet",
    "gemma-4-31b": "runs/vast_fetch_20261017c/runs/gemma-4-31b_think_pilot/index.parquet",
}
END = ("</think>", "<channel|>")
X = r"(?:option\s+)?\**\(?([ab])\b\)?"
STRONG = [
    rf"\bI choose:\s*\**<?{X}",
    rf"\banswer(?:\s+(?:is|must be|would be|should be|will be)|:)\s*{X}",
    rf"\b(?:best|better|safest|right|correct|optimal|logical|rational|sensible|only viable|wiser|prudent)\s+(?:choice|option|answer|investment|one|pick|bet)?\s*(?:is|would be|must be|remains)\s+{X}",
    rf"\b{X}\s+is\s+(?:the\s+|clearly\s+|still\s+|more\s+|therefore\s+)?(?:better|best|correct|right|optimal|only viable|safer|superior|preferable|defensible|more defensible|the (?:better|best|right|correct|logical|rational|only|safer|wiser) (?:choice|option|answer|one))",
    rf"\b{X}\s+(?:preferable|it is|wins)\b",
    rf"\b(?:go with|pick|choose|select|recommend|opt for|choosing|selecting)\s+{X}",
    rf"\bmakes\s+{X}",
    rf"\b(?:therefore|thus|hence|so)[,:]?\s+(?:the answer is\s+)?(?:option\s+)?\**\(?([ab])\)",
    rf"\b(?:therefore|thus|hence|so|likely|then)[,:]?\s+(?:option\s+)?([ab])\s*[.!]",
    rf"(?:^|\n)\s*([ab])\.?\s*$",
]
RX = [re.compile(p, re.I | re.M) for p in STRONG]
HYPO = re.compile(r"\b(?:maybe|if|could|whether|perhaps|unless|might|or|not|why|would they|should we|then maybe|instead of|than)\s+(?:\w+\s+){0,2}$", re.I)
LAB = re.compile(r"(?:\boption\s+\**\(?([ab])\b|\b([ab])\))", re.I)

def split(gen):
    for e in END:
        k = gen.find(e)
        if k >= 0:
            return gen[:k], gen[k + len(e):]
    return None, None

def rule_a(think):
    hits = []
    for rx in RX:
        for m in rx.finditer(think):
            pre = think[max(0, m.start() - 30):m.start()]
            if HYPO.search(pre):
                continue
            lab = next(g for g in m.groups() if g)
            hits.append((m.start(), lab.lower(), m.group(0).strip()))
    if not hits:
        return None, None
    hits.sort()
    return hits[-1][1], hits[-1][2]

def rule_b(think, n=400):
    tail = think[-n:]
    labs = [(m.group(1) or m.group(2)).lower() for m in LAB.finditer(tail)]
    return labs[-1] if labs else None

rows = []
for model, p in RUNS.items():
    df = pd.read_parquet(p)
    for r in df.itertuples():
        think, ans = split(r.gen_text)
        if think is None:
            rows.append(dict(model=model, uid=r.sample_uid, status="unclosed")); continue
        a, atxt = rule_a(think); b = rule_b(think)
        st = "agree" if (a == r.choice and b in (r.choice, None)) else "CHECK"
        rows.append(dict(model=model, uid=r.sample_uid, choice=r.choice, concl=a, concl_txt=atxt, ruleb=b,
                         short_first=r.short_first, hd=float(r.horizon_years) / float(r.long_delay_years),
                         chose_short=r.chose_short, n_generated=r.n_generated, status=st))
out = pd.DataFrame(rows)
out.to_csv(sys.argv[1], index=False)
print(out.groupby(["model", "status"]).size().unstack(fill_value=0))
c = out[out.status == "CHECK"]
print(c.assign(a_vs=c.concl.fillna("-") + "/" + c.ruleb.fillna("-") + " ans " + c.choice).groupby(["model", "a_vs"]).size())
