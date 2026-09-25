#!/usr/bin/env python
"""Compare two run_checksums.py outputs. Usage: compare_checksums.py PRE.json POST.json [--map old=new ...]"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("pre"); ap.add_argument("post"); ap.add_argument("--map", nargs="*", default=[])
a = ap.parse_args()
pre, post = json.load(open(a.pre)), json.load(open(a.post))
m = dict(kv.split("=") for kv in a.map)
ok_all = True
for old, rec in pre.items():
    new = m.get(old, old)
    if new not in post:
        print(f"{old}: no post-run named {new}"); ok_all = False; continue
    q = post[new]; same = {k: rec[k] == q.get(k) for k in ["n", "index_sha256", "T3_sha256", "R0_sha256"]}
    ok = all(same.values()); ok_all &= ok
    print(f"{old} -> {new}: {'IDENTICAL' if ok else 'DIFFERS'}  " + " ".join(f"{k}={'=' if v else 'x'}" for k, v in same.items())
          + ("" if ok else f"  (T3 sums {rec['T3_sum']:.6g} vs {q['T3_sum']:.6g})"))
print("ALL IDENTICAL" if ok_all else "SOME DIFFER")
