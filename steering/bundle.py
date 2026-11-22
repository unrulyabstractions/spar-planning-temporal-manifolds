"""Package fitted methods for hand-over: one self-describing bundle per method (format: steer/methods.py).

    python bundle.py --fit fits/gemma-4-31B-it__t0_s0/default/L38_label --activations activations/gemma-4-31B-it__t0_s0
    python bundle.py --fit ... --activations ... --methods caa curve-snap

Writes methods/<model>/<method>__<activations>__<fit name>__L<layer>_<site>/ for each method, by default all the
fit has: caa, curve-snap, curve-displace (the methods); reward, choice (concept controls fitted by the same recipe);
and the within-regime CAAs caa-none, caa-near_only, caa-both and their pool caa-within (comparison methods). Each
bundle carries method.json (what it is, how to apply it, what it was fitted on, hashes), its arrays, and copies of the
fit's fit.json, fit_uids.jsonl and the activations' search log. Refuses to overwrite a bundle, and refuses a fit whose
recorded activation hash does not match the activations directory.

Load one with steer.methods.load(path).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pbank.local_model import log
from pbank.provenance import utc_now
from steer.methods import METHODS, available_methods, write_bundle
from steer.provenance import code_commit

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None, methods_root: Path = HERE / "methods") -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", type=Path, required=True, help="a fit.py output directory (.../<name>/L<layer>_<site>)")
    p.add_argument("--activations", type=Path, required=True, help="the extract.py directory the fit was made from")
    p.add_argument("--methods", nargs="+", choices=METHODS, help="default: every method the fit has")
    args = p.parse_args(argv)
    have = available_methods(args.fit)
    if args.methods is None:
        args.methods = have
        if set(METHODS) - set(have):
            log(f"not in this fit, so not bundled: {sorted(set(METHODS) - set(have))}")
    elif set(args.methods) - set(have):
        sys.exit(f"{args.fit} has no direction for {sorted(set(args.methods) - set(have))}")

    fit_meta = json.loads((args.fit / "fit.json").read_text())
    extract_meta = json.loads((args.activations / "extract.json").read_text())
    array = fit_meta["inputs"]["array"]
    if extract_meta["arrays"].get(array, {}).get("sha256") != fit_meta["inputs"]["array_sha256"]:
        sys.exit(f"{args.fit} was not fitted from {args.activations} ({array} hash differs or is missing)")
    search_log = args.fit.parents[1] / "search.jsonl"
    if not search_log.exists():
        sys.exit(f"{search_log} is missing: bundles carry the search log of the fit's activations")
    model = extract_meta["model"]["model_id"].split("/")[-1]
    s = fit_meta["settings"]
    stem = f"{fit_meta['inputs']['activations']}__{s['name']}__L{s['layer']:02d}_{s['site']}"
    outs = [Path(methods_root) / model / f"{m}__{stem}" for m in args.methods]
    existing = [o for o in outs if o.exists()]
    if existing:
        sys.exit(f"{existing[0]} exists; bundles are never overwritten")
    commit, now = code_commit(), utc_now()
    for m, out in zip(args.methods, outs):
        write_bundle(out, m, args.fit, fit_meta, extract_meta, search_log, commit, now)
        log(f"wrote {out}")
    return outs


if __name__ == "__main__":
    main()
