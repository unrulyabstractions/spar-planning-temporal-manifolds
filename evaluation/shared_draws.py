"""Shared random-direction draws (null families N1, N2, N2') per (model, layer, site), reproducible bit for bit.

One command turns the extraction activations into the draw file both teams use, so "random" means the same thing in
every report (eval design §4.1 item 6, §6.4). Built without reading the extraction code: it relies only on the file
layout described in `extraction_design/README.md` (activations/<run>/L<layer>_<site>.npy, one row per entry of
index.jsonl, which carries each row's uid) and on the bank's prompts.jsonl (uid, split, regime).

Families (all unit vectors, rows of a [K, d] array):
  isotropic    N1   v ~ N(0, I_d)
  covariance   N2   v ~ N(0, Sigma), Sigma = covariance of the activations at this layer and site
  real_diff    N2'  v = h_i - h_j for random pairs of distinct prompts
The generation tier uses the first 19 rows of each family (K_GEN): no separate set, so the replay and generation
results refer to the same directions.

Why the draws do not depend on the machine or the numpy version:
  - Seeds are not chosen by hand: each (model, layer, site, family) gets its own seed from the SHA-256 of a recipe
    string, so draws for different sites or families are independent and nobody has to agree on seed numbers.
  - Random numbers come from PCG64's raw 64-bit stream (stable across numpy versions by numpy's policy), turned into
    normals by our own Box-Muller code. numpy's Generator.standard_normal is NOT guaranteed stable across versions.
  - Rows are sorted by uid, so the order in which extraction wrote them does not matter.
  - Prompts used: the extraction split, horizon prompts only (no-horizon twins and the verification grid are never
    used; the same prompts the methods are fitted on, including "none" prompts).
  - The one step that can differ in the last bits across machines is the matrix product for `covariance` (BLAS).
    Draws are therefore stored in float32, and `verify` compares with a tolerance; the published file and its SHA-256
    are the reference.

Usage (from evaluation/; needs only numpy):
  python shared_draws.py make   --activations ACT_DIR --bank PROMPTS.jsonl --model google/gemma-4-31B-it \
                                --layer 38 --site label --out draws/
  python shared_draws.py make   --iso-only --d D_MODEL --model ... --layer 38 --site label --out draws/   # N1 now
  python shared_draws.py verify --draws draws/<model>/L38_label [--activations ACT_DIR --bank PROMPTS.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RECIPE = "ptm-shared-draws/v1"
FAMILIES = ("isotropic", "covariance", "real_diff")
K_DEFAULT = 99          # replay tier: smallest p = 1/(K+1) = 0.01
K_GEN = 19              # generation tier uses rows [:19] of the same draws: smallest p = 0.05
MIN_ROWS = 100          # covariance from m rows has rank <= m-1; refuse smoke-sized activations by default
TOL = 1e-6              # verify: max |stored - regenerated| allowed (float32 storage of unit vectors)


# ---------------------------------------------------------------- deterministic randomness

def seed_key(model: str, layer: int, site: str, family: str, recipe: str = RECIPE) -> str:
    return f"{recipe}|{model}|L{layer}|{site}|{family}"


def stream(key: str) -> np.random.PCG64:
    """PCG64 seeded from SHA-256(key). SeedSequence and PCG64's raw output are stable across numpy versions."""
    entropy = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big")
    return np.random.PCG64(np.random.SeedSequence(entropy))


def uniforms(bg: np.random.PCG64, n: int) -> np.ndarray:
    """n doubles in (0, 1) from the top 53 bits of raw 64-bit outputs (never exactly 0, so log() is safe)."""
    raw = bg.random_raw(n).astype(np.uint64)
    return ((raw >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53


def normals(bg: np.random.PCG64, shape: tuple[int, ...]) -> np.ndarray:
    """Standard normals by Box-Muller on our own uniforms: fixed algorithm, independent of numpy's samplers."""
    n = int(np.prod(shape))
    half = (n + 1) // 2
    u1, u2 = uniforms(bg, half), uniforms(bg, half)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
    return z[:n].reshape(shape)


def integers(bg: np.random.PCG64, n: int, high: int) -> np.ndarray:
    """n integers in [0, high). Modulo of 64-bit outputs: bias < high / 2**64, negligible for a few hundred prompts."""
    return (bg.random_raw(n).astype(np.uint64) % np.uint64(high)).astype(np.int64)


def unit(V: np.ndarray) -> np.ndarray:
    return V / np.linalg.norm(V, axis=1, keepdims=True)


# ---------------------------------------------------------------- the families

def draw_isotropic(d: int, k: int, key: str) -> np.ndarray:
    return unit(normals(stream(key), (k, d)))


def draw_covariance(X: np.ndarray, k: int, key: str) -> np.ndarray:
    """v = g @ (X - mean) / sqrt(m - 1), g ~ N(0, I_m): exactly N(0, Sigma) without forming the d x d matrix."""
    Xc = X - X.mean(axis=0)
    g = normals(stream(key), (k, len(X)))
    return unit(g @ Xc / np.sqrt(len(X) - 1))


def draw_real_diff(X: np.ndarray, k: int, key: str) -> tuple[np.ndarray, np.ndarray]:
    """h_i - h_j for k random ordered pairs of distinct rows. Returns the unit vectors and the [k, 2] row indices."""
    bg = stream(key)
    m = len(X)
    i = integers(bg, k, m)
    j = integers(bg, k, m - 1)
    j = j + (j >= i)                       # uniform over rows other than i
    return unit(X[i] - X[j]), np.stack([i, j], axis=1)


# ---------------------------------------------------------------- inputs

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def eligible_uids(bank_prompts: Path) -> set[str]:
    """Extraction split, horizon prompts only: what the methods are fitted on (no-horizon twins, verification out)."""
    return {r["uid"] for r in read_jsonl(bank_prompts)
            if r["split"] == "extraction" and r["regime"] != "no_horizon"}


def load_activations(act_dir: Path, layer: int, site: str, bank_prompts: Path, uid_key: str = "uid",
                     min_rows: int = MIN_ROWS) -> tuple[np.ndarray, list[str], dict]:
    """The [m, d] float64 activations of the eligible prompts, rows sorted by uid, plus the input provenance."""
    npy = act_dir / f"L{layer}_{site}.npy"
    index = act_dir / "index.jsonl"
    A = np.load(npy)
    rows = read_jsonl(index)
    if len(rows) != len(A):
        sys.exit(f"{index} has {len(rows)} rows but {npy} has {len(A)}: cannot align rows to uids.")
    if any(uid_key not in r for r in rows):
        sys.exit(f"{index} rows lack the key {uid_key!r} (keys seen: {sorted(rows[0])}); pass --uid-key.")
    uids = [r[uid_key] for r in rows]
    if len(set(uids)) != len(uids):
        sys.exit(f"{index} lists a uid more than once.")
    keep = eligible_uids(bank_prompts)
    order = sorted((u, n) for n, u in enumerate(uids) if u in keep)
    if len(order) < min_rows:
        sys.exit(f"only {len(order)} eligible rows (< {min_rows}): a covariance from so few prompts spans at most "
                 f"{max(len(order) - 1, 0)} directions. Use the full extraction, or --min-rows for a test.")
    sel_uids = [u for u, _ in order]
    X = A[[n for _, n in order]].astype(np.float64)
    if not np.isfinite(X).all():
        sys.exit(f"{npy} contains non-finite values in the eligible rows.")
    prov = {"activations_file": str(npy), "activations_sha256": sha256_file(npy),
            "index_sha256": sha256_file(index), "bank_prompts_sha256": sha256_file(bank_prompts),
            "n_rows_in_file": len(A), "n_rows_used": len(sel_uids),
            "uids_sha256": hashlib.sha256("\n".join(sel_uids).encode()).hexdigest()}
    return X, sel_uids, prov


# ---------------------------------------------------------------- make / verify

def generate(model: str, layer: int, site: str, k: int, d: int | None = None, X: np.ndarray | None = None,
             recipe: str = RECIPE) -> dict[str, np.ndarray]:
    """All families that the inputs allow: isotropic needs only d; covariance and real_diff need X."""
    if X is not None:
        d = X.shape[1]
    out = {"isotropic": draw_isotropic(d, k, seed_key(model, layer, site, "isotropic", recipe))}
    if X is not None:
        out["covariance"] = draw_covariance(X, k, seed_key(model, layer, site, "covariance", recipe))
        out["real_diff"], out["real_diff_pairs"] = draw_real_diff(
            X, k, seed_key(model, layer, site, "real_diff", recipe))
    return out


def slug(model: str) -> str:
    return model.split("/")[-1]


def make(args) -> Path:
    out = Path(args.out) / slug(args.model) / f"L{args.layer}_{args.site}"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty: draws are never overwritten.")
    if args.iso_only:
        if not args.d:
            sys.exit("--iso-only needs --d (the model's hidden size).")
        X, uids, prov = None, [], {}
    else:
        X, uids, prov = load_activations(Path(args.activations), args.layer, args.site, Path(args.bank),
                                         args.uid_key, args.min_rows)
    draws = generate(args.model, args.layer, args.site, args.k, d=args.d, X=X)
    d = draws["isotropic"].shape[1]
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f: v.astype(np.float32) for f, v in draws.items() if f in FAMILIES}
    if "real_diff_pairs" in draws:
        arrays["real_diff_pairs"] = draws["real_diff_pairs"]
    np.savez(out / "draws.npz", **arrays)
    if uids:
        (out / "uids.txt").write_text("\n".join(uids) + "\n")
    manifest = {
        "format": RECIPE, "model": args.model, "layer": args.layer, "site": args.site, "d": d, "k": args.k,
        "k_generation": K_GEN, "k_generation_rule": f"generation uses rows [:{K_GEN}] of each family",
        "families": {f: {"seed_key": seed_key(args.model, args.layer, args.site, f)} for f in arrays if f in FAMILIES},
        "definitions": {
            "isotropic": "unit(z), z ~ N(0, I_d)",
            "covariance": "unit(g @ (X - mean(X)) / sqrt(m-1)), g ~ N(0, I_m); X = activations, rows sorted by uid",
            "real_diff": "unit(X[i] - X[j]), i != j uniform; pairs in real_diff_pairs index uids.txt (0-based)",
            "randomness": "PCG64(SeedSequence(int(sha256(seed_key)))) raw stream; Box-Muller normals; "
                          "uniform = ((raw >> 11) + 0.5) * 2**-53; integers = raw % high",
            "rows": "bank split == extraction and regime != no_horizon, present in index.jsonl, sorted by uid",
            "storage": "float32 unit vectors; verify tolerance %g" % TOL},
        "inputs": prov,
        "software": {"numpy": np.__version__, "python": platform.python_version()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(p for p in out.iterdir() if p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    print(f"wrote {out}: {', '.join(f for f in arrays if f in FAMILIES)}; K={args.k}, d={d}"
          + (f", {len(uids)} prompts" if uids else ""))
    return out


def verify(args) -> bool:
    """Check file hashes, then regenerate every family in the file and compare within TOL."""
    out = Path(args.draws)
    ok = True
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        good = sha256_file(out / name) == digest
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} sha256 {name}")
    m = json.loads((out / "manifest.json").read_text())
    stored = np.load(out / "draws.npz")
    X = None
    if any(f in stored for f in ("covariance", "real_diff")):
        if not (args.activations and args.bank):
            print("skip covariance/real_diff regeneration: pass --activations and --bank")
        else:
            X, uids, prov = load_activations(Path(args.activations), m["layer"], m["site"], Path(args.bank),
                                             args.uid_key, min_rows=0)
            for k_ in ("activations_sha256", "bank_prompts_sha256", "uids_sha256"):
                good = prov[k_] == m["inputs"].get(k_)
                ok &= good
                print(f"{'ok  ' if good else 'FAIL'} input {k_}")
    fresh = generate(m["model"], m["layer"], m["site"], m["k"], d=m["d"], X=X, recipe=m["format"])
    for f in FAMILIES:
        if f in stored and f in fresh:
            diff = float(np.abs(stored[f].astype(np.float64) - fresh[f]).max())
            good = diff <= TOL
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} regenerate {f}: max |diff| {diff:.2e}")
    if "real_diff_pairs" in stored and "real_diff_pairs" in fresh:
        good = np.array_equal(stored["real_diff_pairs"], fresh["real_diff_pairs"])
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} regenerate real_diff_pairs")
    print("VERIFIED" if ok else "MISMATCH")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--activations"), mk.add_argument("--bank")
    mk.add_argument("--model", required=True), mk.add_argument("--layer", type=int, required=True)
    mk.add_argument("--site", required=True), mk.add_argument("--out", required=True)
    mk.add_argument("--k", type=int, default=K_DEFAULT)
    mk.add_argument("--iso-only", action="store_true"), mk.add_argument("--d", type=int)
    mk.add_argument("--uid-key", default="uid"), mk.add_argument("--min-rows", type=int, default=MIN_ROWS)
    vf = sub.add_parser("verify")
    vf.add_argument("--draws", required=True)
    vf.add_argument("--activations"), vf.add_argument("--bank"), vf.add_argument("--uid-key", default="uid")
    a = p.parse_args(argv)
    if a.cmd == "make":
        if not a.iso_only and not (a.activations and a.bank):
            p.error("make needs --activations and --bank (or --iso-only --d D)")
        make(a)
    else:
        sys.exit(0 if verify(a) else 1)


if __name__ == "__main__":
    main()
