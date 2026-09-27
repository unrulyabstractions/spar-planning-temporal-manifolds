"""Inline pca3.json into the explorer template.

    python analysis/build_explorer.py results/horizon_v1_noprefill
    -> <root>/analysis/pca3_explorer.html (open in a browser, or publish)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(root):
    root = Path(root)
    data = (root / "analysis" / "pca3.json").read_text()
    html = (HERE / "pca3_explorer_template.html").read_text().replace("__DATA__", data.replace("</", "<\\/"))
    out = root / "analysis" / "pca3_explorer.html"
    # The artifact publisher wraps the page itself; the local copy needs a full skeleton.
    out.write_text('<!doctype html><html><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
                   + html + "</body></html>")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "results/horizon")
