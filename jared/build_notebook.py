"""Build a notebook from a `# %%`-delimited .py source.

The .py file is the source of truth (it is importable and testable). Cells are
split on `# %%` markers; `# %% [markdown]` cells become markdown.

    python build_notebook.py                              # horizon_geometry.ipynb
    python build_notebook.py analysis/pca3_explorer_nb.py # analysis/pca3_explorer.ipynb
"""

import re
import sys
from pathlib import Path

import nbformat

DEFAULT_SRC = Path(__file__).with_name("horizon_geometry.py")


def main(src=DEFAULT_SRC):
    src = Path(src)
    dst = src.with_name(src.stem.removesuffix("_nb") + ".ipynb")
    chunks = re.split(r"^# %%(.*)$", src.read_text(), flags=re.M)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                             for line in body.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            if "__main__" in body:
                body = body.replace('if __name__ == "__main__":\n    main()', "main()")
            nb.cells.append(nbformat.v4.new_code_cell(body))
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(*sys.argv[1:])
