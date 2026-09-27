"""Build horizon_geometry.ipynb from horizon_geometry.py.

The .py file is the source of truth (it is importable and testable). Cells are
split on `# %%` markers; `# %% [markdown]` cells become markdown.

    python build_notebook.py
"""

import re
from pathlib import Path

import nbformat

SRC = Path(__file__).with_name("horizon_geometry.py")
DST = SRC.with_suffix(".ipynb")


def main():
    chunks = re.split(r"^# %%(.*)$", SRC.read_text(), flags=re.M)
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
    nbformat.write(nb, DST)
    print(f"wrote {DST} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
