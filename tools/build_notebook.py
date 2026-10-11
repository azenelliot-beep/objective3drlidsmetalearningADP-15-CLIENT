"""Build notebooks/chapter6_v4.ipynb from the original notebook + the cell files in cells/.

Usage: python tools/build_notebook.py
Cells are matched by their '# BLOCK <id>' header, so the mapping is explicit and checked.
"""
import copy, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "original" / "before-viva_1.ipynb"
OUT = ROOT / "notebooks" / "chapter6_v4.ipynb"

KEEP = "__keep_original__"
# original header prefix -> list of replacement cell files (in order)
REPLACE = {
    "# BLOCK 16 ": ["block16_dp_client.py", "block16b_fast_federation.py"],
    "# BLOCK 17 ": [KEEP, "block17b_adpmeta_v2.py"],          # keep Block 17 (v1 controller), add 17b after it
}
# The original FINAL Block 18 starts with a '# ====' banner, matched by this line instead
REPLACE_BLOCK18_MARKER = "# BLOCK 18 (FINAL)"
BLOCK18_FILES = ["block18a_experiment_defs.py", "block18b_run_experiments.py",
                 "block18c_val_summary.py", "block19_final_test.py"]


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    nb = json.loads(SRC.read_text())
    cells, hits = [], {k: 0 for k in REPLACE} | {REPLACE_BLOCK18_MARKER: 0}
    for c in nb["cells"]:
        src = "".join(c["source"])
        new = None
        for prefix, files in REPLACE.items():
            if c["cell_type"] == "code" and src.startswith(prefix):
                new, hits[prefix] = files, hits[prefix] + 1
        if c["cell_type"] == "code" and REPLACE_BLOCK18_MARKER in src.splitlines()[1 if src.startswith("# ===") else 0]:
            new, hits[REPLACE_BLOCK18_MARKER] = BLOCK18_FILES, hits[REPLACE_BLOCK18_MARKER] + 1
        for f in (new or [KEEP]):
            if f == KEEP:
                k = copy.deepcopy(c)
                if k["cell_type"] == "code":
                    k["outputs"], k["execution_count"] = [], None  # strip old outputs: they belong to the old code
                cells.append(k)
            else:
                cells.append(code_cell((ROOT / "cells" / f).read_text()))
    bad = {k: v for k, v in hits.items() if v != 1}
    if bad:
        sys.exit(f"Expected exactly one match per replaced block, got {bad}")
    nb["cells"] = cells
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
