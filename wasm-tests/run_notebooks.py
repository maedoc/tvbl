"""Execute demo notebooks inside the pyjs/emscripten runtime.

Notebooks are mounted at /work/notebooks and executed cell-by-cell in a single
namespace.  IPython magics (``%matplotlib``, ``!cmd``) are display/shell
conveniences that have no meaning here, so they are stripped before execution;
everything else runs unchanged.

Each notebook gets its own browser process, because the wasm runtime is capped
at 2 GiB and the demos keep simulation buffers around.
"""

import json
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

NOTEBOOK_DIR = "/work/notebooks"


def read_notebook(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def code_cells(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        yield index, "\n".join(lines)


def run_notebook(path):
    name = os.path.basename(path)
    print(f"\n=== {name} ===", flush=True)
    namespace = {"__name__": "__main__", "__file__": path}
    failures = []
    for index, source in code_cells(read_notebook(path)):
        if not source.strip():
            continue
        print(f"--- cell {index} ---", flush=True)
        try:
            exec(compile(source, f"{name}:cell{index}", "exec"), namespace)
        except Exception:
            traceback.print_exc()
            failures.append(index)
            print(f"!!! cell {index} failed", flush=True)
    if failures:
        print(f"=== {name}: FAILED cells {failures} ===", flush=True)
        return False
    print(f"=== {name}: OK ===", flush=True)
    return True


def main():
    selection = "/work/notebook.txt"
    if os.path.exists(selection):
        with open(selection, encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
    else:
        names = sorted(f for f in os.listdir(NOTEBOOK_DIR) if f.endswith(".ipynb"))

    results = {}
    for name in names:
        results[name] = run_notebook(os.path.join(NOTEBOOK_DIR, name))

    print("\n=== summary ===", flush=True)
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
