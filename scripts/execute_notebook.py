"""Execute the canonical notebook in this Python environment; keep its source clean.

Usage: .venv/bin/python scripts/execute_notebook.py [--txt] [--scan-id ID] [--all-scans]
The executed notebook and outputs are written only under results/verification/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt", action="store_true", help="Exercise TXT input mode")
    parser.add_argument("--scan-id")
    parser.add_argument(
        "--volumes-only", action="store_true", help="Run full series without manual annotations"
    )
    parser.add_argument(
        "--no-excel", action="store_true", help="Run without optional reference workbooks"
    )
    parser.add_argument("--volumes", action="store_true", help="Also export every DICOM slice")
    parser.add_argument("--all-scans", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mode = "txt" if args.txt or args.no_excel else "excel"
    label = "cohort" if args.all_scans else ("txt-only" if args.no_excel else mode)
    if args.volumes:
        label += "-volumes"
    if args.volumes_only:
        label = "volume-only"
    output = root / "results" / "verification" / label
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(root / "runner.ipynb", as_version=4)
    overrides = [
        "from dataclasses import replace",
        f"settings = replace(settings, output_dir=Path({str(output)!r}))",
    ]
    overrides.append(f"settings = replace(settings, annotation_source={mode!r})")
    if args.no_excel:
        overrides.append(
            "settings = replace(settings, medial_excel=None, lateral_excel=None, kl_excel=None)"
        )
    if args.volumes or args.volumes_only:
        overrides.append("RUN_WHOLE_VOLUMES = True")
    if args.volumes_only:
        overrides.append("RUN_CDI_ANALYSIS = False")
    if args.scan_id:
        overrides.append(f"settings = replace(settings, scan_id={args.scan_id!r})")
    if args.all_scans:
        overrides.append("settings = replace(settings, scan_id=None)")
    config = next(
        cell for cell in notebook.cells if "configuration" in cell.metadata.get("tags", [])
    )
    config.source += (
        "\n\n# Execution-only overrides (the source notebook stays unchanged).\n"
        + "\n".join(overrides)
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=1800, resources={"metadata": {"path": str(root)}}
    )
    start = time.monotonic()
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "runner.executed.ipynb")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    elapsed = time.monotonic() - start
    print(
        f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells in {elapsed:.1f}s"
    )
    print(f"Saved {output / 'runner.executed.ipynb'}")


if __name__ == "__main__":
    main()
