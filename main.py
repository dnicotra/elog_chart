"""Load the ELOG export into a Polars DataFrame and drop into an IPython shell."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl  # noqa: F401  (handy in the embedded shell)
from IPython import embed

from elog_parser import parse_export

EXPORT_PATH = Path(__file__).parent / "data" / "export.xml"


def main() -> None:
    # The DataFrame preview contains non-cp1252 chars (e.g. "°C"); the default
    # Windows console encoding would crash on print().
    sys.stdout.reconfigure(encoding="utf-8")

    df = parse_export(EXPORT_PATH)
    print(f"Loaded {df.height} entries into `df`. Dropping into IPython...\n")

    embed(colors="neutral", header=f"df: {df.shape[0]} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
