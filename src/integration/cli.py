"""CLI: `python -m integration merge left.csv right.csv --keys email,name`

Runs the bundled toolkit end-to-end on two CSVs and prints the match report.
For heavier configuration (schema renames, type coercion), use the library
API directly from Python.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from integration.match import FuzzyMatcher
from integration.merge import Merger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="integration",
        description="Merge two CSV files using fuzzy key matching.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    merge_cmd = sub.add_parser("merge", help="Merge two CSVs on one or more keys.")
    merge_cmd.add_argument("left", help="Path to the left CSV.")
    merge_cmd.add_argument("right", help="Path to the right CSV.")
    merge_cmd.add_argument(
        "--keys",
        required=True,
        help="Comma-separated list of keys to match on (must exist in both).",
    )
    merge_cmd.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum similarity (0-1) to accept as a fuzzy match.",
    )
    merge_cmd.add_argument(
        "--conflict",
        choices=["prefer_left", "prefer_right", "non_null"],
        default="prefer_left",
    )
    merge_cmd.add_argument(
        "--out",
        default=None,
        help="Optional path to write the merged CSV. If omitted, prints to stdout.",
    )

    args = parser.parse_args(argv)

    if args.cmd == "merge":
        left = pd.read_csv(args.left)
        right = pd.read_csv(args.right)
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        matcher = FuzzyMatcher(keys=keys, threshold=args.threshold)
        merger = Merger(matcher, conflict=args.conflict)
        merged, report = merger.merge(left, right)

        print(report.summary())
        if args.out:
            merged.to_csv(args.out, index=False)
            print(f"\nWrote {args.out}")
        else:
            print()
            print(merged.to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
