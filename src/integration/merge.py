"""Merger: apply a FuzzyMatcher, resolve per-column conflicts, and summarize.

Given two canonicalized DataFrames and a matcher, `Merger.merge` returns a
single merged DataFrame and a `MergeReport` with counts of matched /
unmatched rows and per-column conflict details.

Conflict resolution strategies:
    - "prefer_left": when both sides have a value, take left.
    - "prefer_right": take right.
    - "non_null": take whichever side has a value (left wins on ties).
    - callable(left, right, column) -> value: custom resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from integration.match import FuzzyMatcher, Match, _is_null

ConflictStrategy = str | Callable[[Any, Any, str], Any]


@dataclass
class Conflict:
    """One per-row, per-column conflict between matched left/right values."""

    left_idx: int
    right_idx: int
    column: str
    left_value: Any
    right_value: Any
    resolved_value: Any


@dataclass
class MergeReport:
    """Summary of a merge operation."""

    n_left: int
    n_right: int
    matches: list[Match]
    unmatched_left: list[int]
    unmatched_right: list[int]
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return len(self.matches)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_left if self.n_left else 0.0

    def summary(self) -> str:
        method_counts: dict[str, int] = {}
        for m in self.matches:
            method_counts[m.method] = method_counts.get(m.method, 0) + 1
        methods = ", ".join(f"{k}={v}" for k, v in sorted(method_counts.items()))
        return (
            f"matched {self.n_matched}/{self.n_left} left rows "
            f"({self.match_rate:.0%}); "
            f"unmatched: left={len(self.unmatched_left)}, "
            f"right={len(self.unmatched_right)}; "
            f"conflicts={len(self.conflicts)}"
            + (f"; methods: {methods}" if methods else "")
        )


class Merger:
    """Apply a matcher and resolve conflicts into one merged frame."""

    def __init__(
        self,
        matcher: FuzzyMatcher,
        conflict: ConflictStrategy = "prefer_left",
    ) -> None:
        self.matcher = matcher
        self.conflict = conflict

    def merge(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> tuple[pd.DataFrame, MergeReport]:
        matches = self.matcher.match(left, right)
        matched_left = {m.left_idx for m in matches}
        matched_right = {m.right_idx for m in matches}

        all_columns = list(dict.fromkeys(list(left.columns) + list(right.columns)))
        rows: list[dict[str, Any]] = []
        conflicts: list[Conflict] = []

        # Matched rows
        for m in matches:
            l_row = left.loc[m.left_idx]
            r_row = right.loc[m.right_idx]
            merged_row: dict[str, Any] = {
                "_match_confidence": m.confidence,
                "_match_method": m.method,
                "_match_key": m.matched_key,
            }
            for col in all_columns:
                lv = l_row[col] if col in left.columns else None
                rv = r_row[col] if col in right.columns else None
                value = self._resolve(lv, rv, col)
                merged_row[col] = value
                if (
                    col in left.columns
                    and col in right.columns
                    and not _is_null(lv)
                    and not _is_null(rv)
                    and lv != rv
                ):
                    conflicts.append(
                        Conflict(m.left_idx, m.right_idx, col, lv, rv, value)
                    )
            rows.append(merged_row)

        # Unmatched left rows (pass through; no match metadata)
        for l_idx, l_row in left.iterrows():
            if l_idx in matched_left:
                continue
            row: dict[str, Any] = {
                "_match_confidence": None,
                "_match_method": "unmatched_left",
                "_match_key": None,
            }
            for col in all_columns:
                row[col] = l_row[col] if col in left.columns else None
            rows.append(row)

        # Unmatched right rows
        for r_idx, r_row in right.iterrows():
            if r_idx in matched_right:
                continue
            row = {
                "_match_confidence": None,
                "_match_method": "unmatched_right",
                "_match_key": None,
            }
            for col in all_columns:
                row[col] = r_row[col] if col in right.columns else None
            rows.append(row)

        ordered_cols = ["_match_method", "_match_confidence", "_match_key"] + all_columns
        merged_df = pd.DataFrame(rows, columns=ordered_cols)

        report = MergeReport(
            n_left=len(left),
            n_right=len(right),
            matches=matches,
            unmatched_left=[i for i in left.index if i not in matched_left],
            unmatched_right=[i for i in right.index if i not in matched_right],
            conflicts=conflicts,
        )
        return merged_df, report

    def _resolve(self, left_value: Any, right_value: Any, column: str) -> Any:
        if _is_null(left_value) and _is_null(right_value):
            return left_value
        if _is_null(left_value):
            return right_value
        if _is_null(right_value):
            return left_value
        strat = self.conflict
        if callable(strat):
            return strat(left_value, right_value, column)
        if strat == "prefer_left":
            return left_value
        if strat == "prefer_right":
            return right_value
        if strat == "non_null":
            return left_value  # both non-null; left wins on ties
        raise ValueError(f"Unknown conflict strategy: {strat!r}")
