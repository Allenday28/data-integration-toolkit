"""Lightweight validation for canonicalized DataFrames.

Checks the three things that catch the majority of integration bugs:
    - required columns are present and non-null for every row
    - uniqueness constraints on one or more columns
    - type conformance (e.g., all values in `amount` are numeric)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class ValidationError(Exception):
    """Raised when `validate(..., strict=True)` finds any rule violation."""


@dataclass
class ValidationReport:
    """Collected violations from a validate() call."""

    missing_required: dict[str, list[int]] = field(default_factory=dict)
    duplicate_rows: dict[str, list[int]] = field(default_factory=dict)
    type_violations: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_required or self.duplicate_rows or self.type_violations
        )

    def summary(self) -> str:
        if self.ok:
            return "validation passed"
        parts: list[str] = []
        if self.missing_required:
            parts.append(
                "missing required: "
                + ", ".join(f"{c}({len(v)})" for c, v in self.missing_required.items())
            )
        if self.duplicate_rows:
            parts.append(
                "duplicates: "
                + ", ".join(f"{c}({len(v)})" for c, v in self.duplicate_rows.items())
            )
        if self.type_violations:
            parts.append(
                "type errors: "
                + ", ".join(f"{c}({len(v)})" for c, v in self.type_violations.items())
            )
        return "; ".join(parts)


def validate(
    df: pd.DataFrame,
    *,
    required: list[str] | None = None,
    unique: list[str] | list[list[str]] | None = None,
    types: dict[str, type] | None = None,
    strict: bool = False,
) -> ValidationReport:
    """Run validation rules and return a report (or raise when strict)."""
    report = ValidationReport()

    for col in required or []:
        if col not in df.columns:
            report.missing_required[col] = list(df.index)
            continue
        bad = [i for i, v in df[col].items() if _is_null(v)]
        if bad:
            report.missing_required[col] = bad

    for spec in _normalize_unique(unique):
        cols = spec if isinstance(spec, list) else [spec]
        if not all(c in df.columns for c in cols):
            continue
        dup_mask = df.duplicated(subset=cols, keep=False)
        dup_idx = [int(i) for i in df.index[dup_mask]]
        if dup_idx:
            key = "+".join(cols)
            report.duplicate_rows[key] = dup_idx

    for col, py_type in (types or {}).items():
        if col not in df.columns:
            continue
        bad = [i for i, v in df[col].items() if not _conforms(v, py_type)]
        if bad:
            report.type_violations[col] = bad

    if strict and not report.ok:
        raise ValidationError(report.summary())
    return report


def _normalize_unique(unique: Any) -> list[Any]:
    if unique is None:
        return []
    if isinstance(unique, list) and unique and isinstance(unique[0], list):
        return list(unique)
    return [unique] if isinstance(unique, list) else []


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA


def _conforms(v: Any, py_type: type) -> bool:
    if _is_null(v):
        return True  # null handling is the job of `required`, not `types`
    if py_type is int:
        return isinstance(v, int) and not isinstance(v, bool)
    if py_type is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if py_type is str:
        return isinstance(v, str)
    if py_type is bool:
        return isinstance(v, bool)
    return isinstance(v, py_type)
