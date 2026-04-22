"""SchemaMap: rename columns and coerce types into a canonical shape.

Given a mapping like ``{"cust_name": "name", "cust_email": "email"}`` and
optional type coercions, `SchemaMap.apply(df)` returns a new DataFrame with
canonical column names, coerced dtypes, and (by default) only the mapped
columns retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SchemaMap:
    """Canonicalize a DataFrame's columns and types.

    Parameters
    ----------
    rename : dict[str, str]
        Source column name -> canonical column name.
    types : dict[str, type]
        Canonical column name -> Python type (str, int, float, bool).
    keep_extra : bool
        If True, unmapped source columns are preserved (with their original
        names). Defaults to False, which drops unmapped columns.
    """

    rename: dict[str, str]
    types: dict[str, type] = field(default_factory=dict)
    keep_extra: bool = False

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame with canonical columns and coerced types."""
        columns_to_keep = list(self.rename.keys())
        if self.keep_extra:
            extras = [c for c in df.columns if c not in self.rename]
            columns_to_keep += extras

        missing = [c for c in self.rename.keys() if c not in df.columns]
        if missing:
            raise KeyError(
                f"SchemaMap: source columns not found in DataFrame: {missing}"
            )

        out = df[columns_to_keep].rename(columns=self.rename).copy()

        for col, py_type in self.types.items():
            if col not in out.columns:
                continue
            out[col] = _coerce(out[col], py_type)

        return out


def _coerce(series: pd.Series, py_type: type) -> pd.Series:
    """Best-effort type coercion — returns NaN-like for unparseable cells."""
    if py_type is str:
        return series.astype("string").str.strip()
    if py_type is int:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if py_type is float:
        return pd.to_numeric(series, errors="coerce").astype(float)
    if py_type is bool:
        truthy = {"true", "t", "yes", "y", "1"}
        falsy = {"false", "f", "no", "n", "0"}

        def _b(v: Any) -> Any:
            if isinstance(v, bool):
                return v
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return pd.NA
            s = str(v).strip().lower()
            if s in truthy:
                return True
            if s in falsy:
                return False
            return pd.NA

        return series.map(_b).astype("boolean")
    return series.astype(py_type)
