"""FuzzyMatcher: link rows across two DataFrames on one or more keys.

Matching strategy (per key, in order):
    1. Exact equality on the raw values.
    2. Equality after string normalization (strip + lowercase + punctuation
       collapse) — catches "Jane  Doe" vs "jane doe".
    3. Levenshtein similarity above `threshold` — catches typos.

A match's confidence is 1.0 for exact hits, ~0.95 for normalized hits, and
the Levenshtein ratio for fuzzy hits. When multiple keys are given, the best
per-row-pair score is kept, and each left row is matched to its best right
row above threshold (ties broken by right-row index).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Match:
    """A single matched pair of row indices, with which key produced it."""

    left_idx: int
    right_idx: int
    confidence: float
    matched_key: str
    method: str  # "exact" | "normalized" | "fuzzy"


# ----------------------------------------------------------------------- #
# String normalization + Levenshtein
# ----------------------------------------------------------------------- #

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_string(value: Any) -> str:
    """Lowercase, strip, collapse whitespace, strip punctuation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def levenshtein_distance(a: str, b: str) -> int:
    """Classic iterative Levenshtein distance (edit count)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Similarity in [0, 1]. 1.0 means identical; 0.0 means disjoint."""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    longest = max(len(a), len(b))
    return 1.0 - (dist / longest)


# ----------------------------------------------------------------------- #
# Matcher
# ----------------------------------------------------------------------- #


@dataclass
class FuzzyMatcher:
    """Link rows in `left` to rows in `right` based on one or more keys."""

    keys: list[str]
    threshold: float = 0.85

    def match(self, left: pd.DataFrame, right: pd.DataFrame) -> list[Match]:
        """Return the best match for each left row above the threshold."""
        matches: list[Match] = []
        right_used: set[int] = set()

        for l_idx, l_row in left.iterrows():
            best: Match | None = None
            for r_idx, r_row in right.iterrows():
                if r_idx in right_used:
                    continue
                m = self._score_pair(int(l_idx), int(r_idx), l_row, r_row)
                if m and (best is None or m.confidence > best.confidence):
                    best = m
            if best is not None and best.confidence >= self.threshold:
                matches.append(best)
                right_used.add(best.right_idx)

        return matches

    def _score_pair(
        self,
        l_idx: int,
        r_idx: int,
        l_row: pd.Series,
        r_row: pd.Series,
    ) -> Match | None:
        """Score a single pair across all keys, keeping the strongest hit."""
        best: Match | None = None
        for key in self.keys:
            if key not in l_row or key not in r_row:
                continue
            lv, rv = l_row[key], r_row[key]
            if _both_null(lv, rv):
                continue

            # 1. Exact match (handles numeric + strings)
            if not _is_null(lv) and not _is_null(rv) and lv == rv:
                cand = Match(l_idx, r_idx, 1.0, key, "exact")
            else:
                ln, rn = normalize_string(lv), normalize_string(rv)
                if not ln or not rn:
                    continue
                # 2. Normalized equality
                if ln == rn:
                    cand = Match(l_idx, r_idx, 0.95, key, "normalized")
                else:
                    # 3. Fuzzy Levenshtein
                    ratio = levenshtein_ratio(ln, rn)
                    if ratio < self.threshold:
                        continue
                    cand = Match(l_idx, r_idx, ratio, key, "fuzzy")

            if best is None or cand.confidence > best.confidence:
                best = cand
        return best


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA


def _both_null(a: Any, b: Any) -> bool:
    return _is_null(a) and _is_null(b)
