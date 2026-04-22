"""Tests for the data-integration-toolkit. Run with `pytest -q`."""

from __future__ import annotations

import pandas as pd

from integration import (
    FuzzyMatcher,
    Merger,
    SchemaMap,
    ValidationError,
    levenshtein_ratio,
    normalize_string,
    validate,
)


# --------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------- #


def test_schema_map_renames_and_coerces_types():
    df = pd.DataFrame(
        {"cust_name": [" Jane ", "Bob"], "mrr": ["199.5", "50"]}
    )
    mapped = SchemaMap(
        rename={"cust_name": "name", "mrr": "revenue"},
        types={"name": str, "revenue": float},
    ).apply(df)
    assert list(mapped.columns) == ["name", "revenue"]
    assert mapped["name"].tolist() == ["Jane", "Bob"]
    assert mapped["revenue"].tolist() == [199.5, 50.0]


def test_schema_map_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    try:
        SchemaMap(rename={"missing": "x"}).apply(df)
    except KeyError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected KeyError")


# --------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------- #


def test_normalize_string_strips_and_lowercases():
    assert normalize_string(" Jane  Doe! ") == "jane doe"


def test_levenshtein_ratio_handles_identical_and_disjoint():
    assert levenshtein_ratio("jane", "jane") == 1.0
    assert 0.0 <= levenshtein_ratio("jane", "xyz") < 1.0


def test_fuzzy_matcher_exact_and_normalized():
    left = pd.DataFrame(
        {"email": ["a@x.com", "B@X.com"], "name": ["Alice", " Bob "]}
    )
    right = pd.DataFrame(
        {"email": ["a@x.com", "b@x.com"], "name": ["Alice", "Bob"]}
    )
    matches = FuzzyMatcher(keys=["email"], threshold=0.85).match(left, right)
    assert len(matches) == 2
    # First is exact, second is normalized (case differs).
    methods = {m.method for m in matches}
    assert "exact" in methods
    assert "normalized" in methods


def test_fuzzy_matcher_levenshtein_catches_typos():
    left = pd.DataFrame({"name": ["Jane Doe"]})
    right = pd.DataFrame({"name": ["Jane Deo"]})  # transposed letters
    matches = FuzzyMatcher(keys=["name"], threshold=0.7).match(left, right)
    assert len(matches) == 1
    assert matches[0].method == "fuzzy"
    assert matches[0].confidence > 0.7


def test_fuzzy_matcher_below_threshold_is_dropped():
    left = pd.DataFrame({"name": ["Jane Doe"]})
    right = pd.DataFrame({"name": ["Xavier"]})
    matches = FuzzyMatcher(keys=["name"], threshold=0.85).match(left, right)
    assert matches == []


# --------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------- #


def test_merger_basic_match_and_conflict_resolution():
    left = pd.DataFrame(
        {"email": ["a@x.com"], "name": ["Alice"], "region": ["US"]}
    )
    right = pd.DataFrame(
        {"email": ["a@x.com"], "name": ["Alice"], "region": ["EU"]}
    )
    matcher = FuzzyMatcher(keys=["email"], threshold=0.85)
    merged, report = Merger(matcher, conflict="prefer_right").merge(left, right)

    assert report.n_matched == 1
    assert len(report.conflicts) == 1
    assert report.conflicts[0].column == "region"
    row = merged.iloc[0]
    assert row["region"] == "EU"
    assert row["_match_method"] == "exact"


def test_merger_unmatched_rows_are_included():
    left = pd.DataFrame({"email": ["a@x.com", "only_left@x.com"]})
    right = pd.DataFrame({"email": ["a@x.com", "only_right@x.com"]})
    matcher = FuzzyMatcher(keys=["email"], threshold=0.85)
    merged, report = Merger(matcher).merge(left, right)

    assert report.n_matched == 1
    assert len(report.unmatched_left) == 1
    assert len(report.unmatched_right) == 1
    assert len(merged) == 3  # 1 matched + 1 unmatched_left + 1 unmatched_right


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


def test_validate_flags_missing_required_and_duplicates():
    df = pd.DataFrame(
        {"id": [1, 2, 2], "email": ["a@x.com", None, "c@x.com"]}
    )
    report = validate(df, required=["email"], unique=["id"])
    assert "email" in report.missing_required
    assert "id" in report.duplicate_rows
    assert not report.ok


def test_validate_strict_raises():
    df = pd.DataFrame({"a": [None]})
    try:
        validate(df, required=["a"], strict=True)
    except ValidationError:
        pass
    else:
        raise AssertionError("expected ValidationError")
