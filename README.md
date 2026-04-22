# data-integration-toolkit

Reusable building blocks for **joining messy data from multiple sources** —
schema normalization, fuzzy-key matching, conflict resolution, and validation
— with a small declarative API.

The goal: given two CSVs that *should* describe the same entities but don't
share clean keys (different column names, whitespace, casing, typos), produce
a merged dataset plus a report of what matched, what didn't, and where the
two sources disagreed.

## Features

- **Schema mapping** — rename + type-coerce heterogeneous columns into a
  canonical shape.
- **Fuzzy matching** — exact, normalized-string, and Levenshtein-based key
  matchers with per-match confidence scores.
- **Conflict resolution** — when both sides have a value, resolve via
  `prefer_left`, `prefer_right`, `non_null`, or a custom function.
- **Validation** — check for required fields, uniqueness, type conformance.
- **Integration report** — summary of matched rows, unmatched rows, and
  conflicts, for auditing the merge.

## Quick start

```bash
pip install -r requirements.txt
python examples/run_example.py
```

The example merges two customer files (one from CRM, one from billing) that
share no canonical ID, then prints the merged frame and the match report.

## Example

```python
import pandas as pd
from integration import SchemaMap, FuzzyMatcher, Merger

crm = pd.read_csv("crm.csv")
billing = pd.read_csv("billing.csv")

# Normalize both sides to a canonical schema.
canonical_crm = SchemaMap(
    {"cust_name": "name", "cust_email": "email"},
    types={"name": str, "email": str},
).apply(crm)

canonical_billing = SchemaMap(
    {"full_name": "name", "contact_email": "email", "mrr": "revenue"},
    types={"name": str, "email": str, "revenue": float},
).apply(billing)

# Match on normalized email with a fuzzy fallback on name.
matcher = FuzzyMatcher(keys=["email", "name"], threshold=0.85)
merger = Merger(matcher, conflict="prefer_left")
merged, report = merger.merge(canonical_crm, canonical_billing)

print(merged)
print(report.summary())
```

## What's in the toolkit

| Module | Purpose |
| --- | --- |
| `schema.py` | `SchemaMap` — rename columns, coerce types, drop unmapped |
| `match.py` | `FuzzyMatcher` — exact + normalized + Levenshtein key matching |
| `merge.py` | `Merger` — apply matcher, resolve conflicts, produce `MergeReport` |
| `validate.py` | `validate` — required fields / uniqueness / type checks |
| `cli.py` | `python -m integration merge left.csv right.csv --config cfg.yaml` |

## Layout

```
data-integration-toolkit/
├── src/integration/
│   ├── schema.py
│   ├── match.py
│   ├── merge.py
│   ├── validate.py
│   └── cli.py
├── tests/test_integration.py
└── examples/
    ├── run_example.py
    ├── crm.csv
    └── billing.csv
```

## Tech

Python 3.10+ · pandas · pytest. No external fuzzy-matching deps — a compact
Levenshtein implementation is included in `match.py`.

## License

MIT — see LICENSE.
