"""End-to-end example: join CRM and billing CSVs with heterogeneous schemas.

Run:
    python examples/run_example.py

Demonstrates schema canonicalization, fuzzy matching (exact + normalized +
Levenshtein), and conflict-aware merging.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from integration import FuzzyMatcher, Merger, SchemaMap  # noqa: E402


def main() -> int:
    examples = Path(__file__).resolve().parent
    crm_raw = pd.read_csv(examples / "crm.csv")
    billing_raw = pd.read_csv(examples / "billing.csv")

    # Canonicalize both sides to a shared schema (name, email, region, revenue).
    crm = SchemaMap(
        rename={"cust_name": "name", "cust_email": "email", "region": "region"},
        types={"name": str, "email": str, "region": str},
    ).apply(crm_raw)

    billing = SchemaMap(
        rename={"full_name": "name", "contact_email": "email", "mrr": "revenue"},
        types={"name": str, "email": str, "revenue": float},
    ).apply(billing_raw)

    print("=== CRM (canonical) ===")
    print(crm.to_string(index=False))
    print()
    print("=== Billing (canonical) ===")
    print(billing.to_string(index=False))
    print()

    # Match on email, with name as a fuzzy fallback.
    matcher = FuzzyMatcher(keys=["email", "name"], threshold=0.85)
    merger = Merger(matcher, conflict="prefer_left")
    merged, report = merger.merge(crm, billing)

    print("=== Merged ===")
    print(merged.to_string(index=False))
    print()
    print("=== Report ===")
    print(report.summary())
    if report.conflicts:
        print("Conflicts:")
        for c in report.conflicts:
            print(
                f"  {c.column}: left={c.left_value!r} right={c.right_value!r} "
                f"-> {c.resolved_value!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
