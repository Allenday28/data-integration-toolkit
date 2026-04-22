"""Building blocks for joining messy, heterogeneous data sources."""

from integration.schema import SchemaMap
from integration.match import FuzzyMatcher, normalize_string, levenshtein_ratio
from integration.merge import Merger, MergeReport
from integration.validate import validate, ValidationError

__version__ = "0.1.0"
__all__ = [
    "SchemaMap",
    "FuzzyMatcher",
    "Merger",
    "MergeReport",
    "validate",
    "ValidationError",
    "normalize_string",
    "levenshtein_ratio",
]
