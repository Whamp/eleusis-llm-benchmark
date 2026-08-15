"""Validated rule-library records shared by validation and rule selection."""

from typing import NotRequired

from pydantic import TypeAdapter
from typing_extensions import TypedDict


class RuleLibraryEntry(TypedDict):
    """Validated executable rule loaded from the benchmark library."""

    description: str
    code: str
    name: NotRequired[str]
    avg_acceptance_rate: NotRequired[float]


class RuleMetadata(TypedDict):
    """Rule identity and source retained with round results."""

    name: str | None
    description: str
    code: str


_RULE_LIBRARY_ADAPTER = TypeAdapter(list[RuleLibraryEntry])


def parse_rule_library_entries(value: object) -> list[RuleLibraryEntry]:
    """Validate executable rule-library entries from decoded JSON."""
    return _RULE_LIBRARY_ADAPTER.validate_python(value)
