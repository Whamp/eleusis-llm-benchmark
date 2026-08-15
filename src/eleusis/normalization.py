"""Response normalization and schema compliance utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from typing_extensions import TypedDict


class NormalizedActionResponse(TypedDict):
    """Normalized confidence fields and action-schema error tags."""

    confidence_level_raw: object
    confidence_level: int | None
    schema_errors: list[str]


def normalize_confidence(raw_value: object) -> tuple[int | None, str | None]:
    """Normalize a confidence_level value to the 0-10 integer scale.

    Rules:
      - int 0-10: unchanged
      - float 0.0-10.0: rounded half-up to int
      - numeric 11-100: divide by 10, round half-up to int
      - anything else: None + error tag

    Returns:
        (normalized_value, error_tag) — error_tag is None on success.
    """
    if isinstance(raw_value, bool):
        return None, "confidence_type"

    if isinstance(raw_value, int):
        if 0 <= raw_value <= 10:
            return raw_value, None
        if 11 <= raw_value <= 100:
            return _round_half_up(raw_value / 10), None
        return None, "confidence_range"

    if isinstance(raw_value, float):
        if 0.0 <= raw_value <= 10.0:
            return _round_half_up(raw_value), None
        if 10.0 < raw_value <= 100.0:
            return _round_half_up(raw_value / 10), None
        return None, "confidence_range"

    return None, "confidence_type"


def _round_half_up(x: float) -> int:
    """Round to nearest int, with halves rounding up (0.5 -> 1, 7.5 -> 8)."""
    return math.floor(x + 0.5)


def normalize_action_response(
    response: Mapping[str, object] | None,
) -> NormalizedActionResponse:
    """Normalize an LLM action response, preserving raw values and tracking errors.

    Returns a dict with:
      - confidence_level_raw: the original value (or None)
      - confidence_level: the normalized 0-10 int (or None on error)
      - schema_errors: list of error tags (e.g. ["confidence_range", "guess_rule_type"])
    """
    result: NormalizedActionResponse = {
        "confidence_level_raw": None,
        "confidence_level": None,
        "schema_errors": [],
    }

    if response is None:
        return result

    # Confidence normalization
    if "confidence_level" in response:
        raw = response["confidence_level"]
        result["confidence_level_raw"] = raw
        norm, error = normalize_confidence(raw)
        result["confidence_level"] = norm
        if error:
            result["schema_errors"].append(error)

    # guess_rule type validation
    if "guess_rule" in response:
        guess_val = response["guess_rule"]
        if not isinstance(guess_val, bool):
            result["schema_errors"].append("guess_rule_type")

    return result


def compute_schema_compliance_rate(
    turns: Sequence[Mapping[str, object]],
) -> float | None:
    """Compute fraction of turns with no schema errors.

    Args:
        turns: list of turn dicts, each optionally containing 'schema_errors'.

    Returns:
        Float 0.0-1.0, or None if turns is empty.
    """
    if not turns:
        return None
    compliant = sum(1 for t in turns if not t.get("schema_errors"))
    return compliant / len(turns)
