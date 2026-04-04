"""Tests for response normalization and schema compliance metrics."""

from eleusis.normalization import (
    compute_schema_compliance_rate,
    normalize_action_response,
    normalize_confidence,
)


class TestNormalizeConfidence:
    """Confidence normalization per PRD spec."""

    def test_int_in_range_unchanged(self):
        """Int 0-10 returned as-is with no error."""
        value, error = normalize_confidence(8)
        assert value == 8
        assert error is None

    def test_int_zero(self):
        value, error = normalize_confidence(0)
        assert value == 0
        assert error is None

    def test_int_ten(self):
        value, error = normalize_confidence(10)
        assert value == 10
        assert error is None

    def test_float_rounds_half_up(self):
        """Float 7.5 rounds to 8 (half-up)."""
        value, error = normalize_confidence(7.5)
        assert value == 8
        assert error is None

    def test_float_rounds_down(self):
        """Float 7.4 rounds to 7."""
        value, error = normalize_confidence(7.4)
        assert value == 7
        assert error is None

    def test_float_exact(self):
        """Float 8.0 returns 8."""
        value, error = normalize_confidence(8.0)
        assert value == 8
        assert error is None

    def test_int_65_scales_to_7(self):
        """Int 65 (0-100 scale) → 65/10 = 6.5 → rounds to 7."""
        value, error = normalize_confidence(65)
        assert value == 7
        assert error is None

    def test_int_85_scales_to_9(self):
        """Int 85 → 85/10 = 8.5 → rounds to 9."""
        value, error = normalize_confidence(85)
        assert value == 9
        assert error is None

    def test_int_100_scales_to_10(self):
        value, error = normalize_confidence(100)
        assert value == 10
        assert error is None

    def test_int_11_scales(self):
        """Int 11 → 11/10 = 1.1 → rounds to 1."""
        value, error = normalize_confidence(11)
        assert value == 1
        assert error is None

    def test_float_65_scales(self):
        """Float 65.0 → 65/10 = 6.5 → rounds to 7."""
        value, error = normalize_confidence(65.0)
        assert value == 7
        assert error is None

    def test_negative_int_is_error(self):
        value, error = normalize_confidence(-1)
        assert value is None
        assert error == "confidence_range"

    def test_over_100_int_is_error(self):
        value, error = normalize_confidence(101)
        assert value is None
        assert error == "confidence_range"

    def test_string_is_error(self):
        value, error = normalize_confidence("high")
        assert value is None
        assert error == "confidence_type"

    def test_none_is_error(self):
        value, error = normalize_confidence(None)
        assert value is None
        assert error == "confidence_type"

    def test_bool_is_error(self):
        """Booleans are not valid confidence values even though bool is int subclass."""
        value, error = normalize_confidence(True)
        assert value is None
        assert error == "confidence_type"

    def test_negative_float_is_error(self):
        value, error = normalize_confidence(-0.5)
        assert value is None
        assert error == "confidence_range"


class TestNormalizeActionResponse:
    """Full response normalization including schema error tracking."""

    def test_valid_response_no_errors(self):
        response = {
            "card": "5H",
            "confidence_level": 8,
            "guess_rule": False,
            "tentative_rule": "only hearts",
            "reasoning_summary": "test",
        }
        result = normalize_action_response(response)
        assert result["confidence_level_raw"] == 8
        assert result["confidence_level"] == 8
        assert result["schema_errors"] == []

    def test_confidence_normalized_and_raw_preserved(self):
        response = {
            "card": "5H",
            "confidence_level": 7.5,
            "guess_rule": False,
        }
        result = normalize_action_response(response)
        assert result["confidence_level_raw"] == 7.5
        assert result["confidence_level"] == 8
        assert result["schema_errors"] == []

    def test_confidence_65_scaled(self):
        response = {
            "card": "5H",
            "confidence_level": 65,
            "guess_rule": False,
        }
        result = normalize_action_response(response)
        assert result["confidence_level_raw"] == 65
        assert result["confidence_level"] == 7
        assert result["schema_errors"] == []

    def test_invalid_confidence_records_error(self):
        response = {
            "card": "5H",
            "confidence_level": "high",
            "guess_rule": False,
        }
        result = normalize_action_response(response)
        assert result["confidence_level_raw"] == "high"
        assert result["confidence_level"] is None
        assert "confidence_type" in result["schema_errors"]

    def test_nonboolean_guess_rule_records_error(self):
        response = {
            "card": "5H",
            "confidence_level": 5,
            "guess_rule": "yes",
        }
        result = normalize_action_response(response)
        assert "guess_rule_type" in result["schema_errors"]

    def test_none_response_returns_empty(self):
        result = normalize_action_response(None)
        assert result["confidence_level_raw"] is None
        assert result["confidence_level"] is None
        assert result["schema_errors"] == []

    def test_missing_confidence_key(self):
        response = {"card": "5H", "guess_rule": False}
        result = normalize_action_response(response)
        assert result["confidence_level_raw"] is None
        assert result["confidence_level"] is None
        assert result["schema_errors"] == []


class TestSchemaComplianceRate:
    """Aggregated schema compliance metric."""

    def test_all_compliant(self):
        turns = [
            {"schema_errors": []},
            {"schema_errors": []},
            {"schema_errors": []},
        ]
        assert compute_schema_compliance_rate(turns) == 1.0

    def test_none_compliant(self):
        turns = [
            {"schema_errors": ["confidence_range"]},
            {"schema_errors": ["guess_rule_type"]},
        ]
        assert compute_schema_compliance_rate(turns) == 0.0

    def test_partial_compliance(self):
        turns = [
            {"schema_errors": []},
            {"schema_errors": ["confidence_type"]},
            {"schema_errors": []},
            {"schema_errors": ["confidence_range", "guess_rule_type"]},
        ]
        assert compute_schema_compliance_rate(turns) == 0.5

    def test_empty_turns(self):
        assert compute_schema_compliance_rate([]) is None

    def test_turns_without_schema_errors_key(self):
        """Turns missing schema_errors are treated as compliant."""
        turns = [{}, {"schema_errors": ["confidence_range"]}]
        assert compute_schema_compliance_rate(turns) == 0.5


class TestRunnerIntegration:
    """Normalization integrates correctly into runner turn data."""

    def test_turn_data_contains_normalized_fields(self):
        """Turn data from runner should have raw + normalized confidence and schema_errors."""
        # Simulate what runner does: normalize the llm_response and merge into turn_data
        llm_response = {
            "card": "5H",
            "confidence_level": 65,
            "guess_rule": False,
            "tentative_rule": "even cards only",
            "reasoning_summary": "test",
        }

        norm = normalize_action_response(llm_response)

        # Build turn_data the way runner.py should
        turn_data = {
            "turn_number": 1,
            "llm_response": llm_response.copy(),
            "confidence_level_raw": norm["confidence_level_raw"],
            "confidence_level": norm["confidence_level"],
            "schema_errors": norm["schema_errors"],
        }

        assert turn_data["confidence_level_raw"] == 65
        assert turn_data["confidence_level"] == 7
        assert turn_data["schema_errors"] == []

    def test_shadow_uses_normalized_confidence(self):
        """Shadow evaluation should use normalized confidence, not raw."""
        # 7.5 raw -> 8 normalized. Old code: isinstance(7.5, int) == False -> no shadow
        # New code: normalized 8 >= 5 -> shadow triggers
        llm_response = {
            "confidence_level": 7.5,
            "tentative_rule": "only red cards",
            "guess_rule": False,
        }
        norm = normalize_action_response(llm_response)
        MIN_CONFIDENCE_FOR_SHADOW = 5

        normalized_conf = norm["confidence_level"]
        assert normalized_conf == 8
        assert normalized_conf is not None and normalized_conf >= MIN_CONFIDENCE_FOR_SHADOW

    def test_schema_compliance_in_round_result(self):
        """Round result should include schema_compliance_rate computed from turns."""
        turns = [
            {"schema_errors": []},
            {"schema_errors": ["confidence_range"]},
            {"schema_errors": []},
        ]
        rate = compute_schema_compliance_rate(turns)
        # 2 out of 3 turns are compliant
        assert abs(rate - 2 / 3) < 1e-9
