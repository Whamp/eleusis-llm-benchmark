"""Factory for creating rules from various sources."""

import json
import logging
import random
from pathlib import Path

from eleusis.game_engine import Rule
from eleusis.llm_client import HuggingFaceClient
from eleusis.llm_player import LLMRuleMaker
from eleusis.python_rule import PythonRule
from eleusis.rules import RuleValidator

logger = logging.getLogger(__name__)


class RuleFactory:
    """Factory for creating rules from LLM or pre-generated library."""

    def __init__(
        self,
        mode: str,
        library_path: str | None = None,
        selection: str = "random",
        llm_client: HuggingFaceClient | None = None,
        validator: RuleValidator | None = None,
    ) -> None:
        """Initialize rule factory.

        Args:
            mode: "llm" or "library"
            library_path: Path to rules JSON file (required for library mode)
            selection: "random" or "sequential" (for library mode)
            llm_client: LLM client for generating rules (required for llm mode)
            validator: Validator for generated rules (required for llm mode)
        """
        self.mode = mode
        self.library_path = library_path
        self.selection = selection
        self.llm_client = llm_client
        self.validator = validator
        self._library_index = 0
        self._library_rules: list[dict] | None = None

        if mode == "library":
            if not library_path:
                raise ValueError("library_path required for library mode")
            self._load_library()
        elif mode == "llm":
            if not llm_client or not validator:
                raise ValueError("llm_client and validator required for llm mode")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'llm' or 'library'")

    def _load_library(self) -> None:
        """Load rule library from JSON file."""
        path = Path(self.library_path)
        if not path.exists():
            raise FileNotFoundError(f"Rule library not found: {path}")

        with open(path) as f:
            data = json.load(f)

        self._library_rules = data.get("rules", [])
        if not self._library_rules:
            raise ValueError(f"No rules found in library: {path}")

        logger.info(f"Loaded {len(self._library_rules)} rules from {path}")

    def create_rule(self) -> Rule:
        """Create a rule based on the configured mode."""
        if self.mode == "library":
            return self._create_from_library()
        else:
            return self._create_from_llm()

    def _create_from_library(self) -> Rule:
        """Create rule from pre-generated library."""
        if self.selection == "random":
            rule_dict = random.choice(self._library_rules)
        else:  # sequential
            rule_dict = self._library_rules[self._library_index]
            self._library_index = (self._library_index + 1) % len(self._library_rules)

        description = rule_dict["description"]
        code = rule_dict["code"]
        name = rule_dict.get("name", "library_rule")

        logger.info(f"Using library rule: {name}")
        logger.info(f"Description: {description}")
        logger.debug(f"Code:\n{code}")

        return PythonRule(description, code)

    def _create_from_llm(self) -> Rule:
        """Create rule by generating with LLM."""
        logger.info("Generating rule using LLM...")
        rule_maker = LLMRuleMaker(
            self.llm_client,
            self.validator,
            max_attempts=3,
            max_tokens=8192,
        )

        rule = rule_maker.generate_rule()
        if not rule:
            raise RuntimeError("Failed to generate valid rule")

        return rule
