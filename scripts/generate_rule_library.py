"""Generate a library of rules for Eleusis games."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eleusis.game import Rule, RuleValidator
from eleusis.llm import HuggingFaceClient
from eleusis.prompts import get_library_generation_prompt

logger = logging.getLogger(__name__)


def extract_rules_from_response(response: str) -> list[dict]:
    """Extract multiple rules from LLM response.

    Returns:
        List of dicts with 'name', 'description', 'code' keys
    """
    rules = []

    # Find all <RULE>...</RULE> blocks
    rule_blocks = re.findall(r"<RULE>(.*?)</RULE>", response, re.DOTALL | re.IGNORECASE)

    for rule_content in rule_blocks:
        # Extract NAME
        name_match = re.search(
            r"<NAME>(.*?)</NAME>", rule_content, re.DOTALL | re.IGNORECASE
        )
        # Extract DESCRIPTION
        desc_match = re.search(
            r"<DESCRIPTION>(.*?)</DESCRIPTION>", rule_content, re.DOTALL | re.IGNORECASE
        )
        # Extract CODE
        code_match = re.search(
            r"<CODE>(.*?)</CODE>", rule_content, re.DOTALL | re.IGNORECASE
        )

        if name_match and desc_match and code_match:
            rules.append({
                "name": name_match.group(1).strip(),
                "description": desc_match.group(1).strip(),
                "code": code_match.group(1).strip(),
            })
        else:
            logger.warning("Incomplete rule block, skipping")

    return rules


def validate_and_save_rules(
    rules: list[dict],
    output_path: Path,
    validator: RuleValidator,
    num_test_cases: int = 5,
) -> None:
    """Validate rules and save valid ones to JSON.

    Args:
        rules: List of rule dicts with name, description, code
        output_path: Path to output JSON file
        validator: RuleValidator instance
        num_test_cases: Number of test cases for validation
    """
    valid_rules = []
    invalid_rules = []

    for rule_dict in rules:
        name = rule_dict["name"]
        description = rule_dict["description"]
        code = rule_dict["code"]

        logger.info(f"Validating rule: {name}")
        logger.debug(f"Description: {description}")
        logger.debug(f"Code:\n{code}")

        try:
            # Create Rule
            rule = Rule(description, code)

            # Validate
            validation = validator.validate_rule(rule, num_test_cases=num_test_cases)

            if validation.valid:
                logger.info(f"✓ {name}: Valid")
                rule_dict["index"] = len(valid_rules)
                valid_rules.append(rule_dict)
            else:
                logger.warning(f"✗ {name}: Invalid - {', '.join(validation.issues)}")
                invalid_rules.append({
                    **rule_dict,
                    "issues": validation.issues,
                })
        except Exception as e:
            logger.error(f"✗ {name}: Exception during validation - {e}")
            invalid_rules.append({
                **rule_dict,
                "issues": [str(e)],
            })

    # Create output structure
    output = {
        "rules": valid_rules,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_generated": len(rules),
            "valid_count": len(valid_rules),
            "invalid_count": len(invalid_rules),
        },
    }

    if invalid_rules:
        output["invalid_rules"] = invalid_rules

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(valid_rules)} valid rules to {output_path}")
    logger.info(f"Summary: {len(valid_rules)}/{len(rules)} rules valid")


def main():
    """Main entry point for rule library generation."""
    # Example usage : python scripts/generate_rule_library.py --num-rules 20 --output rules.json --model openai/gpt-oss-120b --max-tokens 16384 --test-cases 5
    parser = argparse.ArgumentParser(description="Generate a library of Eleusis rules")
    parser.add_argument(
        "--num-rules",
        type=int,
        default=10,
        help="Number of rules to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rules.json"),
        help="Output JSON file path (default: rules.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max tokens for generation (default: 16384)",
    )
    parser.add_argument(
        "--test-cases",
        type=int,
        default=5,
        help="Number of validation test cases per rule (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s - %(message)s",
    )

    # Initialize clients
    logger.info(f"Initializing LLM client with model: {args.model}")
    llm_client = HuggingFaceClient(model_name=args.model, max_tokens=args.max_tokens)
    validator = RuleValidator()

    # Generate rules
    logger.info(f"Generating {args.num_rules} rules...")
    prompt = get_library_generation_prompt(args.num_rules)

    try:
        response = llm_client.generate(prompt)
        logger.debug(f"Raw response:\n{response}")
    except Exception as e:
        logger.error(f"Failed to generate rules: {e}")
        return 1

    # Extract rules
    logger.info("Extracting rules from response...")
    rules = extract_rules_from_response(response)
    logger.info(f"Extracted {len(rules)} rules")

    if not rules:
        logger.error("No rules extracted from response")
        return 1

    # Validate and save
    logger.info("Validating and saving rules...")
    validate_and_save_rules(rules, args.output, validator, args.test_cases)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
