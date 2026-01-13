"""Compile human-written rules into a JSON library for Eleusis games."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eleusis.game import Rule, RuleEvaluator, RuleValidator
from eleusis.llm import HuggingFaceClient
from eleusis.prompts import get_rule_compile_prompt

logger = logging.getLogger(__name__)


def parse_rules_file(filepath: Path) -> list[str]:
    """Parse rules file, returning list of rule descriptions.

    Skips blank lines and lines starting with #.
    """
    rules = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rules.append(line)
    return rules


def extract_name_and_code(response: str) -> tuple[str, str] | None:
    """Extract name and code from LLM response.

    Returns:
        Tuple of (name, code) or None if extraction fails
    """
    name_match = re.search(r"<NAME>(.*?)</NAME>", response, re.DOTALL | re.IGNORECASE)
    code_match = re.search(r"<CODE>(.*?)</CODE>", response, re.DOTALL | re.IGNORECASE)

    if name_match and code_match:
        return name_match.group(1).strip(), code_match.group(1).strip()
    return None


def compile_rule(
    description: str,
    llm_client: HuggingFaceClient,
) -> dict | None:
    """Compile a single rule description into name and code.

    Returns:
        Dict with name, description, code or None if compilation fails
    """
    prompt = get_rule_compile_prompt(description)

    try:
        response = llm_client.generate(prompt)
        logger.debug(f"LLM response:\n{response}")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

    extracted = extract_name_and_code(response)
    if not extracted:
        logger.warning(f"Failed to extract name/code from response for: {description[:50]}...")
        return None

    name, code = extracted
    return {
        "name": name,
        "description": description,
        "code": code,
    }


def validate_and_save_rules(
    rules: list[dict],
    output_path: Path,
    validator: RuleValidator,
    evaluator: RuleEvaluator,
    num_test_cases: int = 5,
) -> None:
    """Validate and evaluate rules, save results to JSON."""
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
            rule = Rule(description, code)
            validation = validator.validate_rule(rule, num_test_cases=num_test_cases)

            if validation.valid:
                logger.info("  Valid - evaluating...")
                eval_results = evaluator.evaluate(rule)
                rule_dict.update(eval_results)
                logger.info(f"  Acceptance rate: {eval_results['avg_acceptance_rate']:.1%}")
                valid_rules.append(rule_dict)
            else:
                logger.warning(f"  Invalid - {', '.join(validation.issues)}")
                invalid_rules.append({**rule_dict, "issues": validation.issues})
        except Exception as e:
            logger.error(f"  Exception during validation - {e}")
            invalid_rules.append({**rule_dict, "issues": [str(e)]})

    output = {
        "rules": valid_rules,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_compiled": len(rules),
            "valid_count": len(valid_rules),
            "invalid_count": len(invalid_rules),
        },
        "evaluation_params": {
            "num_simulations": evaluator.num_simulations,
            "plays_per_simulation": evaluator.plays_per_simulation,
        },
    }

    if invalid_rules:
        output["invalid_rules"] = invalid_rules

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(valid_rules)} valid rules to {output_path}")
    logger.info(f"Summary: {len(valid_rules)}/{len(rules)} rules valid")


def main():
    """Main entry point for rule library compilation."""
    parser = argparse.ArgumentParser(description="Compile human-written rules into JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rules.txt"),
        help="Input rules file (default: rules.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rules.json"),
        help="Output JSON file (default: rules.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model to use for compilation (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for LLM response (default: 2048)",
    )
    parser.add_argument(
        "--test-cases",
        type=int,
        default=5,
        help="Number of validation test cases per rule (default: 5)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10,
        help="Number of simulations for acceptance rate evaluation (default: 10)",
    )
    parser.add_argument(
        "--plays-per-simulation",
        type=int,
        default=50,
        help="Number of random card plays per simulation (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s - %(message)s",
    )

    # Parse input file
    logger.info(f"Reading rules from {args.input}")
    rule_descriptions = parse_rules_file(args.input)
    logger.info(f"Found {len(rule_descriptions)} rules to compile")

    if not rule_descriptions:
        logger.error("No rules found in input file")
        return 1

    # Initialize clients
    logger.info(f"Initializing LLM client with model: {args.model}")
    llm_client = HuggingFaceClient(model_name=args.model, max_tokens=args.max_tokens)
    validator = RuleValidator()
    evaluator = RuleEvaluator(
        num_simulations=args.num_simulations,
        plays_per_simulation=args.plays_per_simulation,
    )

    # Compile each rule
    compiled_rules = []
    for i, description in enumerate(rule_descriptions, 1):
        logger.info(f"[{i}/{len(rule_descriptions)}] Compiling: {description[:60]}...")
        result = compile_rule(description, llm_client)
        if result:
            compiled_rules.append(result)
            logger.info(f"  -> {result['name']}")
        else:
            logger.warning("  -> Failed to compile")

    logger.info(f"Successfully compiled {len(compiled_rules)}/{len(rule_descriptions)} rules")

    if not compiled_rules:
        logger.error("No rules were successfully compiled")
        return 1

    # Validate, evaluate, and save
    logger.info("Validating and evaluating rules...")
    validate_and_save_rules(compiled_rules, args.output, validator, evaluator, args.test_cases)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
