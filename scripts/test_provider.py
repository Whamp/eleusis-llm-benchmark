"""Test provider reasoning format detection.

Usage:
    uv run scripts/test_provider.py "deepseek-r1"
    uv run scripts/test_provider.py "claude-opus"
    uv run scripts/test_provider.py --all
"""

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from eleusis.llm import create_client

# Test prompt to elicit reasoning
TEST_PROMPT = "What is the derivative of x^2 + 3x + ln(ln(x^2))?"

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default models to test with --all flag (keys from models.yaml)
DEFAULT_MODELS = [
    'deepseek-r1',
    'gpt-oss-120b',
    # 'claude-opus',
    # 'gpt-5.2',
    # 'gemini-3-pro',
    # 'grok-4',
]


def probe_reasoning(model_key: str) -> dict:
    """Probe model for reasoning format using generate() with automatic force-answer."""
    logger.info(f"Probing: {model_key}")

    result = {
        "model": model_key,
        "success": False,
        "reasoning_format": None,
        "thinking_tokens": None,
        "answer_preview": None,
        "reasoning_preview": None,
        "latency": None,
        "error": None,
        "force_answer_used": False,
        "total_calls": 0,
    }

    try:
        start = time.time()
        client = create_client(model_key, max_tokens=2048)

        # Use generate() which handles force-answer automatically
        answer = client.generate(TEST_PROMPT)
        result["latency"] = time.time() - start

        # Get metrics from the call(s)
        metrics_list = client.call_metrics
        result["total_calls"] = len(metrics_list)
        result["force_answer_used"] = len(metrics_list) > 1

        # Aggregate reasoning tokens from all calls
        total_reasoning_tokens = sum(m.reasoning_tokens or 0 for m in metrics_list)
        if total_reasoning_tokens > 0:
            result["thinking_tokens"] = total_reasoning_tokens

        # Check first call's metrics for reasoning format detection
        first_metrics = metrics_list[0] if metrics_list else None

        # Detect reasoning format from first call's has_reasoning flag and token count
        if first_metrics:
            if first_metrics.has_reasoning:
                # Check if reasoning tokens came from API (field-based or hidden)
                if first_metrics.reasoning_tokens and first_metrics.reasoning_tokens > 0:
                    # Could be field-based or hidden - check if answer has think tags
                    if "</think>" in answer or "<think>" in answer:
                        result["reasoning_format"] = "tags:think (inline)"
                    else:
                        result["reasoning_format"] = "field-based or hidden"
                else:
                    # Reasoning detected via content inspection (think tags)
                    result["reasoning_format"] = "tags:think (inline)"
            elif total_reasoning_tokens > 0:
                result["reasoning_format"] = "field-based or hidden"

        # Set answer preview (generate() strips think tags automatically)
        if answer:
            result["answer_preview"] = answer[:300].replace('\n', ' ')
        else:
            result["answer_preview"] = "(empty response)"

        result["success"] = True

    except RuntimeError as e:
        if "truncated" in str(e).lower():
            result["error"] = f"Force-answer failed: {e}"
        else:
            result["error"] = str(e)
        logger.error(f"  Failed: {e}")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  Failed: {e}")

    return result


def print_result(result: dict) -> None:
    """Print probe result."""
    print(f"\n{'='*70}")
    print(f"MODEL: {result['model']}")
    print(f"{'='*70}")

    if not result["success"]:
        print(f"  ERROR: {result['error']}")
        return

    print(f"  Reasoning format: {result['reasoning_format'] or 'none'}")
    print(f"  Thinking tokens: {result['thinking_tokens'] or 'n/a'}")
    print(f"  Latency: {result['latency']:.2f}s")
    print(f"  Total calls: {result['total_calls']}")
    if result['force_answer_used']:
        print(f"  Force-answer: YES (response was truncated)")

    print(f"\n  Answer preview:")
    print(f"    {result['answer_preview']}")

    if result["reasoning_preview"]:
        print(f"\n  Reasoning preview:")
        print(f"    {result['reasoning_preview']}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM provider reasoning format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model key from models.yaml (e.g., 'claude-opus', 'deepseek-r1')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all default models"
    )

    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Must specify a model or use --all")

    if args.model and args.all:
        parser.error("Cannot specify both a model and --all")

    models = DEFAULT_MODELS if args.all else [args.model]

    if args.all:
        logger.info(f"Testing {len(models)} default models")

    results = []
    for model_key in models:
        result = probe_reasoning(model_key)
        results.append(result)
        print_result(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r["success"])
    reasoning_count = sum(1 for r in results if r["success"] and r["reasoning_format"])

    print(f"Total: {len(results)} models, {success_count} succeeded, {reasoning_count} with reasoning")

    sys.exit(0 if success_count == len(results) else 1)


if __name__ == "__main__":
    main()
