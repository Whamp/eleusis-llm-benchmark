"""Unified provider test script for HuggingFace and OpenRouter.

Replaces scripts/test_openrouter.py and scripts/check_model_provider.py.

Usage:
    # Test specific model (default: full probe)
    uv run scripts/test_provider.py "hf:Qwen/Qwen3-4B-Thinking-2507"
    uv run scripts/test_provider.py "openrouter:anthropic/claude-3.5-haiku"

    # Quick connectivity check only
    uv run scripts/test_provider.py "hf:model-name" --quick

    # Test disable_thinking support
    uv run scripts/test_provider.py "openrouter:deepseek/deepseek-r1" --test-disable

    # Test all default models
    uv run scripts/test_provider.py --all
"""

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from eleusis.llm import create_client, probe_model_capabilities

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default models to test with --all flag
DEFAULT_MODELS = [
    'hf:Qwen/Qwen3-4B-Thinking-2507',
    'hf:openai/gpt-oss-20b',
    'hf:openai/gpt-oss-120b',
    'hf:openai/gpt-oss-20b:together',
    'hf:openai/gpt-oss-120b:together',
    'hf:deepseek-ai/DeepSeek-R1:together',
    'hf:zai-org/GLM-4.7:zai-org',
    'hf:moonshotai/Kimi-K2-Thinking:together',
    'openrouter:openai/gpt-5.2',
    'openrouter:google/gemini-3-flash-preview',
    'openrouter:anthropic/claude-4.5-opus',
    'openrouter:x-ai/grok-4'
]


def test_connectivity(model_spec: str) -> bool:
    """Basic connectivity test - verify API responds."""
    logger.info(f"Testing connectivity: {model_spec}")

    try:
        start = time.time()
        client = create_client(model_spec, max_tokens=50)
        messages = [{"role": "user", "content": "Say 'OK'"}]
        response, metrics = client._call_api(messages)
        duration = time.time() - start

        content = response.message.content[:100] if response.message.content else "(empty)"
        logger.info(f"  Response: {content}")
        logger.info(f"  Latency: {duration:.2f}s")
        logger.info(f"  Finish reason: {metrics.finish_reason}")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def test_full_probe(model_spec: str) -> bool:
    """Full capability probe - detect reasoning format and features."""
    logger.info(f"Running full probe: {model_spec}")

    try:
        client = create_client(model_spec, max_tokens=500)
        caps = probe_model_capabilities(client)

        logger.info(f"  Provider: {caps.provider}")
        logger.info(f"  Has reasoning: {caps.has_reasoning}")
        logger.info(f"  Reasoning format: {caps.reasoning_format or 'none'}")
        logger.info(f"  Malformed tags: {caps.thinking_tags_malformed}")
        logger.info(f"  Supports disable_thinking: {caps.supports_disable_thinking}")
        logger.info(f"  Probe latency: {caps.probe_latency_seconds:.2f}s")

        if caps.probe_response_preview:
            preview = caps.probe_response_preview.replace('\n', ' ')[:100]
            logger.info(f"  Response preview: {preview}...")

        return True
    except Exception as e:
        logger.error(f"  Probe failed: {e}")
        return False


def test_disable_thinking(model_spec: str) -> bool:
    """Test if disable_thinking API parameter works."""
    logger.info(f"Testing disable_thinking: {model_spec}")

    try:
        # First, probe to get capabilities
        client = create_client(model_spec, max_tokens=500, run_probe=True)

        if not client.capabilities:
            logger.warning("  No capabilities detected, skipping")
            return True

        if not client.capabilities.has_reasoning:
            logger.info("  Skipping: model doesn't use reasoning")
            return True

        if not client.capabilities.supports_disable_thinking:
            logger.info("  Skipping: model doesn't support disable_thinking API")
            return True

        # Make a call with disable_thinking=True
        logger.info("  Making call with disable_thinking=True...")
        messages = [{"role": "user", "content": "What is 2+2? Think step by step."}]
        response, metrics = client._call_api(messages, disable_thinking=True)

        content = response.message.content or ""

        # Check if thinking was suppressed
        has_thinking = "<think>" in content or "</think>" in content
        has_reasoning_field = (
            (hasattr(response.message, 'reasoning_content') and response.message.reasoning_content) or
            (hasattr(response.message, 'reasoning') and response.message.reasoning)
        )

        if has_thinking or has_reasoning_field:
            logger.warning("  disable_thinking did NOT suppress reasoning")
            logger.warning(f"    has_think_tags: {has_thinking}")
            logger.warning(f"    has_reasoning_field: {has_reasoning_field}")
            return False
        else:
            logger.info("  disable_thinking worked (no reasoning in response)")
            return True

    except Exception as e:
        logger.error(f"  Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM provider connectivity and capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model spec (e.g., 'hf:model-name' or 'openrouter:model-name')"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick connectivity test only"
    )
    parser.add_argument(
        "--test-disable",
        action="store_true",
        help="Test disable_thinking API support"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all default models"
    )

    args = parser.parse_args()

    # Validate args
    if not args.model and not args.all:
        parser.error("Must specify a model or use --all")

    if args.model and args.all:
        parser.error("Cannot specify both a model and --all")

    # Determine models to test
    if args.all:
        models = DEFAULT_MODELS
        logger.info(f"Testing {len(models)} default models")
    else:
        models = [args.model]

    # Run tests
    all_results = {}

    for model_spec in models:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"MODEL: {model_spec}")
        logger.info("=" * 70)

        results = {}

        # Always run connectivity test
        results["connectivity"] = test_connectivity(model_spec)

        # Run full probe unless --quick
        if not args.quick:
            results["probe"] = test_full_probe(model_spec)

        # Run disable_thinking test if requested
        if args.test_disable:
            results["disable_thinking"] = test_disable_thinking(model_spec)

        all_results[model_spec] = results

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    total_passed = 0
    total_failed = 0

    for model_spec, results in all_results.items():
        logger.info(f"\n{model_spec}:")
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {test_name}: {status}")
            if passed:
                total_passed += 1
            else:
                total_failed += 1

    logger.info("")
    logger.info(f"Total: {total_passed + total_failed} tests, {total_passed} passed, {total_failed} failed")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
