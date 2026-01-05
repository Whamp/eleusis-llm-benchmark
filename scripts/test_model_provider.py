"""Test HuggingFace Inference Provider model responsiveness."""

import argparse
import logging
import re
import sys
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default models to test with --all flag
DEFAULT_MODELS = [
    "moonshotai/Kimi-K2-Thinking",

    "meta-llama/Llama-3.3-70B-Instruct",

    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",

    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.2",
    
    "MiniMaxAI/MiniMax-M2",

    "HuggingFaceTB/SmolLM3-3B:hf-inference",

    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507",

    "allenai/Olmo-3-7B-Think"
]


def test_model(model_name: str, provider: str | None = None) -> bool:
    """Test model and report detailed metrics."""
    logger.info("=" * 80)
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Provider: {provider or 'auto (default)'}")

    client = InferenceClient(bill_to="huggingface", provider=provider, model=model_name)

    try:
        messages = [{"role": "user", "content": "Give me a short introduction to the rules of Eleusis card game."}]

        logger.info("Sending test request...")

        # Start timing
        start_time = time.time()
        result = client.chat_completion(
            messages=messages,
            max_tokens=512,
        )
        end_time = time.time()

        # Calculate duration
        duration = end_time - start_time

        logger.info(f"✓ Model responded successfully!")

        # Extract message and response content
        message = result.choices[0].message
        response_content = message.content
        finish_reason = result.choices[0].finish_reason

        # Check for reasoning field
        has_reasoning = hasattr(message, 'reasoning') and message.reasoning is not None and message.reasoning != ""

        # Check for <think> tags in content
        has_think_tags = bool(re.search(r'<think>.*?</think>', response_content, re.DOTALL))

        # Extract token usage
        prompt_tokens = result.usage.prompt_tokens if hasattr(result, 'usage') else None
        completion_tokens = result.usage.completion_tokens if hasattr(result, 'usage') else None
        total_tokens = result.usage.total_tokens if hasattr(result, 'usage') else None

        # Calculate throughput (tokens per second)
        throughput = completion_tokens / duration if completion_tokens else None

        # Log metrics
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Finish reason: {finish_reason}")

        if prompt_tokens is not None:
            logger.info(f"Prompt tokens: {prompt_tokens}")
            logger.info(f"Completion tokens: {completion_tokens}")
            logger.info(f"Total tokens: {total_tokens}")
            logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        else:
            logger.warning("Token usage information not available")

        logger.info(f"Has reasoning field: {has_reasoning}")
        logger.info(f"Has <think> tags: {has_think_tags}")

        logger.info(f"\nResponse preview (first 200 chars):\n{response_content[:200]}...")

        return True

    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def main():
    """Main entry point for testing model responsiveness."""
    parser = argparse.ArgumentParser(
        description="Test HuggingFace Inference Provider model responsiveness"
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Model name to test (e.g., 'meta-llama/Llama-3.3-70B-Instruct')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider name (default: auto). Examples: sambanova, cerebras, groq, together, replicate, fireworks-ai"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all models from the default list"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.all and args.model:
        logger.error("Cannot specify both --all and a specific model")
        sys.exit(1)

    if not args.all and not args.model:
        logger.error("Must specify either a model name or --all flag")
        parser.print_help()
        sys.exit(1)

    # Determine which models to test
    if args.all:
        models_to_test = [model_name + ":fastest" for model_name in DEFAULT_MODELS]
        logger.info(f"Testing {len(models_to_test)} models from default list\n")
    else:
        models_to_test = [args.model]

    # Test each model
    results = {}
    for model in models_to_test:
        success = test_model(model, args.provider)
        results[model] = success

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed

    for model, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {model}")

    logger.info(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
