#!/usr/bin/env python3
"""Run benchmark evaluations in parallel worker processes.

Workers receive different ``batch_round_offset`` values, so each evaluates all
assigned rules with a distinct deterministic deck shuffle.
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from eleusis.benchmark_config import BenchmarkConfig, parse_benchmark_config
from eleusis.suites import resolve_suite


@dataclass(frozen=True)
class ParallelPlan:
    """Resolved work distribution for one parallel benchmark launch."""

    suite_name: str | None
    worker_offsets: list[int]
    total_rules: int
    total_rounds: int


@dataclass(frozen=True)
class WorkerProcess:
    """Started benchmark worker and its identifying metadata."""

    worker_id: int
    process: subprocess.Popen[bytes]
    log_file: Path


def parse_args() -> argparse.Namespace:
    """Parse model, worker-count, suite, and output options."""
    parser = argparse.ArgumentParser(description="Run parallel benchmark evaluations")
    parser.add_argument("--model", required=True, help="Model key from models.yaml")
    parser.add_argument(
        "--config", default="config.yaml", help="Config file (default: config.yaml)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto from suite batch indices)",
    )
    parser.add_argument(
        "--suite", type=str, default=None, help="Named benchmark suite from suites.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    return parser.parse_args()


def _load_config(config_argument: str) -> BenchmarkConfig:
    """Load and validate the requested benchmark configuration."""
    config_path = Path(config_argument)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / config_path
    with config_path.open() as config_file:
        return parse_benchmark_config(yaml.safe_load(config_file))


def _suite_plan(
    suite_name: str,
    requested_workers: int | None,
) -> ParallelPlan:
    """Resolve suite cases into bounded worker offsets."""
    suite_cases = resolve_suite(suite_name)
    batch_indices = sorted({index for _, index in suite_cases})
    total_rules = len(dict.fromkeys(name for name, _ in suite_cases))
    worker_count = requested_workers or len(batch_indices)
    if worker_count > len(batch_indices):
        print(
            f"WARNING: {worker_count} workers requested but suite {suite_name!r} "
            f"has {len(batch_indices)} batch indices."
        )
        print(f"Capping workers to {len(batch_indices)}.")
        worker_count = len(batch_indices)
    return ParallelPlan(
        suite_name=suite_name,
        worker_offsets=batch_indices[:worker_count],
        total_rules=total_rules,
        total_rounds=len(suite_cases),
    )


def _legacy_plan(
    config: BenchmarkConfig, requested_workers: int | None
) -> ParallelPlan:
    """Resolve legacy rounds-per-rule configuration into worker offsets."""
    rules_path_value = config["rules"]["library_path"]
    if rules_path_value is None:
        raise ValueError("Legacy parallel runs require rules.library_path")
    rules_path = Path(rules_path_value)
    if not rules_path.is_absolute():
        rules_path = Path(__file__).parent.parent / rules_path
    with rules_path.open() as rules_file:
        rules_data = json.load(rules_file)
    if not isinstance(rules_data, dict) or not isinstance(
        rules_data.get("rules"), list
    ):
        raise TypeError(f"Rule library {rules_path} must contain a rules list")

    total_rules = len(rules_data["rules"])
    rounds_per_rule = config["game"].get("num_rounds_per_rule", 3)
    worker_count = requested_workers or 3
    if worker_count > rounds_per_rule:
        print(
            f"WARNING: {worker_count} workers requested but only "
            f"{rounds_per_rule} rounds per rule."
        )
        print(f"Capping workers to {rounds_per_rule}.")
        worker_count = rounds_per_rule
    return ParallelPlan(
        suite_name=None,
        worker_offsets=list(range(worker_count)),
        total_rules=total_rules,
        total_rounds=total_rules * rounds_per_rule,
    )


def _print_plan(plan: ParallelPlan, model: str, config_argument: str) -> None:
    """Print the resolved launch plan."""
    suite_suffix = f" (suite: {plan.suite_name})" if plan.suite_name else ""
    print("=" * 70)
    print(f"PARALLEL BENCHMARK - {model}{suite_suffix}")
    print("=" * 70)
    if plan.suite_name:
        print(f"Suite: {plan.suite_name}")
        print(f"Batch indices: {plan.worker_offsets}")
    print(f"Total rules: {plan.total_rules}")
    print(f"Total rounds: {plan.total_rounds}")
    print(f"Workers: {len(plan.worker_offsets)}")
    print(f"Rounds per worker: {plan.total_rules}")
    print(f"Config: {config_argument}\n")


def _worker_command(
    worker_id: int,
    offset: int,
    model: str,
    config_argument: str,
    suite_name: str | None,
) -> list[str]:
    """Build one evaluator worker command."""
    command = [
        "uv",
        "run",
        "python",
        "scripts/evaluate_single.py",
        "--config",
        config_argument,
        "--model",
        model,
        "--batch-round-offset",
        str(offset),
        "--tag",
        f"w{worker_id}_{model}",
    ]
    if suite_name:
        command.extend(["--suite", suite_name])
    return command


def _start_workers(
    plan: ParallelPlan,
    args: argparse.Namespace,
    timestamp: str,
) -> list[WorkerProcess]:
    """Print worker commands and start them unless this is a dry run."""
    workers: list[WorkerProcess] = []
    for worker_id, offset in enumerate(plan.worker_offsets):
        command = _worker_command(
            worker_id,
            offset,
            args.model,
            args.config,
            plan.suite_name,
        )
        print(f"Worker {worker_id}: batch_round_offset={offset}")
        print(f"  cmd: {' '.join(command)}")
        if not args.dry_run:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"worker_{worker_id}_{timestamp}_{args.model}.log"
            with log_file.open("w") as log_stream:
                process = subprocess.Popen(
                    command,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
            workers.append(WorkerProcess(worker_id, process, log_file))
            print(f"  pid: {process.pid}, log: {log_file}")
        print()
    return workers


def _wait_for_workers(workers: list[WorkerProcess]) -> None:
    """Wait for workers and report each exit status."""
    for completed, worker in enumerate(workers, start=1):
        worker.process.wait()
        return_code = worker.process.returncode
        status = "OK" if return_code == 0 else f"FAILED (exit {return_code})"
        print(
            f"  Worker {worker.worker_id}: {status} ({completed}/{len(workers)} done)"
        )

    failed = [
        (worker.worker_id, worker.process.returncode)
        for worker in workers
        if worker.process.returncode != 0
    ]
    print()
    if failed:
        print(f"WARNING: {len(failed)} worker(s) failed: {failed}")
    else:
        print("All workers completed successfully!")
    print("\nResults saved in: results/")


def main() -> None:
    """Launch and monitor parallel benchmark worker processes."""
    args = parse_args()
    config = _load_config(args.config)
    suite_name = args.suite or config.get("suite")
    plan = (
        _suite_plan(suite_name, args.workers)
        if suite_name
        else _legacy_plan(config, args.workers)
    )
    _print_plan(plan, args.model, args.config)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    workers = _start_workers(plan, args, timestamp)
    if args.dry_run:
        print("DRY RUN - no processes started")
        return

    print("=" * 70)
    print(f"All {len(workers)} workers started at {datetime.now(UTC):%H:%M:%S}\n")
    print(f"Monitor progress:\n  tail -f logs/worker_*_{timestamp}_{args.model}.log\n")
    print("Check results:\n  ls -la results/solo_evaluation_*\n")
    print("Waiting for all workers to complete...\n")
    _wait_for_workers(workers)


if __name__ == "__main__":
    main()
