#!/usr/bin/env python3
"""Run parallel benchmark evaluations.

Splits work across N workers by assigning each worker a different
batch_round_offset. Every worker plays ALL rules exactly once, but with
a different deck shuffle (seed). This keeps workers balanced: when a rule
is hard, all workers slow down together.

Usage:
    python scripts/run_parallel_benchmark.py --model rys-qwen3.5-27b-fp8-xl --workers 3
    python scripts/run_parallel_benchmark.py --model rys-qwen3.5-27b-fp8-xl --workers 3 --config config.yaml
    python scripts/run_parallel_benchmark.py --model rys-qwen3.5-27b-fp8-xl --workers 3 --dry-run
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel benchmark evaluations")
    parser.add_argument("--model", required=True, help="Model key from models.yaml")
    parser.add_argument("--config", default="config.yaml", help="Config file (default: config.yaml)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto from suite batch indices)")
    parser.add_argument("--suite", type=str, default=None,
                        help="Named benchmark suite from suites.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config to determine total rules and rounds_per_rule
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / args.config

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve suite: from --suite CLI flag, or suite: key in config
    suite_name = args.suite or config.get('suite')

    if suite_name:
        # Suite-aware mode
        from eleusis.suites import resolve_suite
        suite_cases = resolve_suite(suite_name)
        batch_indices = sorted(set(idx for _, idx in suite_cases))
        unique_rules = list(dict.fromkeys(name for name, _ in suite_cases))
        total_rules = len(unique_rules)
        total_rounds = len(suite_cases)
        rounds_per_batch = total_rules  # each batch index sees all rules in the suite

        if args.workers is not None and args.workers > len(batch_indices):
            print(f"WARNING: {args.workers} workers requested but suite '{suite_name}' "
                  f"has {len(batch_indices)} batch indices.")
            print(f"Capping workers to {len(batch_indices)}.")
            args.workers = len(batch_indices)

        num_workers = args.workers if args.workers is not None else len(batch_indices)
        worker_offsets = batch_indices[:num_workers]

        print("=" * 70)
        print(f"PARALLEL BENCHMARK - {args.model} (suite: {suite_name})")
        print("=" * 70)
        print(f"Suite: {suite_name}")
        print(f"Total rules: {total_rules}")
        print(f"Batch indices: {batch_indices}")
        print(f"Total rounds: {total_rounds}")
        print(f"Workers: {num_workers}")
        print(f"Rounds per worker: {rounds_per_batch}")
        print(f"Config: {args.config}")
        print()
    else:
        # Legacy mode: rounds_per_rule-based splitting
        suite_cases = None
        worker_offsets = None

        rules_path = config["rules"]["library_path"]
        if not Path(rules_path).is_absolute():
            rules_path = Path(__file__).parent.parent / rules_path

        with open(rules_path) as f:
            rules_data = json.load(f)

        total_rules = len(rules_data.get("rules", []))
        rounds_per_rule = config["game"].get("num_rounds_per_rule", 3)
        total_rounds = total_rules * rounds_per_rule
        num_workers = args.workers if args.workers is not None else 3

        if num_workers > rounds_per_rule:
            print(f"WARNING: {num_workers} workers requested but only "
                  f"{rounds_per_rule} rounds per rule.")
            print(f"Capping workers to {rounds_per_rule}.")
            num_workers = rounds_per_rule

        print("=" * 70)
        print(f"PARALLEL BENCHMARK - {args.model}")
        print("=" * 70)
        print(f"Total rules: {total_rules}")
        print(f"Rounds per rule: {rounds_per_rule}")
        print(f"Total rounds: {total_rounds}")
        print(f"Workers: {num_workers}")
        print(f"Rounds per worker: {total_rules} (all rules, 1 round each)")
        print(f"Config: {args.config}")
        print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []

    for worker_id in range(num_workers):
        if worker_offsets:
            offset = worker_offsets[worker_id]
        else:
            offset = worker_id
        tag = f"w{worker_id}_{args.model}"

        cmd = [
            "uv", "run", "python", "scripts/evaluate_single.py",
            "--config", str(args.config),
            "--model", args.model,
            "--batch-round-offset", str(offset),
            "--tag", tag,
        ]
        if suite_name:
            cmd.extend(["--suite", suite_name])

        print(f"Worker {worker_id}: batch_round_offset={offset}")
        print(f"  cmd: {' '.join(cmd)}")

        if not args.dry_run:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"worker_{worker_id}_{timestamp}_{args.model}.log"

            with open(log_file, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
            processes.append((worker_id, proc, log_file))
            print(f"  pid: {proc.pid}, log: {log_file}")

        print()

    if args.dry_run:
        print("DRY RUN - no processes started")
        return

    print("=" * 70)
    print(f"All {len(processes)} workers started at {datetime.now().strftime('%H:%M:%S')}")
    print()
    print("Monitor progress:")
    print(f"  tail -f logs/worker_*_{timestamp}_{args.model}.log")
    print()
    print("Check results:")
    print("  ls -la results/solo_evaluation_*")
    print()
    print("Waiting for all workers to complete...")
    print()

    # Wait for all processes and report status
    completed = 0
    for worker_id, proc, log_file in processes:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        completed += 1
        print(f"  Worker {worker_id}: {status} ({completed}/{len(processes)} done)")

    print()
    all_ok = all(p.returncode == 0 for _, p, _ in processes)
    if all_ok:
        print("All workers completed successfully!")
    else:
        failed = [(wid, p.returncode) for wid, p, _ in processes if p.returncode != 0]
        print(f"WARNING: {len(failed)} worker(s) failed: {failed}")

    print()
    print("Results saved in: results/")


if __name__ == "__main__":
    main()
