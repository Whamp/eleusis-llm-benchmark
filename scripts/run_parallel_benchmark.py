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
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
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

    rules_path = config["rules"]["library_path"]
    if not Path(rules_path).is_absolute():
        rules_path = Path(__file__).parent.parent / rules_path

    with open(rules_path) as f:
        rules_data = json.load(f)

    total_rules = len(rules_data.get("rules", []))
    rounds_per_rule = config["game"].get("num_rounds_per_rule", 3)

    if args.workers > rounds_per_rule:
        print(f"WARNING: {args.workers} workers requested but only {rounds_per_rule} rounds per rule.")
        print(f"Capping workers to {rounds_per_rule}.")
        args.workers = rounds_per_rule

    print("=" * 70)
    print(f"PARALLEL BENCHMARK - {args.model}")
    print("=" * 70)
    print(f"Total rules: {total_rules}")
    print(f"Rounds per rule: {rounds_per_rule}")
    print(f"Total rounds: {total_rules * rounds_per_rule}")
    print(f"Workers: {args.workers}")
    print(f"Rounds per worker: {total_rules} (all rules, 1 round each)")
    print(f"Config: {args.config}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processes = []

    for worker_id in range(args.workers):
        tag = f"w{worker_id}_{args.model}"

        cmd = [
            "uv", "run", "python", "scripts/evaluate_single.py",
            "--config", str(args.config),
            "--model", args.model,
            "--batch-round-offset", str(worker_id),
            "--tag", tag,
        ]

        print(f"Worker {worker_id}: all {total_rules} rules, batch_round_offset={worker_id}")
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
