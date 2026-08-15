"""Fresh-process driver for completed-Turn Round resume tests."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from eleusis import evaluation_orchestrator, evaluation_startup, runner
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.evaluation_startup import resolve_evaluation_startup
from eleusis.evaluation_state import initialize_evaluation_state
from eleusis.llm.base import BaseLLMClient
from tests.conftest import FakeLLMClient, make_action_response


def main() -> None:
    """Resume one active Round and report its remaining prompts and final facts."""
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text())
    os.chdir(payload["working_directory"])
    args = Namespace(
        config=payload["config_path"],
        resume=payload["run_folder"],
        model=None,
        num_rules=None,
        rule_index=None,
        max_turns=None,
        tag=None,
        batch_round_offset=None,
        suite=None,
    )

    patch.object(evaluation_startup, "preflight_check", lambda _model: None).start()
    scientist = FakeLLMClient(
        [make_action_response(card) for card in payload["selected_cards"]]
    )

    def create_test_clients(
        _config: BenchmarkConfig,
        _name: str,
    ) -> tuple[BaseLLMClient, BaseLLMClient]:
        return FakeLLMClient(), scientist

    patch.object(runner, "_create_round_clients", create_test_clients).start()
    startup = resolve_evaluation_startup(args)
    if startup is None:
        raise RuntimeError("Completed-Turn fresh-process resume startup failed")
    state = initialize_evaluation_state(startup)
    if state is None or state.run_store is None:
        raise RuntimeError("Completed-Turn fresh-process resume state failed")
    evaluation_orchestrator._run_evaluation_round(state, state.start_round)
    completed = state.run_store.read_completed_round(state.start_round)
    exported = json.loads(state.run_store.export_path.read_text())
    output_path.write_text(
        json.dumps(
            {
                "prompts": scientist.prompts_seen,
                "record": completed,
                "derived": exported["derived"],
            }
        )
    )


if __name__ == "__main__":
    main()
