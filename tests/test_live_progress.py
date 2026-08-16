"""Contracts for live progress collection and the progress dashboard server."""

import json
import threading
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import pytest
from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator, runner
from eleusis.evaluation_state import _initialize_fresh_state
from tests.conftest import FakeLLMClient
from tests.test_benchmark_run_schedule import (
    _configure_two_round_schedule,
    _first_dealt_card,
)


def _worker_with_completed_and_active_round(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> object:
    """Create one worker store with Round 1 completed and Round 2 active.

    Round 1 plays its single scripted Turn to the turn limit; Round 2 stops
    right after its initial checkpoint, leaving it active with zero committed
    Turns.
    """
    monkeypatch.chdir(tmp_path)
    startup, config, _config_path, rules = _configure_two_round_schedule(tmp_path)
    startup.output_tag = "w0_qwen"
    state = _initialize_fresh_state(startup)

    scientist = FakeLLMClient(
        [
            {
                "card": _first_dealt_card(config, rules[0], batch_round_index=1),
                "reasoning_summary": "play it",
                "tentative_rule": "",
                "confidence_level": 3,
                "guess_rule": False,
            }
        ]
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), scientist),
    )
    evaluation_orchestrator._run_evaluation_round(state, 1)

    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), FakeLLMClient()),
    )

    def stop_after_initial_checkpoint(_runtime: object) -> object:
        raise RuntimeError("stop after initial checkpoint")

    monkeypatch.setattr(runner, "execute_round_turns", stop_after_initial_checkpoint)
    with pytest.raises(RuntimeError, match="stop after initial checkpoint"):
        evaluation_orchestrator._run_evaluation_round(state, 2)
    return state


def test_collect_live_progress_reports_rounds_and_usage(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """C1: SQLite workers expose per-Round rows and derived usage."""
    from eleusis.analysis.live_progress import collect_live_progress

    _worker_with_completed_and_active_round(monkeypatch, tmp_path)

    workers = collect_live_progress("solo_evaluation_*")

    assert len(workers) == 1
    worker = workers[0]
    assert worker.source == "sqlite"
    assert worker.error is None
    assert worker.completed == 1
    assert worker.total == 2
    assert worker.successful == 0
    assert worker.active_round_number == 2
    assert worker.committed_turns == 0

    assert [round_.status for round_ in worker.rounds] == [
        "completed",
        "active",
    ]
    completed = worker.rounds[0]
    assert completed.rule_name == "only_red"
    assert completed.turn_count == 1
    assert completed.terminal_kind == "turn_limit"
    assert completed.score == 0
    active = worker.rounds[1]
    assert active.rule_name == "only_even"
    assert active.score is None

    assert worker.usage is not None
    assert isinstance(worker.usage["reasoning_tokens"], int)


def test_collect_live_progress_isolates_defective_workers(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """C2: one defective SQLite folder becomes an error row, not an abort."""
    from eleusis.analysis.live_progress import collect_live_progress

    _worker_with_completed_and_active_round(monkeypatch, tmp_path)
    bad = Path("results") / "solo_evaluation_990000_zz_badworker"
    bad.mkdir(parents=True)
    (bad / "benchmark_run.sqlite3").write_bytes(b"not a database")

    workers = collect_live_progress("solo_evaluation_*")

    by_name = {worker.name: worker for worker in workers}
    assert by_name["solo_evaluation_990000_zz_badworker"].error is not None
    good = next(worker for worker in workers if worker.error is None)
    assert good.completed == 1


def test_dashboard_serves_html_and_progress_json(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """C3: the dashboard serves one page and one live JSON endpoint."""
    from scripts.dashboard import create_server

    _worker_with_completed_and_active_round(monkeypatch, tmp_path)
    server = create_server(pattern="solo_evaluation_*", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"

        with urllib.request.urlopen(f"{base}/api/progress", timeout=5) as response:
            payload = json.load(response)
        assert response.status == 200
        assert payload["workers"], "JSON payload exposes worker rows"
        worker = payload["workers"][0]
        assert worker["completed"] == 1
        assert worker["rounds"][0]["rule_name"] == "only_red"
        assert payload["overall"]["total"] == 2

        with urllib.request.urlopen(f"{base}/", timeout=5) as response:
            html = response.read().decode()
        assert response.status == 200
        assert "Eleusis Benchmark Dashboard" in html

        with pytest.raises(HTTPError):
            urllib.request.urlopen(f"{base}/nope", timeout=5)
    finally:
        server.shutdown()
        server.server_close()
