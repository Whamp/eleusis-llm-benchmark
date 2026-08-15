"""Fresh-process restoration helper for Round continuation acceptance tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eleusis.game.engine import PlayCardAction
from eleusis.game.state import GameState
from eleusis.player import LLMScientist
from eleusis.round_continuation import (
    capture_round_continuation,
    restore_round_continuation,
)
from tests.conftest import FakeLLMClient


def _unexpected_action_error(
    error: Exception,
    scientist: LLMScientist,
    game_state: GameState,
) -> tuple[PlayCardAction, dict[str, str | None]]:
    """Fail if the restored scripted client unexpectedly needs fallback handling."""
    del error, scientist, game_state
    raise AssertionError("Round continuation subprocess used action fallback")


def main() -> None:
    """Restore one checkpoint, expose its state, and request the next action."""
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    request = json.loads(input_path.read_text())
    scientist_client = FakeLLMClient([request["next_response"]])
    compiler_client = FakeLLMClient()
    restored = restore_round_continuation(
        request["continuation"],
        scientist_client=scientist_client,
        rule_compiler_client=compiler_client,
        handle_action_error=_unexpected_action_error,
    )
    recaptured = capture_round_continuation(
        restored.runtime,
        restored.turn_records,
        next_turn_index=restored.next_turn_index,
    )
    restored.runtime.scientist.get_action(restored.runtime.game_state)
    output_path.write_text(
        json.dumps(
            {
                "next_prompt": scientist_client.prompts_seen[-1],
                "recaptured": recaptured,
            }
        )
    )


if __name__ == "__main__":
    main()
