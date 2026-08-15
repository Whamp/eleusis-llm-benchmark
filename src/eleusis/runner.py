"""Game runner setup and public round-execution entry point."""

import hashlib
import logging
import random
import time
from dataclasses import dataclass

from typing_extensions import TypedDict

from eleusis.benchmark_config import BenchmarkConfig
from eleusis.benchmark_run_store import BenchmarkRunStore
from eleusis.evaluation_results import TurnRecord
from eleusis.game import (
    GameEngine,
    GameState,
    PlayCardAction,
    Rule,
    RuleFactory,
    RuleValidator,
)
from eleusis.game.rule_library import RuleLibraryEntry, RuleMetadata
from eleusis.llm import LLMScientist, create_client, create_client_from_config
from eleusis.llm.base import BaseLLMClient
from eleusis.round_continuation import restore_round_continuation
from eleusis.round_execution import (
    RoundRuntime,
    build_round_result,
    execute_round_turns,
)
from eleusis.utils import model_spec_to_display_name

__all__ = ["RoundResult", "play_round"]

logger = logging.getLogger(__name__)


class RoundResult(TypedDict):
    """Serializable outcome and metrics for one benchmark round."""

    round_number: int
    turn_count: int
    rule_description: str
    rule_code: str
    rule_metadata: RuleMetadata | None
    success: bool
    score: int
    floored_score: int
    no_stakes_score: int
    first_correct_turn: int | None
    first_formal_correct_turn: int | None
    first_shadow_correct_turn: int | None
    failed_guesses: int
    game_over_reason: str
    schema_compliance_rate: float | None
    llm_usage: dict[str, dict[str, object]]
    turns: list[TurnRecord]
    wall_clock_seconds: float


@dataclass(frozen=True)
class RoundSetupRequest:
    """Arguments controlling setup of one benchmark round."""

    config: BenchmarkConfig
    round_number: int
    start_time: float
    rule: Rule | None
    max_turns: int | None
    start_rule_index: int | None
    rules_list: list[RuleLibraryEntry] | None
    batch_round_index: int
    results_folder: str | None


def _handle_action_error(
    error: Exception,
    scientist: LLMScientist,
    game_state: GameState,
) -> tuple[PlayCardAction, dict[str, str | None]]:
    """Produce a deterministic fallback card and serializable error metadata."""
    hand_cards = game_state.player.hand.get_all_cards()
    if not hand_cards:
        raise RuntimeError("Cannot recover from action error with an empty player hand")
    fallback_card = scientist.rng.choice(hand_cards)
    logger.error(
        f"Error getting action from {scientist.name}: {type(error).__name__}: {error}"
    )
    logger.warning(
        f"{scientist.name} using deterministic fallback after unhandled error: "
        f"{fallback_card}"
    )
    return PlayCardAction(fallback_card), {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "fallback_card": str(fallback_card),
    }


def _create_round_clients(
    config: BenchmarkConfig,
    player_display_name: str,
) -> tuple[BaseLLMClient, BaseLLMClient]:
    """Create compiler, fallback, and scientist clients through patchable seams."""
    llm_config = config["llm"]
    compiler_config = config["rule_compiler"]
    max_tokens = llm_config["max_tokens"]
    seed = llm_config["seed"]
    compiler = create_client_from_config(
        compiler_config,
        max_tokens=max_tokens,
        role="rule_compiler",
        seed=seed,
    )
    for fallback_config in compiler_config.get("backup_providers", []):
        try:
            fallback = create_client_from_config(
                fallback_config,
                max_tokens=max_tokens,
                role="rule_compiler_fallback",
                seed=seed,
            )
            compiler.fallback_clients.append(fallback)
            logger.info(
                "Registered fallback rule compiler: "
                f"{fallback.provider_name}/{fallback.model_name}"
            )
        # Optional fallback failures must not prevent the primary compiler from running.
        except Exception as error:  # ruff: ignore[blind-except]
            logger.warning(f"Failed to create fallback provider: {error}")
    scientist = create_client(
        config["model"],
        temperature=llm_config["temperature"],
        max_tokens=max_tokens,
        role=player_display_name,
        seed=seed,
    )
    return compiler, scientist


def _load_round_rule(
    request: RoundSetupRequest,
    compiler: BaseLLMClient,
    scientist: BaseLLMClient,
) -> tuple[Rule, RuleMetadata | None, RuleValidator]:
    """Load a new library rule or retain the caller-provided rule."""
    validator = RuleValidator()
    if request.rule is not None:
        logger.info(f"[Round {request.round_number}] Using provided rule")
        return request.rule, None, validator

    logger.info("=" * 80)
    logger.info(f"[Round {request.round_number}] PHASE 1: RULE LOADING")
    logger.info("=" * 80)
    compiler.reset_usage_stats()
    scientist.reset_usage_stats()
    rules_config = request.config["rules"]
    start_index = (
        request.start_rule_index
        if request.start_rule_index is not None
        else rules_config["index"]
    )
    logger.info(f"Rule factory starting at index: {start_index}")
    factory = RuleFactory(
        library_path=(
            rules_config["library_path"] if request.rules_list is None else None
        ),
        selection=rules_config["selection"],
        start_index=start_index,
        rules_list=request.rules_list,
    )
    rule, metadata = factory.create_rule_with_metadata()
    logger.info("\n" + "=" * 80)
    logger.info(f"SECRET RULE: {rule.description()}")
    logger.info("=" * 80 + "\n")
    return rule, metadata, validator


def _derive_round_seed(
    config: BenchmarkConfig,
    rule: Rule,
    batch_round_index: int,
) -> tuple[int | None, int]:
    """Derive the effective Round seed and low 32-bit secret-rule hash."""
    rule_hash = int(hashlib.md5(rule.get_code().encode()).hexdigest(), 16) & 0xFFFFFFFF
    base_seed = config["game"]["seed"]
    if base_seed is None:
        return None, rule_hash
    return (base_seed + rule_hash + batch_round_index) & 0xFFFFFFFF, rule_hash


def _calculate_round_seed(
    config: BenchmarkConfig,
    rule: Rule,
    batch_round_index: int,
) -> int | None:
    """Calculate and log the deterministic deck seed for one Round."""
    round_seed, rule_hash = _derive_round_seed(config, rule, batch_round_index)
    if round_seed is None:
        return None
    logger.info(
        f"Using round seed: {round_seed} (base={config['game']['seed']}, "
        f"rule_hash={rule_hash}, batch_idx={batch_round_index})"
    )
    return round_seed


def _create_round_engine(
    config: BenchmarkConfig,
    state: GameState,
    rule: Rule,
    compiler: BaseLLMClient,
    validator: RuleValidator,
    batch_round_index: int,
) -> tuple[GameEngine, int | None]:
    """Create and seed the game engine for one round."""
    game_config = config["game"]
    compiler_config = config["rule_compiler"]
    simulation_seed = compiler_config.get("simulation_seed")
    engine = GameEngine(
        state,
        rule,
        rule_compiler_client=compiler,
        rule_validator=validator,
        hand_size=game_config["hand_size"],
        wrong_guess_penalty=game_config["wrong_guess_penalty"],
        num_simulations=compiler_config.get("num_simulations", 100),
        turns_per_simulation=compiler_config.get("turns_per_simulation", 40),
        simulation_seed=42 if simulation_seed is None else simulation_seed,
        compiler_max_retries=compiler_config.get("max_retries", 10),
    )
    round_seed = _calculate_round_seed(config, rule, batch_round_index)
    engine.setup_game(round_seed=round_seed)
    return engine, round_seed


def _prepare_round_runtime(request: RoundSetupRequest) -> RoundRuntime:
    """Initialize all collaborators and immutable settings for one round."""
    player_name = model_spec_to_display_name(request.config["model"])
    compiler, scientist_client = _create_round_clients(request.config, player_name)
    rule, metadata, validator = _load_round_rule(
        request,
        compiler,
        scientist_client,
    )
    state = GameState(player_name)
    engine, round_seed = _create_round_engine(
        request.config,
        state,
        rule,
        compiler,
        validator,
        request.batch_round_index,
    )
    game_config = request.config["game"]
    max_turns = request.max_turns or game_config["max_turns"]
    scientist = LLMScientist(
        player_name,
        scientist_client,
        max_retries=request.config["llm"]["max_llm_retries"],
        engine=engine,
        max_turns=max_turns,
        rng=(random.Random(round_seed) if round_seed is not None else random.Random()),
    )
    logger.info("=" * 80)
    logger.info(f"[Round {request.round_number}] PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info(f"✓ Starter card placed: {state.mainline.get_last()}")
    logger.info(f"✓ Player has {game_config['hand_size']} cards (constant hand size)")
    logger.info(f"✓ Deck has {state.deck.remaining_count()} cards remaining\n")
    logger.info("=" * 80)
    logger.info(f"[Round {request.round_number}] PHASE 3: GAME PLAY")
    logger.info("=" * 80 + "\n")
    runtime = RoundRuntime(
        round_number=request.round_number,
        start_time=request.start_time,
        rule=rule,
        rule_metadata=metadata,
        engine=engine,
        game_state=state,
        scientist=scientist,
        scientist_client=scientist_client,
        rule_compiler_client=compiler,
        player_name=player_name,
        max_turns=max_turns,
        shadow_mode=game_config.get("shadow_mode", "offline"),
        pause_after_turn=game_config.get("pause_after_turn", False),
        results_folder=request.results_folder,
        handle_action_error=_handle_action_error,
    )
    return runtime


def play_round(
    config: BenchmarkConfig,
    round_number: int,
    rule: Rule | None = None,
    max_turns: int | None = None,
    start_rule_index: int | None = None,
    rules_list: list[RuleLibraryEntry] | None = None,
    batch_round_index: int = 0,
    results_folder: str | None = None,
    run_store: BenchmarkRunStore | None = None,
) -> RoundResult:
    """Set up and execute one reproducible pattern-discovery round."""
    request = RoundSetupRequest(
        config=config,
        round_number=round_number,
        start_time=time.time(),
        rule=rule,
        max_turns=max_turns,
        start_rule_index=start_rule_index,
        rules_list=rules_list,
        batch_round_index=batch_round_index,
        results_folder=results_folder,
    )
    active = (
        run_store.read_active_round(round_number) if run_store is not None else None
    )
    if active is not None:
        compiler, scientist_client = _create_round_clients(
            config,
            model_spec_to_display_name(config["model"]),
        )
        restored = restore_round_continuation(
            active.continuation,
            scientist_client=scientist_client,
            rule_compiler_client=compiler,
            handle_action_error=_handle_action_error,
            pause_after_turn=config["game"].get("pause_after_turn", False),
            results_folder=results_folder,
        )
        if restored.next_turn_index != 0 or restored.turn_records:
            raise RuntimeError(
                "Benchmark Run initial resume requires a zero-Turn checkpoint"
            )
        runtime = restored.runtime
    else:
        runtime = _prepare_round_runtime(request)
        if run_store is not None:
            round_seed, _rule_hash = _derive_round_seed(
                config,
                runtime.rule,
                batch_round_index,
            )
            run_store.start_round(
                runtime,
                effective_round_seed=round_seed,
                batch_round_index=batch_round_index,
            )
    if run_store is not None:
        runtime.completed_turn_committer = run_store.commit_completed_turn
    turn_count, game_over_reason, turns = execute_round_turns(runtime)
    result = build_round_result(runtime, turn_count, game_over_reason, turns)
    if run_store is not None:
        run_store.complete_round(
            runtime.round_number,
            game_over_reason=game_over_reason,
        )
    return result
