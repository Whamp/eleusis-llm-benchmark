"""Analyze tournament results and generate visualizations."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_tournament_data(json_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load tournament JSON and create dataframes.

    Returns:
        tournament_dict: Raw tournament data
        turns_df: DataFrame with one row per turn
        players_df: DataFrame with aggregated player statistics
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract all turns into a flat list
    turns_data = []
    for round_data in data['rounds']:
        round_num = round_data['round_number']
        for turn in round_data['turns']:
            turn_record = {
                'round_number': round_num,
                'turn_number': turn['turn_number'],
                'player': turn['player'],
                'mainline_state': turn['mainline_state'],
                'hand_size': len(turn['hand']),
                'reasoning_summary': turn['llm_response'].get('reasoning_summary', ''),
                'action': turn['llm_response'].get('action', ''),
                'tentative_rule': turn['llm_response'].get('tentative_rule', ''),
                'confidence_level': turn['llm_response'].get('confidence_level', None),
                'guess_if_accepted': turn['llm_response'].get('guess_rule_if_accepted', False),
                'action_type': turn['action_result'].get('action', ''),
                'card_played': turn['action_result'].get('card', None),
                'accepted': turn['action_result'].get('accepted', None),
                'correct': turn['action_result'].get('correct', None),
                'success': turn['action_result'].get('success', False),
                'guess_attempt': turn['guess_attempt'] is not None,
                'guess_correct': turn['guess_attempt'].get('correct', None) if turn['guess_attempt'] else None,
                'guess_text': turn['guess_attempt'].get('guess', None) if turn['guess_attempt'] else None,
            }
            turns_data.append(turn_record)

    turns_df = pd.DataFrame(turns_data)

    # Create aggregated player statistics
    players = data['config']['scientists']
    player_stats = []

    for player in players:
        # Collect scores across rounds
        scores = [r['scores'][player] for r in data['rounds'] if player in r['scores']]
        avg_score = np.mean(scores) if scores else 0

        # Collect LLM usage across rounds
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_duration = 0
        total_calls = 0

        for round_data in data['rounds']:
            if player in round_data['llm_usage']['scientists']:
                usage = round_data['llm_usage']['scientists'][player]
                total_prompt_tokens += usage['prompt_tokens']
                total_completion_tokens += usage['completion_tokens']
                total_tokens += usage['total_tokens']
                total_duration += usage['duration_seconds']
                total_calls += usage['call_count']

        avg_throughput = total_completion_tokens / total_duration if total_duration > 0 else 0

        # Get player turns
        player_turns = turns_df[turns_df['player'] == player]

        # Calculate confidence levels by category
        confidence_stats = {}

        # When not guessing at all
        not_guessing = player_turns[~player_turns['guess_if_accepted'] & ~player_turns['guess_attempt']]
        confidence_stats['not_guessing_count'] = len(not_guessing)
        confidence_stats['not_guessing_avg_confidence'] = not_guessing['confidence_level'].mean() if len(not_guessing) > 0 else None

        # When wanting to guess but not allowed (action failed)
        want_guess_not_allowed = player_turns[
            player_turns['guess_if_accepted'] &
            ~player_turns['guess_attempt'] &
            ((player_turns['accepted'] == False) | (player_turns['correct'] == False))
        ]
        confidence_stats['want_guess_not_allowed_count'] = len(want_guess_not_allowed)
        confidence_stats['want_guess_not_allowed_avg_confidence'] = want_guess_not_allowed['confidence_level'].mean() if len(want_guess_not_allowed) > 0 else None

        # When guessing and wrong
        guess_wrong = player_turns[player_turns['guess_attempt'] & (player_turns['guess_correct'] == False)]
        confidence_stats['guess_wrong_count'] = len(guess_wrong)
        confidence_stats['guess_wrong_avg_confidence'] = guess_wrong['confidence_level'].mean() if len(guess_wrong) > 0 else None

        # When guessing and correct
        guess_correct = player_turns[player_turns['guess_attempt'] & (player_turns['guess_correct'] == True)]
        confidence_stats['guess_correct_count'] = len(guess_correct)
        confidence_stats['guess_correct_avg_confidence'] = guess_correct['confidence_level'].mean() if len(guess_correct) > 0 else None

        # Overall confidence
        overall_confidence = player_turns['confidence_level'].mean()

        player_stat = {
            'player': player,
            'avg_score': avg_score,
            'total_rounds': len(scores),
            'wins': sum(1 for r in data['rounds'] if r['winning_player'] == player),
            'avg_confidence': overall_confidence,
            **confidence_stats,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens,
            'total_duration_seconds': total_duration,
            'avg_throughput_tokens_per_sec': avg_throughput,
            'total_llm_calls': total_calls,
            'avg_tokens_per_call': total_tokens / total_calls if total_calls > 0 else 0,
        }
        player_stats.append(player_stat)

    players_df = pd.DataFrame(player_stats)

    return data, turns_df, players_df


def plot_score_comparison(players_df: pd.DataFrame, output_dir: Path):
    """Plot average scores by player."""
    fig, ax = plt.subplots(figsize=(10, 6))

    players_df_sorted = players_df.sort_values('avg_score')

    ax.barh(players_df_sorted['player'], players_df_sorted['avg_score'], color='steelblue')
    ax.set_xlabel('Average Score (lower is better)', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    ax.set_title('Average Score by Player', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (player, score) in enumerate(zip(players_df_sorted['player'], players_df_sorted['avg_score'])):
        ax.text(score + 0.2, i, f'{score:.1f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'score_comparison.png', dpi=150)
    print(f"Saved: {output_dir / 'score_comparison.png'}")
    plt.close()


def plot_confidence_breakdown(players_df: pd.DataFrame, output_dir: Path):
    """Plot confidence levels by guess category."""
    fig, ax = plt.subplots(figsize=(12, 8))

    categories = [
        ('not_guessing_avg_confidence', 'Not Guessing'),
        ('want_guess_not_allowed_avg_confidence', 'Want Guess (Not Allowed)'),
        ('guess_wrong_avg_confidence', 'Guessed Wrong'),
        ('guess_correct_avg_confidence', 'Guessed Correct'),
    ]

    x = np.arange(len(players_df))
    width = 0.2

    for i, (col, label) in enumerate(categories):
        values = players_df[col].fillna(0).astype(float)
        offset = (i - len(categories) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.8)

    ax.set_xlabel('Player', fontsize=12)
    ax.set_ylabel('Average Confidence Level', fontsize=12)
    ax.set_title('Confidence Levels by Guess Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(players_df['player'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_breakdown.png', dpi=150)
    print(f"Saved: {output_dir / 'confidence_breakdown.png'}")
    plt.close()


def plot_llm_usage(players_df: pd.DataFrame, output_dir: Path):
    """Plot LLM usage metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total tokens
    ax = axes[0, 0]
    players_df_sorted = players_df.sort_values('total_tokens', ascending=False)
    ax.barh(players_df_sorted['player'], players_df_sorted['total_tokens'], color='coral')
    ax.set_xlabel('Total Tokens', fontsize=11)
    ax.set_ylabel('Player', fontsize=11)
    ax.set_title('Total Token Usage', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (player, tokens) in enumerate(zip(players_df_sorted['player'], players_df_sorted['total_tokens'])):
        ax.text(tokens + 500, i, f'{tokens:,}', va='center', fontsize=9)

    # Throughput
    ax = axes[0, 1]
    players_df_sorted = players_df.sort_values('avg_throughput_tokens_per_sec', ascending=False)
    ax.barh(players_df_sorted['player'], players_df_sorted['avg_throughput_tokens_per_sec'], color='lightgreen')
    ax.set_xlabel('Throughput (tokens/sec)', fontsize=11)
    ax.set_ylabel('Player', fontsize=11)
    ax.set_title('Average Throughput', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (player, throughput) in enumerate(zip(players_df_sorted['player'], players_df_sorted['avg_throughput_tokens_per_sec'])):
        ax.text(throughput + 10, i, f'{throughput:.1f}', va='center', fontsize=9)

    # Duration
    ax = axes[1, 0]
    players_df_sorted = players_df.sort_values('total_duration_seconds', ascending=False)
    ax.barh(players_df_sorted['player'], players_df_sorted['total_duration_seconds'], color='skyblue')
    ax.set_xlabel('Total Duration (seconds)', fontsize=11)
    ax.set_ylabel('Player', fontsize=11)
    ax.set_title('Total LLM Call Duration', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (player, duration) in enumerate(zip(players_df_sorted['player'], players_df_sorted['total_duration_seconds'])):
        ax.text(duration + 0.5, i, f'{duration:.1f}s', va='center', fontsize=9)

    # Tokens per call
    ax = axes[1, 1]
    players_df_sorted = players_df.sort_values('avg_tokens_per_call', ascending=False)
    ax.barh(players_df_sorted['player'], players_df_sorted['avg_tokens_per_call'], color='plum')
    ax.set_xlabel('Tokens per Call', fontsize=11)
    ax.set_ylabel('Player', fontsize=11)
    ax.set_title('Average Tokens per LLM Call', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (player, tpc) in enumerate(zip(players_df_sorted['player'], players_df_sorted['avg_tokens_per_call'])):
        ax.text(tpc + 50, i, f'{tpc:.0f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'llm_usage_metrics.png', dpi=150)
    print(f"Saved: {output_dir / 'llm_usage_metrics.png'}")
    plt.close()


def plot_confidence_vs_performance(turns_df: pd.DataFrame, output_dir: Path):
    """Plot confidence level vs actual performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to turns with guesses
    guess_turns = turns_df[turns_df['guess_attempt']].copy()

    if len(guess_turns) > 0:
        # Separate correct and incorrect guesses
        correct_guesses = guess_turns[guess_turns['guess_correct'] == True]
        incorrect_guesses = guess_turns[guess_turns['guess_correct'] == False]

        # Scatter plot
        ax.scatter(correct_guesses['confidence_level'],
                  correct_guesses['round_number'],
                  color='green', s=100, alpha=0.6, label='Correct Guess', marker='o')
        ax.scatter(incorrect_guesses['confidence_level'],
                  incorrect_guesses['round_number'],
                  color='red', s=100, alpha=0.6, label='Incorrect Guess', marker='x')

        ax.set_xlabel('Confidence Level', fontsize=12)
        ax.set_ylabel('Round Number', fontsize=12)
        ax.set_title('Confidence Level vs Guess Outcome', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 10)
    else:
        ax.text(0.5, 0.5, 'No guess attempts in this tournament',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_vs_performance.png', dpi=150)
    print(f"Saved: {output_dir / 'confidence_vs_performance.png'}")
    plt.close()


def print_summary_stats(data: dict, players_df: pd.DataFrame, turns_df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("TOURNAMENT SUMMARY")
    print("=" * 80)
    print(f"Timestamp: {data['timestamp']}")
    print(f"Total Rounds: {len(data['rounds'])}")
    print(f"Total Turns: {len(turns_df)}")
    print(f"Players: {len(players_df)}")
    print()

    print("=" * 80)
    print("PLAYER STATISTICS")
    print("=" * 80)

    for _, player in players_df.iterrows():
        print(f"\n{player['player']}")
        print("-" * 80)
        print(f"  Average Score: {player['avg_score']:.2f} (lower is better)")
        print(f"  Wins: {player['wins']}/{player['total_rounds']}")
        print(f"  Average Confidence: {player['avg_confidence']:.2f}/10")
        print()
        print("  Confidence by Category:")
        if pd.notna(player['not_guessing_avg_confidence']):
            print(f"    - Not Guessing: {player['not_guessing_avg_confidence']:.2f} ({player['not_guessing_count']} turns)")
        if pd.notna(player['want_guess_not_allowed_avg_confidence']):
            print(f"    - Wanted to Guess (not allowed): {player['want_guess_not_allowed_avg_confidence']:.2f} ({player['want_guess_not_allowed_count']} turns)")
        if pd.notna(player['guess_wrong_avg_confidence']):
            print(f"    - Guessed Wrong: {player['guess_wrong_avg_confidence']:.2f} ({player['guess_wrong_count']} guesses)")
        if pd.notna(player['guess_correct_avg_confidence']):
            print(f"    - Guessed Correct: {player['guess_correct_avg_confidence']:.2f} ({player['guess_correct_count']} guesses)")
        print()
        print("  LLM Usage:")
        print(f"    - Total Tokens: {player['total_tokens']:,}")
        print(f"    - Prompt Tokens: {player['total_prompt_tokens']:,}")
        print(f"    - Completion Tokens: {player['total_completion_tokens']:,}")
        print(f"    - Total Duration: {player['total_duration_seconds']:.2f}s")
        print(f"    - Average Throughput: {player['avg_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"    - Total Calls: {player['total_llm_calls']}")
        print(f"    - Avg Tokens/Call: {player['avg_tokens_per_call']:.0f}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze Eleusis tournament results')
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to tournament results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save analysis outputs (default: same directory as JSON file)'
    )

    args = parser.parse_args()

    # Load data
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    print(f"Loading tournament data from: {json_path}")
    data, turns_df, players_df = load_tournament_data(str(json_path))

    # Create output directory (default to same directory as JSON file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = json_path.parent
    output_dir.mkdir(exist_ok=True)

    # Save dataframes
    turns_csv = output_dir / f"turns_data_{data['timestamp']}.csv"
    players_csv = output_dir / f"players_data_{data['timestamp']}.csv"

    turns_df.to_csv(turns_csv, index=False)
    players_df.to_csv(players_csv, index=False)

    print(f"\nSaved dataframes:")
    print(f"  - {turns_csv}")
    print(f"  - {players_csv}")

    # Print summary statistics
    print_summary_stats(data, players_df, turns_df)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_score_comparison(players_df, output_dir)
    plot_confidence_breakdown(players_df, output_dir)
    plot_llm_usage(players_df, output_dir)
    plot_confidence_vs_performance(turns_df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
