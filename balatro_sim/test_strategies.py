#!/usr/bin/env python3
"""
Compare different AI strategies for Balatro simulation.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from balatro_sim.engine.game import GameState, simulate_run, simulate_blind, BasicStrategy
from balatro_sim.engine.strategy import SmartStrategy, OptimizedStrategy, AggressiveStrategy


def compare_strategies(num_runs: int = 100, with_jokers: bool = False):
    """Run simulations with different strategies and compare results."""

    strategies = {
        "Basic (naive)": BasicStrategy(),
        "Smart (eval all)": SmartStrategy(),
        "Optimized (flush/straight chase)": OptimizedStrategy(),
        "Aggressive (max score)": AggressiveStrategy(),
    }

    # Load jokers if requested
    jokers = []
    if with_jokers:
        jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"
        try:
            with open(jokers_path) as f:
                all_jokers = json.load(f)
            # Pick some good starting jokers
            for name in ['Joker', 'Jolly Joker', 'Zany Joker', 'Sly Joker', 'Banner']:
                j = next((x for x in all_jokers if x['name'] == name), None)
                if j:
                    jokers.append(j)
        except FileNotFoundError:
            print("Jokers file not found, running without jokers")

    print("=" * 70)
    print(f"STRATEGY COMPARISON ({num_runs} runs each)")
    if jokers:
        print(f"Starting jokers: {[j['name'] for j in jokers]}")
    print("=" * 70)

    results = {}

    for name, strategy in strategies.items():
        print(f"\nTesting: {name}...", end=" ", flush=True)

        start_time = time.time()
        wins = 0
        total_blinds = 0
        total_money = 0
        max_ante = 0
        ante_distribution = {}

        for _ in range(num_runs):
            result = simulate_run(jokers=jokers, strategy=strategy)
            if result.success:
                wins += 1
            total_blinds += result.blinds_beaten
            total_money += result.total_money
            max_ante = max(max_ante, result.final_ante)

            # Track ante distribution
            key = f"Ante {result.final_ante}"
            ante_distribution[key] = ante_distribution.get(key, 0) + 1

        elapsed = time.time() - start_time

        results[name] = {
            "wins": wins,
            "win_rate": wins / num_runs * 100,
            "avg_blinds": total_blinds / num_runs,
            "avg_money": total_money / num_runs,
            "max_ante": max_ante,
            "ante_dist": ante_distribution,
            "time": elapsed
        }

        print(f"Done ({elapsed:.1f}s)")

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<35} {'Win %':>8} {'Avg Blinds':>12} {'Avg $':>8} {'Max Ante':>10}")
    print("-" * 70)

    for name, stats in results.items():
        print(f"{name:<35} {stats['win_rate']:>7.1f}% {stats['avg_blinds']:>12.1f} "
              f"${stats['avg_money']:>7.0f} {stats['max_ante']:>10}")

    # Print ante distribution for best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]['avg_blinds'])
    print(f"\n{best_strategy} - Ante Distribution:")
    dist = results[best_strategy]['ante_dist']
    for ante in sorted(dist.keys(), key=lambda x: int(x.split()[1])):
        pct = dist[ante] / num_runs * 100
        bar = "█" * int(pct / 2)
        print(f"  {ante}: {dist[ante]:>3} ({pct:>5.1f}%) {bar}")

    return results


def detailed_single_run(strategy_name: str = "Smart"):
    """Run a single game with detailed output."""

    strategies = {
        "Basic": BasicStrategy(),
        "Smart": SmartStrategy(),
        "Optimized": OptimizedStrategy(),
        "Aggressive": AggressiveStrategy(),
    }

    strategy = strategies.get(strategy_name, SmartStrategy())

    print("=" * 70)
    print(f"DETAILED RUN - {strategy_name} Strategy")
    print("=" * 70)

    # Load jokers
    jokers = []
    jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"
    try:
        with open(jokers_path) as f:
            all_jokers = json.load(f)
        for name in ['Joker', 'Jolly Joker', 'Zany Joker']:
            j = next((x for x in all_jokers if x['name'] == name), None)
            if j:
                jokers.append(j)
    except FileNotFoundError:
        pass

    from balatro_sim.engine.game import GameState, BlindType

    game = GameState()
    for j in jokers:
        game.add_joker(j)

    print(f"Starting jokers: {[j['name'] for j in game.jokers]}")
    print()

    while True:
        result = simulate_blind(game, strategy)

        print(f"Ante {game.ante} - {game.current_blind.name}")
        print(f"  Required: {result.score_required:,}")
        print(f"  Achieved: {result.score_achieved:,}")
        print(f"  {'✓ SUCCESS' if result.success else '✗ FAILED'}")
        print(f"  Hands: {result.hands_used}, Discards: {result.discards_used}")
        print(f"  Plays: {', '.join(f'{h}:{s:,}' for h, s in result.hands_played)}")
        print(f"  Money: ${game.money} (+${result.money_earned})")
        print()

        if not result.success:
            print("=" * 70)
            print(f"GAME OVER at Ante {game.ante} - {game.current_blind.name}")
            print(f"Total blinds beaten: {game.ante * 3 - 3 + ['SMALL', 'BIG', 'BOSS'].index(game.current_blind.name)}")
            break

        if not game.advance_blind():
            print("=" * 70)
            print("VICTORY! Beat Ante 8!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Balatro AI strategies")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs per strategy")
    parser.add_argument("--jokers", action="store_true", help="Start with jokers")
    parser.add_argument("--detailed", type=str, help="Run detailed single game with strategy")

    args = parser.parse_args()

    if args.detailed:
        detailed_single_run(args.detailed)
    else:
        compare_strategies(num_runs=args.runs, with_jokers=args.jokers)
