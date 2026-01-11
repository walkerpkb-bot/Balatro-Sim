#!/usr/bin/env python3
"""
Demo script for Balatro simulation.
Shows basic hand detection, scoring, and game simulation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from balatro_sim.engine.deck import Deck, Card, Suit, Hand
from balatro_sim.engine.hand_detector import detect_hand, HandType
from balatro_sim.engine.scoring import score_breakdown
from balatro_sim.engine.game import GameState, simulate_blind, simulate_run, GameConfig


def demo_hand_detection():
    """Demonstrate hand detection."""
    print("=" * 60)
    print("HAND DETECTION DEMO")
    print("=" * 60)

    # Create some test hands
    test_hands = [
        # Pair
        [Card("K", Suit.HEARTS), Card("K", Suit.DIAMONDS), Card("5", Suit.CLUBS)],
        # Three of a kind
        [Card("7", Suit.HEARTS), Card("7", Suit.DIAMONDS), Card("7", Suit.CLUBS)],
        # Flush
        [Card("A", Suit.HEARTS), Card("K", Suit.HEARTS), Card("10", Suit.HEARTS),
         Card("7", Suit.HEARTS), Card("2", Suit.HEARTS)],
        # Straight
        [Card("5", Suit.HEARTS), Card("6", Suit.DIAMONDS), Card("7", Suit.CLUBS),
         Card("8", Suit.SPADES), Card("9", Suit.HEARTS)],
        # Full House
        [Card("Q", Suit.HEARTS), Card("Q", Suit.DIAMONDS), Card("Q", Suit.CLUBS),
         Card("9", Suit.SPADES), Card("9", Suit.HEARTS)],
    ]

    for cards in test_hands:
        detected = detect_hand(cards)
        print(f"\nCards: {', '.join(str(c) for c in cards)}")
        print(f"  Hand: {detected.hand_type.name}")
        print(f"  Scoring cards: {', '.join(str(c) for c in detected.scoring_cards)}")
        print(f"  Base: {detected.base_chips} chips × {detected.base_mult} mult")


def demo_scoring():
    """Demonstrate scoring calculation."""
    print("\n" + "=" * 60)
    print("SCORING DEMO")
    print("=" * 60)

    # Create a hand
    cards = [Card("A", Suit.SPADES), Card("A", Suit.HEARTS), Card("A", Suit.DIAMONDS)]
    detected = detect_hand(cards)
    breakdown = score_breakdown(detected)

    print(f"\nPlayed: {', '.join(str(c) for c in cards)}")
    print(f"Detected: {breakdown.hand_type.name}")
    print(f"\nScoring breakdown:")
    print(f"  Base chips: {breakdown.base_chips}")
    print(f"  Card chips: {breakdown.card_chips}")
    print(f"  Total chips: {breakdown.final_chips}")
    print(f"  Base mult: {breakdown.base_mult}")
    print(f"  X-mult: {breakdown.x_mult}")
    print(f"  Final mult: {breakdown.final_mult}")
    print(f"\n  FINAL SCORE: {breakdown.final_score}")


def demo_scoring_with_jokers():
    """Demonstrate scoring with jokers."""
    print("\n" + "=" * 60)
    print("SCORING WITH JOKERS DEMO")
    print("=" * 60)

    # Load parsed jokers
    jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"
    try:
        with open(jokers_path) as f:
            all_jokers = json.load(f)
    except FileNotFoundError:
        print("Jokers file not found. Run joker_parser.py first.")
        return

    # Find some useful jokers
    joker = next((j for j in all_jokers if j['name'] == 'Joker'), None)  # +4 Mult
    jolly = next((j for j in all_jokers if j['name'] == 'Jolly Joker'), None)  # +8 Mult with Pair

    if not joker or not jolly:
        print("Could not find test jokers")
        return

    # Create a pair hand
    cards = [Card("K", Suit.SPADES), Card("K", Suit.HEARTS), Card("7", Suit.DIAMONDS)]
    detected = detect_hand(cards)

    # Score without jokers
    breakdown_no_jokers = score_breakdown(detected, [])
    print(f"\nPair of Kings - NO JOKERS:")
    print(f"  Score: {breakdown_no_jokers.final_score}")

    # Score with basic Joker
    breakdown_with_joker = score_breakdown(detected, [joker])
    print(f"\nPair of Kings - WITH Joker (+4 Mult):")
    print(f"  Score: {breakdown_with_joker.final_score}")

    # Score with both jokers
    breakdown_with_both = score_breakdown(detected, [joker, jolly])
    print(f"\nPair of Kings - WITH Joker + Jolly Joker (+4, +8 Mult):")
    print(f"  Score: {breakdown_with_both.final_score}")
    print(f"  Details: {breakdown_with_both.details}")


def demo_single_blind():
    """Demonstrate simulating a single blind."""
    print("\n" + "=" * 60)
    print("SINGLE BLIND SIMULATION")
    print("=" * 60)

    game = GameState()
    result = simulate_blind(game)

    print(f"\nAnte {game.ante} - {game.current_blind.name} Blind")
    print(f"  Required: {result.score_required}")
    print(f"  Achieved: {result.score_achieved}")
    print(f"  Result: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Hands used: {result.hands_used}")
    print(f"  Discards used: {result.discards_used}")
    print(f"  Money earned: ${result.money_earned}")
    print(f"\nHands played:")
    for hand_type, score in result.hands_played:
        print(f"    {hand_type}: {score}")


def demo_full_run():
    """Demonstrate simulating a full run."""
    print("\n" + "=" * 60)
    print("FULL RUN SIMULATION (no jokers)")
    print("=" * 60)

    result = simulate_run()

    print(f"\nRun Result: {'VICTORY!' if result.success else 'DEFEAT'}")
    print(f"  Made it to: Ante {result.final_ante}, {result.final_blind.name} Blind")
    print(f"  Blinds beaten: {result.blinds_beaten}")
    print(f"  Final money: ${result.total_money}")


def demo_run_with_jokers():
    """Demonstrate a run with jokers."""
    print("\n" + "=" * 60)
    print("FULL RUN SIMULATION (with jokers)")
    print("=" * 60)

    # Load parsed jokers
    jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"
    try:
        with open(jokers_path) as f:
            all_jokers = json.load(f)
    except FileNotFoundError:
        print("Jokers file not found.")
        return

    # Select some good starting jokers
    starter_jokers = []
    for name in ['Joker', 'Jolly Joker', 'Greedy Joker', 'Banner', 'Supernova']:
        joker = next((j for j in all_jokers if j['name'] == name), None)
        if joker:
            starter_jokers.append(joker)

    print(f"Starting jokers: {[j['name'] for j in starter_jokers]}")

    result = simulate_run(jokers=starter_jokers)

    print(f"\nRun Result: {'VICTORY!' if result.success else 'DEFEAT'}")
    print(f"  Made it to: Ante {result.final_ante}, {result.final_blind.name} Blind")
    print(f"  Blinds beaten: {result.blinds_beaten}")
    print(f"  Final money: ${result.total_money}")


def demo_monte_carlo():
    """Run multiple simulations to get win rate."""
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION (100 runs)")
    print("=" * 60)

    wins = 0
    total_blinds = 0
    runs = 100

    for i in range(runs):
        result = simulate_run()
        if result.success:
            wins += 1
        total_blinds += result.blinds_beaten

    print(f"\nResults over {runs} runs:")
    print(f"  Win rate: {wins}/{runs} ({wins/runs*100:.1f}%)")
    print(f"  Avg blinds beaten: {total_blinds/runs:.1f}")


if __name__ == "__main__":
    demo_hand_detection()
    demo_scoring()
    demo_scoring_with_jokers()
    demo_single_blind()
    demo_full_run()
    demo_run_with_jokers()
    demo_monte_carlo()

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
