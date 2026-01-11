"""
Game state and simulation loop for Balatro.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Callable

from .deck import Deck, Hand, Card
from .hand_detector import HandDetector, HandDetectorConfig, HandType, DetectedHand
from .scoring import ScoringEngine, ScoreBreakdown
from .history import RunHistory


class BlindType(Enum):
    SMALL = auto()
    BIG = auto()
    BOSS = auto()


@dataclass
class BlindResult:
    """Result of playing a blind."""
    success: bool
    score_required: int
    score_achieved: int
    hands_used: int
    discards_used: int
    money_earned: int
    hands_played: list  # List of (cards_played, score) tuples


@dataclass
class RunResult:
    """Result of a complete run."""
    success: bool
    final_ante: int
    final_blind: BlindType
    total_money: int
    jokers_collected: list
    blinds_beaten: int


@dataclass
class GameConfig:
    """Configuration for a game run."""
    starting_money: int = 4
    starting_hands: int = 4
    starting_discards: int = 3
    hand_size: int = 8
    joker_slots: int = 5
    consumable_slots: int = 2
    ante_count: int = 8
    enable_shop: bool = True


class GameState:
    """
    Tracks the full state of a Balatro run.
    """

    def __init__(self, config: GameConfig = None, jokers_data: list = None,
                 preset_name: str = "standard", deck_type: str = "standard"):
        self.config = config or GameConfig()
        self.jokers_data = jokers_data or []

        # Run history for narrative tracking
        self.history = RunHistory(preset_name=preset_name, deck_type=deck_type)

        # Run state
        self.money = self.config.starting_money
        self.ante = 1
        self.current_blind = BlindType.SMALL

        # Round state
        self.hands_remaining = self.config.starting_hands
        self.discards_remaining = self.config.starting_discards

        # Deck and hand
        self.deck = Deck.standard_52()
        self.deck.shuffle()
        self.hand = Hand()

        # Jokers and consumables
        self.jokers: list[dict] = []
        self.consumables: list = []  # List of Consumable objects

        # Shop tracking
        self.jokers_purchased: list[str] = []
        self.planets_used: int = 0

        # Hand levels
        self.hand_levels: dict[HandType, int] = {ht: 1 for ht in HandType}

        # Stats
        self.hands_played_count: dict[HandType, int] = {ht: 0 for ht in HandType}

        # Scoring engine
        self.hand_detector = HandDetector(hand_levels=self.hand_levels)
        self.scoring_engine = ScoringEngine()

        # Load blind requirements
        self.blind_requirements = self._load_blind_requirements()

    def _load_blind_requirements(self) -> dict:
        """Load blind score requirements from JSON."""
        try:
            data_path = Path(__file__).parent.parent / "data" / "blind_requirements.json"
            with open(data_path) as f:
                return json.load(f)
        except FileNotFoundError:
            # Default fallback
            return {
                "ante_1": {"small": 300, "big": 450, "boss": 600},
                "ante_2": {"small": 800, "big": 1200, "boss": 1600},
                "ante_3": {"small": 2000, "big": 3000, "boss": 4000},
                "ante_4": {"small": 5000, "big": 7500, "boss": 10000},
                "ante_5": {"small": 11000, "big": 16500, "boss": 22000},
                "ante_6": {"small": 20000, "big": 30000, "boss": 40000},
                "ante_7": {"small": 35000, "big": 52500, "boss": 70000},
                "ante_8": {"small": 50000, "big": 75000, "boss": 100000},
            }

    def get_score_requirement(self) -> int:
        """Get the score needed to beat current blind."""
        ante_key = f"ante_{self.ante}"
        blind_key = self.current_blind.name.lower()

        if ante_key in self.blind_requirements:
            return self.blind_requirements[ante_key].get(blind_key, 999999)

        # Endless mode scaling (rough estimate)
        base = self.blind_requirements.get("ante_8", {}).get(blind_key, 100000)
        scale = 1.6 ** (self.ante - 8)
        return int(base * scale)

    def draw_hand(self) -> None:
        """Draw cards up to hand size."""
        cards_needed = self.config.hand_size - self.hand.size()
        if cards_needed > 0:
            drawn = self.deck.draw(cards_needed)
            self.hand.add(drawn)

    def play_cards(self, card_indices: list[int]) -> tuple[DetectedHand, ScoreBreakdown]:
        """
        Play selected cards from hand.

        Returns the detected hand and score breakdown.
        """
        if self.hands_remaining <= 0:
            raise ValueError("No hands remaining")

        # Get selected cards
        cards_to_play = self.hand.select(card_indices)
        if not cards_to_play:
            raise ValueError("No cards selected")

        # Remove from hand
        played = self.hand.remove(cards_to_play)

        # Detect hand type
        detected = self.hand_detector.detect(played)

        # Score the hand
        breakdown = self.scoring_engine.score_hand(detected, self.jokers)

        # Update state
        self.hands_remaining -= 1
        self.hands_played_count[detected.hand_type] += 1

        # Discard played cards
        self.deck.discard(played)

        return detected, breakdown

    def discard_cards(self, card_indices: list[int]) -> list[Card]:
        """Discard selected cards and draw new ones."""
        if self.discards_remaining <= 0:
            raise ValueError("No discards remaining")

        cards_to_discard = self.hand.select(card_indices)
        if not cards_to_discard:
            return []

        discarded = self.hand.remove(cards_to_discard)
        self.deck.discard(discarded)
        self.discards_remaining -= 1

        # Draw replacement cards
        self.draw_hand()

        return discarded

    def end_blind(self, success: bool) -> int:
        """End the current blind, return money earned."""
        money_earned = 0

        if success:
            # Blind reward
            if self.current_blind == BlindType.SMALL:
                money_earned = 3
            elif self.current_blind == BlindType.BIG:
                money_earned = 4
            else:
                money_earned = 5

            # Interest (20% of money, max $5)
            interest = min(5, int(self.money * 0.2))
            money_earned += interest

        self.money += money_earned
        return money_earned

    def advance_blind(self) -> bool:
        """
        Advance to next blind.
        Returns False if run is complete (beat ante 8 boss).
        """
        if self.current_blind == BlindType.SMALL:
            self.current_blind = BlindType.BIG
        elif self.current_blind == BlindType.BIG:
            self.current_blind = BlindType.BOSS
        else:
            # Beat boss, advance ante
            if self.ante >= self.config.ante_count:
                return False  # Run complete!
            self.ante += 1
            self.current_blind = BlindType.SMALL

        # Reset for new blind
        self.hands_remaining = self.config.starting_hands
        self.discards_remaining = self.config.starting_discards

        # Return hand cards to deck before reset
        remaining_cards = self.hand.clear()
        self.deck.discard(remaining_cards)
        self.deck.reset()

        return True

    def add_joker(self, joker: dict, source: str = "starting") -> bool:
        """Add a joker if there's room."""
        if len(self.jokers) >= self.config.joker_slots:
            return False
        self.jokers.append(joker)
        self.history.add_joker_acquired(self.ante, joker.get("name", "Unknown"), source)
        return True

    def level_up_hand(self, hand_type: HandType) -> None:
        """Increase the level of a hand type."""
        self.hand_levels[hand_type] = self.hand_levels.get(hand_type, 1) + 1
        self.hand_detector.hand_levels = self.hand_levels


class BasicStrategy:
    """
    Simple AI strategy for playing Balatro.
    Used for simulations.
    """

    def select_cards_to_play(self, hand: Hand, game: GameState) -> list[int]:
        """
        Select which cards to play.
        Returns indices of cards to play.
        """
        cards = hand.cards
        if not cards:
            return []

        # Try to find the best hand from available cards
        best_score = 0
        best_indices = []

        # Try different combinations (simplified - just try obvious ones)
        detector = game.hand_detector

        # Try playing all cards (up to 5)
        all_indices = list(range(min(5, len(cards))))
        if all_indices:
            test_cards = [cards[i] for i in all_indices]
            detected = detector.detect(test_cards)
            breakdown = game.scoring_engine.score_hand(detected, game.jokers)
            if breakdown.final_score > best_score:
                best_score = breakdown.final_score
                best_indices = all_indices

        # Try pairs, three of a kind, etc.
        from collections import Counter
        rank_counts = Counter(c.rank for c in cards)

        for rank, count in rank_counts.items():
            if count >= 2:
                # Get indices of cards with this rank
                indices = [i for i, c in enumerate(cards) if c.rank == rank][:min(5, count)]
                test_cards = [cards[i] for i in indices]
                detected = detector.detect(test_cards)
                breakdown = game.scoring_engine.score_hand(detected, game.jokers)
                if breakdown.final_score > best_score:
                    best_score = breakdown.final_score
                    best_indices = indices

        # If nothing good, play highest cards
        if not best_indices:
            sorted_indices = sorted(range(len(cards)),
                                  key=lambda i: cards[i].chip_value, reverse=True)
            best_indices = sorted_indices[:min(5, len(cards))]

        return best_indices

    def select_cards_to_discard(self, hand: Hand, game: GameState) -> list[int]:
        """
        Select which cards to discard.
        Returns indices of cards to discard (empty if should not discard).
        """
        if game.discards_remaining <= 0:
            return []

        cards = hand.cards
        if len(cards) <= 3:
            return []

        # Discard lowest value cards that aren't part of pairs
        from collections import Counter
        rank_counts = Counter(c.rank for c in cards)

        # Find cards that are "lonely" (not part of pairs)
        lonely_indices = [i for i, c in enumerate(cards) if rank_counts[c.rank] == 1]

        # Sort by chip value, discard lowest
        lonely_indices.sort(key=lambda i: cards[i].chip_value)

        # Discard up to 3 of the lowest lonely cards
        return lonely_indices[:min(3, len(lonely_indices))]


def simulate_blind(game: GameState, strategy = None) -> BlindResult:
    """Simulate playing a single blind."""
    if strategy is None:
        strategy = BasicStrategy()

    score_required = game.get_score_requirement()
    total_score = 0
    hands_played = []

    game.draw_hand()

    while game.hands_remaining > 0 and total_score < score_required:
        # Optionally discard first
        if game.discards_remaining > 0 and game.hands_remaining > 1:
            discard_indices = strategy.select_cards_to_discard(game.hand, game)
            if discard_indices:
                game.discard_cards(discard_indices)

        # Play a hand
        play_indices = strategy.select_cards_to_play(game.hand, game)
        if not play_indices:
            break

        detected, breakdown = game.play_cards(play_indices)
        total_score += breakdown.final_score
        hands_played.append((detected.hand_type.name, breakdown.final_score))

        # Draw new cards
        game.draw_hand()

    success = total_score >= score_required
    hands_used = game.config.starting_hands - game.hands_remaining
    discards_used = game.config.starting_discards - game.discards_remaining
    money_earned = game.end_blind(success)

    # Log blind result to history
    best_hand = max(hands_played, key=lambda x: x[1])[0] if hands_played else None
    game.history.add_blind_result(
        ante=game.ante,
        blind_type=game.current_blind.name,
        score=total_score,
        required=score_required,
        success=success,
        hands_used=hands_used,
        discards_used=discards_used,
        best_hand=best_hand
    )

    return BlindResult(
        success=success,
        score_required=score_required,
        score_achieved=total_score,
        hands_used=hands_used,
        discards_used=discards_used,
        money_earned=money_earned,
        hands_played=hands_played
    )


def simulate_shop(game: GameState, all_jokers: list, shop_ai=None) -> dict:
    """
    Simulate a shop visit.
    Returns dict with purchases made.
    """
    from .shop import Shop, ShopAI, apply_planet, ConsumableType

    if shop_ai is None:
        shop_ai = ShopAI()

    shop = Shop(all_jokers)
    shop.generate(owned_jokers=game.jokers)

    decisions = shop_ai.decide_purchases(shop, game)
    results = {
        "jokers_bought": [],
        "consumables_bought": [],
        "planets_used": [],
        "money_spent": 0
    }

    # Buy jokers
    for idx in decisions["jokers_to_buy"]:
        if idx < len(shop.jokers):
            joker = shop.jokers[idx]
            cost = shop.get_joker_cost(joker)
            if cost <= game.money and len(game.jokers) < game.config.joker_slots:
                game.money -= cost
                game.jokers.append(joker)
                game.jokers_purchased.append(joker.get("name", "Unknown"))
                results["jokers_bought"].append(joker.get("name"))
                results["money_spent"] += cost
                # Log joker acquisition
                game.history.add_joker_acquired(game.ante, joker.get("name", "Unknown"), "shop")

    # Buy and use consumables
    for idx in decisions["consumables_to_buy"]:
        if idx < len(shop.consumables):
            consumable = shop.consumables[idx]
            if consumable.cost <= game.money:
                game.money -= consumable.cost
                results["consumables_bought"].append(consumable.name)
                results["money_spent"] += consumable.cost

                # Use planets immediately
                if consumable.type == ConsumableType.PLANET:
                    # Get hand type before applying
                    hand_type = consumable.effect.get("levels_up")
                    old_level = game.hand_levels.get(hand_type, 1) if hand_type else 1
                    msg = apply_planet(consumable, game)
                    results["planets_used"].append(msg)
                    game.planets_used += 1
                    # Log planet usage
                    if hand_type:
                        game.history.add_planet_used(
                            game.ante,
                            consumable.name,
                            hand_type.name,
                            old_level + 1
                        )
                else:
                    # Store other consumables if room
                    if len(game.consumables) < game.config.consumable_slots:
                        game.consumables.append(consumable)

    # Log shop visit
    money_before = game.money + results["money_spent"]
    game.history.add_shop_visit(
        ante=game.ante,
        jokers_bought=results["jokers_bought"],
        planets_used=[p.split(":")[0].strip() for p in results["planets_used"]],  # Extract planet names
        money_spent=results["money_spent"],
        money_remaining=game.money
    )

    return results


def simulate_run(jokers: list = None, config: GameConfig = None, strategy=None,
                 all_jokers: list = None, verbose: bool = False) -> RunResult:
    """Simulate a complete Balatro run."""
    game = GameState(config=config)
    if strategy is None:
        strategy = BasicStrategy()

    # Add starting jokers
    if jokers:
        for joker in jokers[:game.config.joker_slots]:
            game.add_joker(joker)

    # For shop simulation, we need the full joker list
    enable_shop = game.config.enable_shop and all_jokers is not None

    blinds_beaten = 0

    while True:
        result = simulate_blind(game, strategy)

        if verbose:
            status = "WIN" if result.success else "LOSS"
            print(f"Ante {game.ante} {game.current_blind.name}: "
                  f"{result.score_achieved:,}/{result.score_required:,} - {status}")

        if not result.success:
            return RunResult(
                success=False,
                final_ante=game.ante,
                final_blind=game.current_blind,
                total_money=game.money,
                jokers_collected=[j.get('name', 'Unknown') for j in game.jokers],
                blinds_beaten=blinds_beaten
            )

        blinds_beaten += 1

        # Visit shop after beating a blind (before advancing)
        if enable_shop:
            shop_result = simulate_shop(game, all_jokers)
            if verbose and (shop_result["jokers_bought"] or shop_result["planets_used"]):
                if shop_result["jokers_bought"]:
                    print(f"  Shop: Bought {shop_result['jokers_bought']}")
                if shop_result["planets_used"]:
                    print(f"  Shop: {shop_result['planets_used']}")

        if not game.advance_blind():
            # Beat the game!
            return RunResult(
                success=True,
                final_ante=game.ante,
                final_blind=game.current_blind,
                total_money=game.money,
                jokers_collected=[j.get('name', 'Unknown') for j in game.jokers],
                blinds_beaten=blinds_beaten
            )
