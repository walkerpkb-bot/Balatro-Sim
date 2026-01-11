"""
Hand detection for Balatro simulation.
Identifies poker hands from played cards.
"""

from dataclasses import dataclass
from enum import Enum, auto
from collections import Counter
from typing import Optional

from .deck import Card, Suit, Enhancement, RANK_ORDER, RANKS


class HandType(Enum):
    """Poker hand types, ordered by base strength."""
    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()
    ROYAL_FLUSH = auto()
    FIVE_OF_A_KIND = auto()
    FLUSH_HOUSE = auto()
    FLUSH_FIVE = auto()


# Base chips and mult for each hand type (level 1)
HAND_BASE_VALUES = {
    HandType.HIGH_CARD: (5, 1),
    HandType.PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_OF_A_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_OF_A_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.ROYAL_FLUSH: (100, 8),
    HandType.FIVE_OF_A_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

# Chips and mult added per level up
LEVEL_UP_BONUS = {
    HandType.HIGH_CARD: (10, 1),
    HandType.PAIR: (15, 1),
    HandType.TWO_PAIR: (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT: (30, 3),
    HandType.FLUSH: (15, 2),
    HandType.FULL_HOUSE: (25, 2),
    HandType.FOUR_OF_A_KIND: (30, 3),
    HandType.STRAIGHT_FLUSH: (40, 4),
    HandType.ROYAL_FLUSH: (40, 4),
    HandType.FIVE_OF_A_KIND: (35, 3),
    HandType.FLUSH_HOUSE: (40, 4),
    HandType.FLUSH_FIVE: (50, 3),
}


@dataclass
class DetectedHand:
    """Result of hand detection."""
    hand_type: HandType
    scoring_cards: list[Card]  # Cards that contribute to the hand
    all_cards: list[Card]      # All played cards
    level: int = 1

    @property
    def base_chips(self) -> int:
        base, _ = HAND_BASE_VALUES[self.hand_type]
        bonus, _ = LEVEL_UP_BONUS[self.hand_type]
        return base + bonus * (self.level - 1)

    @property
    def base_mult(self) -> int:
        _, base = HAND_BASE_VALUES[self.hand_type]
        _, bonus = LEVEL_UP_BONUS[self.hand_type]
        return base + bonus * (self.level - 1)


@dataclass
class HandDetectorConfig:
    """Configuration for hand detection rules."""
    four_fingers: bool = False      # Flushes/Straights need 4 cards
    shortcut: bool = False          # Straights can have gaps of 1
    smeared_joker: bool = False     # Hearts=Diamonds, Spades=Clubs


class HandDetector:
    """Detects the best poker hand from played cards."""

    def __init__(self, config: HandDetectorConfig = None, hand_levels: dict = None):
        self.config = config or HandDetectorConfig()
        self.hand_levels = hand_levels or {}  # {HandType: level}

    def get_level(self, hand_type: HandType) -> int:
        return self.hand_levels.get(hand_type, 1)

    def detect(self, cards: list[Card]) -> DetectedHand:
        """Detect the best hand from the played cards."""
        if not cards:
            return DetectedHand(HandType.HIGH_CARD, [], [], self.get_level(HandType.HIGH_CARD))

        # Get suits (accounting for wild cards and smeared joker)
        def get_effective_suits(card: Card) -> set[Suit]:
            if card.enhancement == Enhancement.WILD:
                return set(Suit)
            if card.enhancement == Enhancement.STONE:
                return set()  # Stone cards have no suit
            if self.config.smeared_joker:
                if card.suit in [Suit.HEARTS, Suit.DIAMONDS]:
                    return {Suit.HEARTS, Suit.DIAMONDS}
                else:
                    return {Suit.CLUBS, Suit.SPADES}
            return {card.suit}

        # Count ranks
        rank_counts = Counter(c.rank for c in cards if c.enhancement != Enhancement.STONE)

        # Check for flush (all same suit)
        flush_requirement = 4 if self.config.four_fingers else 5
        is_flush = False
        flush_suit = None

        if len(cards) >= flush_requirement:
            # Find if there's a common suit among all cards
            common_suits = None
            for card in cards:
                if card.enhancement == Enhancement.STONE:
                    continue
                suits = get_effective_suits(card)
                if common_suits is None:
                    common_suits = suits
                else:
                    common_suits = common_suits & suits
            if common_suits:
                is_flush = True
                flush_suit = next(iter(common_suits))

        # Check for straight
        straight_requirement = 4 if self.config.four_fingers else 5
        is_straight = False
        straight_cards = []

        if len(cards) >= straight_requirement:
            is_straight, straight_cards = self._check_straight(cards)

        # Check for flush + straight combos
        is_straight_flush = is_flush and is_straight
        is_royal_flush = (is_straight_flush and
                         set(c.rank for c in straight_cards) >= {"10", "J", "Q", "K", "A"})

        # Get the count patterns
        counts = sorted(rank_counts.values(), reverse=True)

        # Determine hand type (from best to worst)
        if is_flush and counts and counts[0] >= 5:
            return self._make_result(HandType.FLUSH_FIVE, cards, rank_counts, 5)

        if is_flush and len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2:
            return self._make_result(HandType.FLUSH_HOUSE, cards, rank_counts, 5)

        if counts and counts[0] >= 5:
            return self._make_result(HandType.FIVE_OF_A_KIND, cards, rank_counts, 5)

        if is_royal_flush:
            return DetectedHand(HandType.ROYAL_FLUSH, straight_cards, cards,
                              self.get_level(HandType.ROYAL_FLUSH))

        if is_straight_flush:
            return DetectedHand(HandType.STRAIGHT_FLUSH, straight_cards, cards,
                              self.get_level(HandType.STRAIGHT_FLUSH))

        if counts and counts[0] >= 4:
            return self._make_result(HandType.FOUR_OF_A_KIND, cards, rank_counts, 4)

        if len(counts) >= 2 and counts[0] >= 3 and counts[1] >= 2:
            return self._make_result(HandType.FULL_HOUSE, cards, rank_counts, 5)

        if is_flush:
            flush_cards = cards[:flush_requirement]
            return DetectedHand(HandType.FLUSH, flush_cards, cards,
                              self.get_level(HandType.FLUSH))

        if is_straight:
            return DetectedHand(HandType.STRAIGHT, straight_cards, cards,
                              self.get_level(HandType.STRAIGHT))

        if counts and counts[0] >= 3:
            return self._make_result(HandType.THREE_OF_A_KIND, cards, rank_counts, 3)

        if len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
            return self._make_result(HandType.TWO_PAIR, cards, rank_counts, 4)

        if counts and counts[0] >= 2:
            return self._make_result(HandType.PAIR, cards, rank_counts, 2)

        # High card - just use the highest card
        highest = max(cards, key=lambda c: RANK_ORDER.get(c.rank, 0))
        return DetectedHand(HandType.HIGH_CARD, [highest], cards,
                          self.get_level(HandType.HIGH_CARD))

    def _check_straight(self, cards: list[Card]) -> tuple[bool, list[Card]]:
        """Check if cards form a straight."""
        # Get unique ranks and their orders
        ranks = set()
        card_by_rank = {}
        for card in cards:
            if card.enhancement != Enhancement.STONE:
                ranks.add(card.rank)
                card_by_rank[card.rank] = card

        if not ranks:
            return False, []

        # Convert to numeric order
        orders = sorted([RANK_ORDER[r] for r in ranks])

        requirement = 4 if self.config.four_fingers else 5
        gap_allowed = 1 if self.config.shortcut else 0

        # Try to find a sequence
        for start_idx in range(len(orders)):
            sequence = [orders[start_idx]]
            for i in range(start_idx + 1, len(orders)):
                gap = orders[i] - sequence[-1]
                if gap == 1 or (gap == 2 and gap_allowed >= 1):
                    sequence.append(orders[i])
                    if len(sequence) >= requirement:
                        # Found a straight, get the cards
                        straight_ranks = [RANKS[o] for o in sequence[:requirement]]
                        straight_cards = [card_by_rank[r] for r in straight_ranks if r in card_by_rank]
                        return True, straight_cards
                elif gap > 2:
                    break

        # Check for wheel (A-2-3-4-5)
        wheel_ranks = ["A", "2", "3", "4", "5"]
        if all(r in ranks for r in wheel_ranks[:requirement]):
            wheel_cards = [card_by_rank[r] for r in wheel_ranks[:requirement] if r in card_by_rank]
            return True, wheel_cards

        return False, []

    def _make_result(self, hand_type: HandType, cards: list[Card],
                     rank_counts: Counter, scoring_count: int) -> DetectedHand:
        """Create a DetectedHand result, selecting scoring cards."""
        scoring_cards = []

        # Sort ranks by count (descending), then by rank value (descending)
        sorted_ranks = sorted(rank_counts.keys(),
                            key=lambda r: (rank_counts[r], RANK_ORDER[r]),
                            reverse=True)

        # Collect scoring cards
        for rank in sorted_ranks:
            rank_cards = [c for c in cards if c.rank == rank]
            for card in rank_cards:
                if len(scoring_cards) < scoring_count:
                    scoring_cards.append(card)

        return DetectedHand(hand_type, scoring_cards, cards, self.get_level(hand_type))


def detect_hand(cards: list[Card], config: HandDetectorConfig = None,
                hand_levels: dict = None) -> DetectedHand:
    """Convenience function to detect a hand."""
    detector = HandDetector(config, hand_levels)
    return detector.detect(cards)
