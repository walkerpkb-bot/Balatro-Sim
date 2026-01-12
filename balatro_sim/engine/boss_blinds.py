"""
Boss blind effects for Balatro simulation.
Each ante has a pool of possible bosses with unique debuffs.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable


class BossEffect(Enum):
    """Types of boss effects."""
    NONE = "none"
    EXTRA_LARGE = "extra_large"           # The Wall - 1.5x score requirement
    VERY_LARGE = "very_large"             # Violet Vessel - 2x score requirement
    NO_DISCARDS = "no_discards"           # The Water - Start with 0 discards
    MINUS_HAND_SIZE = "minus_hand_size"   # The Manacle - -1 hand size
    ONE_HAND_ONLY = "one_hand_only"       # The Needle - Only 1 hand allowed
    HALVE_BASE = "halve_base"             # The Flint - Base chips/mult halved
    MUST_PLAY_5 = "must_play_5"           # The Psychic - Must play exactly 5 cards
    DEBUFF_CLUBS = "debuff_clubs"         # The Club - Clubs don't score
    DEBUFF_SPADES = "debuff_spades"       # The Goad - Spades don't score
    DEBUFF_DIAMONDS = "debuff_diamonds"   # The Window - Diamonds don't score
    DEBUFF_HEARTS = "debuff_hearts"       # The Head - Hearts don't score
    DEBUFF_FACE = "debuff_face"           # The Plant - Face cards don't score
    DISCARD_RANDOM = "discard_random"     # The Hook - Discard 2 random cards per hand
    NO_REPEAT_HANDS = "no_repeat_hands"   # The Eye - Can't play same hand type twice
    ONE_HAND_TYPE = "one_hand_type"       # The Mouth - Only one hand type allowed
    LOSE_MONEY = "lose_money"             # The Tooth - Lose $1 per card played


@dataclass
class BossBlind:
    """A boss blind with its effect."""
    name: str
    ante: int  # Minimum ante this boss appears
    effect: BossEffect
    description: str
    score_mult: float = 1.0  # Multiplier for score requirement


# All boss blinds organized by ante
BOSS_BLINDS = [
    # Ante 1 bosses
    BossBlind("The Hook", 1, BossEffect.DISCARD_RANDOM, "Discards 2 random cards per hand"),
    BossBlind("The House", 1, BossEffect.NONE, "First hand drawn face down"),  # Simplified
    BossBlind("The Ox", 1, BossEffect.NONE, "Playing certain suits sets money to $0"),  # Simplified

    # Ante 2 bosses
    BossBlind("The Wall", 2, BossEffect.EXTRA_LARGE, "Extra large blind", score_mult=1.5),
    BossBlind("The Wheel", 2, BossEffect.NONE, "1 in 7 cards drawn face down"),  # Simplified
    BossBlind("The Arm", 2, BossEffect.NONE, "Decrease level of played hand"),  # Simplified

    # Ante 3 bosses
    BossBlind("The Club", 3, BossEffect.DEBUFF_CLUBS, "All Clubs are debuffed"),
    BossBlind("The Fish", 3, BossEffect.NONE, "Cards drawn face down after each hand"),  # Simplified
    BossBlind("The Psychic", 3, BossEffect.MUST_PLAY_5, "Must play exactly 5 cards"),

    # Ante 4 bosses
    BossBlind("The Goad", 4, BossEffect.DEBUFF_SPADES, "All Spades are debuffed"),
    BossBlind("The Water", 4, BossEffect.NO_DISCARDS, "Start with 0 discards"),
    BossBlind("The Window", 4, BossEffect.DEBUFF_DIAMONDS, "All Diamonds are debuffed"),

    # Ante 5 bosses
    BossBlind("The Manacle", 5, BossEffect.MINUS_HAND_SIZE, "-1 hand size"),
    BossBlind("The Eye", 5, BossEffect.NO_REPEAT_HANDS, "No repeat hand types"),
    BossBlind("The Mouth", 5, BossEffect.ONE_HAND_TYPE, "Play only 1 hand type"),

    # Ante 6 bosses
    BossBlind("The Plant", 6, BossEffect.DEBUFF_FACE, "All face cards are debuffed"),
    BossBlind("The Serpent", 6, BossEffect.NONE, "Always draw 3 cards after play"),  # Simplified
    BossBlind("The Pillar", 6, BossEffect.NONE, "Previously played cards debuffed"),  # Simplified

    # Ante 7 bosses
    BossBlind("The Needle", 7, BossEffect.ONE_HAND_ONLY, "Play only 1 hand"),
    BossBlind("The Head", 7, BossEffect.DEBUFF_HEARTS, "All Hearts are debuffed"),
    BossBlind("The Tooth", 7, BossEffect.LOSE_MONEY, "Lose $1 per card played"),

    # Ante 8 bosses
    BossBlind("The Flint", 8, BossEffect.HALVE_BASE, "Base Chips and Mult are halved"),
    BossBlind("The Mark", 8, BossEffect.NONE, "All face cards drawn face down"),  # Simplified
    BossBlind("Violet Vessel", 8, BossEffect.VERY_LARGE, "Very large blind", score_mult=2.0),
]


def get_boss_for_ante(ante: int) -> BossBlind:
    """Get a random boss appropriate for the given ante."""
    # Filter bosses that can appear at this ante
    eligible = [b for b in BOSS_BLINDS if b.ante <= ante]

    # Weight toward bosses matching current ante
    weights = []
    for boss in eligible:
        if boss.ante == ante:
            weights.append(3)  # Higher weight for matching ante
        elif boss.ante == ante - 1:
            weights.append(2)  # Medium weight for previous ante
        else:
            weights.append(1)  # Lower weight for earlier antes

    return random.choices(eligible, weights=weights, k=1)[0]


class BossBlindState:
    """Tracks boss blind state during a round."""

    def __init__(self, boss: BossBlind):
        self.boss = boss
        self.hands_played_types: set = set()  # For The Eye
        self.first_hand_type: Optional[str] = None  # For The Mouth
        self.cards_played_count: int = 0  # For The Tooth

    def get_score_multiplier(self) -> float:
        """Get the score requirement multiplier."""
        return self.boss.score_mult

    def get_discard_modifier(self) -> int:
        """Get modifier to starting discards."""
        if self.boss.effect == BossEffect.NO_DISCARDS:
            return -99  # Effectively 0
        return 0

    def get_hand_size_modifier(self) -> int:
        """Get modifier to hand size."""
        if self.boss.effect == BossEffect.MINUS_HAND_SIZE:
            return -1
        return 0

    def get_hands_modifier(self) -> int:
        """Get modifier to number of hands."""
        if self.boss.effect == BossEffect.ONE_HAND_ONLY:
            return -99  # Effectively 1
        return 0

    def get_base_score_multiplier(self) -> float:
        """Get multiplier for base chips/mult (The Flint)."""
        if self.boss.effect == BossEffect.HALVE_BASE:
            return 0.5
        return 1.0

    def is_suit_debuffed(self, suit) -> bool:
        """Check if a suit is debuffed."""
        # Handle both Suit enum and string
        suit_value = suit.value if hasattr(suit, 'value') else str(suit)
        suit_lower = suit_value.lower()

        if self.boss.effect == BossEffect.DEBUFF_CLUBS and "club" in suit_lower:
            return True
        if self.boss.effect == BossEffect.DEBUFF_SPADES and "spade" in suit_lower:
            return True
        if self.boss.effect == BossEffect.DEBUFF_DIAMONDS and "diamond" in suit_lower:
            return True
        if self.boss.effect == BossEffect.DEBUFF_HEARTS and "heart" in suit_lower:
            return True
        return False

    def is_face_debuffed(self) -> bool:
        """Check if face cards are debuffed."""
        return self.boss.effect == BossEffect.DEBUFF_FACE

    def must_play_5_cards(self) -> bool:
        """Check if must play exactly 5 cards."""
        return self.boss.effect == BossEffect.MUST_PLAY_5

    def check_hand_allowed(self, hand_type_name: str) -> bool:
        """Check if playing this hand type is allowed."""
        # The Eye: No repeat hand types
        if self.boss.effect == BossEffect.NO_REPEAT_HANDS:
            if hand_type_name in self.hands_played_types:
                return False

        # The Mouth: Only one hand type allowed
        if self.boss.effect == BossEffect.ONE_HAND_TYPE:
            if self.first_hand_type is not None and hand_type_name != self.first_hand_type:
                return False

        return True

    def record_hand_played(self, hand_type_name: str):
        """Record that a hand type was played."""
        self.hands_played_types.add(hand_type_name)
        if self.first_hand_type is None:
            self.first_hand_type = hand_type_name

    def get_random_discards(self) -> int:
        """Get number of random cards to discard (The Hook)."""
        if self.boss.effect == BossEffect.DISCARD_RANDOM:
            return 2
        return 0

    def get_money_loss_per_card(self) -> int:
        """Get money lost per card played (The Tooth)."""
        if self.boss.effect == BossEffect.LOSE_MONEY:
            return 1
        return 0
