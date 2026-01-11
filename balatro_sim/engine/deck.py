"""
Deck management for Balatro simulation.
Handles card creation, shuffling, drawing, and deck modifications.
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Suit(Enum):
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Enhancement(Enum):
    NONE = "None"
    BONUS = "Bonus"      # +30 Chips
    MULT = "Mult"        # +4 Mult
    WILD = "Wild"        # Any suit
    GLASS = "Glass"      # X2 Mult, can break
    STEEL = "Steel"      # X1.5 Mult in hand
    STONE = "Stone"      # +50 Chips, no rank/suit
    GOLD = "Gold"        # $3 at end of round
    LUCKY = "Lucky"      # 1/5 +20 Mult, 1/15 $20


class Edition(Enum):
    BASE = "Base"
    FOIL = "Foil"           # +50 Chips
    HOLOGRAPHIC = "Holo"    # +10 Mult
    POLYCHROME = "Poly"     # X1.5 Mult
    NEGATIVE = "Negative"   # +1 Joker slot


class Seal(Enum):
    NONE = "None"
    GOLD = "Gold"      # $3 when played
    RED = "Red"        # Retrigger
    BLUE = "Blue"      # Create planet card
    PURPLE = "Purple"  # Create tarot card


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 10, "Q": 10, "K": 10, "A": 11
}
RANK_ORDER = {rank: i for i, rank in enumerate(RANKS)}


@dataclass
class Card:
    rank: str
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.BASE
    seal: Seal = Seal.NONE

    @property
    def chip_value(self) -> int:
        """Base chip value of this card."""
        if self.enhancement == Enhancement.STONE:
            return 50  # Stone cards have fixed 50 chips
        return RANK_VALUES.get(self.rank, 0)

    @property
    def rank_order(self) -> int:
        """Numeric order for sorting/straights."""
        return RANK_ORDER.get(self.rank, 0)

    @property
    def is_face_card(self) -> bool:
        return self.rank in ["J", "Q", "K"]

    def __str__(self) -> str:
        base = f"{self.rank}{self.suit.value[0]}"
        mods = []
        if self.enhancement != Enhancement.NONE:
            mods.append(self.enhancement.value)
        if self.edition != Edition.BASE:
            mods.append(self.edition.value)
        if self.seal != Seal.NONE:
            mods.append(f"{self.seal.value} Seal")
        if mods:
            return f"{base} [{', '.join(mods)}]"
        return base

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Deck:
    cards: list[Card] = field(default_factory=list)
    discard_pile: list[Card] = field(default_factory=list)

    @classmethod
    def standard_52(cls) -> "Deck":
        """Create a standard 52-card deck."""
        cards = []
        for suit in Suit:
            for rank in RANKS:
                cards.append(Card(rank=rank, suit=suit))
        return cls(cards=cards)

    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def draw(self, n: int = 1) -> list[Card]:
        """Draw n cards from the deck."""
        drawn = []
        for _ in range(n):
            if self.cards:
                drawn.append(self.cards.pop())
            elif self.discard_pile:
                # Reshuffle discard pile into deck
                self.cards = self.discard_pile
                self.discard_pile = []
                self.shuffle()
                if self.cards:
                    drawn.append(self.cards.pop())
        return drawn

    def discard(self, cards: list[Card]) -> None:
        """Move cards to discard pile."""
        self.discard_pile.extend(cards)

    def add_card(self, card: Card) -> None:
        """Add a card to the deck."""
        self.cards.append(card)

    def remove_card(self, card: Card) -> bool:
        """Remove a specific card from deck. Returns True if found."""
        if card in self.cards:
            self.cards.remove(card)
            return True
        return False

    def size(self) -> int:
        """Total cards in deck + discard."""
        return len(self.cards) + len(self.discard_pile)

    def cards_remaining(self) -> int:
        """Cards left to draw before reshuffle."""
        return len(self.cards)

    def reset(self) -> None:
        """Combine deck and discard, shuffle."""
        self.cards.extend(self.discard_pile)
        self.discard_pile = []
        self.shuffle()

    def count_by_suit(self, suit: Suit) -> int:
        """Count cards of a specific suit in full deck."""
        all_cards = self.cards + self.discard_pile
        return sum(1 for c in all_cards if c.suit == suit)

    def count_by_rank(self, rank: str) -> int:
        """Count cards of a specific rank in full deck."""
        all_cards = self.cards + self.discard_pile
        return sum(1 for c in all_cards if c.rank == rank)

    def count_enhanced(self, enhancement: Enhancement) -> int:
        """Count cards with a specific enhancement."""
        all_cards = self.cards + self.discard_pile
        return sum(1 for c in all_cards if c.enhancement == enhancement)


class Hand:
    """Represents cards currently held in hand."""

    def __init__(self, cards: list[Card] = None):
        self.cards: list[Card] = cards or []

    def add(self, cards: list[Card]) -> None:
        self.cards.extend(cards)

    def remove(self, cards: list[Card]) -> list[Card]:
        """Remove and return specified cards from hand."""
        removed = []
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)
                removed.append(card)
        return removed

    def select(self, indices: list[int]) -> list[Card]:
        """Get cards at specified indices."""
        return [self.cards[i] for i in indices if 0 <= i < len(self.cards)]

    def clear(self) -> list[Card]:
        """Remove and return all cards."""
        cards = self.cards
        self.cards = []
        return cards

    def size(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        return ", ".join(str(c) for c in self.cards)

    def __repr__(self) -> str:
        return f"Hand({self.__str__()})"
