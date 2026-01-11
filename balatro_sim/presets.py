"""
Preset configurations for Balatro simulation.
Allows easy setup of different playstyles and starting conditions.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .engine.game import GameConfig


class StrategyType(Enum):
    BASIC = "basic"
    SMART = "smart"
    OPTIMIZED = "optimized"
    AGGRESSIVE = "aggressive"


class DeckType(Enum):
    RED = "red"           # +1 discard per round
    BLUE = "blue"         # +1 hand per round
    YELLOW = "yellow"     # +$10 starting money
    GREEN = "green"       # No interest cap
    BLACK = "black"       # +1 joker slot, -1 hand
    MAGIC = "magic"       # Crystal Ball + Illusion vouchers
    NEBULA = "nebula"     # Telescope voucher, -1 consumable slot
    GHOST = "ghost"       # Hex + Polychrome cards
    ABANDONED = "abandoned"  # No face cards
    CHECKERED = "checkered"  # 26 Spades, 26 Hearts
    ZODIAC = "zodiac"     # Start with vouchers
    PAINTED = "painted"   # +2 hand size, -1 joker slot
    ANAGLYPH = "anaglyph" # Double tags
    PLASMA = "plasma"     # Chips and mult balanced
    ERRATIC = "erratic"   # Random ranks/suits
    STANDARD = "standard" # Default deck


@dataclass
class Preset:
    """A complete preset configuration for a run."""
    name: str
    description: str
    deck_type: DeckType = DeckType.STANDARD
    strategy: StrategyType = StrategyType.SMART
    starting_jokers: list[str] = field(default_factory=list)
    hand_levels: dict = field(default_factory=dict)  # HandType name -> level
    config_overrides: dict = field(default_factory=dict)


# Built-in presets
PRESETS = {
    "standard": Preset(
        name="Standard",
        description="Default run with no modifiers",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.SMART,
    ),

    "flush_build": Preset(
        name="Flush Build",
        description="Focus on flush hands with suit-based jokers",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.OPTIMIZED,
        starting_jokers=["The Tribe", "Greedy Joker", "Lusty Joker"],
        hand_levels={"FLUSH": 3, "FLUSH_FIVE": 2},
    ),

    "pair_spam": Preset(
        name="Pair Spam",
        description="Maximize pair hands with The Duo",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.SMART,
        starting_jokers=["The Duo", "Jolly Joker", "Sly Joker"],
        hand_levels={"PAIR": 5, "TWO_PAIR": 3},
    ),

    "mult_stacker": Preset(
        name="Mult Stacker",
        description="Stack X-mult jokers for big scores",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.AGGRESSIVE,
        starting_jokers=["The Duo", "The Trio", "The Family", "Joker"],
        hand_levels={"PAIR": 3, "THREE_OF_A_KIND": 3, "FOUR_OF_A_KIND": 3},
    ),

    "economy": Preset(
        name="Economy",
        description="Money-focused build for shop advantage",
        deck_type=DeckType.YELLOW,
        strategy=StrategyType.SMART,
        starting_jokers=["Golden Joker", "To the Moon", "Egg"],
        config_overrides={"starting_money": 14},
    ),

    "blue_deck": Preset(
        name="Blue Deck",
        description="Extra hand per round",
        deck_type=DeckType.BLUE,
        strategy=StrategyType.SMART,
        config_overrides={"starting_hands": 5},
    ),

    "red_deck": Preset(
        name="Red Deck",
        description="Extra discard per round",
        deck_type=DeckType.RED,
        strategy=StrategyType.OPTIMIZED,
        config_overrides={"starting_discards": 4},
    ),

    "speedrun": Preset(
        name="Speedrun",
        description="Aggressive play with strong starting jokers",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.AGGRESSIVE,
        starting_jokers=["The Family", "The Trio", "The Duo", "Blackboard", "Baron"],
        hand_levels={"PAIR": 8, "THREE_OF_A_KIND": 8, "FOUR_OF_A_KIND": 8, "FULL_HOUSE": 6},
    ),

    "no_jokers": Preset(
        name="No Jokers Challenge",
        description="Win without any jokers",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.SMART,
        starting_jokers=[],
        config_overrides={"enable_shop": False},
    ),
}


def get_preset(name: str) -> Optional[Preset]:
    """Get a preset by name."""
    return PRESETS.get(name.lower().replace(" ", "_"))


def list_presets() -> list[str]:
    """List all available preset names."""
    return list(PRESETS.keys())


def get_preset_info(name: str) -> Optional[dict]:
    """Get info about a preset."""
    preset = get_preset(name)
    if preset:
        return {
            "name": preset.name,
            "description": preset.description,
            "deck": preset.deck_type.value,
            "strategy": preset.strategy.value,
            "starting_jokers": preset.starting_jokers,
        }
    return None
