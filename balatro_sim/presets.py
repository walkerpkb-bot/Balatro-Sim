"""
Preset configurations for Balatro simulation.
Reflects actual Balatro deck choices and their bonuses.
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
    STANDARD = "standard"   # Default deck
    RED = "red"             # +1 discard per round
    BLUE = "blue"           # +1 hand per round
    YELLOW = "yellow"       # +$10 starting money
    GREEN = "green"         # No interest cap (not simulated yet)
    BLACK = "black"         # +1 joker slot, -1 hand
    PAINTED = "painted"     # +2 hand size, -1 joker slot
    ABANDONED = "abandoned" # No face cards in deck
    CHECKERED = "checkered" # Only Spades and Hearts
    PLASMA = "plasma"       # Chips and mult balanced
    ERRATIC = "erratic"     # Random ranks/suits


@dataclass
class Preset:
    """A complete preset configuration for a run."""
    name: str
    description: str
    deck_type: DeckType = DeckType.STANDARD
    strategy: StrategyType = StrategyType.SMART
    starting_jokers: list[str] = field(default_factory=list)
    hand_levels: dict = field(default_factory=dict)
    config_overrides: dict = field(default_factory=dict)


# Official Balatro Deck Presets
PRESETS = {
    "standard": Preset(
        name="Standard Deck",
        description="No modifiers. The default Balatro experience.",
        deck_type=DeckType.STANDARD,
        strategy=StrategyType.SMART,
    ),

    "red": Preset(
        name="Red Deck",
        description="+1 discard every round",
        deck_type=DeckType.RED,
        strategy=StrategyType.OPTIMIZED,
        config_overrides={"starting_discards": 4},
    ),

    "blue": Preset(
        name="Blue Deck",
        description="+1 hand every round",
        deck_type=DeckType.BLUE,
        strategy=StrategyType.SMART,
        config_overrides={"starting_hands": 5},
    ),

    "yellow": Preset(
        name="Yellow Deck",
        description="Start with extra $10",
        deck_type=DeckType.YELLOW,
        strategy=StrategyType.SMART,
        config_overrides={"starting_money": 14},
    ),

    "black": Preset(
        name="Black Deck",
        description="+1 Joker slot, -1 hand every round",
        deck_type=DeckType.BLACK,
        strategy=StrategyType.AGGRESSIVE,
        config_overrides={"joker_slots": 6, "starting_hands": 3},
    ),

    "painted": Preset(
        name="Painted Deck",
        description="+2 hand size, -1 Joker slot",
        deck_type=DeckType.PAINTED,
        strategy=StrategyType.SMART,
        config_overrides={"hand_size": 10, "joker_slots": 4},
    ),

    "abandoned": Preset(
        name="Abandoned Deck",
        description="No face cards in deck",
        deck_type=DeckType.ABANDONED,
        strategy=StrategyType.SMART,
        config_overrides={"no_face_cards": True},
    ),

    "checkered": Preset(
        name="Checkered Deck",
        description="Only Spades and Hearts in deck",
        deck_type=DeckType.CHECKERED,
        strategy=StrategyType.OPTIMIZED,
        config_overrides={"suits": ["Spades", "Hearts"]},
    ),
}


def get_preset(name: str) -> Optional[Preset]:
    """Get a preset by name."""
    return PRESETS.get(name.lower().replace(" ", "_").replace("-", "_"))


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
