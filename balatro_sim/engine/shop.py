"""
Shop simulation for Balatro.
Handles joker/consumable generation, buying, selling, and rerolling.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .hand_detector import HandType


class ConsumableType(Enum):
    TAROT = "Tarot"
    PLANET = "Planet"
    SPECTRAL = "Spectral"


@dataclass
class Consumable:
    """A consumable card (Tarot, Planet, or Spectral)."""
    name: str
    type: ConsumableType
    effect: dict
    cost: int = 3

    def __str__(self):
        return f"{self.name} ({self.type.value})"


# Planet cards and their effects
PLANET_CARDS = {
    "Pluto": HandType.HIGH_CARD,
    "Mercury": HandType.PAIR,
    "Uranus": HandType.TWO_PAIR,
    "Venus": HandType.THREE_OF_A_KIND,
    "Saturn": HandType.STRAIGHT,
    "Jupiter": HandType.FLUSH,
    "Earth": HandType.FULL_HOUSE,
    "Mars": HandType.FOUR_OF_A_KIND,
    "Neptune": HandType.STRAIGHT_FLUSH,
    "Planet X": HandType.FIVE_OF_A_KIND,
    "Ceres": HandType.FLUSH_HOUSE,
    "Eris": HandType.FLUSH_FIVE,
}

# Rarity weights for joker spawning
RARITY_WEIGHTS = {
    "Common": 70,
    "Uncommon": 25,
    "Rare": 5,
    "Legendary": 0,  # Only from Soul card
}


@dataclass
class ShopConfig:
    """Configuration for shop behavior."""
    joker_slots: int = 2
    consumable_slots: int = 2
    pack_slots: int = 2
    base_reroll_cost: int = 5
    joker_base_costs: dict = field(default_factory=lambda: {
        "Common": 4,
        "Uncommon": 6,
        "Rare": 8,
        "Legendary": 20
    })


class Shop:
    """
    Generates and manages shop contents.
    """

    def __init__(self, all_jokers: list, config: ShopConfig = None):
        self.all_jokers = all_jokers
        self.config = config or ShopConfig()

        # Organize jokers by rarity
        self.jokers_by_rarity = {
            "Common": [],
            "Uncommon": [],
            "Rare": [],
            "Legendary": []
        }
        for j in all_jokers:
            rarity = j.get("rarity", "Common")
            if rarity in self.jokers_by_rarity:
                self.jokers_by_rarity[rarity].append(j)

        # Current shop contents
        self.jokers: list[dict] = []
        self.consumables: list[Consumable] = []
        self.vouchers: list[dict] = []

        # Track what's been bought this run (for Showman joker)
        self.purchased_joker_names: set = set()

    def generate(self, owned_jokers: list = None, allow_duplicates: bool = False):
        """Generate new shop contents."""
        owned_names = {j.get("name") for j in (owned_jokers or [])}

        # Generate jokers
        self.jokers = []
        for _ in range(self.config.joker_slots):
            joker = self._pick_random_joker(owned_names, allow_duplicates)
            if joker:
                self.jokers.append(joker)

        # Generate consumables (mix of planets and tarots)
        self.consumables = []
        for _ in range(self.config.consumable_slots):
            if random.random() < 0.5:
                self.consumables.append(self._generate_planet())
            else:
                self.consumables.append(self._generate_tarot())

    def _pick_random_joker(self, owned_names: set, allow_duplicates: bool) -> Optional[dict]:
        """Pick a random joker based on rarity weights."""
        # Build weighted list
        candidates = []
        weights = []

        for rarity, weight in RARITY_WEIGHTS.items():
            if weight <= 0:
                continue
            for joker in self.jokers_by_rarity.get(rarity, []):
                name = joker.get("name")
                # Skip if already owned (unless duplicates allowed)
                if not allow_duplicates and name in owned_names:
                    continue
                # Skip if already in shop
                if name in [j.get("name") for j in self.jokers]:
                    continue
                candidates.append(joker)
                weights.append(weight)

        if not candidates:
            return None

        return random.choices(candidates, weights=weights, k=1)[0]

    def _generate_planet(self) -> Consumable:
        """Generate a random planet card."""
        name = random.choice(list(PLANET_CARDS.keys()))
        hand_type = PLANET_CARDS[name]
        return Consumable(
            name=name,
            type=ConsumableType.PLANET,
            effect={"levels_up": hand_type},
            cost=3
        )

    def _generate_tarot(self) -> Consumable:
        """Generate a random tarot card (simplified)."""
        tarots = [
            ("The Fool", {"copy_last_tarot": True}),
            ("The Magician", {"enhance": "Lucky", "count": 2}),
            ("The High Priestess", {"create_planets": 2}),
            ("The Empress", {"enhance": "Mult", "count": 2}),
            ("The Emperor", {"create_tarots": 2}),
            ("The Hierophant", {"enhance": "Bonus", "count": 2}),
            ("The Lovers", {"enhance": "Wild", "count": 1}),
            ("The Chariot", {"enhance": "Steel", "count": 1}),
            ("Justice", {"enhance": "Glass", "count": 2}),
            ("The Hermit", {"double_money": True, "max": 20}),
            ("Wheel of Fortune", {"random_edition": True}),
            ("Strength", {"rank_up": 2}),
            ("The Hanged Man", {"destroy_cards": 2}),
            ("Death", {"convert_cards": 2}),
            ("Temperance", {"sell_value_to_money": True}),
            ("The Devil", {"enhance": "Gold", "count": 2}),
            ("The Tower", {"enhance": "Stone", "count": 2}),
            ("The Star", {"convert_to_suit": "Diamonds"}),
            ("The Moon", {"convert_to_suit": "Clubs"}),
            ("The Sun", {"convert_to_suit": "Hearts"}),
            ("The World", {"convert_to_suit": "Spades"}),
            ("Judgement", {"create_joker": True}),
        ]
        name, effect = random.choice(tarots)
        return Consumable(
            name=name,
            type=ConsumableType.TAROT,
            effect=effect,
            cost=3
        )

    def get_joker_cost(self, joker: dict) -> int:
        """Get the cost of a joker."""
        rarity = joker.get("rarity", "Common")
        base = self.config.joker_base_costs.get(rarity, 5)
        # Use the joker's own cost if available
        cost_str = joker.get("cost", f"${base}")
        if isinstance(cost_str, str):
            return int(cost_str.replace("$", ""))
        return cost_str

    def reroll_cost(self, rerolls_used: int) -> int:
        """Get the cost of rerolling (may increase with vouchers, etc.)."""
        return self.config.base_reroll_cost


class ShopAI:
    """
    AI for making shop decisions.
    """

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy  # "balanced", "aggressive", "economy"

    def decide_purchases(self, shop: Shop, game_state) -> dict:
        """
        Decide what to buy from the shop.

        Returns dict with:
            - jokers_to_buy: list of joker indices
            - consumables_to_buy: list of consumable indices
            - use_consumables: list of (consumable, target) tuples
            - reroll: bool
        """
        decisions = {
            "jokers_to_buy": [],
            "consumables_to_buy": [],
            "use_consumables": [],
            "reroll": False
        }

        money = game_state.money
        joker_slots_free = game_state.config.joker_slots - len(game_state.jokers)
        consumable_slots_free = game_state.config.consumable_slots - len(game_state.consumables)

        # Evaluate jokers
        joker_scores = []
        for i, joker in enumerate(shop.jokers):
            cost = shop.get_joker_cost(joker)
            if cost > money or joker_slots_free <= 0:
                continue
            score = self._score_joker(joker, game_state)
            joker_scores.append((i, joker, cost, score))

        # Sort by score/cost ratio
        joker_scores.sort(key=lambda x: x[3] / max(1, x[2]), reverse=True)

        # Buy best jokers we can afford
        for i, joker, cost, score in joker_scores:
            if cost <= money and joker_slots_free > 0 and score > 0:
                decisions["jokers_to_buy"].append(i)
                money -= cost
                joker_slots_free -= 1

        # Evaluate consumables (prioritize planets for leveling)
        for i, consumable in enumerate(shop.consumables):
            if consumable.cost > money:
                continue

            if consumable.type == ConsumableType.PLANET:
                # Always try to buy planets
                if consumable_slots_free > 0 or self._should_use_immediately(consumable, game_state):
                    decisions["consumables_to_buy"].append(i)
                    money -= consumable.cost
                    if consumable_slots_free > 0:
                        consumable_slots_free -= 1
                    # Plan to use it
                    decisions["use_consumables"].append((consumable, None))

        return decisions

    def _score_joker(self, joker: dict, game_state) -> float:
        """Score a joker based on current game state."""
        score = 0
        effect = joker.get("effect", {})
        modifiers = effect.get("modifiers", [])

        for mod in modifiers:
            mod_type = mod.get("type", "")

            if mod_type == "x_mult":
                # X-mult is very valuable
                score += mod.get("value", 1) * 50

            elif mod_type == "add_mult":
                score += mod.get("value", 0) * 2

            elif mod_type == "add_chips":
                score += mod.get("value", 0) * 0.5

            elif mod_type in ("earn_money", "give_money"):
                score += mod.get("value", 0) * 3

        # Bonus for rarity
        rarity = joker.get("rarity", "Common")
        rarity_bonus = {"Common": 0, "Uncommon": 5, "Rare": 15, "Legendary": 50}
        score += rarity_bonus.get(rarity, 0)

        return score

    def _should_use_immediately(self, consumable: Consumable, game_state) -> bool:
        """Check if consumable should be used right away."""
        if consumable.type == ConsumableType.PLANET:
            return True  # Always use planets immediately
        return False


def apply_planet(consumable: Consumable, game_state) -> str:
    """Apply a planet card's effect (level up a hand)."""
    if consumable.type != ConsumableType.PLANET:
        return f"{consumable.name} is not a planet card"

    hand_type = consumable.effect.get("levels_up")
    if hand_type:
        old_level = game_state.hand_levels.get(hand_type, 1)
        game_state.hand_levels[hand_type] = old_level + 1
        game_state.hand_detector.hand_levels = game_state.hand_levels
        return f"{consumable.name}: {hand_type.name} leveled up to {old_level + 1}"

    return f"{consumable.name} had no effect"
