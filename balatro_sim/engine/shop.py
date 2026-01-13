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


class PackType(Enum):
    ARCANA = "Arcana"          # Tarot cards
    CELESTIAL = "Celestial"    # Planet cards
    SPECTRAL = "Spectral"      # Spectral cards
    STANDARD = "Standard"      # Playing cards
    BUFFOON = "Buffoon"        # Jokers


@dataclass
class Pack:
    """A booster pack that offers choices."""
    pack_type: PackType
    cost: int
    choices: int      # How many you can pick
    options: list     # The cards/jokers to choose from
    is_mega: bool = False

    def __str__(self):
        mega = "Mega " if self.is_mega else ""
        return f"{mega}{self.pack_type.value} Pack (${self.cost})"


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


# Voucher definitions
@dataclass
class Voucher:
    """A voucher that provides a permanent upgrade."""
    name: str
    cost: int
    effect: dict
    description: str
    tier: int = 1  # 1 = base, 2 = upgraded version

    def __str__(self):
        return f"{self.name} (${self.cost})"


# Available vouchers and their effects
VOUCHERS = [
    # Tier 1 vouchers
    Voucher("Overstock", 10, {"extra_shop_slots": 1}, "+1 card slot in shop", tier=1),
    Voucher("Clearance Sale", 10, {"shop_discount": 0.25}, "25% off all items in shop", tier=1),
    Voucher("Hone", 10, {"uncommon_rate_boost": 0.04}, "Better joker rarities", tier=1),
    Voucher("Reroll Surplus", 10, {"reroll_discount": 2}, "-$2 per reroll", tier=1),
    Voucher("Crystal Ball", 10, {"consumable_slots": 1}, "+1 consumable slot", tier=1),
    Voucher("Telescope", 10, {"celestial_rate": 1.5}, "More planet cards appear", tier=1),
    Voucher("Grabber", 10, {"extra_hands": 1}, "+1 hand per round", tier=1),
    Voucher("Wasteful", 10, {"extra_discards": 1}, "+1 discard per round", tier=1),
    Voucher("Seed Money", 10, {"interest_cap": 10}, "Raise interest cap to $10", tier=1),
    Voucher("Blank", 10, {"starting_draw": 1}, "+1 hand size", tier=1),

    # Tier 2 vouchers (require tier 1 to unlock)
    Voucher("Overstock Plus", 10, {"extra_shop_slots": 1}, "+1 more card slot", tier=2),
    Voucher("Liquidation", 10, {"shop_discount": 0.25}, "Additional 25% off", tier=2),
    Voucher("Glow Up", 10, {"rare_rate_boost": 0.04}, "Even better rarities", tier=2),
    Voucher("Reroll Glut", 10, {"reroll_discount": 2}, "-$2 more per reroll", tier=2),
    Voucher("Omen Globe", 10, {"spectral_rate": 1.5}, "More spectral cards", tier=2),
    Voucher("Nacho Tong", 10, {"extra_hands": 1}, "+1 more hand", tier=2),
    Voucher("Recyclomancy", 10, {"extra_discards": 1}, "+1 more discard", tier=2),
    Voucher("Money Tree", 10, {"interest_cap": 15}, "Raise cap to $15", tier=2),
    Voucher("Antimatter", 10, {"joker_slots": 1}, "+1 joker slot", tier=2),
    Voucher("Magic Trick", 10, {"play_hand_anywhere": True}, "Cards can go anywhere", tier=2),
]


def get_available_vouchers(owned_vouchers: list[str], ante: int = 1) -> list[Voucher]:
    """Get vouchers available for purchase based on owned vouchers and ante."""
    available = []
    owned_names = set(owned_vouchers)

    for voucher in VOUCHERS:
        if voucher.name in owned_names:
            continue
        # Tier 2 requires owning the tier 1 version first (simplified check)
        if voucher.tier == 2:
            # Check if corresponding tier 1 is owned
            tier1_names = [v.name for v in VOUCHERS if v.tier == 1]
            if not any(name in owned_names for name in tier1_names):
                continue
        available.append(voucher)

    return available

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
    pack_base_cost: int = 4
    mega_pack_cost: int = 6


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
        self.packs: list[Pack] = []

        # Track what's been bought this run (for Showman joker)
        self.purchased_joker_names: set = set()

    def generate(self, owned_jokers: list = None, allow_duplicates: bool = False,
                 owned_vouchers: list = None, ante: int = 1):
        """Generate new shop contents."""
        owned_names = {j.get("name") for j in (owned_jokers or [])}
        owned_vouchers = owned_vouchers or []

        # Generate jokers
        self.jokers = []
        for _ in range(self.config.joker_slots):
            joker = self._pick_random_joker(owned_names, allow_duplicates)
            if joker:
                self.jokers.append(joker)

        # Generate consumables (mix of planets, tarots, and rarely spectrals)
        self.consumables = []
        for _ in range(self.config.consumable_slots):
            roll = random.random()
            if roll < 0.05:  # 5% chance for spectral
                self.consumables.append(self._generate_spectral())
            elif roll < 0.55:  # 50% chance for planet
                self.consumables.append(self._generate_planet())
            else:  # 45% chance for tarot
                self.consumables.append(self._generate_tarot())

        # Generate voucher (one per shop)
        self.vouchers = []
        available = get_available_vouchers(owned_vouchers, ante)
        if available:
            voucher = random.choice(available)
            self.vouchers.append(voucher)

        # Generate booster packs
        self.packs = []
        for _ in range(self.config.pack_slots):
            pack = self._generate_pack(owned_names)
            if pack:
                self.packs.append(pack)

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

        joker = random.choices(candidates, weights=weights, k=1)[0]

        # Copy joker dict so we don't mutate the original
        joker = dict(joker)

        # Small chance for edition (2% Foil, 1% Holo, 0.5% Poly)
        edition_roll = random.random()
        if edition_roll < 0.005:
            joker['edition'] = 'Polychrome'
        elif edition_roll < 0.015:
            joker['edition'] = 'Holographic'
        elif edition_roll < 0.035:
            joker['edition'] = 'Foil'

        return joker

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

    def _generate_spectral(self) -> Consumable:
        """Generate a random spectral card (rare and powerful)."""
        spectrals = [
            ("Familiar", {"create_joker": True, "type": "random"}),
            ("Grim", {"destroy_for_money": True, "money_per_card": 4}),
            ("Incantation", {"add_copies": 4, "rank": "random"}),
            ("Talisman", {"add_seal": "Gold", "count": 1}),
            ("Aura", {"add_edition": "random", "count": 1}),
            ("Wraith", {"create_rare_joker": True, "cost": "all_money"}),
            ("Sigil", {"convert_to_suit": "single", "count": "all"}),
            ("Ouija", {"add_enhancement": "random", "reduce_hand_size": 1}),
            ("Ectoplasm", {"add_negative": True, "reduce_hand_size": 1}),
            ("Immolate", {"destroy_cards": 5, "gain_money": 20}),
            ("Ankh", {"copy_random_joker": True, "destroy_others": True}),
            ("Deja Vu", {"add_seal": "Red", "count": 1}),
            ("Hex", {"add_polychrome": True, "destroy_other_jokers": True}),
            ("Trance", {"add_seal": "Blue", "count": 1}),
            ("Medium", {"add_seal": "Purple", "count": 1}),
            ("Cryptid", {"copy_cards": 2}),
            ("The Soul", {"create_legendary_joker": True}),
            ("Black Hole", {"level_up_all": 1}),
        ]
        name, effect = random.choice(spectrals)
        return Consumable(
            name=name,
            type=ConsumableType.SPECTRAL,
            effect=effect,
            cost=4
        )

    def _generate_pack(self, owned_joker_names: set = None) -> Pack:
        """Generate a random booster pack with its contents."""
        owned_joker_names = owned_joker_names or set()

        # 20% chance for mega pack
        is_mega = random.random() < 0.20
        cost = self.config.mega_pack_cost if is_mega else self.config.pack_base_cost

        # Pack type weights: Arcana/Celestial more common, Buffoon rarer
        pack_weights = {
            PackType.ARCANA: 30,
            PackType.CELESTIAL: 30,
            PackType.STANDARD: 20,
            PackType.SPECTRAL: 10,
            PackType.BUFFOON: 10,
        }

        pack_type = random.choices(
            list(pack_weights.keys()),
            weights=list(pack_weights.values()),
            k=1
        )[0]

        # Generate options based on pack type
        if pack_type == PackType.ARCANA:
            num_options = 5 if is_mega else 3
            options = [self._generate_tarot() for _ in range(num_options)]
            choices = 1 if not is_mega else 2

        elif pack_type == PackType.CELESTIAL:
            num_options = 5 if is_mega else 3
            options = [self._generate_planet() for _ in range(num_options)]
            choices = 1 if not is_mega else 2

        elif pack_type == PackType.SPECTRAL:
            num_options = 4 if is_mega else 2
            options = [self._generate_spectral() for _ in range(num_options)]
            choices = 1 if not is_mega else 2

        elif pack_type == PackType.STANDARD:
            num_options = 5 if is_mega else 3
            options = self._generate_playing_cards(num_options)
            choices = 1 if not is_mega else 2

        elif pack_type == PackType.BUFFOON:
            num_options = 4 if is_mega else 2
            options = []
            for _ in range(num_options):
                joker = self._pick_random_joker(owned_joker_names, allow_duplicates=False)
                if joker:
                    options.append(joker)
            choices = 1

        return Pack(
            pack_type=pack_type,
            cost=cost,
            choices=choices,
            options=options,
            is_mega=is_mega
        )

    def _generate_playing_cards(self, count: int) -> list:
        """Generate random playing cards for Standard packs."""
        from .deck import Card, Suit, RANKS, Enhancement, Edition, Seal

        cards = []
        for _ in range(count):
            suit = random.choice(list(Suit))
            rank = random.choice(RANKS)
            card = Card(suit=suit, rank=rank)

            # 15% chance for enhancement
            if random.random() < 0.15:
                card.enhancement = random.choice(list(Enhancement))

            # 5% chance for edition
            if random.random() < 0.05:
                card.edition = random.choice([Edition.FOIL, Edition.HOLOGRAPHIC, Edition.POLYCHROME])

            # 3% chance for seal
            if random.random() < 0.03:
                card.seal = random.choice(list(Seal))

            cards.append(card)

        return cards

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
            - vouchers_to_buy: list of voucher indices
            - packs_to_buy: list of pack indices
            - pack_choices: dict of pack_idx -> list of chosen option indices
            - use_consumables: list of (consumable, target) tuples
            - reroll: bool
        """
        decisions = {
            "jokers_to_buy": [],
            "consumables_to_buy": [],
            "vouchers_to_buy": [],
            "packs_to_buy": [],
            "pack_choices": {},
            "use_consumables": [],
            "reroll": False
        }

        money = game_state.money
        extra_joker_slots = game_state.voucher_bonuses.get('extra_joker_slots', 0)
        joker_slots_free = game_state.config.joker_slots + extra_joker_slots - len(game_state.jokers)
        extra_cons_slots = game_state.voucher_bonuses.get('extra_consumable_slots', 0)
        consumable_slots_free = game_state.config.consumable_slots + extra_cons_slots - len(game_state.consumables)

        # Evaluate vouchers first (permanent upgrades are valuable)
        for i, voucher in enumerate(shop.vouchers):
            if voucher.cost > money:
                continue
            voucher_score = self._score_voucher(voucher, game_state)
            if voucher_score > 5:  # Worth buying
                decisions["vouchers_to_buy"].append(i)
                money -= voucher.cost

        # PRIORITY: Buy jokers BEFORE packs in early game (ante <= 4)
        # Direct jokers are more reliable than pack RNG
        if game_state.ante <= 4 or joker_slots_free >= 3:
            # Evaluate jokers first
            joker_scores = []
            for i, joker in enumerate(shop.jokers):
                cost = shop.get_joker_cost(joker)
                if cost > money or joker_slots_free <= 0:
                    continue
                score = self._score_joker(joker, game_state)
                joker_scores.append((i, joker, cost, score))

            joker_scores.sort(key=lambda x: x[3] / max(1, x[2]), reverse=True)

            for i, joker, cost, score in joker_scores:
                if cost <= money and joker_slots_free > 0 and score > 0:
                    decisions["jokers_to_buy"].append(i)
                    money -= cost
                    joker_slots_free -= 1

        # Evaluate packs (better in late game when we have money)
        for i, pack in enumerate(shop.packs):
            if pack.cost > money:
                continue

            pack_score = self._score_pack(pack, game_state)
            if pack_score > 3:  # Worth buying
                decisions["packs_to_buy"].append(i)
                choices = self._choose_from_pack(pack, game_state, joker_slots_free)
                decisions["pack_choices"][i] = choices
                money -= pack.cost

                if pack.pack_type == PackType.BUFFOON:
                    joker_slots_free -= len(choices)

        # Evaluate remaining jokers (late game or after packs)
        joker_scores = []
        for i, joker in enumerate(shop.jokers):
            if i in decisions["jokers_to_buy"]:  # Skip already bought
                continue
            cost = shop.get_joker_cost(joker)
            if cost > money or joker_slots_free <= 0:
                continue
            score = self._score_joker(joker, game_state)
            joker_scores.append((i, joker, cost, score))

        # Sort by score/cost ratio
        joker_scores.sort(key=lambda x: x[3] / max(1, x[2]), reverse=True)

        # Buy best jokers we can afford
        for i, joker, cost, score in joker_scores:
            if i in decisions["jokers_to_buy"]:  # Double-check not already bought
                continue
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

    def _score_voucher(self, voucher: Voucher, game_state) -> float:
        """Score a voucher based on its effect and game state."""
        score = 0
        effect = voucher.effect

        # Extra hands/discards are very valuable
        if 'extra_hands' in effect:
            score += effect['extra_hands'] * 15
        if 'extra_discards' in effect:
            score += effect['extra_discards'] * 10
        if 'starting_draw' in effect:
            score += effect['starting_draw'] * 8
        if 'joker_slots' in effect:
            score += effect['joker_slots'] * 20
        if 'interest_cap' in effect:
            score += (effect['interest_cap'] - 5) * 2

        return score

    def _score_joker(self, joker: dict, game_state) -> float:
        """Score a joker based on current game state and synergies."""
        score = 0
        effect = joker.get("effect", {})
        modifiers = effect.get("modifiers", [])
        conditions = effect.get("conditions", [])
        name = joker.get("name", "")

        # Base scoring from modifiers
        for mod in modifiers:
            mod_type = mod.get("type", "")

            if mod_type == "x_mult":
                score += mod.get("value", 1) * 50
            elif mod_type == "add_mult":
                score += mod.get("value", 0) * 2
            elif mod_type == "add_chips":
                score += mod.get("value", 0) * 0.5
            elif mod_type in ("earn_money", "give_money"):
                score += mod.get("value", 0) * 3

        # Rarity bonus
        rarity = joker.get("rarity", "Common")
        rarity_bonus = {"Common": 0, "Uncommon": 5, "Rare": 15, "Legendary": 50}
        score += rarity_bonus.get(rarity, 0)

        # SYNERGY SCORING

        # Check if joker synergizes with leveled hand types
        for cond in conditions:
            if cond.get("type") == "hand_contains":
                required_hand = cond.get("hand", "").upper().replace(" ", "_")
                hand_level = game_state.hand_levels.get(required_hand, 1)
                if hand_level >= 2:
                    score += 15 * hand_level  # Bonus for leveled hands

        # Suit synergy with existing jokers
        existing_suits = set()
        for existing in game_state.jokers:
            ex_effect = existing.get("effect", {})
            for cond in ex_effect.get("conditions", []):
                if cond.get("type") == "suit":
                    existing_suits.add(cond.get("suit", "").lower())

        for cond in conditions:
            if cond.get("type") == "suit":
                req_suit = cond.get("suit", "").lower()
                if req_suit in existing_suits:
                    score += 20  # Same suit synergy

        # Scaling jokers are valuable early
        scaling_jokers = {"Ride the Bus", "Green Joker", "Ice Cream", "Runner", "Square Joker", "Supernova"}
        if name in scaling_jokers and game_state.ante <= 3:
            score += 15

        # X-mult jokers synergize with high-mult existing jokers
        has_xmult = any(m.get("type") == "x_mult" for m in modifiers)
        existing_mult = sum(
            m.get("value", 0) for j in game_state.jokers
            for m in j.get("effect", {}).get("modifiers", [])
            if m.get("type") == "add_mult"
        )
        if has_xmult and existing_mult >= 10:
            score += 25  # x-mult amplifies existing mult

        # Retrigger jokers (Hack, Dusk, Sock and Buskin) synergize with chip-heavy cards
        retrigger_jokers = {"Hack", "Dusk", "Sock and Buskin"}
        if name in retrigger_jokers:
            score += 10  # Generally good

        return score

    def _should_use_immediately(self, consumable: Consumable, game_state) -> bool:
        """Check if consumable should be used right away."""
        if consumable.type == ConsumableType.PLANET:
            return True  # Always use planets immediately
        return False

    def _score_pack(self, pack: Pack, game_state) -> float:
        """Score a pack based on type and game state."""
        score = 0

        # Base value by type
        type_scores = {
            PackType.CELESTIAL: 8,   # Planets are always good
            PackType.BUFFOON: 7,     # Jokers are valuable
            PackType.ARCANA: 5,      # Tarots are situational
            PackType.SPECTRAL: 6,    # Spectrals can be powerful
            PackType.STANDARD: 3,    # Cards are least exciting
        }
        score = type_scores.get(pack.pack_type, 3)

        # Bonus for mega packs (more choices)
        if pack.is_mega:
            score += 2

        # Celestial packs more valuable if we have leveled hands
        if pack.pack_type == PackType.CELESTIAL:
            leveled_hands = sum(1 for lv in game_state.hand_levels.values() if lv > 1)
            score += leveled_hands * 0.5

        # Buffoon packs less valuable if joker slots full
        if pack.pack_type == PackType.BUFFOON:
            extra_slots = game_state.voucher_bonuses.get('extra_joker_slots', 0)
            slots_free = game_state.config.joker_slots + extra_slots - len(game_state.jokers)
            if slots_free <= 0:
                score = 0  # Can't take jokers

        return score

    def _choose_from_pack(self, pack: Pack, game_state, joker_slots_free: int) -> list[int]:
        """Choose the best options from a pack. Returns indices of chosen options."""
        if not pack.options:
            return []

        # Score each option
        scored_options = []
        for i, option in enumerate(pack.options):
            score = self._score_pack_option(option, pack.pack_type, game_state)
            scored_options.append((i, score))

        # Sort by score descending
        scored_options.sort(key=lambda x: x[1], reverse=True)

        # Pick top N choices (limited by pack.choices and available slots for jokers)
        num_to_pick = pack.choices
        if pack.pack_type == PackType.BUFFOON:
            num_to_pick = min(num_to_pick, joker_slots_free)

        chosen = [idx for idx, score in scored_options[:num_to_pick] if score > 0]
        return chosen

    def _score_pack_option(self, option, pack_type: PackType, game_state) -> float:
        """Score a single option from a pack."""

        if pack_type == PackType.CELESTIAL:
            # Score planet based on how much we use that hand type
            consumable = option
            hand_type = consumable.effect.get("levels_up")
            if hand_type:
                current_level = game_state.hand_levels.get(hand_type, 1)
                # Prefer leveling already-leveled hands, or common hands
                base_score = 5
                if current_level > 1:
                    base_score += current_level * 2  # Compound gains
                # Bonus for common hand types
                common_types = [HandType.PAIR, HandType.TWO_PAIR, HandType.FLUSH, HandType.STRAIGHT]
                if hand_type in common_types:
                    base_score += 3
                return base_score
            return 3

        elif pack_type == PackType.ARCANA:
            # Score tarot by effect usefulness
            consumable = option
            effect = consumable.effect

            if effect.get("double_money"):
                return 8 if game_state.money >= 10 else 4
            if "enhance" in effect:
                return 6
            if "convert_to_suit" in effect:
                return 5
            if effect.get("random_edition"):
                return 7 if game_state.jokers else 2
            return 4

        elif pack_type == PackType.SPECTRAL:
            consumable = option
            effect = consumable.effect

            if effect.get("level_up_all"):  # Black Hole
                return 15
            if effect.get("create_legendary_joker"):  # The Soul
                return 12
            if effect.get("gain_money"):  # Immolate
                return 8
            return 5

        elif pack_type == PackType.BUFFOON:
            # Use existing joker scoring
            return self._score_joker(option, game_state)

        elif pack_type == PackType.STANDARD:
            # Score playing cards by enhancement/edition/seal
            card = option
            score = 2  # Base value of adding a card

            if hasattr(card, 'enhancement') and card.enhancement:
                score += 4
            if hasattr(card, 'edition') and card.edition:
                score += 6
            if hasattr(card, 'seal') and card.seal:
                score += 5

            return score

        return 3


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


def apply_tarot(consumable: Consumable, game_state, target_cards: list = None) -> str:
    """
    Apply a tarot card's effect.

    Args:
        consumable: The tarot card to use
        game_state: Current game state
        target_cards: Cards to target (from hand or deck)

    Returns:
        Description of what happened
    """
    from .deck import Enhancement, Suit

    if consumable.type != ConsumableType.TAROT:
        return f"{consumable.name} is not a tarot card"

    effect = consumable.effect
    name = consumable.name

    # Enhance cards (Magician, Empress, Hierophant, Lovers, Chariot, Justice, Devil, Tower)
    if "enhance" in effect:
        enhancement_name = effect["enhance"]
        count = effect.get("count", 1)

        # Get enhancement enum
        try:
            enhancement = Enhancement[enhancement_name.upper()]
        except KeyError:
            return f"{name}: Unknown enhancement {enhancement_name}"

        # Apply to random cards in hand
        cards_enhanced = 0
        if game_state.hand and game_state.hand.cards:
            targets = random.sample(
                game_state.hand.cards,
                min(count, len(game_state.hand.cards))
            )
            for card in targets:
                card.enhancement = enhancement
                cards_enhanced += 1

        return f"{name}: Enhanced {cards_enhanced} cards with {enhancement_name}"

    # Convert to suit (Star, Moon, Sun, World)
    if "convert_to_suit" in effect:
        suit_name = effect["convert_to_suit"]
        try:
            new_suit = Suit[suit_name.upper()]
        except KeyError:
            return f"{name}: Unknown suit {suit_name}"

        cards_converted = 0
        if game_state.hand and game_state.hand.cards:
            # Convert up to 3 random cards
            targets = random.sample(
                game_state.hand.cards,
                min(3, len(game_state.hand.cards))
            )
            for card in targets:
                card.suit = new_suit
                cards_converted += 1

        return f"{name}: Converted {cards_converted} cards to {suit_name}"

    # Double money (Hermit)
    if effect.get("double_money"):
        max_gain = effect.get("max", 20)
        gain = min(game_state.money, max_gain)
        game_state.money += gain
        return f"{name}: Doubled ${gain}"

    # Rank up cards (Strength)
    if "rank_up" in effect:
        from .deck import RANKS, RANK_ORDER
        count = effect["rank_up"]
        cards_upgraded = 0

        if game_state.hand and game_state.hand.cards:
            targets = random.sample(
                game_state.hand.cards,
                min(count, len(game_state.hand.cards))
            )
            for card in targets:
                current_idx = RANK_ORDER.get(card.rank, 0)
                if current_idx < len(RANKS) - 1:
                    card.rank = RANKS[current_idx + 1]
                    cards_upgraded += 1

        return f"{name}: Upgraded {cards_upgraded} cards"

    # Destroy cards (Hanged Man)
    if "destroy_cards" in effect:
        count = effect["destroy_cards"]
        cards_destroyed = 0

        if game_state.hand and game_state.hand.cards:
            to_destroy = random.sample(
                game_state.hand.cards,
                min(count, len(game_state.hand.cards))
            )
            for card in to_destroy:
                game_state.hand.cards.remove(card)
                cards_destroyed += 1

        return f"{name}: Destroyed {cards_destroyed} cards"

    # Create planets (High Priestess)
    if "create_planets" in effect:
        # Simplified - just adds planet uses counter
        count = effect["create_planets"]
        return f"{name}: Created {count} planet cards (use later)"

    # Create tarots (Emperor)
    if "create_tarots" in effect:
        count = effect["create_tarots"]
        return f"{name}: Created {count} tarot cards (use later)"

    # Random edition on joker (Wheel of Fortune)
    if effect.get("random_edition"):
        if game_state.jokers:
            joker = random.choice(game_state.jokers)
            editions = ['Foil', 'Holographic', 'Polychrome']
            edition = random.choice(editions)
            joker['edition'] = edition
            return f"{name}: Added {edition} edition to {joker.get('name', 'joker')}"
        return f"{name}: No jokers to enhance"

    return f"{name}: Effect not implemented"


def apply_spectral(consumable: Consumable, game_state, all_jokers: list = None) -> str:
    """
    Apply a spectral card's effect.

    Args:
        consumable: The spectral card to use
        game_state: Current game state
        all_jokers: List of all available jokers (for creating new ones)

    Returns:
        Description of what happened
    """
    from .deck import Seal, Enhancement, Edition
    from .hand_detector import HandType

    if consumable.type != ConsumableType.SPECTRAL:
        return f"{consumable.name} is not a spectral card"

    effect = consumable.effect
    name = consumable.name

    # Immolate: Destroy 5 cards, gain $20
    if "gain_money" in effect and "destroy_cards" in effect:
        cards_to_destroy = effect["destroy_cards"]
        money_gain = effect["gain_money"]

        destroyed = 0
        if game_state.hand and game_state.hand.cards:
            to_destroy = random.sample(
                game_state.hand.cards,
                min(cards_to_destroy, len(game_state.hand.cards))
            )
            for card in to_destroy:
                game_state.hand.cards.remove(card)
                destroyed += 1

        game_state.money += money_gain
        return f"{name}: Destroyed {destroyed} cards, gained ${money_gain}"

    # Add seal to cards
    if "add_seal" in effect:
        seal_name = effect["add_seal"]
        count = effect.get("count", 1)

        try:
            seal = Seal[seal_name.upper()]
        except KeyError:
            return f"{name}: Unknown seal {seal_name}"

        cards_sealed = 0
        if game_state.hand and game_state.hand.cards:
            targets = random.sample(
                game_state.hand.cards,
                min(count, len(game_state.hand.cards))
            )
            for card in targets:
                card.seal = seal
                cards_sealed += 1

        return f"{name}: Added {seal_name} seal to {cards_sealed} cards"

    # Black Hole: Level up all hands
    if "level_up_all" in effect:
        levels = effect["level_up_all"]
        for ht in HandType:
            old = game_state.hand_levels.get(ht, 1)
            game_state.hand_levels[ht] = old + levels
        game_state.hand_detector.hand_levels = game_state.hand_levels
        return f"{name}: Leveled up all hands by {levels}"

    # Create a random joker
    if effect.get("create_joker") and all_jokers:
        owned_names = {j.get("name") for j in game_state.jokers}
        available = [j for j in all_jokers if j.get("name") not in owned_names]
        if available:
            new_joker = random.choice(available).copy()
            extra_slots = game_state.voucher_bonuses.get('extra_joker_slots', 0)
            max_slots = game_state.config.joker_slots + extra_slots
            if len(game_state.jokers) < max_slots:
                game_state.jokers.append(new_joker)
                return f"{name}: Created {new_joker.get('name', 'joker')}"
        return f"{name}: No joker slot available"

    # Grim: Destroy cards for money
    if effect.get("destroy_for_money"):
        money_per = effect.get("money_per_card", 4)
        destroyed = 0
        money_gained = 0

        if game_state.hand and game_state.hand.cards:
            # Destroy up to 2 cards
            to_destroy = random.sample(
                game_state.hand.cards,
                min(2, len(game_state.hand.cards))
            )
            for card in to_destroy:
                game_state.hand.cards.remove(card)
                destroyed += 1
                money_gained += money_per

        game_state.money += money_gained
        return f"{name}: Destroyed {destroyed} cards for ${money_gained}"

    # Aura: Add random edition to card
    if "add_edition" in effect:
        editions = [Edition.FOIL, Edition.HOLOGRAPHIC, Edition.POLYCHROME]
        edition = random.choice(editions)

        if game_state.hand and game_state.hand.cards:
            card = random.choice(game_state.hand.cards)
            card.edition = edition
            return f"{name}: Added {edition.value} to {card}"

        return f"{name}: No cards to enhance"

    return f"{name}: Effect not implemented"
