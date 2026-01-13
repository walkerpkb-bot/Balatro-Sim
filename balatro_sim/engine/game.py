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
from .boss_blinds import BossBlind, BossBlindState, BossEffect, get_boss_for_ante


class BlindType(Enum):
    SMALL = auto()
    BIG = auto()
    BOSS = auto()


class Tag(Enum):
    """Tags earned from skipping blinds."""
    UNCOMMON = "Uncommon Tag"      # Free uncommon joker
    RARE = "Rare Tag"              # Free rare joker
    NEGATIVE = "Negative Tag"      # Free negative joker
    FOIL = "Foil Tag"              # Free foil joker
    HOLOGRAPHIC = "Holographic Tag"
    POLYCHROME = "Polychrome Tag"
    INVESTMENT = "Investment Tag"  # Double blind money
    VOUCHER = "Voucher Tag"        # Free voucher
    BOSS = "Boss Tag"              # Reroll boss blind
    STANDARD = "Standard Tag"      # Free Standard pack
    CHARM = "Charm Tag"            # Free Arcana pack
    METEOR = "Meteor Tag"          # Free Celestial pack
    BUFFOON = "Buffoon Tag"        # Free Buffoon pack
    HANDY = "Handy Tag"            # +$1 per hand played this run
    GARBAGE = "Garbage Tag"        # +$1 per unused hand/discard
    COUPON = "Coupon Tag"          # Free shop items
    DOUBLE = "Double Tag"          # Copy next tag
    JUGGLE = "Juggle Tag"          # +3 hand size next round
    D6 = "D6 Tag"                  # Free rerolls next shop
    SKIP = "Skip Tag"              # +$5


@dataclass
class BlindInfo:
    """Information about an upcoming blind."""
    blind_type: BlindType
    base_chips: int
    reward_money: int
    boss_name: str = None
    boss_effect: str = None
    skip_tag: Tag = None


@dataclass
class AntePreview:
    """Preview of all blinds in current ante."""
    ante: int
    small_blind: BlindInfo
    big_blind: BlindInfo
    boss_blind: BlindInfo


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
    # Deck modifications
    no_face_cards: bool = False  # Abandoned deck
    suits: list = None  # Checkered deck (e.g., ["Spades", "Hearts"])


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

        # Deck and hand - apply deck modifications
        if self.config.no_face_cards or self.config.suits:
            self.deck = Deck.custom(
                suits=self.config.suits,
                no_face_cards=self.config.no_face_cards
            )
        else:
            self.deck = Deck.standard_52()
        self.deck.shuffle()
        self.hand = Hand()

        # Jokers and consumables
        self.jokers: list[dict] = []
        self.consumables: list = []  # List of Consumable objects

        # Shop tracking
        self.jokers_purchased: list[str] = []
        self.planets_used: int = 0
        self.vouchers_owned: list[str] = []  # Names of owned vouchers

        # Voucher bonuses (accumulated from purchased vouchers)
        self.voucher_bonuses = {
            'extra_hands': 0,
            'extra_discards': 0,
            'extra_hand_size': 0,
            'extra_joker_slots': 0,
            'extra_consumable_slots': 0,
            'shop_discount': 0.0,
            'reroll_discount': 0,
            'interest_cap': 5,  # Default interest cap
        }

        # Hand levels
        self.hand_levels: dict[HandType, int] = {ht: 1 for ht in HandType}

        # Stats
        self.hands_played_count: dict[HandType, int] = {ht: 0 for ht in HandType}
        self.total_hands_played: int = 0
        self.total_discards_used: int = 0

        # Joker state tracking (for scaling jokers)
        self.joker_state: dict = {
            'ride_the_bus': 0,           # Current mult (resets on face card)
            'ice_cream': 100,            # Chips (starts 100, -5 per hand)
            'green_joker': 0,            # Mult from hands+discards
            'red_card': 0,               # Mult from skipped boosters
            'blue_joker_mult': 0,        # +1 per remaining card in deck
            'supernova': {},             # {hand_type: times_played}
            'constellation': {},         # {hand_type: x_mult_accumulated}
            'square_joker': 0,           # Chips (starts 0, +4 per card played if 4 cards)
            'runner': 0,                 # Chips gained from straights
            'obelisk': 1.0,              # X-mult (grows if most played hand not played)
            'loyalty_card': 0,           # Hands since last X4
        }

        # Scoring engine
        self.hand_detector = HandDetector(hand_levels=self.hand_levels)
        self.scoring_engine = ScoringEngine()

        # Boss blind state (set when entering boss blind)
        self.current_boss: Optional[BossBlindState] = None

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
            base_score = self.blind_requirements[ante_key].get(blind_key, 999999)
        else:
            # Endless mode scaling (rough estimate)
            base = self.blind_requirements.get("ante_8", {}).get(blind_key, 100000)
            scale = 1.6 ** (self.ante - 8)
            base_score = int(base * scale)

        # Apply boss multiplier if active
        if self.current_boss:
            base_score = int(base_score * self.current_boss.get_score_multiplier())

        return base_score

    def get_ante_preview(self) -> AntePreview:
        """Get preview of all blinds in current ante."""
        ante_key = f"ante_{self.ante}"
        reqs = self.blind_requirements.get(ante_key, {"small": 999, "big": 999, "boss": 999})

        # Get boss info
        boss = get_boss_for_ante(self.ante)

        # Base rewards scale with ante
        small_reward = 3 + self.ante
        big_reward = 4 + self.ante

        # Generate random tags for skip rewards
        skip_tags = [Tag.SKIP, Tag.UNCOMMON, Tag.CHARM, Tag.METEOR, Tag.STANDARD,
                     Tag.BUFFOON, Tag.HANDY, Tag.D6, Tag.COUPON, Tag.JUGGLE]

        return AntePreview(
            ante=self.ante,
            small_blind=BlindInfo(
                blind_type=BlindType.SMALL,
                base_chips=reqs["small"],
                reward_money=small_reward,
                skip_tag=random.choice(skip_tags)
            ),
            big_blind=BlindInfo(
                blind_type=BlindType.BIG,
                base_chips=reqs["big"],
                reward_money=big_reward,
                skip_tag=random.choice(skip_tags)
            ),
            boss_blind=BlindInfo(
                blind_type=BlindType.BOSS,
                base_chips=reqs["boss"],
                reward_money=self.ante + 5,
                boss_name=boss.name,
                boss_effect=boss.description if boss else None
            )
        )

    def skip_blind(self, tag: Tag) -> dict:
        """Skip current blind and apply tag reward. Returns reward details."""
        reward = {"tag": tag.value, "money": 0, "joker": None, "pack": None}

        if tag == Tag.SKIP:
            reward["money"] = 5
            self.money += 5
        elif tag == Tag.UNCOMMON:
            reward["joker"] = "uncommon"  # Shop will give free uncommon
        elif tag == Tag.RARE:
            reward["joker"] = "rare"
        elif tag == Tag.HANDY:
            reward["money"] = self.total_hands_played
            self.money += self.total_hands_played
        elif tag == Tag.D6:
            reward["free_rerolls"] = 2
        elif tag in (Tag.CHARM, Tag.METEOR, Tag.STANDARD, Tag.BUFFOON):
            reward["pack"] = tag.value.replace(" Tag", " Pack")
        elif tag == Tag.JUGGLE:
            reward["hand_size_bonus"] = 3
        elif tag == Tag.COUPON:
            reward["free_shop_item"] = True

        # Advance to next blind
        self._advance_blind_after_skip()

        return reward

    def _advance_blind_after_skip(self):
        """Advance blind state after skipping."""
        if self.current_blind == BlindType.SMALL:
            self.current_blind = BlindType.BIG
        elif self.current_blind == BlindType.BIG:
            self.current_blind = BlindType.BOSS

    def draw_hand(self) -> None:
        """Draw cards up to hand size."""
        hand_size = self.config.hand_size + self.voucher_bonuses.get('extra_hand_size', 0)
        # Apply boss hand size modifier
        if self.current_boss:
            hand_size += self.current_boss.get_hand_size_modifier()
        cards_needed = hand_size - self.hand.size()
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

        # Cards still held (for Baron, Steel, etc.)
        held_cards = self.hand.cards

        # Detect hand type
        detected = self.hand_detector.detect(played)

        # Check if this is the final hand (for Dusk retrigger)
        is_final_hand = self.hands_remaining == 1

        # Score the hand (pass joker_state for scaling jokers, boss_state for debuffs)
        breakdown = self.scoring_engine.score_hand(
            detected,
            self.jokers,
            held_cards=held_cards,
            is_final_hand=is_final_hand,
            joker_state=self.joker_state,
            deck_size=len(self.deck.cards) + len(self.deck.discard_pile),
            boss_state=self.current_boss
        )

        # Update joker state after scoring
        has_face_card = any(c.is_face_card for c in played)

        # Ride the Bus: +1 mult per hand without face cards, resets on face
        if has_face_card:
            self.joker_state['ride_the_bus'] = 0
        else:
            self.joker_state['ride_the_bus'] += 1

        # Ice Cream: -5 chips per hand
        self.joker_state['ice_cream'] = max(0, self.joker_state['ice_cream'] - 5)

        # Green Joker: +1 mult per hand
        self.joker_state['green_joker'] += 1

        # Square Joker: +4 chips if exactly 4 cards played
        if len(played) == 4:
            self.joker_state['square_joker'] += 4

        # Runner: +15 chips if straight was played
        if detected.hand_type in (HandType.STRAIGHT, HandType.STRAIGHT_FLUSH):
            self.joker_state['runner'] += 15

        # Supernova: track hand plays
        ht_name = detected.hand_type.name
        self.joker_state['supernova'][ht_name] = self.joker_state['supernova'].get(ht_name, 0) + 1

        # Update state
        self.hands_remaining -= 1
        self.hands_played_count[detected.hand_type] += 1
        self.total_hands_played += 1

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
        self.total_discards_used += 1

        # Green Joker: +1 mult per discard
        self.joker_state['green_joker'] += 1

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

            # Interest (20% of money, capped by vouchers)
            interest_cap = self.voucher_bonuses.get('interest_cap', 5)
            interest = min(interest_cap, int(self.money * 0.2))
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

        # Reset for new blind (apply voucher bonuses)
        self.hands_remaining = self.config.starting_hands + self.voucher_bonuses.get('extra_hands', 0)
        self.discards_remaining = self.config.starting_discards + self.voucher_bonuses.get('extra_discards', 0)

        # Return hand cards to deck before reset
        remaining_cards = self.hand.clear()
        self.deck.discard(remaining_cards)
        self.deck.reset()

        return True

    def add_joker(self, joker: dict, source: str = "starting") -> bool:
        """Add a joker if there's room."""
        max_slots = self.config.joker_slots + self.voucher_bonuses.get('extra_joker_slots', 0)
        if len(self.jokers) >= max_slots:
            return False
        self.jokers.append(joker)
        self.history.add_joker_acquired(self.ante, joker.get("name", "Unknown"), source)
        return True

    def add_voucher(self, voucher_name: str, voucher_effect: dict) -> None:
        """Add a voucher and apply its effects."""
        self.vouchers_owned.append(voucher_name)

        # Apply voucher effects to bonuses
        for key, value in voucher_effect.items():
            if key == 'extra_hands':
                self.voucher_bonuses['extra_hands'] += value
            elif key == 'extra_discards':
                self.voucher_bonuses['extra_discards'] += value
            elif key == 'starting_draw':  # Hand size
                self.voucher_bonuses['extra_hand_size'] += value
            elif key == 'joker_slots':
                self.voucher_bonuses['extra_joker_slots'] += value
            elif key == 'consumable_slots':
                self.voucher_bonuses['extra_consumable_slots'] += value
            elif key == 'shop_discount':
                self.voucher_bonuses['shop_discount'] += value
            elif key == 'reroll_discount':
                self.voucher_bonuses['reroll_discount'] += value
            elif key == 'interest_cap':
                self.voucher_bonuses['interest_cap'] = max(
                    self.voucher_bonuses['interest_cap'], value
                )

    def level_up_hand(self, hand_type: HandType) -> None:
        """Increase the level of a hand type."""
        self.hand_levels[hand_type] = self.hand_levels.get(hand_type, 1) + 1
        self.hand_detector.hand_levels = self.hand_levels


class BasicStrategy:
    """
    Simple AI strategy for playing Balatro.
    Used for simulations.
    """

    def select_cards_to_play(self, hand: Hand, game: GameState, must_play_count: int = None) -> list[int]:
        """
        Select which cards to play.
        Returns indices of cards to play.

        Args:
            must_play_count: If set (e.g., The Psychic boss), must play exactly this many cards
        """
        cards = hand.cards
        if not cards:
            return []

        # Determine how many cards to play
        play_count = must_play_count if must_play_count else 5

        # Try to find the best hand from available cards
        best_score = 0
        best_indices = []

        # Try different combinations (simplified - just try obvious ones)
        detector = game.hand_detector

        # Try playing cards (up to play_count)
        all_indices = list(range(min(play_count, len(cards))))
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
                indices = [i for i, c in enumerate(cards) if c.rank == rank][:min(play_count, count)]
                # If must_play_count, pad with other cards
                if must_play_count and len(indices) < must_play_count:
                    other_indices = [i for i in range(len(cards)) if i not in indices]
                    indices.extend(other_indices[:must_play_count - len(indices)])
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
            best_indices = sorted_indices[:min(play_count, len(cards))]

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


def use_consumables_before_blind(game: GameState) -> list[str]:
    """
    Strategically use stored consumables before a blind.
    Returns list of consumables used.
    """
    from .shop import apply_planet, apply_tarot, ConsumableType

    used = []
    score_req = game.get_score_requirement()
    is_boss = game.current_blind == BlindType.BOSS

    # Use planets if score requirement is tough (ante 4+) or facing boss
    if game.ante >= 4 or is_boss:
        planets_to_use = []
        for i, cons in enumerate(game.consumables):
            if hasattr(cons, 'type') and cons.type == ConsumableType.PLANET:
                planets_to_use.append(i)

        # Use planets (reverse order to not mess up indices)
        for i in reversed(planets_to_use):
            cons = game.consumables.pop(i)
            apply_planet(cons, game)
            used.append(cons.name)

    # Use strength tarots before boss blinds (those that give mult/chips to cards)
    if is_boss:
        tarots_to_use = []
        for i, cons in enumerate(game.consumables):
            if hasattr(cons, 'type') and cons.type == ConsumableType.TAROT:
                effect = getattr(cons, 'effect', {})
                # Use tarots that enhance cards
                if effect.get('enhancement') or effect.get('gives'):
                    tarots_to_use.append(i)

        for i in reversed(tarots_to_use):
            cons = game.consumables.pop(i)
            apply_tarot(cons, game)
            used.append(cons.name)

    return used


def simulate_blind(game: GameState, strategy = None) -> BlindResult:
    """Simulate playing a single blind."""
    if strategy is None:
        strategy = BasicStrategy()

    # Set up boss blind if this is a boss fight
    boss_name = None
    if game.current_blind == BlindType.BOSS:
        boss = get_boss_for_ante(game.ante)
        game.current_boss = BossBlindState(boss)
        boss_name = boss.name

        # Apply boss modifiers to starting resources
        # The Water: start with 0 discards
        game.discards_remaining = max(0, game.discards_remaining + game.current_boss.get_discard_modifier())
        # The Needle: only 1 hand
        game.hands_remaining = max(1, game.hands_remaining + game.current_boss.get_hands_modifier())
    else:
        game.current_boss = None

    score_required = game.get_score_requirement()
    total_score = 0
    hands_played = []

    game.draw_hand()

    while game.hands_remaining > 0 and total_score < score_required:
        # The Hook: discard random cards at start of each hand
        if game.current_boss:
            random_discards = game.current_boss.get_random_discards()
            if random_discards > 0 and game.hand.size() > random_discards:
                import random as rng
                discard_indices = rng.sample(range(game.hand.size()), min(random_discards, game.hand.size()))
                # Don't use normal discard (it decrements counter), just remove cards
                cards_to_remove = game.hand.select(discard_indices)
                removed = game.hand.remove(cards_to_remove)
                game.deck.discard(removed)

        # Optionally discard first
        if game.discards_remaining > 0 and game.hands_remaining > 1:
            discard_indices = strategy.select_cards_to_discard(game.hand, game)
            if discard_indices:
                game.discard_cards(discard_indices)

        # The Psychic: must play exactly 5 cards
        if game.current_boss and game.current_boss.must_play_5_cards():
            play_indices = strategy.select_cards_to_play(game.hand, game, must_play_count=5)
        else:
            play_indices = strategy.select_cards_to_play(game.hand, game)

        if not play_indices:
            break

        # Check hand type allowed (The Eye, The Mouth)
        if game.current_boss:
            # Preview what hand type would be played
            preview_cards = game.hand.select(play_indices)
            preview_hand = game.hand_detector.detect(preview_cards)
            if not game.current_boss.check_hand_allowed(preview_hand.hand_type.name):
                # Try to find a different hand, or just play anyway (AI limitation)
                pass  # For now, allow it - real AI improvement would find alternative

        detected, breakdown = game.play_cards(play_indices)
        total_score += breakdown.final_score
        hands_played.append((detected.hand_type.name, breakdown.final_score))

        # Record hand type for The Eye/The Mouth tracking
        if game.current_boss:
            game.current_boss.record_hand_played(detected.hand_type.name)

        # The Tooth: lose money per card played
        if game.current_boss:
            money_loss = game.current_boss.get_money_loss_per_card() * len(detected.all_cards)
            game.money = max(0, game.money - money_loss)

        # Draw new cards
        game.draw_hand()

    success = total_score >= score_required
    hands_used = game.config.starting_hands - game.hands_remaining
    discards_used = game.config.starting_discards - game.discards_remaining
    money_earned = game.end_blind(success)

    # Clear boss state after blind
    game.current_boss = None

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
        best_hand=best_hand,
        boss_name=boss_name,
        hands_played=hands_played
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
    from .shop import Shop, ShopAI, apply_planet, apply_tarot, apply_spectral, ConsumableType, PackType

    if shop_ai is None:
        shop_ai = ShopAI()

    shop = Shop(all_jokers)
    shop.generate(
        owned_jokers=game.jokers,
        owned_vouchers=game.vouchers_owned,
        ante=game.ante
    )

    decisions = shop_ai.decide_purchases(shop, game)
    results = {
        "jokers_bought": [],
        "consumables_bought": [],
        "vouchers_bought": [],
        "packs_opened": [],
        "planets_used": [],
        "money_spent": 0
    }

    # Buy vouchers first (permanent upgrades)
    for idx in decisions.get("vouchers_to_buy", []):
        if idx < len(shop.vouchers):
            voucher = shop.vouchers[idx]
            if voucher.cost <= game.money:
                game.money -= voucher.cost
                game.add_voucher(voucher.name, voucher.effect)
                results["vouchers_bought"].append(voucher.name)
                results["money_spent"] += voucher.cost

    # Buy and open packs
    for pack_idx in decisions.get("packs_to_buy", []):
        if pack_idx >= len(shop.packs):
            continue
        pack = shop.packs[pack_idx]
        if pack.cost > game.money:
            continue

        game.money -= pack.cost
        results["money_spent"] += pack.cost

        # Get chosen options for this pack
        chosen_indices = decisions.get("pack_choices", {}).get(pack_idx, [])
        if not chosen_indices and pack.options:
            # Default to first option if AI didn't choose
            chosen_indices = [0]

        pack_result = {"type": pack.pack_type.value, "choices": []}

        for opt_idx in chosen_indices:
            if opt_idx >= len(pack.options):
                continue
            option = pack.options[opt_idx]

            if pack.pack_type == PackType.CELESTIAL:
                # Apply planet
                hand_type = option.effect.get("levels_up")
                old_level = game.hand_levels.get(hand_type, 1) if hand_type else 1
                msg = apply_planet(option, game)
                results["planets_used"].append(msg)
                game.planets_used += 1
                pack_result["choices"].append(option.name)
                if hand_type:
                    game.history.add_planet_used(game.ante, option.name, hand_type.name, old_level + 1)

            elif pack.pack_type == PackType.ARCANA:
                # Apply tarot
                msg = apply_tarot(option, game)
                if "tarots_used" not in results:
                    results["tarots_used"] = []
                results["tarots_used"].append(msg)
                pack_result["choices"].append(option.name)

            elif pack.pack_type == PackType.SPECTRAL:
                # Apply spectral
                msg = apply_spectral(option, game, all_jokers)
                if "spectrals_used" not in results:
                    results["spectrals_used"] = []
                results["spectrals_used"].append(msg)
                pack_result["choices"].append(option.name)

            elif pack.pack_type == PackType.BUFFOON:
                # Add joker
                extra_slots = game.voucher_bonuses.get('extra_joker_slots', 0)
                max_slots = game.config.joker_slots + extra_slots
                if len(game.jokers) < max_slots:
                    game.jokers.append(option)
                    joker_name = option.get("name", "Unknown")
                    results["jokers_bought"].append(joker_name)
                    pack_result["choices"].append(joker_name)
                    game.history.add_joker_acquired(game.ante, joker_name, "buffoon_pack")

            elif pack.pack_type == PackType.STANDARD:
                # Add card to deck
                game.deck.cards.append(option)
                card_str = str(option)
                pack_result["choices"].append(card_str)

        results["packs_opened"].append(pack_result)

        # Log pack opening with full details
        option_names = []
        for opt in pack.options:
            if hasattr(opt, 'name'):
                option_names.append(opt.name)
            elif isinstance(opt, dict):
                option_names.append(opt.get('name', str(opt)))
            else:
                option_names.append(str(opt))

        game.history.add_pack_opened(
            ante=game.ante,
            pack_type=pack.pack_type.value,
            is_mega=pack.is_mega,
            options=option_names,
            choices=pack_result["choices"]
        )

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
                elif consumable.type == ConsumableType.TAROT:
                    # Use tarots that provide immediate benefit
                    effect = consumable.effect
                    # Use Hermit (money) immediately
                    if effect.get("double_money") or effect.get("random_edition"):
                        msg = apply_tarot(consumable, game)
                        if "tarots_used" not in results:
                            results["tarots_used"] = []
                        results["tarots_used"].append(msg)
                    else:
                        # Store other tarots for later use
                        extra_slots = game.voucher_bonuses.get('extra_consumable_slots', 0)
                        max_slots = game.config.consumable_slots + extra_slots
                        if len(game.consumables) < max_slots:
                            game.consumables.append(consumable)
                elif consumable.type == ConsumableType.SPECTRAL:
                    # Use spectrals that provide immediate benefit
                    effect = consumable.effect
                    # Use money-generating spectrals immediately
                    if effect.get("gain_money") or effect.get("destroy_for_money") or effect.get("level_up_all"):
                        msg = apply_spectral(consumable, game, all_jokers)
                        if "spectrals_used" not in results:
                            results["spectrals_used"] = []
                        results["spectrals_used"].append(msg)
                    else:
                        # Store for later
                        extra_slots = game.voucher_bonuses.get('extra_consumable_slots', 0)
                        max_slots = game.config.consumable_slots + extra_slots
                        if len(game.consumables) < max_slots:
                            game.consumables.append(consumable)
                else:
                    # Store other consumables if room
                    extra_slots = game.voucher_bonuses.get('extra_consumable_slots', 0)
                    max_slots = game.config.consumable_slots + extra_slots
                    if len(game.consumables) < max_slots:
                        game.consumables.append(consumable)

    # Log shop visit with all consumable types
    game.history.add_shop_visit(
        ante=game.ante,
        jokers_bought=results["jokers_bought"],
        planets_used=[p.split(":")[0].strip() for p in results["planets_used"]],
        money_spent=results["money_spent"],
        money_remaining=game.money,
        vouchers_bought=results.get("vouchers_bought"),
        tarots_used=results.get("tarots_used"),
        spectrals_used=results.get("spectrals_used"),
        packs_opened=results.get("packs_opened")
    )

    # Log individual voucher acquisitions for narrative
    for voucher_name in results.get("vouchers_bought", []):
        game.history.add_voucher_acquired(game.ante, voucher_name)

    return results


def simulate_run(jokers: list = None, config: GameConfig = None, strategy=None,
                 all_jokers: list = None, verbose: bool = False,
                 enable_skips: bool = True) -> RunResult:
    """Simulate a complete Balatro run with skip decisions and pivot detection."""
    from .strategy import SkipDecisionAI, PivotDetector, BuildDetector

    game = GameState(config=config)
    if strategy is None:
        strategy = BasicStrategy()

    # Initialize strategic AI components
    skip_ai = SkipDecisionAI()
    pivot_detector = PivotDetector()
    build_detector = BuildDetector()

    # Add starting jokers
    if jokers:
        for joker in jokers[:game.config.joker_slots]:
            game.add_joker(joker)

    # For shop simulation, we need the full joker list
    enable_shop = game.config.enable_shop and all_jokers is not None

    blinds_beaten = 0

    while True:
        # Get ante preview (see all upcoming blinds)
        preview = game.get_ante_preview()

        # Get current blind info
        if game.current_blind == BlindType.SMALL:
            current_blind_info = preview.small_blind
        elif game.current_blind == BlindType.BIG:
            current_blind_info = preview.big_blind
        else:
            current_blind_info = preview.boss_blind

        # Check if we should skip this blind (not boss)
        if enable_skips and game.current_blind != BlindType.BOSS:
            if skip_ai.should_skip(game, current_blind_info, preview):
                tag = current_blind_info.skip_tag
                reward = game.skip_blind(tag)
                if verbose:
                    print(f"Ante {game.ante}: SKIPPED {current_blind_info.blind_type.name} - {tag.value}")
                    if reward.get("money"):
                        print(f"  Reward: +${reward['money']}")
                continue  # Move to next blind without playing

        # Use stored consumables strategically before blind
        consumables_used = use_consumables_before_blind(game)
        if verbose and consumables_used:
            print(f"  Used before blind: {consumables_used}")

        result = simulate_blind(game, strategy)

        # Record result for pivot detection
        pivot_detector.record_blind_result(result.score_achieved, result.score_required)

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

        # Check for pivot after each blind
        should_pivot, old_build, new_build = pivot_detector.should_pivot(game)
        if should_pivot:
            if verbose:
                print(f"  PIVOT: {old_build} -> {new_build}")
            game.history.add_pivot(game.ante, old_build, new_build, "struggling")

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
