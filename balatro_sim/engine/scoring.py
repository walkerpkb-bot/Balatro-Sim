"""
Scoring engine for Balatro simulation.
Calculates final score from cards, jokers, and modifiers.
"""

from dataclasses import dataclass, field
from typing import Callable

from .deck import Card, Enhancement, Edition, Seal
from .hand_detector import DetectedHand, HandType


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how score was calculated."""
    hand_type: HandType
    base_chips: int
    base_mult: int
    card_chips: int
    bonus_chips: int
    bonus_mult: int
    x_mult: float
    final_chips: int
    final_mult: float
    final_score: int
    details: list[str] = field(default_factory=list)

    def add_detail(self, msg: str):
        self.details.append(msg)


@dataclass
class ScoringContext:
    """Context passed through scoring pipeline."""
    hand: DetectedHand
    chips: int = 0
    mult: float = 0
    x_mult: float = 1.0
    money_earned: int = 0
    cards_triggered: list[Card] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def add_chips(self, amount: int, source: str = ""):
        self.chips += amount
        if source:
            self.details.append(f"+{amount} Chips ({source})")

    def add_mult(self, amount: float, source: str = ""):
        self.mult += amount
        if source:
            self.details.append(f"+{amount} Mult ({source})")

    def multiply_mult(self, factor: float, source: str = ""):
        self.x_mult *= factor
        if source:
            self.details.append(f"x{factor} Mult ({source})")

    def add_money(self, amount: int, source: str = ""):
        self.money_earned += amount
        if source:
            self.details.append(f"+${amount} ({source})")


class ScoringEngine:
    """
    Calculates scores following Balatro's scoring rules.

    Score = (Base Chips + Card Chips + Bonus Chips) × (Base Mult + Bonus Mult) × X-Mult
    """

    def __init__(self):
        self.joker_scorers: list[Callable] = []

    def register_joker_scorer(self, scorer: Callable):
        """Register a joker scoring function."""
        self.joker_scorers.append(scorer)

    def score_hand(self, hand: DetectedHand, jokers: list = None,
                   all_cards_score: bool = False,
                   held_cards: list = None) -> ScoreBreakdown:
        """
        Calculate the score for a played hand.

        Args:
            hand: The detected poker hand
            jokers: List of active jokers (with parsed effects)
            all_cards_score: If True, all played cards contribute chips (Splash joker)
            held_cards: Cards still held in hand (for Baron, Steel, etc.)
        """
        ctx = ScoringContext(hand=hand)
        held_cards = held_cards or []

        # 1. Base chips and mult from hand type
        ctx.add_chips(hand.base_chips, f"{hand.hand_type.name} base")
        ctx.add_mult(hand.base_mult, f"{hand.hand_type.name} base")

        # 2. Add chips from scoring cards
        scoring_cards = hand.all_cards if all_cards_score else hand.scoring_cards
        for card in scoring_cards:
            self._score_card(card, ctx)

        # 3. Score held-in-hand effects (Steel cards, etc.)
        for card in held_cards:
            if card.enhancement == Enhancement.STEEL:
                ctx.multiply_mult(1.5, f"{card} Steel (held)")

        # 4. Apply joker effects (if any)
        jokers = jokers or []
        for joker in jokers:
            self._apply_joker(joker, ctx, held_cards=held_cards)

        # 4. Calculate final score
        final_chips = ctx.chips
        final_mult = ctx.mult * ctx.x_mult
        final_score = int(final_chips * final_mult)

        return ScoreBreakdown(
            hand_type=hand.hand_type,
            base_chips=hand.base_chips,
            base_mult=hand.base_mult,
            card_chips=sum(c.chip_value for c in scoring_cards),
            bonus_chips=ctx.chips - hand.base_chips - sum(c.chip_value for c in scoring_cards),
            bonus_mult=int(ctx.mult - hand.base_mult),
            x_mult=ctx.x_mult,
            final_chips=final_chips,
            final_mult=final_mult,
            final_score=final_score,
            details=ctx.details
        )

    def _score_card(self, card: Card, ctx: ScoringContext):
        """Add scoring from a single card."""
        # Base chip value
        ctx.add_chips(card.chip_value, f"{card}")

        # Enhancement bonuses
        if card.enhancement == Enhancement.BONUS:
            ctx.add_chips(30, f"{card} Bonus")
        elif card.enhancement == Enhancement.MULT:
            ctx.add_mult(4, f"{card} Mult")
        elif card.enhancement == Enhancement.GLASS:
            ctx.multiply_mult(2.0, f"{card} Glass")
        elif card.enhancement == Enhancement.LUCKY:
            # Lucky: 1/5 chance for +20 mult (simplified - always apply expected value)
            ctx.add_mult(4, f"{card} Lucky (avg)")
            # 1/15 chance for $20 - add expected $1.33
            ctx.add_money(1, f"{card} Lucky (avg)")

        # Edition bonuses
        if card.edition == Edition.FOIL:
            ctx.add_chips(50, f"{card} Foil")
        elif card.edition == Edition.HOLOGRAPHIC:
            ctx.add_mult(10, f"{card} Holo")
        elif card.edition == Edition.POLYCHROME:
            ctx.multiply_mult(1.5, f"{card} Poly")

        # Seal bonuses (Gold seal gives money, not score)
        if card.seal == Seal.GOLD:
            ctx.add_money(3, f"{card} Gold Seal")

        ctx.cards_triggered.append(card)

    def _apply_joker(self, joker: dict, ctx: ScoringContext, held_cards: list = None):
        """Apply a parsed joker's effects to the scoring context."""
        if not joker or 'effect' not in joker:
            return

        held_cards = held_cards or []
        effect = joker['effect']
        name = joker.get('name', 'Unknown')
        modifiers = effect.get('modifiers', [])
        conditions = effect.get('conditions', [])
        flags = effect.get('flags', [])

        # Special handling for held-in-hand jokers
        # Baron: Each King held in hand gives X1.5 Mult
        if name == "Baron":
            kings_held = sum(1 for c in held_cards if c.rank == "K")
            if kings_held > 0:
                for _ in range(kings_held):
                    ctx.multiply_mult(1.5, f"Baron (King held)")
            return

        # Shoot the Moon: Each Queen held in hand gives +13 Mult
        if name == "Shoot the Moon":
            queens_held = sum(1 for c in held_cards if c.rank == "Q")
            if queens_held > 0:
                ctx.add_mult(13 * queens_held, f"Shoot the Moon ({queens_held} Queens)")
            return

        # Raised Fist: Adds double the rank of lowest ranked card held to Mult
        if name == "Raised Fist" and held_cards:
            from .deck import RANK_VALUES
            min_rank_value = min(RANK_VALUES.get(c.rank, 0) for c in held_cards)
            ctx.add_mult(min_rank_value * 2, f"Raised Fist")
            return

        # Check conditions
        if not self._check_conditions(conditions, ctx):
            return

        # Apply modifiers
        for mod in modifiers:
            mod_type = mod.get('type', '')

            if mod_type == 'add_mult':
                ctx.add_mult(mod['value'], name)

            elif mod_type == 'add_chips':
                ctx.add_chips(mod['value'], name)

            elif mod_type == 'x_mult':
                ctx.multiply_mult(mod['value'], name)

            elif mod_type == 'random_mult':
                # Use expected value (average of min and max)
                avg = (mod['min'] + mod['max']) / 2
                ctx.add_mult(avg, f"{name} (avg)")

            elif mod_type == 'earn_money' or mod_type == 'give_money':
                ctx.add_money(mod['value'], name)

            elif mod_type == 'gains_mult':
                # Scaling joker - use base value (state tracking would improve this)
                ctx.add_mult(mod['value'], f"{name} (base)")

            elif mod_type == 'gains_x_mult':
                ctx.multiply_mult(1 + mod['value'], f"{name} (base)")

            elif mod_type == 'gains_chips':
                # Scaling chips joker
                ctx.add_chips(mod['value'], f"{name} (base)")

            elif mod_type == 'dynamic_mult':
                # Context-dependent mult (e.g., "Adds sell value of jokers to Mult")
                # Approximate with fixed value since we don't have full state
                ctx.add_mult(10, f"{name} (approx)")

            elif mod_type == 'all_cards_score':
                # Splash joker - handled via all_cards_score flag in score_hand
                pass

            elif mod_type == 'hand_size':
                # Hand size modifier - applied via game config, not scoring
                pass

            elif mod_type == 'hands':
                # Hands per round - applied via game config
                pass

            elif mod_type == 'discards':
                # Discards per round - applied via game config
                pass

            elif mod_type == 'hand_requirement_change':
                # Four Fingers - handled in hand_detector
                pass

            elif mod_type == 'straight_gaps':
                # Shortcut - handled in hand_detector
                pass

            elif mod_type == 'considered_as':
                # Pareidolia - handled in hand_detector
                pass

            elif mod_type == 'suit_equivalence':
                # Smeared Joker - handled in hand_detector
                pass

            elif mod_type == 'create_card':
                # Card generation - handled in shop/game loop
                pass

            elif mod_type == 'create_cards':
                # Multiple card generation - handled in shop/game loop
                pass

            elif mod_type == 'add_to_deck':
                # Deck modification - handled in game state
                pass

            elif mod_type == 'upgrade_hand_level':
                # Hand level up - handled in game state with planets
                pass

            elif mod_type == 'transform_card':
                # Midas Mask etc - card enhancement, handled separately
                pass

            elif mod_type == 'duplicate':
                # Copy joker - handled in game state
                pass

            elif mod_type == 'create_copy':
                # Perkeo etc - handled in game state
                pass

            elif mod_type == 'prevent_death':
                # Mr. Bones - would need death prevention logic
                pass

            elif mod_type == 'disable_effect':
                # Chicot - boss blind negation
                pass

            elif mod_type == 'double_probability':
                # Oops All 6s - probability doubling
                pass

            elif mod_type == 'free_reroll':
                # Chaos the Clown - shop modifier
                pass

            elif mod_type == 'free_items':
                # Astronomer - shop modifier
                pass

            elif mod_type == 'allow_duplicates':
                # Showman - shop modifier
                pass

            elif mod_type == 'debt_limit':
                # Credit Card - economy modifier
                pass

    def _check_conditions(self, conditions: list, ctx: ScoringContext) -> bool:
        """Check if all conditions are met."""
        for cond in conditions:
            cond_type = cond.get('type', '')

            if cond_type == 'hand_contains':
                required_hand = cond.get('hand', '')
                if required_hand.lower() not in ctx.hand.hand_type.name.lower().replace('_', ' '):
                    # Simple string matching - could be improved
                    return False

            elif cond_type == 'suit':
                required_suit = cond.get('suit', '')
                # Check if any scoring card has this suit
                has_suit = any(required_suit.lower() in c.suit.value.lower()
                             for c in ctx.hand.scoring_cards)
                if not has_suit:
                    return False

            elif cond_type == 'rank':
                required_rank = cond.get('rank', '')
                if required_rank == 'face_card':
                    has_face = any(c.is_face_card for c in ctx.hand.scoring_cards)
                    if not has_face:
                        return False
                else:
                    has_rank = any(c.rank == required_rank for c in ctx.hand.scoring_cards)
                    if not has_rank:
                        return False

            elif cond_type == 'max_cards':
                if len(ctx.hand.all_cards) > cond['value']:
                    return False

            elif cond_type == 'exact_cards':
                if len(ctx.hand.all_cards) != cond['value']:
                    return False

            elif cond_type == 'probability':
                # For simulation, use expected value (always "pass" with fractional effect)
                # The actual effect is scaled in the modifier application
                pass

        return True


def calculate_score(hand: DetectedHand, jokers: list = None, held_cards: list = None) -> int:
    """Convenience function to calculate score."""
    engine = ScoringEngine()
    breakdown = engine.score_hand(hand, jokers, held_cards=held_cards)
    return breakdown.final_score


def score_breakdown(hand: DetectedHand, jokers: list = None, held_cards: list = None) -> ScoreBreakdown:
    """Get detailed score breakdown."""
    engine = ScoringEngine()
    return engine.score_hand(hand, jokers, held_cards=held_cards)
