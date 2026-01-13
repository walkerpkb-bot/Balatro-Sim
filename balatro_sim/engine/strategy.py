"""
Smart card selection strategies for Balatro simulation.
"""

from itertools import combinations
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .deck import Card, Hand, Suit, RANK_ORDER
from .hand_detector import HandDetector, HandType, DetectedHand
from .scoring import ScoringEngine, ScoreBreakdown


@dataclass
class PlayOption:
    """A possible play with its expected score."""
    indices: list[int]
    cards: list[Card]
    hand_type: HandType
    score: int
    breakdown: ScoreBreakdown


class BasicStrategy:
    """
    Simple strategy that plays the highest scoring 5-card hand.
    Doesn't consider discards strategically.
    """

    def __init__(self):
        self.hand_detector = HandDetector()
        self.scoring_engine = ScoringEngine()

    def select_cards_to_play(self, hand: Hand, game, must_play_count: int = None) -> list[int]:
        """Just play the best 5-card hand available."""
        cards = hand.cards
        jokers = getattr(game, 'jokers', [])

        if len(cards) <= 5:
            return list(range(len(cards)))

        num_cards = must_play_count or 5

        # Try all N-card combinations, pick the best
        best_score = -1
        best_indices = list(range(min(num_cards, len(cards))))

        for indices in combinations(range(len(cards)), min(num_cards, len(cards))):
            selected = [cards[i] for i in indices]
            detected = self.hand_detector.detect(selected)
            breakdown = self.scoring_engine.score_hand(detected, jokers or [])

            if breakdown.final_score > best_score:
                best_score = breakdown.final_score
                best_indices = list(indices)

        return best_indices

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """Discard lowest value cards."""
        discards_left = getattr(game, 'discards_remaining', 0)
        if discards_left <= 0 or len(hand.cards) <= 5:
            return []

        # Just discard the 3 lowest-value cards
        cards = hand.cards
        sorted_indices = sorted(range(len(cards)),
                               key=lambda i: cards[i].chip_value)
        return sorted_indices[:min(3, len(cards) - 5)]


class SmartStrategy:
    """
    Intelligent card selection strategy.
    Evaluates all possible plays and picks the best one.
    Considers joker synergies and hand levels for optimal selection.
    """

    def __init__(self):
        self.hand_detector = HandDetector()
        self.scoring_engine = ScoringEngine()

    def _count_joker_synergies(self, cards: list[Card], jokers: list) -> int:
        """Count how many joker conditions this card selection triggers."""
        if not jokers:
            return 0

        synergy_count = 0
        suits_in_play = {c.suit.value.lower() for c in cards}
        ranks_in_play = {c.rank for c in cards}
        has_face = any(c.is_face_card for c in cards)

        for joker in jokers:
            effect = joker.get('effect', {})
            conditions = effect.get('conditions', [])

            for cond in conditions:
                cond_type = cond.get('type', '')

                if cond_type == 'suit':
                    req_suit = cond.get('suit', '').lower()
                    if any(req_suit in s for s in suits_in_play):
                        synergy_count += 1

                elif cond_type == 'rank':
                    req_rank = cond.get('rank', '')
                    if req_rank == 'face_card' and has_face:
                        synergy_count += 1
                    elif req_rank in ranks_in_play:
                        synergy_count += 1

            # Special jokers that benefit from specific cards
            name = joker.get('name', '')
            if name == 'Hack' and any(r in ranks_in_play for r in ('2', '3', '4', '5')):
                synergy_count += 1
            elif name == 'Sock and Buskin' and has_face:
                synergy_count += 1
            elif name == 'Splash':
                synergy_count += len(cards)  # More cards = more chips with Splash

        return synergy_count

    def evaluate_all_plays(self, hand: Hand, jokers: list = None,
                           min_cards: int = 1, max_cards: int = 5,
                           hand_levels: dict = None) -> list[PlayOption]:
        """
        Evaluate all possible card combinations and return scored options.
        Considers joker synergies and hand levels for smarter ranking.
        """
        cards = hand.cards
        jokers = jokers or []
        hand_levels = hand_levels or {}
        options = []

        # Try all combinations from min_cards to max_cards
        for num_cards in range(min_cards, min(max_cards + 1, len(cards) + 1)):
            for indices in combinations(range(len(cards)), num_cards):
                selected = [cards[i] for i in indices]
                detected = self.hand_detector.detect(selected)
                breakdown = self.scoring_engine.score_hand(detected, jokers)

                options.append(PlayOption(
                    indices=list(indices),
                    cards=selected,
                    hand_type=detected.hand_type,
                    score=breakdown.final_score,
                    breakdown=breakdown
                ))

        # Sort by effective score (base score + synergy bonus + level bonus)
        def effective_score(opt: PlayOption) -> float:
            base = opt.score
            # Joker synergy bonus: +5% per synergy triggered
            synergies = self._count_joker_synergies(opt.cards, jokers)
            synergy_bonus = base * 0.05 * synergies
            # Hand level bonus: +10% per level above 1
            level = hand_levels.get(opt.hand_type.name, 1)
            level_bonus = base * 0.10 * (level - 1)
            # Card economy: slight preference for fewer cards at similar scores
            economy_bonus = (5 - len(opt.cards)) * 10
            return base + synergy_bonus + level_bonus + economy_bonus

        options.sort(key=effective_score, reverse=True)
        return options

    def select_cards_to_play(self, hand: Hand, game, must_play_count: int = None) -> list[int]:
        """
        Select the best cards to play.
        Evaluates all combinations and picks highest scoring.
        Considers joker synergies and hand levels for optimal selection.

        Args:
            must_play_count: If set (e.g., The Psychic boss), must play exactly this many cards
        """
        if not hand.cards:
            return []

        # If boss requires exactly N cards, only evaluate N-card plays
        if must_play_count:
            min_cards = must_play_count
            max_cards = must_play_count
        else:
            min_cards = 1
            max_cards = 5

        # Get hand levels from game state
        hand_levels = getattr(game, 'hand_levels', {})

        options = self.evaluate_all_plays(
            hand,
            jokers=game.jokers,
            min_cards=min_cards,
            max_cards=max_cards,
            hand_levels=hand_levels
        )

        if not options:
            return []

        return options[0].indices

    def _get_joker_preferred_suits(self, jokers: list) -> set:
        """Get suits that jokers give bonuses for."""
        preferred = set()
        for joker in jokers:
            effect = joker.get('effect', {})
            for cond in effect.get('conditions', []):
                if cond.get('type') == 'suit':
                    preferred.add(cond.get('suit', '').lower())
        return preferred

    def _get_joker_preferred_ranks(self, jokers: list) -> set:
        """Get ranks that jokers give bonuses for."""
        preferred = set()
        for joker in jokers:
            effect = joker.get('effect', {})
            for cond in effect.get('conditions', []):
                if cond.get('type') == 'rank':
                    rank = cond.get('rank', '')
                    if rank == 'face_card':
                        preferred.update(['J', 'Q', 'K'])
                    else:
                        preferred.add(rank)
            # Special jokers
            name = joker.get('name', '')
            if name == 'Hack':
                preferred.update(['2', '3', '4', '5'])
            elif name in ('Sock and Buskin', 'Baron', 'Shoot the Moon'):
                preferred.update(['J', 'Q', 'K'])
        return preferred

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """
        Smart discard strategy.
        Considers joker synergies, hand levels, and draw potential.
        """
        if game.discards_remaining <= 0:
            return []

        cards = hand.cards
        if len(cards) <= 3:
            return []

        rank_counts = Counter(c.rank for c in cards)
        suit_counts = Counter(c.suit for c in cards)
        hand_levels = getattr(game, 'hand_levels', {})
        jokers = game.jokers or []

        # Find the best current play
        best_options = self.evaluate_all_plays(hand, jokers, min_cards=1, max_cards=5, hand_levels=hand_levels)
        if not best_options:
            return []

        best_play = best_options[0]
        best_indices = set(best_play.indices)

        # Score needed vs what we can make
        score_needed = game.get_score_requirement()
        hands_left = game.hands_remaining
        score_per_hand_needed = score_needed / max(1, hands_left)

        # If our best play is good enough, don't discard
        if best_play.score >= score_per_hand_needed * 1.5:
            return []

        # Get joker preferences
        preferred_suits = self._get_joker_preferred_suits(jokers)
        preferred_ranks = self._get_joker_preferred_ranks(jokers)

        # Check for draws worth chasing
        dominated_suit = max(suit_counts, key=suit_counts.get)
        flush_level = hand_levels.get('FLUSH', 1)
        flush_potential = suit_counts[dominated_suit] >= 3

        # Prioritize flush if leveled or joker prefers suit
        flush_priority = flush_level >= 2 or dominated_suit.value.lower() in preferred_suits

        # Score each card for discard priority (higher = more likely to discard)
        discard_scores = []
        for i, card in enumerate(cards):
            if i in best_indices:
                continue  # Never discard cards in best play

            score = 0

            # Lonely cards (no pairs) are discard candidates
            if rank_counts[card.rank] == 1:
                score += 50

            # Cards not helping flush draw
            if flush_potential and card.suit != dominated_suit:
                score += 30 if flush_priority else 15

            # Low chip value cards more expendable
            score += max(0, 11 - card.chip_value)

            # BUT protect joker-synergy cards
            if card.suit.value.lower() in preferred_suits:
                score -= 40
            if card.rank in preferred_ranks:
                score -= 40
            if card.is_face_card and any(j.get('name') in ('Sock and Buskin', 'Baron') for j in jokers):
                score -= 50

            discard_scores.append((i, score))

        # Sort by discard score (highest = discard first)
        discard_scores.sort(key=lambda x: x[1], reverse=True)

        # Only discard cards with positive scores
        candidates = [(i, s) for i, s in discard_scores if s > 0]

        # Discard up to 3 cards
        max_discard = min(3, len(candidates))
        if hands_left <= 2:
            max_discard = min(2, max_discard)

        return [idx for idx, _ in candidates[:max_discard]]

    def should_discard(self, hand: Hand, game) -> bool:
        """
        Decide whether to discard at all.
        """
        if game.discards_remaining <= 0:
            return False

        hand_levels = getattr(game, 'hand_levels', {})

        # Evaluate current best play
        best_options = self.evaluate_all_plays(hand, game.jokers, min_cards=1, max_cards=5, hand_levels=hand_levels)
        if not best_options:
            return False

        best_score = best_options[0].score
        score_needed = game.get_score_requirement()
        hands_left = game.hands_remaining

        # If we can easily beat the requirement, don't bother discarding
        if best_score >= score_needed / hands_left * 2:
            return False

        # If this is our last hand, don't discard
        if hands_left <= 1:
            return False

        # Check if we have discard candidates
        discard_indices = self.select_cards_to_discard(hand, game)
        return len(discard_indices) > 0


class OptimizedStrategy(SmartStrategy):
    """
    Further optimized strategy with flush/straight chasing.
    """

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """
        Enhanced discard that considers flush/straight potential.
        """
        if game.discards_remaining <= 0:
            return []

        cards = hand.cards
        if len(cards) <= 3:
            return []

        rank_counts = Counter(c.rank for c in cards)
        suit_counts = Counter(c.suit for c in cards)

        # Check for flush draw (4 of same suit)
        best_suit = max(suit_counts, key=suit_counts.get)
        flush_draw = suit_counts[best_suit] == 4

        # Check for straight draw
        ranks_present = set(c.rank for c in cards)
        straight_draw = self._check_straight_draw(ranks_present)

        # Check for strong made hands
        hand_levels = getattr(game, 'hand_levels', {})
        best_options = self.evaluate_all_plays(hand, game.jokers, min_cards=1, max_cards=5, hand_levels=hand_levels)
        if not best_options:
            return []

        best_play = best_options[0]
        best_type = best_play.hand_type

        # Don't discard if we have a strong hand already
        strong_hands = {HandType.FLUSH, HandType.STRAIGHT, HandType.FULL_HOUSE,
                       HandType.FOUR_OF_A_KIND, HandType.STRAIGHT_FLUSH}
        if best_type in strong_hands:
            return []

        # If we have a flush draw, discard non-flush cards
        if flush_draw:
            discard_indices = []
            for i, card in enumerate(cards):
                if card.suit != best_suit and rank_counts[card.rank] == 1:
                    discard_indices.append(i)
            if discard_indices:
                return discard_indices[:3]

        # If we have a straight draw, keep the straight cards
        if straight_draw:
            straight_ranks = self._get_straight_draw_ranks(ranks_present)
            discard_indices = []
            for i, card in enumerate(cards):
                if card.rank not in straight_ranks and rank_counts[card.rank] == 1:
                    discard_indices.append(i)
            if discard_indices:
                return discard_indices[:3]

        # Fall back to parent logic
        return super().select_cards_to_discard(hand, game)

    def _check_straight_draw(self, ranks: set) -> bool:
        """Check if we have 4 cards to a straight."""
        orders = sorted([RANK_ORDER[r] for r in ranks])

        for i in range(len(orders) - 3):
            window = orders[i:i+4]
            if window[-1] - window[0] <= 4:  # 4 cards within 5 positions
                gaps = sum(1 for j in range(3) if window[j+1] - window[j] > 1)
                if gaps <= 1:
                    return True
        return False

    def _get_straight_draw_ranks(self, ranks: set) -> set:
        """Get ranks that are part of the straight draw."""
        orders = sorted([RANK_ORDER[r] for r in ranks])
        rank_list = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        best_window = []
        for i in range(len(orders) - 3):
            window = orders[i:i+4]
            if window[-1] - window[0] <= 4:
                if len(window) > len(best_window):
                    best_window = window

        return {rank_list[o] for o in best_window}


class AggressiveStrategy(SmartStrategy):
    """
    Strategy that prioritizes high-scoring plays even if risky.
    Good for when you're behind on score.
    """

    def select_cards_to_play(self, hand: Hand, game, must_play_count: int = None) -> list[int]:
        """Always play the maximum scoring option."""
        hand_levels = getattr(game, 'hand_levels', {})
        if must_play_count:
            options = self.evaluate_all_plays(hand, game.jokers, min_cards=must_play_count, max_cards=must_play_count, hand_levels=hand_levels)
        else:
            options = self.evaluate_all_plays(hand, game.jokers, min_cards=1, max_cards=5, hand_levels=hand_levels)

        if not options:
            return []

        # Filter to only 5-card plays for maximum scoring potential (unless boss forces different)
        if not must_play_count:
            five_card_options = [o for o in options if len(o.cards) == 5]
            if five_card_options:
                return five_card_options[0].indices

        return options[0].indices

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """Aggressively discard to chase big hands."""
        if game.discards_remaining <= 0 or game.hands_remaining <= 1:
            return []

        cards = hand.cards
        if not cards:
            return []

        suit_counts = Counter(c.suit for c in cards)
        rank_counts = Counter(c.rank for c in cards)

        if not suit_counts:
            return []

        # Chase flushes aggressively
        best_suit = max(suit_counts, key=suit_counts.get)
        if suit_counts[best_suit] >= 3:
            discard_indices = [i for i, c in enumerate(cards)
                             if c.suit != best_suit and rank_counts[c.rank] == 1]
            return discard_indices[:min(4, len(discard_indices))]

        # Chase pairs/trips
        good_ranks = {r for r, c in rank_counts.items() if c >= 2}
        if good_ranks:
            discard_indices = [i for i, c in enumerate(cards)
                             if c.rank not in good_ranks]
            # Keep at least 3 cards
            max_discard = len(cards) - 3
            return discard_indices[:min(3, max_discard)]

        return super().select_cards_to_discard(hand, game)


# ============================================================================
# BUILD ARCHETYPES AND STRATEGIC PLANNING
# ============================================================================

class BuildArchetype:
    """Identified build archetypes."""
    FLUSH = "flush"           # Focused on flush hands
    STRAIGHT = "straight"     # Focused on straights
    PAIRS = "pairs"           # Two pair, full house focused
    MULT_STACK = "mult_stack" # High mult jokers
    CHIP_STACK = "chip_stack" # High chip jokers
    RETRIGGER = "retrigger"   # Retrigger-focused (Hack, Dusk, etc)
    ECONOMY = "economy"       # Money generation focused
    HYBRID = "hybrid"         # Mixed strategy


class BuildDetector:
    """Detects current build archetype from game state."""

    def detect_build(self, game) -> tuple[str, float]:
        """
        Analyze game state and return (archetype, confidence).
        Confidence is 0-1 indicating how strong the build is.
        """
        scores = {
            BuildArchetype.FLUSH: self._score_flush_build(game),
            BuildArchetype.STRAIGHT: self._score_straight_build(game),
            BuildArchetype.PAIRS: self._score_pairs_build(game),
            BuildArchetype.MULT_STACK: self._score_mult_build(game),
            BuildArchetype.RETRIGGER: self._score_retrigger_build(game),
            BuildArchetype.ECONOMY: self._score_economy_build(game),
        }

        best_archetype = max(scores, key=scores.get)
        best_score = scores[best_archetype]

        # Confidence based on how dominant the best archetype is
        total = sum(scores.values()) or 1
        confidence = best_score / total

        if confidence < 0.3:
            return BuildArchetype.HYBRID, confidence

        return best_archetype, confidence

    def _score_flush_build(self, game) -> float:
        score = 0
        # Check hand levels
        flush_level = game.hand_levels.get(HandType.FLUSH, 1)
        score += (flush_level - 1) * 10

        # Check for flush-synergy jokers
        for joker in game.jokers:
            effect = joker.get('effect', {})
            for cond in effect.get('conditions', []):
                if cond.get('type') == 'suit':
                    score += 15
                if cond.get('type') == 'hand_contains' and 'flush' in cond.get('hand', '').lower():
                    score += 20

        return score

    def _score_straight_build(self, game) -> float:
        score = 0
        straight_level = game.hand_levels.get(HandType.STRAIGHT, 1)
        score += (straight_level - 1) * 10

        # Four Fingers, Shortcut jokers
        for joker in game.jokers:
            name = joker.get('name', '')
            if name in ('Four Fingers', 'Shortcut'):
                score += 25

        return score

    def _score_pairs_build(self, game) -> float:
        score = 0
        pair_level = game.hand_levels.get(HandType.PAIR, 1)
        two_pair_level = game.hand_levels.get(HandType.TWO_PAIR, 1)
        fh_level = game.hand_levels.get(HandType.FULL_HOUSE, 1)

        score += (pair_level - 1) * 5
        score += (two_pair_level - 1) * 8
        score += (fh_level - 1) * 12

        return score

    def _score_mult_build(self, game) -> float:
        score = 0
        for joker in game.jokers:
            effect = joker.get('effect', {})
            for mod in effect.get('modifiers', []):
                if mod.get('type') == 'x_mult':
                    score += mod.get('value', 1) * 20
                elif mod.get('type') == 'add_mult':
                    score += mod.get('value', 0) * 2

        return score

    def _score_retrigger_build(self, game) -> float:
        score = 0
        retrigger_jokers = {'Hack', 'Dusk', 'Sock and Buskin', 'Seltzer', 'Hanging Chad'}
        for joker in game.jokers:
            if joker.get('name') in retrigger_jokers:
                score += 25

        return score

    def _score_economy_build(self, game) -> float:
        score = 0
        for joker in game.jokers:
            effect = joker.get('effect', {})
            for mod in effect.get('modifiers', []):
                if mod.get('type') in ('earn_money', 'give_money'):
                    score += mod.get('value', 0) * 5

        return score


class SkipDecisionAI:
    """Decides whether to skip blinds based on game state and preview."""

    def __init__(self):
        self.build_detector = BuildDetector()

    def should_skip(self, game, blind_info, preview) -> bool:
        """
        Decide whether to skip the current blind.
        """
        # Never skip boss blinds
        if blind_info.blind_type.name == 'BOSS':
            return False

        # Get current build strength
        archetype, confidence = self.build_detector.detect_build(game)

        skip_score = 0

        # Tag value
        tag_value = self._evaluate_tag(blind_info.skip_tag, game, archetype)
        skip_score += tag_value

        # Money considerations
        if game.money >= 20:
            skip_score += 10
        elif game.money < 5:
            skip_score -= 20

        # Boss consideration
        boss = preview.boss_blind
        if boss.boss_name in ('The Needle', 'The Water', 'The Psychic', 'The Eye'):
            skip_score += 5

        # Build confidence
        if confidence > 0.6:
            skip_score += 10
        elif confidence < 0.3:
            skip_score -= 10

        # Early game penalty
        if game.ante <= 2:
            skip_score -= 15

        # Late game bonus
        if game.ante >= 5:
            skip_score += 5

        return skip_score > 15

    def _evaluate_tag(self, tag, game, archetype: str) -> float:
        """Score a tag's value for current game state."""
        from .game import Tag

        base_values = {
            Tag.SKIP: 5,
            Tag.UNCOMMON: 15,
            Tag.RARE: 25,
            Tag.NEGATIVE: 35,
            Tag.CHARM: 12,
            Tag.METEOR: 15,
            Tag.STANDARD: 8,
            Tag.BUFFOON: 18,
            Tag.HANDY: 0,
            Tag.D6: 10,
            Tag.COUPON: 20,
            Tag.JUGGLE: 8,
        }

        value = base_values.get(tag, 5)

        # Adjust based on build
        if tag == Tag.METEOR and archetype in (BuildArchetype.FLUSH, BuildArchetype.STRAIGHT):
            value += 10
        if tag == Tag.BUFFOON and len(game.jokers) < game.config.joker_slots:
            value += 10
        if tag == Tag.HANDY:
            value = game.total_hands_played

        return value


class PivotDetector:
    """Detects when a build isn't working and suggests pivots."""

    def __init__(self):
        self.build_detector = BuildDetector()
        self.recent_scores = []

    def record_blind_result(self, score_achieved: int, score_required: int):
        """Record a blind result for analysis."""
        margin = score_achieved / max(score_required, 1)
        self.recent_scores.append(margin)
        if len(self.recent_scores) > 6:
            self.recent_scores.pop(0)

    def should_pivot(self, game) -> tuple[bool, str, str]:
        """
        Check if current build should be pivoted.
        Returns (should_pivot, current_build, suggested_build).
        """
        current_build, confidence = self.build_detector.detect_build(game)

        if len(self.recent_scores) < 3:
            return False, current_build, current_build

        avg_margin = sum(self.recent_scores) / len(self.recent_scores)
        recent_margin = sum(self.recent_scores[-3:]) / 3

        struggling = recent_margin < 1.3 or (recent_margin < avg_margin * 0.8)

        if not struggling:
            return False, current_build, current_build

        suggested = self._suggest_pivot(game, current_build)

        if suggested != current_build:
            return True, current_build, suggested

        return False, current_build, current_build

    def _suggest_pivot(self, game, current: str) -> str:
        """Suggest a pivot based on available resources."""
        level_scores = {}
        for ht, level in game.hand_levels.items():
            if level > 1:
                if 'FLUSH' in ht.name:
                    level_scores[BuildArchetype.FLUSH] = level_scores.get(BuildArchetype.FLUSH, 0) + level
                elif 'STRAIGHT' in ht.name:
                    level_scores[BuildArchetype.STRAIGHT] = level_scores.get(BuildArchetype.STRAIGHT, 0) + level
                elif ht.name in ('PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'FULL_HOUSE', 'FOUR_OF_A_KIND'):
                    level_scores[BuildArchetype.PAIRS] = level_scores.get(BuildArchetype.PAIRS, 0) + level

        if current in level_scores:
            del level_scores[current]

        if level_scores:
            return max(level_scores, key=level_scores.get)

        return BuildArchetype.PAIRS if current != BuildArchetype.PAIRS else BuildArchetype.HYBRID
