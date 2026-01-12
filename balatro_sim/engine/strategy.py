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

    def select_play(self, hand: Hand, jokers: list = None,
                    score_to_beat: int = 0, hands_left: int = 4) -> list[int]:
        """Just play the best 5-card hand available."""
        cards = hand.cards
        if len(cards) <= 5:
            return list(range(len(cards)))

        # Try all 5-card combinations, pick the best
        best_score = -1
        best_indices = list(range(5))

        for indices in combinations(range(len(cards)), 5):
            selected = [cards[i] for i in indices]
            detected = self.hand_detector.detect(selected)
            breakdown = self.scoring_engine.score_hand(detected, jokers or [])

            if breakdown.final_score > best_score:
                best_score = breakdown.final_score
                best_indices = list(indices)

        return best_indices

    def select_discard(self, hand: Hand, jokers: list = None,
                       discards_left: int = 3) -> list[int]:
        """Discard lowest value cards."""
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
