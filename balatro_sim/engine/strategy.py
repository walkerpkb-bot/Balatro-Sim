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


import json
from pathlib import Path


class CoachMemory:
    """
    Persistent memory for CoachStrategy.
    Tracks what worked and what didn't across runs.

    This is NOT machine learning - it's statistical bookkeeping:
    - Track joker appearances in wins vs losses
    - Track build lead success rates
    - Track which bosses kill runs

    Then use ratios to nudge future decisions.
    """

    DEFAULT_PATH = Path(__file__).parent.parent.parent / "coach_memory.json"

    def __init__(self, path: Path = None):
        self.path = path or self.DEFAULT_PATH
        self.data = self._load()

    def _load(self) -> dict:
        """Load memory from disk, or create fresh."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "total_runs": 0,
            "total_wins": 0,
            "joker_stats": {},      # name -> {"wins": n, "appearances": n}
            "build_lead_stats": {}, # name -> {"wins": n, "runs": n, "total_ante": n}
            "boss_deaths": {},      # name -> {"deaths": n, "total_ante": n}
        }

    def save(self):
        """Persist memory to disk."""
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_run(self, victory: bool, jokers: list[str], build_lead: str,
                   ante_reached: int, killer_boss: str = None):
        """
        Record outcome of a run.

        Args:
            victory: Did we win?
            jokers: List of joker names collected during run
            build_lead: Detected build lead (or None)
            ante_reached: Final ante
            killer_boss: Boss that killed us (if loss)
        """
        self.data["total_runs"] += 1
        if victory:
            self.data["total_wins"] += 1

        # Track joker appearances
        for joker in jokers:
            if joker not in self.data["joker_stats"]:
                self.data["joker_stats"][joker] = {"wins": 0, "appearances": 0}
            self.data["joker_stats"][joker]["appearances"] += 1
            if victory:
                self.data["joker_stats"][joker]["wins"] += 1

        # Track build lead success
        if build_lead:
            if build_lead not in self.data["build_lead_stats"]:
                self.data["build_lead_stats"][build_lead] = {"wins": 0, "runs": 0, "total_ante": 0}
            self.data["build_lead_stats"][build_lead]["runs"] += 1
            self.data["build_lead_stats"][build_lead]["total_ante"] += ante_reached
            if victory:
                self.data["build_lead_stats"][build_lead]["wins"] += 1

        # Track boss deaths
        if killer_boss and not victory:
            if killer_boss not in self.data["boss_deaths"]:
                self.data["boss_deaths"][killer_boss] = {"deaths": 0, "total_ante": 0}
            self.data["boss_deaths"][killer_boss]["deaths"] += 1
            self.data["boss_deaths"][killer_boss]["total_ante"] += ante_reached

        self.save()

    def get_joker_win_rate(self, joker_name: str) -> tuple[float, int]:
        """
        Get historical win rate for a joker.

        Returns: (win_rate, sample_size)
        - win_rate: 0.0 to 1.0, or -1 if no data
        - sample_size: number of runs with this joker
        """
        stats = self.data["joker_stats"].get(joker_name)
        if not stats or stats["appearances"] == 0:
            return -1, 0
        return stats["wins"] / stats["appearances"], stats["appearances"]

    def get_build_success_rate(self, build_lead: str) -> tuple[float, float, int]:
        """
        Get historical success metrics for a build lead.

        Returns: (win_rate, avg_ante, sample_size)
        """
        stats = self.data["build_lead_stats"].get(build_lead)
        if not stats or stats["runs"] == 0:
            return -1, -1, 0
        win_rate = stats["wins"] / stats["runs"]
        avg_ante = stats["total_ante"] / stats["runs"]
        return win_rate, avg_ante, stats["runs"]

    def get_boss_threat_level(self, boss_name: str) -> tuple[float, int]:
        """
        Get how dangerous a boss is based on death frequency.

        Returns: (death_rate, deaths)
        - death_rate: deaths / total_losses
        """
        stats = self.data["boss_deaths"].get(boss_name)
        total_losses = self.data["total_runs"] - self.data["total_wins"]
        if not stats or total_losses == 0:
            return 0, 0
        return stats["deaths"] / total_losses, stats["deaths"]

    def get_baseline_win_rate(self) -> float:
        """Get overall win rate across all runs."""
        if self.data["total_runs"] == 0:
            return 0
        return self.data["total_wins"] / self.data["total_runs"]

    def get_top_jokers(self, min_appearances: int = 5) -> list[tuple[str, float, int]]:
        """
        Get jokers sorted by win rate (with minimum sample size).

        Returns: List of (name, win_rate, appearances)
        """
        results = []
        for name, stats in self.data["joker_stats"].items():
            if stats["appearances"] >= min_appearances:
                win_rate = stats["wins"] / stats["appearances"]
                results.append((name, win_rate, stats["appearances"]))
        return sorted(results, key=lambda x: x[1], reverse=True)


class CoachStrategy(SmartStrategy):
    """
    Human-coached strategy developed through iterative natural language feedback.

    This strategy is built from the expertise of a 100+ hour Balatro player,
    translating human intuition and heuristics into code logic.

    VERSION: 0.3 - Build leads concept

    ============================================================================
    COACHING KNOWLEDGE BASE
    ============================================================================

    GENERAL:
    --------
    - Don't force a build. Let RNG present "build leads" to you.
    - A build lead is a joker that, when it appears, signals a direction.
    - Once you have a lead, it influences ALL future decisions.
    - Early game: stay open, wait for a lead to present itself.
    - Stronger leads can override weaker ones (natural pivot).

    EARLY GAME (Ante 1-2):
    ----------------------
    - You DON'T KNOW your build yet. Stay flexible.
    - Wait for RNG to present a build lead.
    - Take "staying alive" jokers while waiting.

    MID GAME (Ante 3-5):
    --------------------
    - Build lead should be identified by now.
    - Commit to the lead's requirements.
    - Can still pivot if a stronger lead appears.

    LATE GAME (Ante 6-8):
    ---------------------
    - Rarely get new build leads here.
    - Focus on maximizing existing build.
    - "Staying alive" becomes critical.

    JOKER SELECTION:
    ----------------
    [pending]

    SHOP CHOICE:
    ------------
    Three distinct shop behavior profiles, plus emergency pivot:

    1. EARLY GAME - "Scanning for leads / waiting for power / seeding build"
       - Not much money - hopefully saved/generated some (staying alive)
       - CRITICAL: Sets tone for rest of run
       - Looking for your ARC - picking power and running with it
       - Clear obvious choices based on tier/rank
       - BUT ALSO: less obvious jokers if they SEED a build idea
       - Squinting to see what build awaits for high scores

    2. MID GAME - "Build fill-out / complimenting / stacking / active synergizing"
       - Choices should ADD to existing build idea
       - Filling out the path you've highlighted
       - Flush stuff? Face cards? One suit? One hand type?
       - Narrowed down from wide options - now BUYING INTO IT
       - Hopefully generated funding along the way

    3. LATE GAME - "Topping off / over-powering / scaling"
       - Build is hopefully complete and locked in
       - ONLY looking for jokers that BEAT current AND don't break anything
       - Spending on: standard packs, celestials, supportive tarots, vouchers
       - Very selective on jokers - must be strict upgrade

    4. PIVOT (emergency, any phase)
       - Something coming will BREAK your build, causing death
       - Scramble mode - come up with something quick
       - Involves: selling jokers, steering toward something new
       - Thinking quick on your feet, probably getting lucky
       - [more coaching needed on pivot specifics]

    SKIP CONSIDERATION:
    -------------------
    - Throwback: If acquired, commit to skipping EVERY blind to pump it.

    JOKER KNOWLEDGE:
    ----------------
    - Human-ranked tier list integrated (JOKER_POWER_TIERS)
    - IMPORTANT: Tier list is SUBJECTIVE and CONTEXT-DEPENDENT
    - It's a REFERENCE for strength, not biblical truth
    - Many jokers move up/down tiers based on:
      * Current build direction
      * What synergies are available
      * Game phase (early vs late)
      * Deck composition
    - S+ Tier: Blueprint, Brainstorm, Triboulet (usually game-winning)
    - S Tier: Vampire, Cavendish, The Duo/Trio/Family, Canio, Campfire
    - A Tier: Strong anchors - Bloodstone, Onyx Agate, Throwback, etc.
    - B Tier: Solid context-dependent jokers
    - C Tier: Often situational - can be great WITH the right build
    - Use tier as baseline, let context adjust heavily

    BUILD MAXIMIZATION / SYNERGY:
    -----------------------------
    - Pareidolia + face card jokers (Smiley Face, Sock and Buskin)
    - Suit-specific jokers (Bloodstone/Hearts, Onyx Agate/Spades) = commit to suit
    - When build is WORKING (healthy margins), SCALE IT UP - don't diversify
    - When build is STRUGGLING, consider pivot opportunities
    - Track performance via score margins: crushing > healthy > struggling > critical

    JOKER ORDERING:
    ---------------
    - Jokers trigger LEFT to RIGHT - order matters for scoring
    - Optimal order: Chips -> Mult -> Retriggers -> X-Mult -> Utility
    - Math: (base + chip_jokers) * (mult + mult_jokers) * x_mult_jokers
    - Retriggers before x_mult so retriggered cards benefit from multiplication
    - Always check and optimize joker order

    SELLING JOKERS:
    ---------------
    - Sell to make room for something better
    - Sell to fund critical purchases
    - Sell jokers that conflict with build lead
    - Sell jokers whose conditions we're not meeting
    - Never sell the build lead itself
    - Stencil special: selling other jokers HELPS (it wants to be alone)

    STAYING ALIVE:
    --------------
    - NO HARDCODED LIST. Being too rigid kills adaptability.
    - Instead: evaluate "unconditional value" fluidly.
    - Ask: does this joker need setup? Does it have conditions?
    - Unconditional x_mult > conditional x_mult
    - Retriggers generally strong. Economy helps survive.
    - Scaling jokers: risky early, good mid, too late by late game.

    FINANCING:
    ----------
    - Money is your friend early on. It buys OPTIONS.
    - Fund a strong build once realized = clearer path.
    - Fund TRYING ideas before committing = flexibility.
    - Generate funds along the way = ability to strengthen/upgrade.
    - Early game priority: build a money cushion.
    - Interest mechanic: $1 per $5 held (up to $5 at $25). $25 = passive income.
    - Don't spend everything. Leave room for future opportunities.
    - Econ jokers early = compound returns over the run.
    - Money tarots (Hermit) are valuable early.
    - BUT: Econ doesn't WIN games - scoring power does. Econ is secondary.
    - Don't prioritize econ OVER power jokers. Econ is a tiebreaker.

    ============================================================================
    """

    # ========================================================================
    # BUILD LEADS REGISTRY
    # A build lead is a joker that defines your strategic direction.
    # Once acquired, it influences all future decisions.
    # ========================================================================

    BUILD_LEADS = {
        # Format: "Joker Name": {
        #     "archetype": what kind of build this creates
        #     "commitment": behavior to adopt once this lead is chosen
        #     "strength": 1-10, how compelling this lead is
        #     "synergies": jokers that work with this lead
        #     "anti_synergies": jokers that conflict
        # }

        # ===== MINIMALIST =====
        "Stencil": {
            "archetype": "minimalist",
            "commitment": "keep_joker_count_low",  # Sell jokers, don't buy more
            "strength": 8,
            "synergies": [],  # Works alone
            "anti_synergies": ["*"],  # Conflicts with having many jokers
        },

        # ===== FACE CARDS =====
        "Pareidolia": {
            "archetype": "face_cards",
            "commitment": "prioritize_face_cards",  # Keep face cards, use tarots to create more
            "strength": 7,
            "synergies": ["Smiley Face", "Sock and Buskin", "Photograph", "Scary Face",
                         "Business Card", "Hanging Chad", "Baron", "Triboulet"],
            "anti_synergies": ["Even Steven", "Hack"],
        },
        "Baron": {
            "archetype": "kings",
            "commitment": "prioritize_kings",  # Hold kings in hand, don't discard them
            "strength": 6,
            "synergies": ["Sock and Buskin", "Smiley Face", "Scary Face", "Pareidolia", "Mime"],
            "anti_synergies": ["Even Steven", "Hack"],
        },

        # ===== SUIT BUILDS =====
        "Bloodstone": {
            "archetype": "hearts",
            "commitment": "convert_to_hearts",  # Use tarots to change suits, prioritize hearts
            "strength": 6,
            "synergies": ["Lusty Joker", "Rough Gem", "Smeared Joker", "Fibonacci"],
            "anti_synergies": ["Onyx Agate", "Arrowhead", "Greedy Joker", "Gluttonous Joker"],
        },
        "Onyx Agate": {
            "archetype": "spades",
            "commitment": "convert_to_spades",  # Use tarots to change suits, prioritize spades
            "strength": 6,
            "synergies": ["Greedy Joker", "Arrowhead", "Smeared Joker"],
            "anti_synergies": ["Bloodstone", "Lusty Joker", "Rough Gem"],
        },
        "Arrowhead": {
            "archetype": "diamonds",
            "commitment": "convert_to_diamonds",  # Use tarots to change suits, prioritize diamonds
            "strength": 5,
            "synergies": ["Onyx Agate", "Greedy Joker", "Smeared Joker"],
            "anti_synergies": ["Bloodstone", "Lusty Joker"],
        },

        # ===== HAND TYPE BUILDS =====
        "The Duo": {
            "archetype": "pairs",
            "commitment": "always_play_pairs",  # Every hand should contain a pair
            "strength": 8,
            "synergies": ["The Trio", "The Family", "Mime", "Sock and Buskin", "Hanging Chad"],
            "anti_synergies": [],
        },
        "The Trio": {
            "archetype": "three_of_kind",
            "commitment": "always_play_three_of_kind",  # Aim for trips in every hand
            "strength": 8,
            "synergies": ["The Duo", "The Family", "Mime", "Photograph"],
            "anti_synergies": [],
        },
        "The Family": {
            "archetype": "four_of_kind",
            "commitment": "always_play_four_of_kind",  # Aim for quads
            "strength": 8,
            "synergies": ["The Duo", "The Trio", "Mime"],
            "anti_synergies": [],
        },
        "Shortcut": {
            "archetype": "straights",
            "commitment": "play_straights",  # Keep run-building cards, discard outliers
            "strength": 5,
            "synergies": ["Four Fingers", "Hack", "Runner"],
            "anti_synergies": [],
        },
        "Four Fingers": {
            "archetype": "four_card_hands",
            "commitment": "play_four_card_flushes_straights",  # Exploit 4-card hands
            "strength": 5,
            "synergies": ["Shortcut", "Smeared Joker"],
            "anti_synergies": [],
        },

        # ===== RETRIGGER BUILDS =====
        "Hack": {
            "archetype": "low_cards",
            "commitment": "keep_low_cards",  # Prioritize 2,3,4,5 - discard high cards
            "strength": 6,
            "synergies": ["Fibonacci", "Even Steven", "Raised Fist", "Walkie Talkie", "Seltzer"],
            "anti_synergies": ["Baron", "Photograph", "Sock and Buskin"],
        },
        "Seltzer": {
            "archetype": "retrigger",
            "commitment": "maximize_retriggers",  # Position cards to retrigger, play triggerable cards
            "strength": 7,
            "synergies": ["Hack", "Sock and Buskin", "Hanging Chad", "Dusk", "Mime"],
            "anti_synergies": [],
        },

        # ===== PARITY BUILDS =====
        "Even Steven": {
            "archetype": "even_ranks",
            "commitment": "keep_even_cards",  # Discard odd-ranked cards
            "strength": 5,
            "synergies": ["Walkie Talkie", "Scholar", "Hack"],
            "anti_synergies": ["Odd Todd", "Baron", "Photograph"],
        },
        "Odd Todd": {
            "archetype": "odd_ranks",
            "commitment": "keep_odd_cards",  # Discard even-ranked cards
            "strength": 5,
            "synergies": ["Fibonacci", "Raised Fist"],
            "anti_synergies": ["Even Steven", "Walkie Talkie"],
        },

        # ===== ENHANCED CARD BUILDS =====
        "Steel Joker": {
            "archetype": "steel_cards",
            "commitment": "create_steel_cards",  # Use tarots to make steel cards, protect them
            "strength": 7,
            "synergies": ["Mime", "Hanging Chad", "Dusk"],
            "anti_synergies": [],
        },
        "Glass Joker": {
            "archetype": "glass_cards",
            "commitment": "create_glass_cards",  # Use tarots to make glass cards, accept the risk
            "strength": 6,
            "synergies": ["Hanging Chad", "Dusk", "Lucky Cat"],
            "anti_synergies": [],
        },

        # ===== SPECIAL =====
        "Throwback": {
            "archetype": "skipper",
            "commitment": "skip_every_blind",  # Always skip small/big blinds for x_mult stacking
            "strength": 6,
            "synergies": [],
            "anti_synergies": [],
        },
        "Blueprint": {
            "archetype": "copy",
            "commitment": "position_rightmost_joker",  # Best joker goes to Blueprint's right
            "strength": 10,
            "synergies": ["Brainstorm"],
            "anti_synergies": [],
        },
        "Brainstorm": {
            "archetype": "copy",
            "commitment": "position_leftmost_joker",  # Brainstorm copies leftmost, order matters
            "strength": 10,
            "synergies": ["Blueprint"],
            "anti_synergies": [],
        },

        # ===== SCALING =====
        "Hologram": {
            "archetype": "scaling_mult",
            "commitment": "add_cards_to_deck",  # Use tarots/planets that add cards
            "strength": 7,
            "synergies": ["Mime", "Dusk", "Seltzer"],
            "anti_synergies": [],
        },
        "Ride The Bus": {
            "archetype": "scaling_mult",
            "commitment": "avoid_face_cards",  # Never play face cards, discard them
            "strength": 7,
            "synergies": ["Even Steven", "Hack"],
            "anti_synergies": ["Pareidolia", "Baron"],
        },
        "Campfire": {
            "archetype": "scaling_mult",
            "commitment": "sell_jokers_strategically",  # Sell weak jokers to pump Campfire
            "strength": 8,
            "synergies": [],
            "anti_synergies": [],
        },
    }

    # NO HARDCODED "STAYING ALIVE" LIST
    # Instead, we evaluate jokers fluidly based on how unconditional their value is.
    # A joker that "just works" without setup is more valuable when we have no direction.
    # But we don't commit to a list - we evaluate in context.

    # Interest thresholds - money milestones that generate passive income
    INTEREST_THRESHOLDS = [5, 10, 15, 20, 25]  # $1 interest per threshold reached

    # ========================================================================
    # JOKER POWER TIERS - Human-ranked joker power list
    # Based on 100+ hours of Balatro experience
    #
    # NOTE: This is SUBJECTIVE and CONTEXT-DEPENDENT - not biblical truth!
    # A C-tier joker can be S-tier with the right build.
    # An S-tier joker can be useless if it doesn't fit your direction.
    # Use as a BASELINE reference, let build context adjust heavily.
    # ========================================================================

    JOKER_POWER_TIERS = {
        # S+ Tier (100) - Build-defining, game-winning jokers
        "Blueprint": 100, "Brainstorm": 100, "Triboulet": 100,

        # S Tier (85) - Extremely powerful, often carry runs
        "Vampire": 85, "Cavendish": 85, "The Duo": 85, "The Trio": 85,
        "The Family": 85, "Spare Trousers": 85, "Canio": 85, "Campfire": 85,

        # A Tier (70) - Strong jokers that can anchor builds
        "Hiker": 70, "Fortune Teller": 70, "Rocket": 70, "Seltzer": 70,
        "Trading Card": 70, "Bloodstone": 70, "Perkeo": 70, "Fibonacci": 70,
        "Onyx Agate": 70, "Arrowhead": 70, "Sixth Sense": 70, "Space Joker": 70,
        "Burnt Joker": 70, "Hologram": 70, "Driver's License": 70, "Steel Joker": 70,
        "Ancient Joker": 70, "Card Sharp": 70, "Baseball Card": 70, "To Do List": 70,
        "Business Card": 70, "Mail-In Rebate": 70, "Cloud 9": 70, "Golden Joker": 70,
        "To The Moon": 70, "DNA": 70, "Green Joker": 70, "Gros Michel": 70,
        "Ramen": 70, "Ride The Bus": 70, "Stuntman": 70, "The Tribe": 70,
        "Throwback": 70, "Vagabond": 70,

        # B Tier (50) - Solid jokers, good in right context
        "Supernova": 50, "Scholar": 50, "Walkie Talkie": 50, "Sock and Buskin": 50,
        "Smiley Face": 50, "Scary Face": 50, "Wee Joker": 50, "Square Joker": 50,
        "Riff Raff": 50, "Half Joker": 50, "Invisible Joker": 50, "Constellation": 50,
        "Certificate": 50, "Ceremonial Dagger": 50, "Raised Fist": 50, "Yorick": 50,
        "Blackboard": 50, "Shoot The Moon": 50, "Egg": 50, "Abstract Joker": 50,
        "Swashbuckler": 50, "Misprint": 50, "Turtle Bean": 50, "Madness": 50,
        "Hack": 50, "Hit The Road": 50, "Rough Gem": 50, "Gluttonous Joker": 50,
        "Wrathful Joker": 50, "Lusty Joker": 50, "Greedy Joker": 50, "Diet Cola": 50,
        "Blue Joker": 50, "Bootstraps": 50, "Burglar": 50, "Acrobat": 50,
        "Baron": 50, "Seeing Double": 50, "The Order": 50, "Cartomancer": 50,
        "Flash Card": 50, "Delayed Gratification": 50, "Even Steven": 50, "Mime": 50,
        "Popcorn": 50, "Castle": 50, "Odd Todd": 50, "Ice Cream": 50,
        "Runner": 50, "Faceless Joker": 50, "Hanging Chad": 50, "8 Ball": 50,
        "Photograph": 50, "Erosion": 50, "Lucky Cat": 50, "Glass Joker": 50,
        "Flower Pot": 50, "Obelisk": 50, "Joker Stencil": 50, "Reserved Parking": 50,
        "Joker": 50, "Oops! All 6s": 50, "Midas Mask": 50, "Mystic Summit": 50,
        "Superposition": 50, "Satellite": 50,

        # C Tier (30) - Situational, weak, or trap jokers
        "Matador": 30, "The Idol": 30, "Juggler": 30, "Splash": 30,
        "Pareidolia": 30, "Loyalty Card": 30, "Dusk": 30, "Jolly Joker": 30,
        "Zany Joker": 30, "Wily Joker": 30, "Mad Joker": 30, "Clever Joker": 30,
        "Sly Joker": 30, "Bull": 30, "Banner": 30, "Smeared Joker": 30,
        "Astronomer": 30, "Drunkard": 30, "Droll Joker": 30, "Crafty Joker": 30,
        "Crazy Joker": 30, "Devious Joker": 30, "Troubadour": 30, "Hallucination": 30,
        "Chaos The Clown": 30, "Mr. Bones": 30, "Merry Andy": 30, "Red Card": 30,
        "Showman": 30, "Stone Joker": 30, "Marble Joker": 30, "Gift Card": 30,
        "Luchador": 30, "Golden Ticket": 30, "Credit Card": 30, "Shortcut": 30,
        "Four Fingers": 30, "Séance": 30,
    }

    def __init__(self, memory: CoachMemory = None):
        super().__init__()
        self.build_detector = BuildDetector()
        # Experience memory - persists across runs
        self.memory = memory or CoachMemory()
        # Track current build lead
        self.current_lead = None
        self.lead_strength = 0
        # Track build performance over time
        self.recent_margins = []  # List of (score_achieved / score_required) ratios
        self.blinds_played = 0

    def _detect_build_lead(self, game) -> tuple[Optional[str], int]:
        """
        Scan owned jokers for build leads.
        Returns (lead_joker_name, strength) or (None, 0).
        """
        best_lead = None
        best_strength = 0

        for joker in game.jokers:
            name = joker.get("name", "")
            if name in self.BUILD_LEADS:
                lead_info = self.BUILD_LEADS[name]
                if lead_info["strength"] > best_strength:
                    best_lead = name
                    best_strength = lead_info["strength"]

        return best_lead, best_strength

    def _get_lead_archetype(self, game) -> Optional[str]:
        """Get the archetype of current build lead, if any."""
        lead, _ = self._detect_build_lead(game)
        if lead and lead in self.BUILD_LEADS:
            return self.BUILD_LEADS[lead]["archetype"]
        return None

    def _joker_fits_lead(self, joker_name: str, game) -> bool:
        """Check if a joker fits with current build lead."""
        lead, _ = self._detect_build_lead(game)
        if not lead:
            return True  # No lead = everything is fine

        lead_info = self.BUILD_LEADS[lead]

        # Check anti-synergies
        if "*" in lead_info.get("anti_synergies", []):
            # This lead (like Stencil) conflicts with adding more jokers
            return False
        if joker_name in lead_info.get("anti_synergies", []):
            return False

        # Check if it's a synergy
        if joker_name in lead_info.get("synergies", []):
            return True  # Definitely take synergies

        return True  # Neutral - doesn't conflict

    def _is_potential_lead(self, joker_name: str, game) -> tuple[bool, int]:
        """
        Check if a joker could become a new build lead.
        Returns (is_lead, strength).
        """
        if joker_name in self.BUILD_LEADS:
            return True, self.BUILD_LEADS[joker_name]["strength"]
        return False, 0

    def _evaluate_unconditional_value(self, joker: dict, game) -> float:
        """
        Evaluate how "unconditionally useful" a joker is right now.

        Instead of a hardcoded list, we look at:
        - Does it require specific suits/ranks? (conditional = less valuable without build)
        - Does it require specific hand types? (conditional)
        - Does it have x_mult? (scaling = good)
        - Does it make money? (always useful)
        - Does it require setup/accumulation? (risky early)

        Returns a fluid score, not a binary yes/no.
        """
        effect = joker.get("effect", {})
        conditions = effect.get("conditions", [])
        modifiers = effect.get("modifiers", [])
        name = joker.get("name", "")

        score = 0
        has_conditions = False

        # Check what conditions this joker requires
        for cond in conditions:
            cond_type = cond.get("type", "")
            if cond_type in ("suit", "rank", "hand_contains"):
                has_conditions = True
                # Conditional jokers without a build direction are RISKY
                # They might not pay off - penalize them
                score -= 10

        # Unconditional multipliers are great
        for mod in modifiers:
            mod_type = mod.get("type", "")
            if mod_type == "x_mult" and not has_conditions:
                score += 30  # Unconditional x_mult is strong
            elif mod_type == "x_mult" and has_conditions:
                score += 10  # Conditional x_mult needs the condition met
            elif mod_type == "add_mult" and not has_conditions:
                score += 15
            elif mod_type == "add_mult" and has_conditions:
                score += 5
            elif mod_type in ("earn_money", "give_money"):
                score += mod.get("value", 0) * 2  # Money is always good

        # Retrigger jokers are generally strong (Hanging Chad, Dusk, etc.)
        if "retrigger" in str(effect).lower():
            score += 20

        # Economy jokers help us stay in the game
        if any(mod.get("type") in ("earn_money", "give_money") for mod in modifiers):
            score += 10

        # Jokers that scale over time are risky early, good mid
        scaling_jokers = {"Ride the Bus", "Green Joker", "Red Card", "Blue Joker",
                         "Runner", "Ice Cream", "Obelisk", "Lucky Cat"}
        phase = self._get_game_phase(game)
        if name in scaling_jokers:
            if phase == "early":
                score += 5   # Risky - might not pay off
            elif phase == "mid":
                score += 15  # Good time to start scaling
            else:
                score -= 5   # Too late to scale

        return score

    # ========================================================================
    # FINANCING LOGIC
    # ========================================================================

    def _get_current_interest(self, money: int) -> int:
        """Calculate current interest earned per round."""
        return min(money // 5, 5)  # $1 per $5, max $5

    def _get_next_interest_threshold(self, money: int) -> int:
        """Get the next interest threshold to aim for."""
        for threshold in self.INTEREST_THRESHOLDS:
            if money < threshold:
                return threshold
        return 25  # Already at max

    def _would_break_interest(self, current_money: int, cost: int) -> bool:
        """Check if a purchase would drop us below an interest threshold."""
        current_interest = self._get_current_interest(current_money)
        after_interest = self._get_current_interest(current_money - cost)
        return after_interest < current_interest

    def _evaluate_economy_value(self, joker: dict, game) -> float:
        """
        Evaluate economy/money generation value of a joker.

        Coaching insight: Money is your friend early on.
        - Econ jokers early = compound returns over the run
        - The earlier you get econ, the more rounds it pays off
        - $4/round joker in ante 1 = potentially $30+ over the run
        """
        effect = joker.get("effect", {})
        modifiers = effect.get("modifiers", [])
        name = joker.get("name", "")
        phase = self._get_game_phase(game)

        econ_score = 0

        # Direct money generation
        # IMPORTANT: Econ is nice but SCORING POWER wins games.
        # Econ is a tiebreaker, not a primary driver.
        for mod in modifiers:
            if mod.get("type") in ("earn_money", "give_money"):
                base_value = mod.get("value", 0)

                # Modest bonus - econ helps but doesn't win alone
                if phase == "early":
                    econ_score += base_value * 3  # Nice early but not over power
                elif phase == "mid":
                    econ_score += base_value * 2
                else:
                    econ_score += base_value * 1

        # Known strong econ jokers - modest bonuses
        # These help but shouldn't override scoring jokers
        strong_econ = {
            "Golden Joker": 12,      # $4 at end of round - reliable
            "Rocket": 10,            # Scales with boss defeats
            "Delayed Gratification": 8,   # $2 per discard unused
            "Business Card": 6,      # $2 per face card
            "Faceless Joker": 6,     # $5 if 3+ face cards discarded
            "To the Moon": 10,       # Extra $1 interest per $5
            "Satellite": 8,          # $ per unique planet used
            "Egg": 5,                # Gains $3 sell value per round
        }

        if name in strong_econ:
            base = strong_econ[name]
            if phase == "early":
                econ_score += base
            elif phase == "mid":
                econ_score += base * 0.7
            else:
                econ_score += base * 0.3

        return econ_score

    def _should_save_money(self, game, potential_cost: int) -> tuple[bool, str]:
        """
        Decide if we should save money instead of spending.

        Returns (should_save, reason).

        Coaching insight: Don't spend everything. Leave room for:
        - Future build opportunities
        - Interest generation
        - Flexibility to try ideas
        """
        money = game.money
        phase = self._get_game_phase(game)

        # Early game: prioritize building toward $25 interest cap
        if phase == "early":
            # If we're close to next interest threshold, consider saving
            next_threshold = self._get_next_interest_threshold(money)
            if money >= next_threshold - 3 and potential_cost > (money - next_threshold):
                return True, f"close_to_interest_{next_threshold}"

            # Don't drop below $5 early if possible (lose all interest)
            if money - potential_cost < 5 and money >= 5:
                return True, "preserve_base_interest"

        # Mid game: balance spending and saving
        elif phase == "mid":
            # Try to maintain at least $15 for flexibility
            if money - potential_cost < 15 and money >= 20:
                return True, "maintain_flexibility"

        # Late game: spend more freely to win
        # (less reason to save, need power now)

        return False, ""

    # ========================================================================
    # PHASE DETECTION
    # ========================================================================

    def _get_game_phase(self, game) -> str:
        """Determine current game phase."""
        if game.ante <= 2:
            return "early"
        elif game.ante <= 5:
            return "mid"
        else:
            return "late"

    # ========================================================================
    # CARD SELECTION - What to play
    # ========================================================================

    def select_cards_to_play(self, hand: Hand, game, must_play_count: int = None) -> list[int]:
        """
        Select cards to play.

        Coaching notes:
        - Commitment system influences hand selection preferences
        - The Duo/Trio/Family require specific hand types to trigger
        - Ride The Bus requires avoiding face cards
        - Suit builds prefer cards of the target suit
        """
        # Get commitment-based play preferences
        prefs = self._apply_commitment_to_play(hand, game)

        # Get all possible plays
        if must_play_count:
            min_cards = must_play_count
            max_cards = must_play_count
        else:
            min_cards = 1
            max_cards = 5

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

        # Filter/boost options based on commitment preferences
        scored_options = []
        for opt in options:
            score = opt.score
            dominated_suit = self._get_dominated_suit(opt.cards)
            has_face = any(c.is_face_card for c in opt.cards)
            has_pair = self._has_pair(opt.cards)
            has_three = self._has_three_of_kind(opt.cards)
            has_four = self._has_four_of_kind(opt.cards)

            # Apply preference adjustments
            if prefs["require_pair"] and has_pair:
                score *= 1.5  # Strong bonus for triggering The Duo
            elif prefs["require_pair"] and not has_pair:
                score *= 0.3  # Heavy penalty - won't trigger x_mult

            if prefs["require_three_of_kind"] and has_three:
                score *= 1.5
            elif prefs["require_three_of_kind"] and not has_three:
                score *= 0.3

            if prefs["require_four_of_kind"] and has_four:
                score *= 1.5
            elif prefs["require_four_of_kind"] and not has_four:
                score *= 0.3

            if prefs["avoid_face_cards"]:
                if has_face:
                    score *= 0.1  # Severely penalize - resets Ride The Bus
                else:
                    score *= 1.3  # Bonus for maintaining the mult

            if prefs["prefer_suit"] and dominated_suit:
                if dominated_suit.lower() == prefs["prefer_suit"]:
                    score *= 1.2  # Bonus for playing target suit

            if prefs["prefer_flush"] and opt.hand_type == HandType.FLUSH:
                score *= 1.3

            if prefs["prefer_straight"] and opt.hand_type == HandType.STRAIGHT:
                score *= 1.3

            if prefs["prefer_low_cards"]:
                low_count = sum(1 for c in opt.cards if c.rank in ('2', '3', '4', '5'))
                score *= (1 + 0.1 * low_count)  # More low cards = better

            scored_options.append((opt, score))

        # Sort by adjusted score
        scored_options.sort(key=lambda x: x[1], reverse=True)

        return scored_options[0][0].indices if scored_options else []

    def _get_dominated_suit(self, cards: list) -> Optional[str]:
        """Get the most common suit in the cards."""
        if not cards:
            return None
        from collections import Counter
        suits = Counter(c.suit.value for c in cards)
        if suits:
            return suits.most_common(1)[0][0]
        return None

    def _has_pair(self, cards: list) -> bool:
        """Check if cards contain a pair."""
        from collections import Counter
        ranks = Counter(c.rank for c in cards)
        return any(count >= 2 for count in ranks.values())

    def _has_three_of_kind(self, cards: list) -> bool:
        """Check if cards contain three of a kind."""
        from collections import Counter
        ranks = Counter(c.rank for c in cards)
        return any(count >= 3 for count in ranks.values())

    def _has_four_of_kind(self, cards: list) -> bool:
        """Check if cards contain four of a kind."""
        from collections import Counter
        ranks = Counter(c.rank for c in cards)
        return any(count >= 4 for count in ranks.values())

    # ========================================================================
    # DISCARD STRATEGY - What to throw away
    # ========================================================================

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """
        Select cards to discard.

        Coaching notes:
        - Commitment system influences what to discard
        - Build lead dictates which cards to prioritize keeping
        """
        if game.discards_remaining <= 0:
            return []

        cards = hand.cards
        if len(cards) <= 3:
            return []

        # Get commitment-based discard recommendations
        commitment_discards = self._apply_commitment_to_discard(cards, game)

        if commitment_discards:
            # Commitment has opinions - respect them but limit to 3
            # Also don't discard cards that are part of our best play
            best_options = self.evaluate_all_plays(hand, game.jokers, min_cards=1, max_cards=5)
            best_indices = set(best_options[0].indices) if best_options else set()

            # Filter out cards in best play
            safe_discards = [i for i in commitment_discards if i not in best_indices]

            # Limit to max discards
            max_discard = min(3, game.discards_remaining)
            return safe_discards[:max_discard]

        # No commitment opinion - fall back to parent logic
        return super().select_cards_to_discard(hand, game)

    # ========================================================================
    # JOKER EVALUATION - For shop decisions
    # ========================================================================

    def evaluate_joker(self, joker: dict, game) -> float:
        """
        Score a joker for purchase consideration.

        Coaching notes:
        - PRIMARY: Use human-ranked JOKER_POWER_TIERS as baseline
        - SECONDARY: Apply shop behavior profile adjustments
        - TERTIARY: Apply commitment-based shop preferences
        - Shop behaviors: scanning (early), filling (mid), topping (late), pivot (emergency)
        """
        joker_name = joker.get("name", "")
        current_lead, current_strength = self._detect_build_lead(game)

        # Check commitment-based shop preferences first
        shop_prefs = self._apply_commitment_to_shop(game)
        if shop_prefs.get("avoid_jokers"):
            # Stencil commitment - don't buy more jokers
            return -100

        # === PRIMARY: Start with human-ranked tier score ===
        tier_score = self.JOKER_POWER_TIERS.get(joker_name, 40)
        score = tier_score

        # === EXPERIENCE BONUS: Adjust based on historical win rate ===
        experience_bonus = self._get_joker_experience_bonus(joker_name)
        score += experience_bonus

        # Get current shop behavior profile
        behavior = self._get_shop_behavior(game)

        # Check if this joker seeds/leads a build
        is_lead, lead_strength = self._is_potential_lead(joker_name, game)
        seeds_build, seed_type = self._joker_seeds_build(joker, game)

        # Economy value as small modifier (more relevant when scanning)
        economy_value = self._evaluate_economy_value(joker, game)

        # === SCANNING: Early game, looking for arc ===
        if behavior == "scanning":
            # Clear obvious choices based on tier
            # Plus: less obvious jokers that SEED a build idea
            if seeds_build or is_lead:
                score += 25  # Potential to define our arc

            if is_lead:
                score += lead_strength * 5  # Build leads are exciting

            # Economy helps us stay alive while scanning
            score += economy_value * 0.4

        # === FILLING: Mid game, buying into the path ===
        elif behavior == "filling":
            # Choices should ADD to existing build
            fits_build = self._joker_fits_current_build(joker, game)

            if current_lead and current_lead in self.BUILD_LEADS:
                lead_info = self.BUILD_LEADS[current_lead]

                # Synergies are highly valuable - filling out the build
                if joker_name in lead_info.get("synergies", []):
                    score += 50  # Strongly favor - this is what we're here for

                # Anti-synergies are bad - don't break the path
                elif joker_name in lead_info.get("anti_synergies", []):
                    score -= 40
                elif "*" in lead_info.get("anti_synergies", []):
                    score -= 100  # Stencil etc

                # Neutral but fits = modest bonus
                elif fits_build:
                    score += 15

            elif fits_build:
                score += 20  # No formal lead but joker fits direction

            # Economy less critical now, build matters more
            score += economy_value * 0.2

        # === TOPPING: Late game, only strict upgrades ===
        elif behavior == "topping":
            # ONLY take jokers that BEAT current AND don't break anything
            is_upgrade = self._joker_is_strict_upgrade(joker, game)

            if not is_upgrade:
                # Not a strict upgrade - heavily penalize
                score -= 30

            # Must fit the locked-in build
            if not self._joker_fits_current_build(joker, game):
                score -= 50  # Can't break what's working

            # Very selective - only high tier matters
            if tier_score < 70:
                score -= 20  # Not worth considering late

            # Economy basically irrelevant now
            score += economy_value * 0.1

        # === PIVOT: Emergency scramble ===
        elif behavior == "pivot":
            # Looking for anything that can save the run
            # High tier unconditional power is king
            if tier_score >= 85:
                score += 30  # S/S+ tier could save us

            # Build leads offer new direction
            if is_lead:
                score += lead_strength * 6

            # Economy helps fund the pivot
            score += economy_value * 0.3

            # Don't care about current build fit - we're pivoting away
            # (no penalty for anti-synergy with failing build)

        return score

    # ========================================================================
    # SHOP DECISIONS
    # ========================================================================

    def _get_shop_behavior(self, game) -> str:
        """
        Determine current shop behavior profile.

        Returns: "scanning", "filling", "topping", or "pivot"
        """
        phase = self._get_game_phase(game)
        performance = self._get_build_performance()
        current_lead, _ = self._detect_build_lead(game)

        # Check for pivot conditions first (emergency overrides phase)
        if performance == "critical":
            return "pivot"

        # Phase-based behavior
        if phase == "early":
            return "scanning"  # Looking for arc, seeding build
        elif phase == "mid":
            if current_lead:
                return "filling"  # Have direction, fill it out
            else:
                return "scanning"  # Still looking for direction
        else:  # late
            if performance in ("struggling", "critical"):
                return "pivot"  # Emergency mode
            return "topping"  # Lock in, only strict upgrades

    def _joker_seeds_build(self, joker: dict, game) -> tuple[bool, str]:
        """
        Check if a joker could seed/start a new build direction.

        Returns: (seeds_build, build_type)

        Used in early game "scanning" to identify less obvious
        jokers that could define an arc.
        """
        name = joker.get("name", "")
        effect = joker.get("effect", {})
        conditions = effect.get("conditions", [])

        # Check if it's a known build lead
        if name in self.BUILD_LEADS:
            return True, self.BUILD_LEADS[name]["archetype"]

        # Check for suit-specific jokers (could seed suit build)
        for cond in conditions:
            if cond.get("type") == "suit":
                suit = cond.get("suit", "").lower()
                return True, f"{suit}_build"

        # Check for hand-type specific jokers
        for cond in conditions:
            if cond.get("type") == "hand_contains":
                hand = cond.get("hand", "").lower()
                if "flush" in hand:
                    return True, "flush_build"
                if "straight" in hand:
                    return True, "straight_build"
                if "pair" in hand or "two pair" in hand:
                    return True, "pairs_build"

        # Check for face card jokers
        face_jokers = {"Sock and Buskin", "Smiley Face", "Scary Face",
                       "Photograph", "Business Card", "Baron"}
        if name in face_jokers:
            return True, "face_card_build"

        return False, ""

    def _joker_fits_current_build(self, joker: dict, game) -> bool:
        """
        Check if a joker complements/adds to the current build.

        Used in mid game "filling" to ensure purchases ADD to the path.
        """
        current_lead, _ = self._detect_build_lead(game)
        if not current_lead:
            return True  # No build = anything goes

        joker_name = joker.get("name", "")

        # Check if it's a synergy with our lead
        if current_lead in self.BUILD_LEADS:
            lead_info = self.BUILD_LEADS[current_lead]
            if joker_name in lead_info.get("synergies", []):
                return True
            if joker_name in lead_info.get("anti_synergies", []):
                return False

        # Check if joker matches the archetype direction
        archetype = self._get_lead_archetype(game)
        seeds, seed_type = self._joker_seeds_build(joker, game)

        if archetype and seeds:
            # Does this joker's direction match our archetype?
            if archetype in seed_type or seed_type in archetype:
                return True

        return self._joker_fits_current_direction(joker, game)

    def _joker_is_strict_upgrade(self, joker: dict, game) -> bool:
        """
        Check if a joker is a strict upgrade over what we have.

        Used in late game "topping" - must BEAT current AND not break anything.
        """
        joker_name = joker.get("name", "")
        tier_score = self.JOKER_POWER_TIERS.get(joker_name, 40)

        # Must not conflict with build
        if not self._joker_fits_current_build(joker, game):
            return False

        # Must be high tier (A or above) to be worth considering late
        if tier_score < 70:
            return False

        # Check if we have a weaker joker to replace
        for owned in game.jokers:
            owned_tier = self.JOKER_POWER_TIERS.get(owned.get("name", ""), 40)
            if tier_score > owned_tier + 20:  # Significantly better
                return True

        # If we have open slots, any A+ tier that fits is an upgrade
        if len(game.jokers) < game.config.joker_slots:
            return True

        return False

    def should_buy_pack(self, pack, game) -> bool:
        """
        Decide whether to buy a pack.

        Coaching notes by shop behavior:
        - SCANNING: Packs are RNG, prefer direct joker purchases if available
        - FILLING: Packs less useful unless they support the build (planets)
        - TOPPING: Standard packs, celestials are good here
        - PIVOT: Depends on what we need
        """
        cost = getattr(pack, 'cost', 4)
        pack_type = getattr(pack, 'type', 'standard').lower()
        behavior = self._get_shop_behavior(game)

        # Check if we should save instead
        should_save, reason = self._should_save_money(game, cost)
        if should_save and behavior != "pivot":
            return False

        # Behavior-specific pack preferences
        if behavior == "scanning":
            # Early game - be conservative, building money cushion
            # Packs are RNG, prefer seeing direct joker options
            if game.money < 10:
                return False
            # Celestial/planet packs can help level hands
            if "celestial" in pack_type or "planet" in pack_type:
                return True
            return game.money >= 15  # Only if we have cushion

        elif behavior == "filling":
            # Mid game - packs less useful unless supporting build
            # Planet packs help level our key hand types
            if "celestial" in pack_type or "planet" in pack_type:
                return True
            # Arcana can help modify deck
            if "arcana" in pack_type:
                return True
            # Standard/buffoon packs are lower priority
            return game.money >= 20

        elif behavior == "topping":
            # Late game - celestials and standard packs are good
            # User said: "spending on standard packs, celestials, supportive tarots"
            if "celestial" in pack_type or "planet" in pack_type:
                return True
            if "standard" in pack_type:
                return True  # Can enhance cards
            if "arcana" in pack_type:
                return True  # Supportive tarots
            # Buffoon packs - only if we need jokers
            if "buffoon" in pack_type:
                return len(game.jokers) < game.config.joker_slots
            return True

        elif behavior == "pivot":
            # Emergency - any pack that might help
            return True

        return True

    def should_reroll(self, shop, game) -> bool:
        """
        Decide whether to reroll the shop.

        Coaching notes:
        - FINANCING: Rerolls cost money. Early game, that money could compound.
        - Only reroll if current options are truly bad.
        - Don't reroll below interest thresholds.
        """
        reroll_cost = 5  # Base cost

        # Never reroll if it would break interest
        if self._would_break_interest(game.money, reroll_cost):
            return False

        # Early game: be conservative with rerolls
        phase = self._get_game_phase(game)
        if phase == "early" and game.money < 15:
            return False

        return False  # Default conservative - more logic can be added

    def should_buy_joker(self, joker: dict, cost: int, game) -> bool:
        """
        Final check on whether to buy a specific joker.

        Even if a joker scores well, we might pass to preserve money.
        """
        # Get the joker's score
        joker_score = self.evaluate_joker(joker, game)

        # Check if we should save money instead
        should_save, reason = self._should_save_money(game, cost)

        # If we should save, only buy if the joker is REALLY good
        if should_save:
            # High-value threshold to override saving
            if joker_score < 50:
                return False

        # Otherwise, buy if score is positive
        return joker_score > 0

    # ========================================================================
    # BUILD PERFORMANCE TRACKING
    # ========================================================================

    def record_blind_result(self, score_achieved: int, score_required: int):
        """
        Record the result of a blind for performance tracking.
        Call this after each blind to track how the build is performing.
        """
        if score_required > 0:
            margin = score_achieved / score_required
            self.recent_margins.append(margin)
            # Keep last 6 blinds (roughly 2 antes)
            if len(self.recent_margins) > 6:
                self.recent_margins.pop(0)
        self.blinds_played += 1

    def _get_build_performance(self) -> str:
        """
        Assess how well the current build is performing.

        Returns: "crushing", "healthy", "struggling", "critical"

        Coaching insight:
        - If build is working well, SCALE IT UP (more of what's working)
        - If struggling, consider pivot or emergency measures
        """
        if len(self.recent_margins) < 2:
            return "unknown"

        avg_margin = sum(self.recent_margins) / len(self.recent_margins)
        recent_avg = sum(self.recent_margins[-3:]) / min(3, len(self.recent_margins))

        # Crushing it - scores are way above requirements
        if avg_margin >= 2.0 and recent_avg >= 1.8:
            return "crushing"

        # Healthy - comfortable margins, build is working
        if avg_margin >= 1.4 and recent_avg >= 1.3:
            return "healthy"

        # Struggling - tight margins, might need to adjust
        if avg_margin >= 1.1 or recent_avg >= 1.0:
            return "struggling"

        # Critical - barely surviving or failing
        return "critical"

    def _should_scale_build(self, game) -> bool:
        """
        Determine if we should focus on scaling the current build.

        When build is performing well, double down on what's working
        rather than diversifying or pivoting.
        """
        performance = self._get_build_performance()
        return performance in ("crushing", "healthy")

    # ========================================================================
    # JOKER SELLING DECISIONS
    # ========================================================================

    def evaluate_joker_for_sale(self, joker: dict, game) -> tuple[bool, str, int]:
        """
        Evaluate if a joker should be sold.

        Returns: (should_sell, reason, priority)
        - should_sell: True if this joker is a sell candidate
        - reason: Why we'd sell it
        - priority: Higher = more eager to sell (for ranking)

        Reasons to sell:
        - Make room for something better
        - Fund a critical purchase
        - Doesn't fit the build anymore (pivot)
        - Build lead changed, this conflicts
        - Low tier joker taking up a slot
        """
        joker_name = joker.get("name", "")
        sell_value = self._get_joker_sell_value(joker)
        current_lead, lead_strength = self._detect_build_lead(game)

        # Never sell our build lead
        if joker_name == current_lead:
            return False, "is_build_lead", 0

        # Never sell S+ or S tier jokers (unless they conflict with lead)
        tier_score = self.JOKER_POWER_TIERS.get(joker_name, 40)

        priority = 0
        reasons = []

        # Check if joker conflicts with current build lead
        if current_lead and current_lead in self.BUILD_LEADS:
            lead_info = self.BUILD_LEADS[current_lead]
            if joker_name in lead_info.get("anti_synergies", []):
                reasons.append("conflicts_with_lead")
                priority += 50

        # Check if joker is underperforming (has conditions we're not meeting)
        effect = joker.get("effect", {})
        conditions = effect.get("conditions", [])
        if conditions:
            # Joker has conditions - is our build meeting them?
            fits_build = self._joker_fits_current_direction(joker, game)
            if not fits_build:
                reasons.append("conditions_not_met")
                priority += 30

        # Low tier jokers are sell candidates (C tier = 30)
        if tier_score <= 30:
            reasons.append("low_tier")
            priority += 25
        elif tier_score <= 50:
            # B tier - slight sell consideration if we need room
            priority += 10

        # Check commitment-based selling preferences
        shop_prefs = self._apply_commitment_to_shop(game)
        if shop_prefs.get("want_to_sell_jokers"):
            # Stencil or Campfire - actively want to sell jokers
            commitment = self._get_current_commitment(game)
            if commitment == "keep_joker_count_low":
                # Stencil - sell everything except Stencil itself
                reasons.append("stencil_wants_fewer_jokers")
                priority += 40
            elif commitment == "sell_jokers_strategically":
                # Campfire - sell weak jokers to pump x_mult
                if tier_score <= 50:  # B tier or below
                    reasons.append("campfire_fuel")
                    priority += 30

        # High tier protection - reduce priority for good jokers
        if tier_score >= 85:
            priority -= 30  # S and S+ tier - very reluctant to sell
        elif tier_score >= 70:
            priority -= 15  # A tier - somewhat reluctant

        should_sell = priority >= 30  # Threshold for recommending sale
        reason = reasons[0] if reasons else "none"

        return should_sell, reason, priority

    def get_jokers_to_sell(self, game, need_slots: int = 0, need_money: int = 0) -> list[dict]:
        """
        Get a list of jokers that should be sold, prioritized.

        Args:
            need_slots: How many joker slots we need to free up
            need_money: How much money we need to raise

        Returns list of jokers to sell, in priority order.
        """
        candidates = []

        for joker in game.jokers:
            should_sell, reason, priority = self.evaluate_joker_for_sale(joker, game)
            if should_sell or need_slots > 0 or need_money > 0:
                sell_value = self._get_joker_sell_value(joker)
                candidates.append({
                    "joker": joker,
                    "reason": reason,
                    "priority": priority,
                    "sell_value": sell_value
                })

        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x["priority"], reverse=True)

        # If we need specific slots or money, ensure we return enough
        result = []
        slots_freed = 0
        money_raised = 0

        for c in candidates:
            if c["priority"] >= 30:  # Genuinely should sell
                result.append(c["joker"])
                slots_freed += 1
                money_raised += c["sell_value"]
            elif slots_freed < need_slots or money_raised < need_money:
                # Reluctant sale to meet requirements
                result.append(c["joker"])
                slots_freed += 1
                money_raised += c["sell_value"]

        return result

    def _get_joker_sell_value(self, joker: dict) -> int:
        """Get the sell value of a joker."""
        # Base sell value is roughly half the buy cost
        rarity = joker.get("rarity", "Common")
        base_values = {"Common": 2, "Uncommon": 3, "Rare": 4, "Legendary": 10}
        value = base_values.get(rarity, 2)

        # Editions add value
        edition = joker.get("edition", "")
        if edition == "Foil":
            value += 2
        elif edition == "Holographic":
            value += 3
        elif edition == "Polychrome":
            value += 5

        return value

    def _joker_fits_current_direction(self, joker: dict, game) -> bool:
        """Check if a joker fits the current build direction."""
        current_lead, _ = self._detect_build_lead(game)
        if not current_lead:
            return True  # No direction = everything fits

        joker_name = joker.get("name", "")

        # Check if it's a synergy
        if current_lead in self.BUILD_LEADS:
            lead_info = self.BUILD_LEADS[current_lead]
            if joker_name in lead_info.get("synergies", []):
                return True
            if joker_name in lead_info.get("anti_synergies", []):
                return False

        # Check if joker's conditions align with build archetype
        archetype = self._get_lead_archetype(game)
        effect = joker.get("effect", {})
        conditions = effect.get("conditions", [])

        for cond in conditions:
            cond_type = cond.get("type", "")
            if cond_type == "suit":
                req_suit = cond.get("suit", "").lower()
                # Does our archetype care about this suit?
                if archetype == "hearts" and req_suit != "hearts":
                    return False
                if archetype == "spades" and req_suit != "spades":
                    return False

        return True

    # ========================================================================
    # JOKER ORDERING (Left to Right Optimization)
    # ========================================================================

    def get_optimal_joker_order(self, jokers: list, game=None) -> list:
        """
        Return jokers in optimal scoring order (left to right).

        In Balatro, jokers trigger left to right. The optimal order is:
        1. Chip-adding jokers (add to base)
        2. Mult-adding jokers (add to mult)
        3. X-mult jokers (multiply the total)
        4. Retrigger jokers (position depends on what they retrigger)
        5. Economy/utility jokers (order doesn't matter much)

        Math example:
        - Base: 10 chips, 5 mult
        - Joker A: +30 chips, Joker B: +10 mult, Joker C: x2 mult
        - Optimal (A, B, C): (10+30) * (5+10) * 2 = 40 * 15 * 2 = 1200
        - Suboptimal (C, B, A): Still 1200 (x_mult is always last in calc)
        - But with retriggers, order REALLY matters
        """
        if not jokers:
            return jokers

        # Categorize jokers by their primary effect
        chip_jokers = []
        mult_jokers = []
        xmult_jokers = []
        retrigger_jokers = []
        utility_jokers = []

        for joker in jokers:
            category = self._categorize_joker_for_ordering(joker)
            if category == "chips":
                chip_jokers.append(joker)
            elif category == "mult":
                mult_jokers.append(joker)
            elif category == "xmult":
                xmult_jokers.append(joker)
            elif category == "retrigger":
                retrigger_jokers.append(joker)
            else:
                utility_jokers.append(joker)

        # Optimal order: chips -> mult -> retriggers -> xmult -> utility
        # Retriggers before xmult so retriggered cards get multiplied
        optimal_order = chip_jokers + mult_jokers + retrigger_jokers + xmult_jokers + utility_jokers

        # Apply commitment-based adjustments (Blueprint/Brainstorm positioning)
        if game is not None:
            optimal_order = self._apply_commitment_to_joker_order(optimal_order, game)

        return optimal_order

    def _categorize_joker_for_ordering(self, joker: dict) -> str:
        """
        Categorize a joker for ordering purposes.

        Returns: "chips", "mult", "xmult", "retrigger", "utility"
        """
        effect = joker.get("effect", {})
        modifiers = effect.get("modifiers", [])
        name = joker.get("name", "")

        # Check for retrigger first (specific jokers)
        retrigger_jokers = {
            "Hanging Chad", "Dusk", "Sock and Buskin", "Hack",
            "Seltzer", "Mime", "Baseball Card"
        }
        if name in retrigger_jokers:
            return "retrigger"

        # Check modifiers for primary effect type
        has_xmult = False
        has_mult = False
        has_chips = False

        for mod in modifiers:
            mod_type = mod.get("type", "")
            if mod_type == "x_mult":
                has_xmult = True
            elif mod_type == "add_mult":
                has_mult = True
            elif mod_type == "add_chips":
                has_chips = True

        # Priority: xmult > mult > chips > utility
        if has_xmult:
            return "xmult"
        if has_mult:
            return "mult"
        if has_chips:
            return "chips"

        return "utility"

    def should_reorder_jokers(self, game) -> bool:
        """Check if current joker order is suboptimal."""
        current_order = game.jokers
        optimal_order = self.get_optimal_joker_order(current_order)

        # Compare orders
        for i, (current, optimal) in enumerate(zip(current_order, optimal_order)):
            if current.get("name") != optimal.get("name"):
                return True

        return False

    def get_joker_reorder_actions(self, game) -> list[tuple[int, int]]:
        """
        Get the swap actions needed to achieve optimal joker order.

        Returns list of (from_idx, to_idx) swaps to perform.
        """
        current = [j.get("name") for j in game.jokers]
        optimal = [j.get("name") for j in self.get_optimal_joker_order(game.jokers)]

        swaps = []

        # Simple bubble-sort style swaps to achieve optimal order
        working = current.copy()
        for i, target_name in enumerate(optimal):
            if working[i] != target_name:
                # Find where target currently is
                for j in range(i + 1, len(working)):
                    if working[j] == target_name:
                        swaps.append((j, i))
                        working[i], working[j] = working[j], working[i]
                        break

        return swaps

    # ========================================================================
    # SKIP DECISIONS
    # ========================================================================

    def should_skip_blind(self, game, blind_info, tag) -> bool:
        """
        Decide whether to skip current blind for the tag reward.

        Coaching notes:
        - Commitment system can dictate skip behavior
        - Throwback = skip EVERY blind to pump it
        """
        # Check if commitment dictates skip behavior
        commitment_skip = self._apply_commitment_to_skip(game, blind_info)
        if commitment_skip is not None:
            return commitment_skip

        # Default: don't skip (more logic to be added)
        return False

    # ========================================================================
    # COMMITMENT SYSTEM - Behavioral changes based on build lead
    # ========================================================================

    def _get_current_commitment(self, game) -> Optional[str]:
        """Get the commitment string for current build lead."""
        lead, _ = self._detect_build_lead(game)
        if lead and lead in self.BUILD_LEADS:
            return self.BUILD_LEADS[lead].get("commitment")
        return None

    def _apply_commitment_to_discard(self, cards: list, game) -> list[int]:
        """
        Get discard recommendations based on current commitment.
        Returns indices of cards that should be discarded.
        """
        commitment = self._get_current_commitment(game)
        if not commitment:
            return []

        discard_indices = []

        for i, card in enumerate(cards):
            should_discard = False

            # Face card commitments
            if commitment == "prioritize_face_cards":
                # Keep face cards, discard non-face
                if not card.is_face_card:
                    should_discard = True

            elif commitment == "prioritize_kings":
                # Never discard kings - but other non-kings can go
                pass  # Don't actively discard, just protect kings

            elif commitment == "avoid_face_cards":
                # Ride The Bus - face cards reset mult, always discard them
                if card.is_face_card:
                    should_discard = True

            # Suit commitments
            elif commitment == "convert_to_hearts":
                if card.suit.value.lower() != "hearts":
                    should_discard = True

            elif commitment == "convert_to_spades":
                if card.suit.value.lower() != "spades":
                    should_discard = True

            elif commitment == "convert_to_diamonds":
                if card.suit.value.lower() != "diamonds":
                    should_discard = True

            # Rank/parity commitments
            elif commitment == "keep_low_cards":
                # Hack - keep 2,3,4,5
                if card.rank not in ('2', '3', '4', '5'):
                    should_discard = True

            elif commitment == "keep_even_cards":
                # Even Steven - keep 2,4,6,8,10,Q
                even_ranks = {'2', '4', '6', '8', '10', 'Q'}
                if card.rank not in even_ranks:
                    should_discard = True

            elif commitment == "keep_odd_cards":
                # Odd Todd - keep A,3,5,7,9,J,K
                odd_ranks = {'A', '3', '5', '7', '9', 'J', 'K'}
                if card.rank not in odd_ranks:
                    should_discard = True

            if should_discard:
                discard_indices.append(i)

        return discard_indices

    def _apply_commitment_to_play(self, hand, game) -> dict:
        """
        Get play preferences based on current commitment.
        Returns dict with preferences that can influence hand selection.
        """
        commitment = self._get_current_commitment(game)
        prefs = {
            "require_pair": False,
            "require_three_of_kind": False,
            "require_four_of_kind": False,
            "prefer_flush": False,
            "prefer_straight": False,
            "avoid_face_cards": False,
            "prefer_low_cards": False,
            "prefer_suit": None,
        }

        if not commitment:
            return prefs

        if commitment == "always_play_pairs":
            prefs["require_pair"] = True

        elif commitment == "always_play_three_of_kind":
            prefs["require_three_of_kind"] = True

        elif commitment == "always_play_four_of_kind":
            prefs["require_four_of_kind"] = True

        elif commitment == "play_straights":
            prefs["prefer_straight"] = True

        elif commitment == "play_four_card_flushes_straights":
            prefs["prefer_flush"] = True
            prefs["prefer_straight"] = True

        elif commitment == "avoid_face_cards":
            prefs["avoid_face_cards"] = True

        elif commitment == "keep_low_cards":
            prefs["prefer_low_cards"] = True

        elif commitment in ("convert_to_hearts", "prioritize_face_cards"):
            # Pareidolia doesn't care about suit, but Bloodstone does
            if "hearts" in commitment:
                prefs["prefer_suit"] = "hearts"

        elif commitment == "convert_to_spades":
            prefs["prefer_suit"] = "spades"

        elif commitment == "convert_to_diamonds":
            prefs["prefer_suit"] = "diamonds"

        return prefs

    def _apply_commitment_to_shop(self, game) -> dict:
        """
        Get shop preferences based on current commitment.
        Returns dict with what to prioritize in shop.
        """
        commitment = self._get_current_commitment(game)
        prefs = {
            "want_suit_changing_tarots": False,
            "want_card_creating_tarots": False,
            "want_steel_tarots": False,
            "want_glass_tarots": False,
            "avoid_jokers": False,
            "want_to_sell_jokers": False,
            "target_suit": None,
        }

        if not commitment:
            return prefs

        if commitment == "keep_joker_count_low":
            prefs["avoid_jokers"] = True
            prefs["want_to_sell_jokers"] = True

        elif commitment in ("convert_to_hearts", "convert_to_spades", "convert_to_diamonds"):
            prefs["want_suit_changing_tarots"] = True
            prefs["target_suit"] = commitment.replace("convert_to_", "")

        elif commitment == "create_steel_cards":
            prefs["want_steel_tarots"] = True

        elif commitment == "create_glass_cards":
            prefs["want_glass_tarots"] = True

        elif commitment == "add_cards_to_deck":
            prefs["want_card_creating_tarots"] = True

        elif commitment == "sell_jokers_strategically":
            prefs["want_to_sell_jokers"] = True

        return prefs

    def _apply_commitment_to_skip(self, game, blind_info) -> Optional[bool]:
        """
        Check if commitment dictates skip behavior.
        Returns True/False if commitment has opinion, None otherwise.
        """
        commitment = self._get_current_commitment(game)

        if commitment == "skip_every_blind":
            # Throwback - always skip non-boss blinds
            if blind_info.blind_type.name != "BOSS":
                return True

        return None

    def _apply_commitment_to_joker_order(self, jokers: list, game) -> list:
        """
        Adjust joker ordering based on commitment.
        Blueprint/Brainstorm have specific positioning needs.
        """
        commitment = self._get_current_commitment(game)
        if not commitment or not jokers:
            return jokers

        joker_names = [j.get("name", "") for j in jokers]

        if commitment == "position_rightmost_joker":
            # Blueprint copies the joker to its RIGHT
            # We want the best non-Blueprint joker to Blueprint's right
            if "Blueprint" in joker_names:
                bp_idx = joker_names.index("Blueprint")
                # Find the best joker (highest tier) that isn't Blueprint
                best_joker = None
                best_tier = 0
                best_idx = -1
                for i, j in enumerate(jokers):
                    if j.get("name") != "Blueprint":
                        tier = self.JOKER_POWER_TIERS.get(j.get("name", ""), 40)
                        if tier > best_tier:
                            best_tier = tier
                            best_joker = j
                            best_idx = i
                # Move best joker to right of Blueprint
                if best_joker and best_idx != bp_idx + 1:
                    jokers = jokers.copy()
                    jokers.pop(best_idx)
                    # Insert after Blueprint
                    insert_pos = bp_idx + 1 if best_idx > bp_idx else bp_idx
                    jokers.insert(insert_pos, best_joker)

        elif commitment == "position_leftmost_joker":
            # Brainstorm copies the LEFTMOST joker
            # We want the best non-Brainstorm joker in position 0
            if "Brainstorm" in joker_names:
                # Find the best joker that isn't Brainstorm
                best_joker = None
                best_tier = 0
                best_idx = -1
                for i, j in enumerate(jokers):
                    if j.get("name") != "Brainstorm":
                        tier = self.JOKER_POWER_TIERS.get(j.get("name", ""), 40)
                        if tier > best_tier:
                            best_tier = tier
                            best_joker = j
                            best_idx = i
                # Move best joker to position 0
                if best_joker and best_idx != 0:
                    jokers = jokers.copy()
                    jokers.pop(best_idx)
                    jokers.insert(0, best_joker)

        return jokers

    # ========================================================================
    # TAROT/CONSUMABLE EVALUATION - Based on commitment
    # ========================================================================

    def evaluate_tarot(self, tarot_name: str, game) -> float:
        """
        Score a tarot card for use/purchase based on current commitment.

        Returns a score - higher = more valuable for current build.
        """
        commitment = self._get_current_commitment(game)
        shop_prefs = self._apply_commitment_to_shop(game)

        base_score = 20  # Default tarot value

        # Suit-changing tarots
        suit_changers = {
            "The Lovers": "hearts",      # Changes suit to Hearts
            "The Chariot": "spades",     # Changes suit to Spades
            "The Hermit": None,          # Money (not suit-changing but valuable)
            "Justice": "diamonds",       # Changes suit to Diamonds
            "The Hanged Man": None,      # Destroys cards
            "Death": None,               # Converts cards
            "Temperance": None,          # Money based on jokers
            "The Tower": None,           # Enhances to Stone
            "The Star": "diamonds",      # Changes to Diamonds
            "The Moon": "clubs",         # Changes to Clubs
            "The Sun": "hearts",         # Changes to Hearts
            "The World": "spades",       # Changes to Spades
        }

        # Enhancement tarots
        enhancement_tarots = {
            "The Magician": "lucky",
            "The High Priestess": None,  # Creates planet cards
            "The Empress": "mult",       # Enhances to Mult cards
            "The Emperor": None,         # Creates tarot cards
            "The Hierophant": "bonus",   # Enhances to Bonus cards
            "Strength": None,            # Increases rank
            "The Wheel of Fortune": "edition",  # Adds edition
            "The Devil": "gold",         # Enhances to Gold cards
            "Judgement": None,           # Creates joker
        }

        # Steel and Glass specific
        steel_glass = {
            "The Stars": "steel",        # Note: not a real tarot name, placeholder
        }

        # Check if this tarot matches our commitment needs
        if shop_prefs.get("want_suit_changing_tarots"):
            target = shop_prefs.get("target_suit")
            if tarot_name in suit_changers:
                tarot_suit = suit_changers[tarot_name]
                if tarot_suit == target:
                    base_score += 60  # Exactly what we need
                elif tarot_suit is not None:
                    base_score -= 20  # Wrong suit

        if shop_prefs.get("want_steel_tarots"):
            # In Balatro, there's no direct "make steel" tarot
            # But some consumables can help - this is a placeholder
            pass

        if shop_prefs.get("want_glass_tarots"):
            # Similar - placeholder for glass enhancement
            pass

        if shop_prefs.get("want_card_creating_tarots"):
            # Tarots that add cards to deck
            if tarot_name in ("The Fool", "The High Priestess", "The Emperor"):
                base_score += 30

        # General value adjustments based on commitment
        if commitment == "avoid_face_cards":
            # The Hanged Man can destroy face cards
            if tarot_name == "The Hanged Man":
                base_score += 40

        if commitment in ("keep_even_cards", "keep_odd_cards"):
            # Strength can change ranks
            if tarot_name == "Strength":
                base_score += 20

        # The Hermit is always good for money
        if tarot_name == "The Hermit":
            phase = self._get_game_phase(game)
            if phase == "early":
                base_score += 30
            else:
                base_score += 15

        return base_score

    def evaluate_planet(self, planet_name: str, game) -> float:
        """
        Score a planet card based on current build direction.

        Planets level up specific hand types - we want ones that match our build.
        """
        commitment = self._get_current_commitment(game)
        archetype = self._get_lead_archetype(game)

        base_score = 25  # Planets are generally useful

        # Map planets to hand types
        planet_hands = {
            "Mercury": "PAIR",
            "Venus": "THREE_OF_A_KIND",
            "Earth": "FULL_HOUSE",
            "Mars": "FOUR_OF_A_KIND",
            "Jupiter": "FLUSH",
            "Saturn": "STRAIGHT",
            "Uranus": "TWO_PAIR",
            "Neptune": "STRAIGHT_FLUSH",
            "Pluto": "HIGH_CARD",
            "Planet X": "FIVE_OF_A_KIND",
            "Ceres": "FLUSH_HOUSE",
            "Eris": "FLUSH_FIVE",
        }

        planet_hand = planet_hands.get(planet_name, "")

        # Boost planets that match our commitment
        if commitment == "always_play_pairs" and planet_hand in ("PAIR", "TWO_PAIR"):
            base_score += 40
        elif commitment == "always_play_three_of_kind" and planet_hand == "THREE_OF_A_KIND":
            base_score += 50
        elif commitment == "always_play_four_of_kind" and planet_hand == "FOUR_OF_A_KIND":
            base_score += 50
        elif commitment in ("play_straights", "play_four_card_flushes_straights"):
            if planet_hand in ("STRAIGHT", "STRAIGHT_FLUSH"):
                base_score += 40
        elif commitment == "play_four_card_flushes_straights":
            if planet_hand in ("FLUSH", "FLUSH_HOUSE", "FLUSH_FIVE"):
                base_score += 40

        # Archetype-based adjustments
        if archetype == "pairs" and planet_hand in ("PAIR", "TWO_PAIR", "FULL_HOUSE"):
            base_score += 30
        elif archetype in ("three_of_kind", "four_of_kind"):
            if planet_hand in ("THREE_OF_A_KIND", "FOUR_OF_A_KIND", "FULL_HOUSE"):
                base_score += 30
        elif archetype == "straights" and planet_hand in ("STRAIGHT", "STRAIGHT_FLUSH"):
            base_score += 30

        return base_score

    def evaluate_pack(self, pack_type: str, game) -> float:
        """
        Score a pack type based on current commitment and game state.
        """
        commitment = self._get_current_commitment(game)
        shop_prefs = self._apply_commitment_to_shop(game)
        behavior = self._get_shop_behavior(game)

        base_score = 15

        pack_type_lower = pack_type.lower()

        # Arcana packs (tarots)
        if "arcana" in pack_type_lower:
            if shop_prefs.get("want_suit_changing_tarots"):
                base_score += 30  # Need tarots for suit conversion
            if shop_prefs.get("want_steel_tarots") or shop_prefs.get("want_glass_tarots"):
                base_score += 20
            if commitment == "avoid_face_cards":
                base_score += 15  # Hanged Man can help

        # Celestial packs (planets)
        elif "celestial" in pack_type_lower:
            # Good for leveling hand types
            if commitment in ("always_play_pairs", "always_play_three_of_kind",
                            "always_play_four_of_kind", "play_straights"):
                base_score += 35  # Want to level our key hand type
            if behavior == "topping":
                base_score += 25  # Late game planet scaling

        # Buffoon packs (jokers)
        elif "buffoon" in pack_type_lower:
            if shop_prefs.get("avoid_jokers"):
                base_score -= 50  # Stencil doesn't want more jokers
            elif behavior == "scanning":
                base_score += 20  # Looking for build leads
            elif behavior == "topping":
                base_score += 10  # Only if we have room

        # Standard packs (playing cards)
        elif "standard" in pack_type_lower:
            if shop_prefs.get("want_card_creating_tarots"):
                base_score += 25  # Hologram wants more cards
            if behavior == "topping":
                base_score += 20  # Can enhance deck

        # Spectral packs
        elif "spectral" in pack_type_lower:
            if shop_prefs.get("want_steel_tarots") or shop_prefs.get("want_glass_tarots"):
                base_score += 40  # Spectral can create steel/glass
            base_score += 15  # Generally powerful

        return base_score

    # ========================================================================
    # EXPERIENCE-BASED LEARNING
    # ========================================================================

    def _get_joker_experience_bonus(self, joker_name: str) -> float:
        """
        Get bonus/penalty based on historical win rate for this joker.

        If joker appears more in wins than baseline, boost it.
        If it appears more in losses, penalize it.

        Returns: -15 to +15 adjustment
        """
        win_rate, sample_size = self.memory.get_joker_win_rate(joker_name)

        # Need minimum sample size for confidence
        if sample_size < 5:
            return 0

        baseline = self.memory.get_baseline_win_rate()
        if baseline == 0:
            return 0

        # Compare joker's win rate to baseline
        # If joker appears in wins 2x more than baseline, that's significant
        ratio = win_rate / baseline if baseline > 0 else 1

        # Cap the adjustment to avoid runaway effects
        if ratio > 1.5:
            return min(15, (ratio - 1) * 20)  # Up to +15
        elif ratio < 0.5:
            return max(-15, (ratio - 1) * 20)  # Down to -15
        else:
            return (ratio - 1) * 10  # Modest adjustment

    def _get_build_experience_bonus(self, build_lead: str) -> float:
        """
        Get bonus/penalty based on historical success of this build lead.

        Returns: -10 to +10 adjustment to lead strength
        """
        win_rate, avg_ante, sample_size = self.memory.get_build_success_rate(build_lead)

        if sample_size < 3:
            return 0

        baseline = self.memory.get_baseline_win_rate()
        if baseline == 0:
            return 0

        ratio = win_rate / baseline if baseline > 0 else 1

        if ratio > 1.5:
            return min(10, (ratio - 1) * 15)
        elif ratio < 0.5:
            return max(-10, (ratio - 1) * 15)
        else:
            return (ratio - 1) * 8

    def record_run_outcome(self, game, victory: bool):
        """
        Record the outcome of a run for future learning.

        Call this after each run completes.
        """
        # Get jokers collected
        jokers = [j.get("name", "") for j in game.jokers if j.get("name")]

        # Get detected build lead
        build_lead, _ = self._detect_build_lead(game)

        # Get killer boss if we lost
        killer_boss = None
        if not victory and game.history and game.history.events:
            # Find the last blind_result that was a loss
            for event in reversed(game.history.events):
                if event.event_type == "blind_result":
                    if not event.data.get("success") and event.data.get("boss_name"):
                        killer_boss = event.data["boss_name"]
                        break

        # Record to memory
        self.memory.record_run(
            victory=victory,
            jokers=jokers,
            build_lead=build_lead,
            ante_reached=game.ante,
            killer_boss=killer_boss
        )

    # ========================================================================
    # COACHING HELPERS - Supporting methods for coached logic
    # ========================================================================

    # Methods will be added here as coaching reveals needed functionality


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
