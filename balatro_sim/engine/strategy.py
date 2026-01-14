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
    [pending]

    SKIP CONSIDERATION:
    -------------------
    - Throwback: If acquired, commit to skipping EVERY blind to pump it.

    PATIENCE:
    ---------
    [pending]

    JOKER KNOWLEDGE:
    ----------------
    [pending - specific joker interactions]

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
        #     "commitment": what you must do to maximize it
        #     "strength": 1-10, how compelling this lead is
        #     "synergies": jokers that work with this lead
        #     "anti_synergies": jokers that conflict
        # }

        "Stencil": {
            "archetype": "minimalist",
            "commitment": "keep_joker_count_low",
            "strength": 8,
            "synergies": [],  # Works alone
            "anti_synergies": ["*"],  # Conflicts with having many jokers
        },

        "Pareidolia": {
            "archetype": "face_cards",
            "commitment": "get_face_card_jokers",
            "strength": 7,
            "synergies": ["Smiley Face", "Sock and Buskin", "Photograph", "Scary Face", "Business Card", "Hanging Chad"],
            "anti_synergies": [],
        },

        "Throwback": {
            "archetype": "skipper",
            "commitment": "skip_every_blind",
            "strength": 6,
            "synergies": [],
            "anti_synergies": [],
        },

        "Bloodstone": {
            "archetype": "hearts",
            "commitment": "convert_deck_to_hearts",
            "strength": 6,
            "synergies": ["Lusty Joker", "Rough Gem"],
            "anti_synergies": ["Onyx Agate", "Arrowhead", "Smeared Joker"],
        },

        "Onyx Agate": {
            "archetype": "spades",
            "commitment": "convert_deck_to_spades",
            "strength": 6,
            "synergies": ["Greedy Joker", "Arrowhead"],
            "anti_synergies": ["Bloodstone", "Rough Gem"],
        },

        # More will be added through coaching...
    }

    # NO HARDCODED "STAYING ALIVE" LIST
    # Instead, we evaluate jokers fluidly based on how unconditional their value is.
    # A joker that "just works" without setup is more valuable when we have no direction.
    # But we don't commit to a list - we evaluate in context.

    # Interest thresholds - money milestones that generate passive income
    INTEREST_THRESHOLDS = [5, 10, 15, 20, 25]  # $1 interest per threshold reached

    def __init__(self):
        super().__init__()
        self.build_detector = BuildDetector()
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
        - GENERAL: [pending]
        - EARLY: [pending]
        - MID: [pending]
        - LATE: [pending]
        """
        phase = self._get_game_phase(game)

        # Phase-specific logic will be added through coaching
        # For now, use parent SmartStrategy logic
        return super().select_cards_to_play(hand, game, must_play_count)

    # ========================================================================
    # DISCARD STRATEGY - What to throw away
    # ========================================================================

    def select_cards_to_discard(self, hand: Hand, game) -> list[int]:
        """
        Select cards to discard.

        Coaching notes:
        - GENERAL: [pending]
        - PATIENCE: [pending]
        - BUILD MAXIMIZATION: [pending]
        """
        phase = self._get_game_phase(game)

        # Phase-specific logic will be added through coaching
        # For now, use parent SmartStrategy logic
        return super().select_cards_to_discard(hand, game)

    # ========================================================================
    # JOKER EVALUATION - For shop decisions
    # ========================================================================

    def evaluate_joker(self, joker: dict, game) -> float:
        """
        Score a joker for purchase consideration.

        Coaching notes:
        - GENERAL: Let RNG present build leads. Don't force a build.
        - EARLY: No lead yet - be FLUID. Evaluate unconditional value.
        - MID/LATE: Have a lead - favor synergies, reject anti-synergies.
        - A stronger lead can override current lead (pivot opportunity).
        - NO HARDCODED LISTS. Evaluate each joker in context.
        """
        phase = self._get_game_phase(game)
        joker_name = joker.get("name", "")
        current_lead, current_strength = self._detect_build_lead(game)

        score = 0

        # Check if this joker is a potential build lead
        is_lead, lead_strength = self._is_potential_lead(joker_name, game)

        # Always start with the fluid unconditional value assessment
        unconditional_value = self._evaluate_unconditional_value(joker, game)

        # Add economy value - money generation is especially good early
        economy_value = self._evaluate_economy_value(joker, game)

        # === EARLY GAME: No lead yet ===
        if phase == "early" and not current_lead:
            # Be fluid - weight unconditional value highly
            score += unconditional_value

            # Economy is nice but secondary - don't let it override power
            score += economy_value * 0.5

            # Potential build leads are exciting - they give us direction
            if is_lead:
                score += lead_strength * 8

        # === WE HAVE A BUILD LEAD ===
        elif current_lead:
            lead_info = self.BUILD_LEADS[current_lead]
            performance = self._get_build_performance()

            # Is this joker a synergy with our lead?
            if joker_name in lead_info.get("synergies", []):
                score += 70  # Strongly favor synergies
                score += unconditional_value * 0.5
                score += economy_value * 0.3

                # BUILD IS WORKING - SCALE IT UP
                # If we're crushing/healthy, synergies are even MORE valuable
                if performance in ("crushing", "healthy"):
                    score += 30  # Double down on what's working

            # Does this joker conflict with our lead?
            elif "*" in lead_info.get("anti_synergies", []):
                # Lead like Stencil - don't add more jokers
                score -= 100
            elif joker_name in lead_info.get("anti_synergies", []):
                score -= 40  # Conflicts with our direction

            # Is this a STRONGER lead that could pivot us?
            elif is_lead and lead_strength > current_strength:
                # Only consider pivot if we're NOT performing well
                if performance in ("struggling", "critical", "unknown"):
                    score += lead_strength * 6  # Consider the pivot
                    score += unconditional_value * 0.3
                    score += economy_value * 0.3
                else:
                    # Build is working - don't pivot just because something shiny appeared
                    score += lead_strength * 2  # Small consideration only

            # No special relationship - evaluate on merit
            else:
                score += unconditional_value * 0.7
                score += economy_value * 0.5

        # === MID/LATE WITHOUT LEAD (drifting - need direction) ===
        else:
            # Really want a lead now - but don't ignore good jokers
            if is_lead:
                score += lead_strength * 10
            score += unconditional_value
            score += economy_value * 0.7  # Econ less valuable late but still helps

        return score

    # ========================================================================
    # SHOP DECISIONS
    # ========================================================================

    def should_buy_pack(self, pack, game) -> bool:
        """
        Decide whether to buy a pack.

        Coaching notes:
        - FINANCING: Consider if buying would hurt interest income.
        - Early game: be more conservative, build money cushion.
        - Packs are RNG - sometimes saving for direct joker is better.
        """
        cost = getattr(pack, 'cost', 4)

        # Check if we should save instead
        should_save, reason = self._should_save_money(game, cost)
        if should_save:
            return False

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
        """
        joker_name = joker.get("name", "")
        sell_value = self._get_joker_sell_value(joker)
        current_lead, lead_strength = self._detect_build_lead(game)

        # Never sell our build lead
        if joker_name == current_lead:
            return False, "is_build_lead", 0

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
            archetype = self._get_lead_archetype(game)
            fits_build = self._joker_fits_current_direction(joker, game)
            if not fits_build:
                reasons.append("conditions_not_met")
                priority += 30

        # Check if we're at joker cap and this is the weakest
        joker_score = self.evaluate_joker(joker, game)
        if joker_score < 10:
            reasons.append("low_value")
            priority += 20

        # Stencil special case - if we have Stencil, selling others HELPS
        if current_lead == "Stencil" and joker_name != "Stencil":
            reasons.append("stencil_wants_fewer_jokers")
            priority += 40

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

    def get_optimal_joker_order(self, jokers: list) -> list:
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
        - SKIP CONSIDERATION: Throwback = skip EVERY blind to pump it.
        - Other skip decisions pending more coaching.
        """
        # Check for Throwback commitment
        lead, _ = self._detect_build_lead(game)
        if lead == "Throwback":
            # Throwback archetype: skip every small/big blind
            if blind_info.blind_type.name != "BOSS":
                return True

        # Default: don't skip (more logic to be added)
        return False

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
