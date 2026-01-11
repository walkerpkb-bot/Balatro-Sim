"""
Balatro Joker Effect Parser
Converts natural language joker effects into structured JSON
"""

import csv
import json
import re
from pathlib import Path

# Regex patterns for extracting effect components
PATTERNS = {
    # Numeric effects
    'add_mult': r'\+(\d+) Mult',
    'add_chips': r'\+(\d+) Chips',
    'x_mult': r'X([\d.]+) Mult',
    'earn_money': r'[Ee]arn \$(\d+)',
    'give_money': r'give \$(\d+)',
    'random_mult': r'\+(\d+)-(\d+) Mult',

    # Probability
    'probability': r'(\d+) in (\d+) chance',

    # Conditions - hand types
    'hand_contains': r'(?:if )?(?:played )?hand contains (?:a )?([A-Za-z\s]+?)(?:\s*,|\s*$|when)',

    # Conditions - card properties
    'suit_condition': r'(?:with |cards? )(Diamond|Heart|Spade|Club)s? suit',
    'rank_condition': r'played (?:cards? with )?(Ace|King|Queen|Jack|[0-9]+|face card)s?',
    'face_cards': r'face cards?',

    # Triggers
    'when_scored': r'when scored',
    'when_blind_selected': r'[Ww]hen (?:Small\/Big )?[Bb]lind (?:is )?selected',
    'end_of_round': r'(?:at )?end of round',
    'each_round': r'each round',
    'round_begins': r'[Ww]hen round begins',
    'booster_opened': r'[Bb]ooster [Pp]ack opened',
    'booster_skipped': r'[Bb]ooster [Pp]ack skipped',

    # Scaling
    'gains': r'[Gg]ains? \+?(X?[\d.]+) (Mult|Chips)',
    'per_condition': r'(?:for each|per) (.+?)(?:\.|$|,)',
    'adds_to_mult': r'[Aa]dds? (.+?) to Mult',

    # Retrigger
    'retrigger': r'[Rr]etrigger',

    # Card counts
    'card_count': r'(\d+) or fewer cards',
    'exactly_cards': r'exactly (\d+) cards',

    # Card generation
    'create_card': r'[Cc]reate[s]? (?:a |an )?(?:random )?(Tarot|Spectral|Planet|Joker)',
    'create_n_cards': r'[Cc]reate[s]? (\d+) (\w+) (Jokers?|cards?)',
    'add_to_deck': r'[Aa]dds? (?:one |a )?([\w\s]+?) (?:card )?to (?:your )?(?:deck|hand)',
    'duplicate': r'duplicate',
    'copy': r'[Cc]reates? (?:\w+ )?copy',

    # Rule modifiers
    'can_be_made': r'can be made with (\d+) cards',
    'considered_as': r'(?:are |is )?considered (?:as )?([\w\s]+)',
    'counts_in_scoring': r'counts? (?:as|in) scoring',

    # Economy modifiers
    'free_reroll': r'(\d+) free [Rr]erolls?',
    'sell_value': r'sell value',
    'debt': r'[Gg]o up to -\$(\d+) in debt',
    'interest': r'interest',

    # Hand size
    'hand_size_plus': r'\+(\d+) hand size',
    'hand_size_minus': r'-(\d+) hand size',

    # Discards/Hands per round
    'plus_discards': r'\+(\d+) discards?',
    'plus_hands': r'\+(\d+) [Hh]ands',
    'minus_hands': r'-(\d+) [Hh]and',

    # Copy effects
    'copies_ability': r'[Cc]opies? (?:the )?ability',

    # Destroy effects
    'destroy': r'[Dd]estroy',

    # Upgrade/level up
    'upgrade_level': r'[Uu]pgrade[s]? (?:the )?level',

    # Become/transform
    'become': r'become[s]? (\w+)',

    # Sell to activate
    'sell_to': r'[Ss]ell this (?:card )?to',

    # Defensive
    'prevents_death': r'[Pp]revents? death',

    # Probability modifier
    'doubles_probability': r'[Dd]oubles? (?:all )?(?:listed )?probabilities?',

    # Free items
    'are_free': r'(?:are|is) free',

    # Disable effects
    'disable': r'[Dd]isable[s]?',

    # Gaps in straights
    'gaps': r'gaps? of (\d+) rank',

    # Same suit rule
    'same_suit': r'same suit',

    # Appear multiple times
    'multiple_appearances': r'appear multiple times',
}

# Hand types in Balatro
HAND_TYPES = [
    'Flush Five', 'Flush House', 'Five of a Kind', 'Royal Flush',
    'Straight Flush', 'Four of a Kind', 'Full House', 'Flush',
    'Straight', 'Three of a Kind', 'Two Pair', 'Pair', 'High Card'
]

SUITS = ['Diamond', 'Heart', 'Spade', 'Club']
RANKS = ['Ace', 'King', 'Queen', 'Jack', '10', '9', '8', '7', '6', '5', '4', '3', '2']


def parse_effect(effect_text: str) -> dict:
    """Parse a joker effect string into structured data."""
    parsed = {
        'raw': effect_text,
        'modifiers': [],
        'conditions': [],
        'triggers': [],
        'flags': []
    }

    # Extract additive mult
    match = re.search(PATTERNS['add_mult'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'add_mult',
            'value': int(match.group(1))
        })

    # Extract additive chips
    match = re.search(PATTERNS['add_chips'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'add_chips',
            'value': int(match.group(1))
        })

    # Extract multiplicative mult
    match = re.search(PATTERNS['x_mult'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'x_mult',
            'value': float(match.group(1))
        })

    # Extract money earned
    match = re.search(PATTERNS['earn_money'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'earn_money',
            'value': int(match.group(1))
        })

    # Extract money given (chance-based)
    match = re.search(PATTERNS['give_money'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'give_money',
            'value': int(match.group(1))
        })

    # Extract probability
    match = re.search(PATTERNS['probability'], effect_text)
    if match:
        parsed['conditions'].append({
            'type': 'probability',
            'numerator': int(match.group(1)),
            'denominator': int(match.group(2))
        })

    # Check for hand type conditions
    for hand in HAND_TYPES:
        if hand.lower() in effect_text.lower():
            parsed['conditions'].append({
                'type': 'hand_contains',
                'hand': hand
            })
            break

    # Check for suit conditions
    for suit in SUITS:
        if suit in effect_text:
            parsed['conditions'].append({
                'type': 'suit',
                'suit': suit
            })
            break

    # Check for face card references
    if re.search(PATTERNS['face_cards'], effect_text, re.IGNORECASE):
        parsed['conditions'].append({
            'type': 'rank',
            'rank': 'face_card'
        })

    # Check for specific rank references
    # First, mask out numeric values that are part of modifiers to avoid false positives
    masked_text = re.sub(r'[+X][\d.]+', '', effect_text)  # Remove "+4", "X1.5" etc
    masked_text = re.sub(r'\$\d+', '', masked_text)  # Remove "$5" etc
    masked_text = re.sub(r'\d+ in \d+', '', masked_text)  # Remove "1 in 4" etc

    for rank in RANKS:
        # For numeric ranks, require word boundaries and context
        if rank.isdigit():
            # Look for patterns like "played 8", "each 9", "every 2"
            pattern = rf'(?:played|each|every|scored?)\s+{rank}s?\b'
            if re.search(pattern, masked_text, re.IGNORECASE):
                parsed['conditions'].append({
                    'type': 'rank',
                    'rank': rank
                })
        else:
            # For face cards (Ace, King, etc), just check presence
            pattern = rf'\b{rank}s?\b'
            if re.search(pattern, effect_text):
                if not any(c.get('rank') == rank for c in parsed['conditions']):
                    parsed['conditions'].append({
                        'type': 'rank',
                        'rank': rank
                    })

    # Extract triggers
    if re.search(PATTERNS['when_scored'], effect_text):
        parsed['triggers'].append('on_score')
    if re.search(PATTERNS['when_blind_selected'], effect_text):
        parsed['triggers'].append('on_blind_select')
    if re.search(PATTERNS['end_of_round'], effect_text):
        parsed['triggers'].append('end_of_round')

    # Check for scaling effects
    match = re.search(PATTERNS['gains'], effect_text)
    if match:
        parsed['flags'].append('scaling')
        value_str = match.group(1)
        modifier_type = match.group(2).lower()

        if value_str.startswith('X'):
            parsed['modifiers'].append({
                'type': f'gains_x_{modifier_type}',
                'value': float(value_str[1:])
            })
        else:
            parsed['modifiers'].append({
                'type': f'gains_{modifier_type}',
                'value': float(value_str)
            })

    # Check for retrigger
    if re.search(PATTERNS['retrigger'], effect_text):
        parsed['flags'].append('retrigger')

    # Check for card count conditions
    match = re.search(PATTERNS['card_count'], effect_text)
    if match:
        parsed['conditions'].append({
            'type': 'max_cards',
            'value': int(match.group(1))
        })

    match = re.search(PATTERNS['exactly_cards'], effect_text)
    if match:
        parsed['conditions'].append({
            'type': 'exact_cards',
            'value': int(match.group(1))
        })

    # === NEW PATTERNS ===

    # Random mult range (e.g., "+0-23 Mult")
    match = re.search(PATTERNS['random_mult'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'random_mult',
            'min': int(match.group(1)),
            'max': int(match.group(2))
        })
        parsed['flags'].append('random')

    # Card generation - single card
    match = re.search(PATTERNS['create_card'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'create_card',
            'card_type': match.group(1)
        })
        parsed['flags'].append('card_generation')

    # Card generation - multiple cards (e.g., "create 2 Common Jokers")
    match = re.search(PATTERNS['create_n_cards'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'create_cards',
            'count': int(match.group(1)),
            'rarity': match.group(2),
            'card_type': match.group(3)
        })
        parsed['flags'].append('card_generation')

    # Duplicate
    if re.search(PATTERNS['duplicate'], effect_text):
        parsed['modifiers'].append({'type': 'duplicate'})
        parsed['flags'].append('copy_effect')

    # Creates copy
    if re.search(PATTERNS['copy'], effect_text):
        parsed['modifiers'].append({'type': 'create_copy'})
        parsed['flags'].append('copy_effect')

    # Add card to deck
    match = re.search(PATTERNS['add_to_deck'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'add_to_deck',
            'card': match.group(1).strip()
        })
        parsed['flags'].append('deck_modification')

    # Rule modifiers - can be made with N cards
    match = re.search(PATTERNS['can_be_made'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'hand_requirement_change',
            'cards_needed': int(match.group(1))
        })
        parsed['flags'].append('rule_modifier')

    # Considered as
    match = re.search(PATTERNS['considered_as'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'considered_as',
            'as': match.group(1).strip()
        })
        parsed['flags'].append('rule_modifier')

    # Counts in scoring
    if re.search(PATTERNS['counts_in_scoring'], effect_text):
        parsed['modifiers'].append({'type': 'all_cards_score'})
        parsed['flags'].append('scoring_modifier')

    # Economy - free rerolls
    match = re.search(PATTERNS['free_reroll'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'free_reroll',
            'count': int(match.group(1))
        })
        parsed['flags'].append('shop_modifier')

    # Economy - debt
    match = re.search(PATTERNS['debt'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'debt_limit',
            'amount': int(match.group(1))
        })
        parsed['flags'].append('economy')

    # Economy - sell value
    if re.search(PATTERNS['sell_value'], effect_text):
        parsed['flags'].append('sell_value_modifier')

    # Economy - interest
    if re.search(PATTERNS['interest'], effect_text):
        parsed['flags'].append('interest_modifier')

    # Hand size changes
    match = re.search(PATTERNS['hand_size_plus'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'hand_size',
            'value': int(match.group(1))
        })

    match = re.search(PATTERNS['hand_size_minus'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'hand_size',
            'value': -int(match.group(1))
        })

    # Discards
    match = re.search(PATTERNS['plus_discards'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'discards',
            'value': int(match.group(1))
        })

    # Hands per round
    match = re.search(PATTERNS['plus_hands'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'hands',
            'value': int(match.group(1))
        })

    match = re.search(PATTERNS['minus_hands'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'hands',
            'value': -int(match.group(1))
        })

    # Copy effects
    if re.search(PATTERNS['copies_ability'], effect_text):
        parsed['flags'].append('copy_effect')

    # Destroy effects
    if re.search(PATTERNS['destroy'], effect_text):
        parsed['flags'].append('destroy')

    # Additional triggers
    if re.search(PATTERNS['round_begins'], effect_text):
        parsed['triggers'].append('round_start')
    if re.search(PATTERNS['booster_opened'], effect_text):
        parsed['triggers'].append('booster_opened')
    if re.search(PATTERNS['booster_skipped'], effect_text):
        parsed['triggers'].append('booster_skipped')

    # Adds X to Mult (dynamic)
    match = re.search(PATTERNS['adds_to_mult'], effect_text)
    if match and 'add_mult' not in [m['type'] for m in parsed['modifiers']]:
        parsed['modifiers'].append({
            'type': 'dynamic_mult',
            'source': match.group(1).strip()
        })
        parsed['flags'].append('dynamic')

    # === FINAL BATCH OF PATTERNS ===

    # Upgrade level (hand level up)
    if re.search(PATTERNS['upgrade_level'], effect_text):
        parsed['modifiers'].append({'type': 'upgrade_hand_level'})
        parsed['flags'].append('hand_level')

    # Become/transform cards
    match = re.search(PATTERNS['become'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'transform_card',
            'to': match.group(1)
        })
        parsed['flags'].append('card_transform')

    # Sell to activate
    if re.search(PATTERNS['sell_to'], effect_text):
        parsed['flags'].append('sell_to_activate')
        parsed['triggers'].append('on_sell')

    # Prevents death
    if re.search(PATTERNS['prevents_death'], effect_text):
        parsed['modifiers'].append({'type': 'prevent_death'})
        parsed['flags'].append('defensive')

    # Doubles probability
    if re.search(PATTERNS['doubles_probability'], effect_text):
        parsed['modifiers'].append({'type': 'double_probability'})
        parsed['flags'].append('probability_modifier')

    # Free items in shop
    if re.search(PATTERNS['are_free'], effect_text):
        parsed['modifiers'].append({'type': 'free_items'})
        parsed['flags'].append('shop_modifier')

    # Disable effects
    if re.search(PATTERNS['disable'], effect_text):
        parsed['modifiers'].append({'type': 'disable_effect'})
        parsed['flags'].append('negation')

    # Gaps in straights
    match = re.search(PATTERNS['gaps'], effect_text)
    if match:
        parsed['modifiers'].append({
            'type': 'straight_gaps',
            'gap_size': int(match.group(1))
        })
        parsed['flags'].append('rule_modifier')

    # Same suit rule modification
    if re.search(PATTERNS['same_suit'], effect_text):
        parsed['modifiers'].append({'type': 'suit_equivalence'})
        parsed['flags'].append('rule_modifier')

    # Multiple appearances in shop
    if re.search(PATTERNS['multiple_appearances'], effect_text):
        parsed['modifiers'].append({'type': 'allow_duplicates'})
        parsed['flags'].append('shop_modifier')

    # Set default trigger if none found but has modifiers
    if not parsed['triggers'] and (parsed['modifiers'] or parsed['flags']):
        # Most jokers trigger on scoring
        parsed['triggers'].append('on_score')

    return parsed


def parse_joker_csv(csv_path: str) -> list:
    """Parse the joker CSV file and return structured data."""
    jokers = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            joker = {
                'id': int(row['number']),
                'name': row['name'],
                'cost': int(row['cost'].replace('$', '')),
                'rarity': row['rarity'],
                'unlock': row['unlock'],
                'type': row['type'],
                'category': row['category'],
                'effect': parse_effect(row['effect'])
            }
            jokers.append(joker)

    return jokers


def analyze_parsing_coverage(jokers: list) -> dict:
    """Analyze how well the parser covered the joker effects."""
    stats = {
        'total': len(jokers),
        'has_modifiers': 0,
        'has_conditions': 0,
        'has_triggers': 0,
        'fully_parsed': 0,  # Has at least one modifier and trigger
        'unparsed': []  # Jokers with no extracted data
    }

    for joker in jokers:
        effect = joker['effect']
        has_mod = len(effect['modifiers']) > 0
        has_cond = len(effect['conditions']) > 0
        has_trig = len(effect['triggers']) > 0

        if has_mod:
            stats['has_modifiers'] += 1
        if has_cond:
            stats['has_conditions'] += 1
        if has_trig:
            stats['has_triggers'] += 1
        if has_mod and has_trig:
            stats['fully_parsed'] += 1

        if not has_mod and not effect['flags']:
            stats['unparsed'].append(joker['name'])

    return stats


def main():
    # Parse the joker CSV
    csv_path = Path(__file__).parent / 'balatro_data' / 'balatro_all_jokers_complete.csv'
    jokers = parse_joker_csv(csv_path)

    # Analyze coverage
    stats = analyze_parsing_coverage(jokers)

    print("=" * 60)
    print("BALATRO JOKER PARSER RESULTS")
    print("=" * 60)
    print(f"\nTotal jokers: {stats['total']}")
    print(f"With modifiers extracted: {stats['has_modifiers']} ({stats['has_modifiers']/stats['total']*100:.1f}%)")
    print(f"With conditions extracted: {stats['has_conditions']} ({stats['has_conditions']/stats['total']*100:.1f}%)")
    print(f"With triggers extracted: {stats['has_triggers']} ({stats['has_triggers']/stats['total']*100:.1f}%)")
    print(f"Fully parsed (modifier + trigger): {stats['fully_parsed']} ({stats['fully_parsed']/stats['total']*100:.1f}%)")

    print(f"\nJokers needing manual review ({len(stats['unparsed'])}):")
    for name in stats['unparsed'][:10]:
        print(f"  - {name}")
    if len(stats['unparsed']) > 10:
        print(f"  ... and {len(stats['unparsed']) - 10} more")

    # Show some example parses
    print("\n" + "=" * 60)
    print("EXAMPLE PARSED JOKERS")
    print("=" * 60)

    examples = ['Joker', 'Jolly Joker', 'Greedy Joker', 'Half Joker', 'Baron', 'Blackboard']
    for name in examples:
        joker = next((j for j in jokers if j['name'] == name), None)
        if joker:
            print(f"\n{joker['name']} (${joker['cost']}, {joker['rarity']})")
            print(f"  Raw: {joker['effect']['raw']}")
            print(f"  Modifiers: {joker['effect']['modifiers']}")
            print(f"  Conditions: {joker['effect']['conditions']}")
            print(f"  Triggers: {joker['effect']['triggers']}")
            print(f"  Flags: {joker['effect']['flags']}")

    # Save full parsed output to JSON
    output_path = Path(__file__).parent / 'jokers_parsed.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(jokers, f, indent=2)

    print(f"\n\nFull parsed data saved to: {output_path}")


if __name__ == '__main__':
    main()
