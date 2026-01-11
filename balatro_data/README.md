# Balatro Game Content Data - CSV Collection

## Overview
This collection contains comprehensive data for the roguelike deckbuilder game **Balatro** (as of update 1.0.1o-FULL). All data is organized into CSV files for easy analysis and use.

## Files Included

### 1. **balatro_all_jokers_complete.csv** (18 KB)
Complete dataset of all 150 Jokers in the game.

**Columns:**
- `number` - Joker ID number (1-150)
- `name` - Joker name
- `effect` - What the joker does
- `cost` - Shop purchase price
- `rarity` - Common (61), Uncommon (65), Rare (19), or Legendary (5)
- `unlock` - Unlock requirement
- `type` - Scoring type (Additive Mult, Multiplicative Mult, Chips, Effect, etc.)
- `category` - Thematic category (Hand-based, Suit-based, Economy, etc.)

**Rarity Breakdown:**
- Common: 61 jokers
- Uncommon: 65 jokers  
- Rare: 19 jokers
- Legendary: 5 jokers

---

### 2. **balatro_tarot_cards.csv** (1.8 KB)
All 22 Tarot cards (consumables).

**Columns:**
- `name` - Tarot card name
- `effect` - What the card does when used
- `availability` - When it becomes available

---

### 3. **balatro_planet_cards.csv** (487 B)
All 12 Planet cards for leveling poker hands.

**Columns:**
- `name` - Planet name
- `poker_hand` - Which poker hand it levels up
- `effect` - Level up effect

**List:** Mercury (Pair), Venus (Three of a Kind), Earth (Full House), Mars (Two Pair), Jupiter (Flush Five), Saturn (Straight Flush), Uranus (Royal Flush), Neptune (Straight), Pluto (High Card), Planet X (Four of a Kind), Ceres (Flush House), Eris (Flush)

---

### 4. **balatro_spectral_cards.csv** (1.6 KB)
All 18 Spectral cards (powerful but risky consumables).

**Columns:**
- `name` - Spectral card name
- `effect` - What the card does (often with drawbacks)
- `availability` - When it becomes available

**Notable cards:** The Soul (creates Legendary Joker), Black Hole (upgrades all poker hands), Ankh, Hex, Cryptid

---

### 5. **balatro_decks.csv** (1.6 KB)
All 15 playable deck types.

**Columns:**
- `name` - Deck name
- `effect` - Starting bonus/modifier
- `unlock` - How to unlock the deck

**Decks include:** Red, Blue, Yellow, Green, Black, Magic, Nebula, Ghost, Abandoned, Checkered, Zodiac, Painted, Anaglyph, Plasma, Erratic

---

### 6. **balatro_vouchers.csv**
All 32 vouchers (permanent upgrades).

**Columns:**
- `name` - Voucher name
- `tier` - Base or Tier 2
- `effect` - Permanent effect
- `requires` - Prerequisite voucher (for Tier 2)

**Examples:** Overstock (+shop slots), Crystal Ball (+consumable slots), Telescope (planet targeting), Grabber (+hands per round)

---

### 7. **balatro_enhancements.csv**
All 8 card enhancements.

**Columns:**
- `name` - Enhancement name
- `effect` - What it does to the card
- `symbol` - Visual indicator

**Enhancements:** Bonus, Mult, Wild, Glass, Steel, Stone, Gold, Lucky

---

### 8. **balatro_seals.csv**
All 4 card seal types.

**Columns:**
- `name` - Seal name
- `color` - Visual color
- `effect` - Special ability

**Seals:** Gold (earn $3), Red (retrigger), Blue (create Planet), Purple (create Tarot)

---

### 9. **balatro_editions.csv**
All 5 card edition types.

**Columns:**
- `name` - Edition name
- `effect` - Bonus provided

**Editions:** Base, Foil (+50 Chips), Holographic (+10 Mult), Polychrome (X1.5 Mult), Negative (+1 Joker slot)

---

### 10. **balatro_boss_blinds.csv**
All 28 Boss Blinds (challenge rounds).

**Columns:**
- `name` - Boss Blind name
- `ante` - Which ante it appears in
- `effect` - Challenge modifier

**Examples:** The Hook (discards 2 cards), The Psychic (must play 5 cards), The Flint (halves base values)

---

### 11. **balatro_poker_hands.csv**
All 13 poker hand types with base values.

**Columns:**
- `hand` - Hand name
- `base_chips` - Starting chip value
- `base_mult` - Starting multiplier
- `example` - How to make the hand

**Ranked from best to worst:** Flush Five, Flush House, Five of a Kind, Royal Flush, Straight Flush, Four of a Kind, Full House, Flush, Straight, Three of a Kind, Two Pair, Pair, High Card

---

### 12. **balatro_stakes.csv**
All 8 difficulty stakes.

**Columns:**
- `name` - Stake name
- `color` - Visual color
- `difficulty` - Difficulty rating (1-8)
- `effect` - Difficulty modifier

**Stakes:** White (base), Red, Green, Black, Blue, Purple, Orange, Gold (hardest)

---

## Data Sources

This data was compiled from:
- Balatro Wiki (Fandom)
- Balatro Wiki (balatrowiki.org)
- Official game information (as of update 1.0.1o-FULL)
- Community documentation

## Use Cases

This dataset is ideal for:
- Game analysis and strategy development
- Joker combination optimization
- Statistical analysis of game balance
- Building Balatro tools and calculators
- Data science projects
- Machine learning applications
- Community resources and guides

## Notes

- All 150 jokers are included in the complete dataset
- Legendary jokers can only be obtained via The Soul spectral card
- Some jokers have conditional unlock requirements
- Vouchers come in base + tier 2 pairs
- Stakes increase in difficulty from White to Gold

## Updates

**Current Version:** 1.0.1o-FULL
**Date Compiled:** January 10, 2026

For the most up-to-date information, consult the official Balatro wiki or game documentation.

---

## File Sizes Summary

```
balatro_all_jokers_complete.csv    18 KB  (150 jokers)
balatro_tarot_cards.csv             1.8 KB (22 cards)
balatro_planet_cards.csv            487 B  (12 cards)
balatro_spectral_cards.csv          1.6 KB (18 cards)
balatro_decks.csv                   1.6 KB (15 decks)
balatro_vouchers.csv                2.4 KB (32 vouchers)
balatro_enhancements.csv            450 B  (8 types)
balatro_seals.csv                   280 B  (4 types)
balatro_editions.csv                250 B  (5 types)
balatro_boss_blinds.csv             2.1 KB (28 bosses)
balatro_poker_hands.csv             620 B  (13 hands)
balatro_stakes.csv                  510 B  (8 stakes)
```

**Total:** ~30 KB of comprehensive Balatro game data

---

Created for analysis and research purposes. All content belongs to LocalThunk and Playstack.
