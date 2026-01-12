# Balatro Narrative Engine

**Transforming simulation data into human stories**

## What Is This?

The Narrative Engine sits on top of the Balatro simulator and extracts **story beats** from raw event logs. It identifies pivots, crises, triumphs, and defeats—then generates narratives in multiple styles.

## The Philosophy

> "Roguelikes are great storytellers. Each run is different and they tell a whole mini story."

This project explores the space between:
- **Engineering and Art**
- **Technical and Inspirational**  
- **Machine and Human Spirit**

The simulator generates deterministic data. The narrative engine finds the *meaning*.

## How It Works

```
Simulation → Event Log → Narrative Analysis → Story Generation
  (data)    (structure)    (interpretation)     (prose)
```

### 1. Event Logging

The simulator captures key events:
- Run start (jokers, deck, money)
- Blind attempts (score, margin, success)
- Shop visits (purchases, planets)
- Joker acquisitions
- Hand level-ups
- Run end (victory/defeat)

### 2. Narrative Analysis

The engine extracts **story beats**:

| Beat Type | What It Means |
|-----------|---------------|
| OPENING | How the run began |
| PIVOT | Strategic shift (key joker acquired) |
| CRISIS | Dangerous moment (blind failed or close call) |
| TRIUMPH | Crushing victory |
| CLOSE_CALL | Survived by the skin of teeth (<20% margin) |
| POWER_SPIKE | Significant power increase (planet card) |
| DEFEAT | The run ended |

### 3. Story Generation

Four narrative styles:

**Literary** - Prose narrative with emotional weight
```
THE RUN OF NARROW MARGINS

A humble beginning. No jokers. Just hope and a deck of cards.

SMALL blind. Score: 328. Required: 300. Margin: 9.3%. Barely.
```

**Haiku** - Distilled to three lines
```
Cards fall like rain drops
Each blind a mountain to climb
Victory, barely
```

**Dev Commentary** - Technical analysis
```
Archetype: Nail Biter
Tension Score: 1.00/1.0
Development Score: 0.50/1.0
```

**Comic Script** - Visual panel format
```
PANEL 1:
[Ante 1]
SMALL blind. Score: 328. Required: 300. Margin: 9.3%. Barely.
```

## Usage

### Basic

```python
from narrative_engine import process_log

story, analysis = process_log('run_log.json', style='literary')
print(story)
```

### CLI

```bash
python narrative_engine.py run_log.json literary
python narrative_engine.py run_log.json haiku
python narrative_engine.py run_log.json dev_commentary
python narrative_engine.py run_log.json comic
```

## Story Quality Metrics

The engine assesses narrative quality:

**Tension Score** (0-1)
- Based on close calls (<20% margin)
- High tension = nail-biter story

**Development Score** (0-1)
- Based on pivots and strategic shifts
- High development = interesting arc

**Archetypes**
- **Nail Biter** - Multiple close calls, high tension
- **Adaptation Saga** - Multiple pivots, strategic evolution
- **Straightforward Run** - Low drama, decisive outcome
- **Classic Roguelike** - Balanced tension and development

## Themes

Detected automatically:

- **Survival Against Odds** - Multiple close calls
- **Pure Skill** - No jokers (or minimal help)
- **Building the Machine** - Many jokers, complex build
- **Adaptation** - Mid-range complexity

## Character Arc

Jokers are treated as characters:

```python
CHARACTER ARC (JOKERS):
  Ante 1: Flower Pot (support)
  Ante 1: To the Moon (economy)
  Ante 1: Abstract Joker (scoring)
  Ante 2: Wrathful Joker (scoring)
```

Each joker has a **narrative role**:
- Economy (money generation)
- Scoring (mult/chips)
- Support (utility, hand size)
- Utility (rule changes)

## The Analysis Output

```json
{
  "beats": [...],
  "theme": "survival_against_odds",
  "character_arc": [...],
  "emotional_arc": [0.3, 0.9, 0.9, 0.7, ...],
  "story_quality": {
    "overall": 0.73,
    "tension": 1.00,
    "development": 0.50,
    "archetype": "nail_biter"
  }
}
```

## Future Directions

Potential expansions:

1. **Visual Timeline** - Graph emotional arc over time
2. **Comparative Analysis** - Compare runs, find patterns
3. **AI Narrator** - Live commentary during simulation
4. **Tarot Reading** - Mystical interpretation of runs
5. **Gallery Mode** - Beautiful visual summaries
6. **Music Generation** - Soundtracks based on emotional arc
7. **Twitter Bot** - Share great stories automatically

## The Meta

This is an exploration of **emergent narrative**. 

The simulation is deterministic. The story is interpretive.

The same event log could be read a thousand ways. This engine represents *one* interpretation—shaped by design choices, thematic emphasis, and aesthetic values.

**The art is in the interpretation layer.**

## Examples

See `example_narratives.md` for sample outputs from a real run.

## Requirements

- Python 3.10+
- Standard library only (no dependencies!)

## License

Created for exploration of the space between technical and inspirational.

---

*"Each run is a story. This engine finds it."*
