"""
Balatro Narrative Engine
Transforms simulation logs into human stories.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class BeatType(Enum):
    """Types of narrative beats"""
    OPENING = "opening"
    PIVOT = "pivot"
    CRISIS = "crisis"
    TRIUMPH = "triumph"
    DEFEAT = "defeat"
    CLOSE_CALL = "close_call"
    POWER_SPIKE = "power_spike"
    DESPERATION = "desperation"


@dataclass
class StoryBeat:
    """A moment in the narrative"""
    beat_type: BeatType
    ante: int
    description: str
    data: Dict[str, Any]
    emotional_weight: float  # 0.0 to 1.0


class NarrativeEngine:
    """Extracts narrative structure from simulation logs"""
    
    def __init__(self):
        self.beats: List[StoryBeat] = []
        self.theme = None
        self.character_arc = []  # The jokers as characters
        
    def analyze_run(self, log_data: Dict) -> Dict[str, Any]:
        """Main analysis - extract narrative structure"""
        events = log_data['events']
        metadata = log_data['metadata']
        
        # Extract story beats
        self.beats = []
        self._extract_opening(events)
        self._extract_journey(events)
        self._extract_ending(events)
        
        # Identify theme
        self.theme = self._identify_theme(events)
        
        # Build character arc (jokers)
        self.character_arc = self._build_character_arc(events)
        
        return {
            "beats": [self._beat_to_dict(b) for b in self.beats],
            "theme": self.theme,
            "character_arc": self.character_arc,
            "emotional_arc": self._build_emotional_arc(),
            "story_quality": self._assess_story_quality()
        }
    
    def _extract_opening(self, events: List[Dict]):
        """The beginning - how did it start?"""
        run_start = next(e for e in events if e['event_type'] == 'run_start')
        starting_jokers = run_start['data']['starting_jokers']
        
        if not starting_jokers:
            desc = "A humble beginning. No jokers. Just hope and a deck of cards."
            weight = 0.3
        else:
            desc = f"The run began with {', '.join(starting_jokers)}. The path was set."
            weight = 0.5
            
        self.beats.append(StoryBeat(
            beat_type=BeatType.OPENING,
            ante=1,
            description=desc,
            data={"starting_jokers": starting_jokers},
            emotional_weight=weight
        ))
    
    def _extract_journey(self, events: List[Dict]):
        """The middle - pivots, crises, triumphs"""
        current_jokers = []
        
        for event in events:
            event_type = event['event_type']
            
            # Joker acquisitions (potential pivots)
            if event_type == 'joker_acquired':
                joker = event['data']['joker']
                current_jokers.append(joker)
                
                # Is this a pivot moment?
                if self._is_pivot(joker, current_jokers):
                    self.beats.append(StoryBeat(
                        beat_type=BeatType.PIVOT,
                        ante=event['ante'],
                        description=f"{joker} appeared. Everything changed.",
                        data={"joker": joker, "ante": event['ante']},
                        emotional_weight=0.7
                    ))
            
            # Blind results (crises, triumphs, close calls)
            elif event_type == 'blind_result':
                data = event['data']
                
                if data['close_call'] and data['success']:
                    # Close call - survived by the skin of teeth
                    margin_pct = abs(data['margin_pct'])
                    self.beats.append(StoryBeat(
                        beat_type=BeatType.CLOSE_CALL,
                        ante=event['ante'],
                        description=f"{event['blind_type']} blind. Score: {data['score']:,}. Required: {data['required']:,}. Margin: {margin_pct:.1f}%. Barely.",
                        data=data,
                        emotional_weight=0.9
                    ))
                
                elif data['success'] and data['margin_pct'] > 100:
                    # Crushing victory
                    self.beats.append(StoryBeat(
                        beat_type=BeatType.TRIUMPH,
                        ante=event['ante'],
                        description=f"{event['blind_type']} blind crushed. {data['score']:,} points. The build was working.",
                        data=data,
                        emotional_weight=0.6
                    ))
                
                elif not data['success']:
                    # The end
                    self.beats.append(StoryBeat(
                        beat_type=BeatType.CRISIS,
                        ante=event['ante'],
                        description=f"{event['blind_type']} blind. Scored {data['score']:,}. Needed {data['required']:,}. Not enough.",
                        data=data,
                        emotional_weight=1.0
                    ))
            
            # Hand level ups (power spikes)
            elif event_type == 'hand_level_up':
                data = event['data']
                self.beats.append(StoryBeat(
                    beat_type=BeatType.POWER_SPIKE,
                    ante=event['ante'],
                    description=f"{data['planet']} leveled {data['hand_type'].replace('_', ' ')} to {data['new_level']}.",
                    data=data,
                    emotional_weight=0.5
                ))
    
    def _extract_ending(self, events: List[Dict]):
        """The resolution"""
        run_end = next(e for e in events if e['event_type'] == 'run_end')
        data = run_end['data']
        
        if data['victory']:
            desc = f"Victory. Ante {run_end['ante']} conquered. The dream realized."
            beat_type = BeatType.TRIUMPH
            weight = 1.0
        else:
            desc = f"Defeat at Ante {run_end['ante']}. {data['blinds_beaten']} blinds beaten. The run ended, but the lesson remained."
            beat_type = BeatType.DEFEAT
            weight = 0.8
            
        self.beats.append(StoryBeat(
            beat_type=beat_type,
            ante=run_end['ante'],
            description=desc,
            data=data,
            emotional_weight=weight
        ))
    
    def _is_pivot(self, joker: str, current_jokers: List[str]) -> bool:
        """Determine if this joker represents a strategic pivot"""
        # First joker is always significant
        if len(current_jokers) == 1:
            return True
        
        # Certain jokers are pivot-worthy
        pivot_jokers = [
            "Blueprint", "Brainstorm", "The Duo", "The Trio", "The Family",
            "Bloodstone", "Smeared Joker", "Four Fingers", "Splash"
        ]
        
        return joker in pivot_jokers
    
    def _identify_theme(self, events: List[Dict]) -> str:
        """What is this run about?"""
        jokers = [e['data']['joker'] for e in events if e['event_type'] == 'joker_acquired']
        close_calls = sum(1 for e in events if e['event_type'] == 'blind_result' and e['data'].get('close_call'))
        
        if close_calls >= 3:
            return "survival_against_odds"
        elif len(jokers) == 0:
            return "pure_skill"
        elif len(jokers) >= 4:
            return "building_the_machine"
        else:
            return "adaptation"
    
    def _build_character_arc(self, events: List[Dict]) -> List[Dict]:
        """Jokers as characters in the story"""
        arc = []
        for event in events:
            if event['event_type'] == 'joker_acquired':
                arc.append({
                    "ante": event['ante'],
                    "joker": event['data']['joker'],
                    "role": self._classify_joker_role(event['data']['joker'])
                })
        return arc
    
    def _classify_joker_role(self, joker: str) -> str:
        """What narrative role does this joker play?"""
        # This could be expanded with joker data
        if "Moon" in joker:
            return "economy"
        elif any(word in joker.lower() for word in ["mult", "chip", "joker"]):
            return "scoring"
        elif any(word in joker.lower() for word in ["flower", "pot", "abstract"]):
            return "support"
        else:
            return "utility"
    
    def _build_emotional_arc(self) -> List[float]:
        """Emotional intensity over time"""
        return [beat.emotional_weight for beat in self.beats]
    
    def _assess_story_quality(self) -> Dict[str, Any]:
        """How good is this story?"""
        close_calls = sum(1 for b in self.beats if b.beat_type == BeatType.CLOSE_CALL)
        pivots = sum(1 for b in self.beats if b.beat_type == BeatType.PIVOT)
        
        # Good stories have tension (close calls) and development (pivots)
        tension_score = min(close_calls / 2, 1.0)
        development_score = min(pivots / 2, 1.0)
        
        avg_emotional_weight = sum(b.emotional_weight for b in self.beats) / len(self.beats)
        
        quality = (tension_score + development_score + avg_emotional_weight) / 3
        
        return {
            "overall": quality,
            "tension": tension_score,
            "development": development_score,
            "emotional_intensity": avg_emotional_weight,
            "archetype": self._determine_archetype(close_calls, pivots)
        }
    
    def _determine_archetype(self, close_calls: int, pivots: int) -> str:
        """What kind of story is this?"""
        if close_calls >= 3:
            return "nail_biter"
        elif pivots >= 2:
            return "adaptation_saga"
        elif close_calls == 0 and pivots <= 1:
            return "straightforward_run"
        else:
            return "classic_roguelike"
    
    def _beat_to_dict(self, beat: StoryBeat) -> Dict:
        """Convert beat to dict for serialization"""
        return {
            "type": beat.beat_type.value,
            "ante": beat.ante,
            "description": beat.description,
            "data": beat.data,
            "emotional_weight": beat.emotional_weight
        }


class StoryGenerator:
    """Generates prose from narrative structure"""
    
    def __init__(self, style: str = "literary"):
        self.style = style
    
    def generate(self, analysis: Dict, log_data: Dict) -> str:
        """Generate narrative in specified style"""
        if self.style == "literary":
            return self._literary_prose(analysis, log_data)
        elif self.style == "comic":
            return self._comic_script(analysis, log_data)
        elif self.style == "haiku":
            return self._haiku(analysis, log_data)
        elif self.style == "dev_commentary":
            return self._dev_commentary(analysis, log_data)
        else:
            return self._basic_summary(analysis, log_data)
    
    def _literary_prose(self, analysis: Dict, log_data: Dict) -> str:
        """Literary narrative style"""
        beats = analysis['beats']
        theme = analysis['theme']
        quality = analysis['story_quality']
        
        lines = []
        
        # Title based on theme
        titles = {
            "survival_against_odds": "THE RUN OF NARROW MARGINS",
            "pure_skill": "THE ASCETIC'S PATH",
            "building_the_machine": "THE ARCHITECT'S DREAM",
            "adaptation": "THE PIVOT"
        }
        lines.append(titles.get(theme, "A BALATRO RUN"))
        lines.append("")
        
        # Opening
        for beat in beats:
            if beat['type'] == 'opening':
                lines.append(beat['description'])
                lines.append("")
        
        # Journey - group by ante
        current_ante = 1
        for beat in beats[1:-1]:  # Skip opening and ending
            if beat['ante'] > current_ante:
                lines.append("")
                lines.append(f"Ante {beat['ante']}.")
                current_ante = beat['ante']
            
            # Add beat with emphasis for high emotional weight
            if beat['emotional_weight'] > 0.8:
                lines.append(f"{beat['description']}")
            else:
                lines.append(beat['description'])
        
        # Ending
        lines.append("")
        for beat in beats:
            if beat['type'] in ['triumph', 'defeat']:
                lines.append(beat['description'])
        
        # Story quality note
        lines.append("")
        lines.append(f"[Story Quality: {quality['archetype'].replace('_', ' ').title()}]")
        
        return "\n".join(lines)
    
    def _comic_script(self, analysis: Dict, log_data: Dict) -> str:
        """Comic book panel format"""
        beats = analysis['beats']
        
        lines = ["COMIC SCRIPT: BALATRO RUN", "=" * 40, ""]
        
        panel = 1
        for beat in beats:
            if beat['emotional_weight'] > 0.6:  # Only significant moments
                lines.append(f"PANEL {panel}:")
                lines.append(f"[Ante {beat['ante']}]")
                lines.append(f"{beat['description']}")
                lines.append("")
                panel += 1
        
        return "\n".join(lines)
    
    def _haiku(self, analysis: Dict, log_data: Dict) -> str:
        """Distill to haiku"""
        beats = analysis['beats']
        theme = analysis['theme']
        
        # Create haiku based on theme
        haikus = {
            "survival_against_odds": [
                "Cards fall like rain drops",
                "Each blind a mountain to climb",
                "Victory, barely"
            ],
            "pure_skill": [
                "No jokers needed",
                "Pure strategy prevails",
                "Or does it falter?"
            ],
            "building_the_machine": [
                "Jokers gather round",
                "Each piece clicks into place",
                "The engine roars loud"
            ],
            "adaptation": [
                "The path was not clear",
                "But the pivot came in time",
                "Hope found its way through"
            ]
        }
        
        return "\n".join(haikus.get(theme, ["Run attempted", "Cards were played", "Outcome occurred"]))
    
    def _dev_commentary(self, analysis: Dict, log_data: Dict) -> str:
        """Developer analysis style"""
        quality = analysis['story_quality']
        theme = analysis['theme']
        character_arc = analysis['character_arc']
        
        lines = [
            "RUN ANALYSIS: POST-MORTEM",
            "=" * 60,
            "",
            f"Archetype: {quality['archetype'].replace('_', ' ').title()}",
            f"Theme: {theme.replace('_', ' ').title()}",
            f"Tension Score: {quality['tension']:.2f}/1.0",
            f"Development Score: {quality['development']:.2f}/1.0",
            f"Overall Quality: {quality['overall']:.2f}/1.0",
            "",
            "NARRATIVE STRUCTURE:",
        ]
        
        for beat in analysis['beats']:
            lines.append(f"  Ante {beat['ante']}: {beat['type'].upper()} - {beat['description'][:60]}...")
        
        lines.append("")
        lines.append("CHARACTER ARC (JOKERS):")
        for char in character_arc:
            lines.append(f"  Ante {char['ante']}: {char['joker']} ({char['role']})")
        
        lines.append("")
        lines.append("DESIGNER NOTES:")
        if quality['archetype'] == 'nail_biter':
            lines.append("  Player experienced high tension throughout. Good pacing.")
        elif quality['archetype'] == 'straightforward_run':
            lines.append("  Low tension. Either crushing victory or early defeat.")
        else:
            lines.append("  Classic roguelike experience with meaningful decisions.")
        
        return "\n".join(lines)
    
    def _basic_summary(self, analysis: Dict, log_data: Dict) -> str:
        """Simple summary"""
        summary = log_data.get('summary', {})
        
        return f"""
Run Summary:
- Blinds Won: {summary.get('blinds_won', 0)}/{summary.get('total_blinds_attempted', 0)}
- Close Calls: {summary.get('close_calls', 0)}
- Jokers Acquired: {summary.get('jokers_acquired', 0)}
- Victory: {summary.get('victory', False)}
"""


def process_log(log_path: str, style: str = "literary") -> str:
    """Main entry point - process a log file and generate narrative"""
    with open(log_path) as f:
        log_data = json.load(f)
    
    # Analyze
    engine = NarrativeEngine()
    analysis = engine.analyze_run(log_data)
    
    # Generate story
    generator = StoryGenerator(style=style)
    story = generator.generate(analysis, log_data)
    
    return story, analysis


if __name__ == "__main__":
    # Test with the uploaded log
    import sys
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        style = sys.argv[2] if len(sys.argv) > 2 else "literary"
        story, analysis = process_log(log_path, style)
        print(story)
        print("\n" + "=" * 60)
        print(f"\nAnalysis saved to {log_path.replace('.json', '_analysis.json')}")
        
        with open(log_path.replace('.json', '_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
