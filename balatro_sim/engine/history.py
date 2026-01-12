"""
Run history tracking for narrative generation.
Captures key events during simulation for storytelling.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class RunEvent:
    """Single event in a run."""
    ante: int
    blind_type: Optional[str]
    event_type: str  # "run_start", "shop_visit", "blind_result", "joker_acquired", etc.
    data: dict
    timestamp: int = 0  # event sequence number


class RunHistory:
    """Captures the narrative of a run."""

    def __init__(self, preset_name: str, deck_type: str):
        self.events: list[RunEvent] = []
        self.metadata = {
            "preset": preset_name,
            "deck": deck_type,
        }
        self._event_counter = 0

    def add_event(self, ante: int, event_type: str, data: dict, blind_type: str = None):
        """Add an event to the history."""
        self.events.append(RunEvent(
            ante=ante,
            blind_type=blind_type,
            event_type=event_type,
            data=data,
            timestamp=self._event_counter
        ))
        self._event_counter += 1

    def add_run_start(self, money: int, jokers: list, hand_levels: dict = None):
        """Log the start of a run."""
        self.add_event(
            ante=1,
            event_type="run_start",
            data={
                "starting_money": money,
                "starting_jokers": jokers,
                "starting_hand_levels": hand_levels or {}
            }
        )

    def add_shop_visit(self, ante: int, jokers_bought: list, planets_used: list,
                       money_spent: int, money_remaining: int,
                       vouchers_bought: list = None, tarots_used: list = None,
                       spectrals_used: list = None):
        """Log shop purchases."""
        data = {
            "jokers_bought": jokers_bought,
            "planets_used": planets_used,
            "money_spent": money_spent,
            "money_remaining": money_remaining
        }
        if vouchers_bought:
            data["vouchers_bought"] = vouchers_bought
        if tarots_used:
            data["tarots_used"] = tarots_used
        if spectrals_used:
            data["spectrals_used"] = spectrals_used

        self.add_event(
            ante=ante,
            event_type="shop_visit",
            data=data
        )

    def add_voucher_acquired(self, ante: int, voucher_name: str, effect_summary: str = None):
        """Log voucher purchase."""
        self.add_event(
            ante=ante,
            event_type="voucher_acquired",
            data={
                "voucher": voucher_name,
                "effect": effect_summary
            }
        )

    def add_consumable_used(self, ante: int, consumable_name: str, consumable_type: str,
                            effect_description: str):
        """Log tarot/spectral usage."""
        self.add_event(
            ante=ante,
            event_type="consumable_used",
            data={
                "name": consumable_name,
                "type": consumable_type,
                "effect": effect_description
            }
        )

    def add_blind_result(self, ante: int, blind_type: str, score: int,
                         required: int, success: bool, hands_used: int,
                         discards_used: int = 0, best_hand: str = None,
                         boss_name: str = None, hands_played: list = None):
        """Log blind attempt."""
        margin = score - required
        margin_pct = (margin / required * 100) if required > 0 else 0

        data = {
            "score": score,
            "required": required,
            "success": success,
            "margin": margin,
            "margin_pct": round(margin_pct, 1),
            "hands_used": hands_used,
            "discards_used": discards_used,
            "best_hand": best_hand,
            "close_call": abs(margin_pct) < 20
        }

        if boss_name:
            data["boss_name"] = boss_name

        if hands_played:
            data["hands_played"] = hands_played  # List of (hand_type, score) tuples

        self.add_event(
            ante=ante,
            blind_type=blind_type,
            event_type="blind_result",
            data=data
        )

    def add_joker_acquired(self, ante: int, joker_name: str, source: str = "shop"):
        """Log joker acquisition."""
        self.add_event(
            ante=ante,
            event_type="joker_acquired",
            data={
                "joker": joker_name,
                "source": source
            }
        )

    def add_joker_destroyed(self, ante: int, joker_name: str, reason: str = "sold"):
        """Log joker removal."""
        self.add_event(
            ante=ante,
            event_type="joker_destroyed",
            data={
                "joker": joker_name,
                "reason": reason
            }
        )

    def add_planet_used(self, ante: int, planet: str, hand_type: str, new_level: int):
        """Log hand level up."""
        self.add_event(
            ante=ante,
            event_type="hand_level_up",
            data={
                "planet": planet,
                "hand_type": hand_type,
                "new_level": new_level
            }
        )

    def add_pivot(self, ante: int, old_strategy: str, new_strategy: str, trigger: str):
        """Log strategy pivot (when build direction changes)."""
        self.add_event(
            ante=ante,
            event_type="pivot",
            data={
                "from": old_strategy,
                "to": new_strategy,
                "trigger": trigger
            }
        )

    def add_run_end(self, success: bool, final_ante: int, final_blind: str,
                    blinds_beaten: int, final_money: int, jokers: list,
                    hand_levels: dict = None):
        """Log run completion."""
        self.add_event(
            ante=final_ante,
            blind_type=final_blind,
            event_type="run_end",
            data={
                "victory": success,
                "blinds_beaten": blinds_beaten,
                "final_money": final_money,
                "final_jokers": jokers,
                "final_hand_levels": hand_levels or {}
            }
        )

    def get_close_calls(self) -> list[RunEvent]:
        """Get all close call events."""
        return [e for e in self.events
                if e.event_type == "blind_result" and e.data.get("close_call")]

    def get_joker_timeline(self) -> list[dict]:
        """Get timeline of joker acquisitions/losses."""
        return [{"ante": e.ante, "type": e.event_type, **e.data}
                for e in self.events
                if e.event_type in ("joker_acquired", "joker_destroyed")]

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "metadata": self.metadata,
            "events": [asdict(e) for e in self.events],
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> dict:
        """Generate a quick summary of the run."""
        blind_results = [e for e in self.events if e.event_type == "blind_result"]
        close_calls = [e for e in blind_results if e.data.get("close_call")]
        shop_visits = [e for e in self.events if e.event_type == "shop_visit"]

        run_end = next((e for e in self.events if e.event_type == "run_end"), None)

        # Count boss fights
        boss_fights = [e for e in blind_results if e.data.get("boss_name")]

        return {
            "total_blinds_attempted": len(blind_results),
            "blinds_won": sum(1 for e in blind_results if e.data.get("success")),
            "close_calls": len(close_calls),
            "jokers_acquired": sum(1 for e in self.events if e.event_type == "joker_acquired"),
            "vouchers_acquired": sum(1 for e in self.events if e.event_type == "voucher_acquired"),
            "planets_used": sum(len(e.data.get("planets_used", [])) for e in shop_visits),
            "tarots_used": sum(len(e.data.get("tarots_used", []) or []) for e in shop_visits),
            "spectrals_used": sum(len(e.data.get("spectrals_used", []) or []) for e in shop_visits),
            "boss_fights": len(boss_fights),
            "bosses_defeated": sum(1 for e in boss_fights if e.data.get("success")),
            "victory": run_end.data.get("victory") if run_end else False
        }

    def save(self, filepath: str):
        """Save run history to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'RunHistory':
        """Load run history from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        history = cls(
            preset_name=data["metadata"]["preset"],
            deck_type=data["metadata"]["deck"]
        )
        history.metadata = data["metadata"]

        for event_data in data["events"]:
            history.events.append(RunEvent(**event_data))
            history._event_counter = max(history._event_counter, event_data["timestamp"] + 1)

        return history
