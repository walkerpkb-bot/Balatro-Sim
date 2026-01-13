"""
Main API for Balatro simulation.
Provides clean interface for running simulations.
"""

__version__ = "2.0.0"  # Added strategy_override support

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .engine.game import simulate_run, simulate_blind, simulate_shop, GameConfig, GameState
from .engine.strategy import BasicStrategy, SmartStrategy, OptimizedStrategy, AggressiveStrategy, CoachStrategy
from .engine.hand_detector import HandType
from .presets import Preset, StrategyType, DeckType, get_preset, list_presets, PRESETS


@dataclass
class BlindDetail:
    """Details of a single blind attempt."""
    ante: int
    blind_type: str
    boss_name: str  # None for small/big blinds
    score: int
    required: int
    success: bool
    hands_played: list  # List of (hand_type, score) tuples
    hands_used: int
    discards_used: int

    @property
    def margin_pct(self) -> float:
        if self.required == 0:
            return 0
        return (self.score - self.required) / self.required * 100


@dataclass
class ShopDetail:
    """Details of a shop visit."""
    ante: int
    jokers_bought: list[str]
    vouchers_bought: list[str]
    planets_used: list[str]
    money_spent: int
    packs_opened: list[dict] = None  # List of {"type": str, "is_mega": bool, "choices": list}


@dataclass
class RunSummary:
    """Summary of a simulation run."""
    victory: bool
    ante_reached: int
    blind_reached: str
    blinds_beaten: int
    final_money: int
    jokers_collected: list[str]
    planets_used: int
    hand_levels: dict[str, int]
    preset_used: str
    # Detailed history
    blind_history: list[BlindDetail] = None
    shop_history: list[ShopDetail] = None
    vouchers_acquired: list[str] = None
    bosses_encountered: list[str] = None

    def __str__(self):
        result = "VICTORY!" if self.victory else "DEFEAT"
        lines = [
            f"{'='*50}",
            f"  {result} - Ante {self.ante_reached} {self.blind_reached}",
            f"{'='*50}",
            f"  Blinds beaten: {self.blinds_beaten}/24",
            f"  Final money: ${self.final_money}",
            f"  Jokers: {', '.join(self.jokers_collected) if self.jokers_collected else 'None'}",
            f"  Planets used: {self.planets_used}",
        ]

        # Show leveled hands
        leveled = {k: v for k, v in self.hand_levels.items() if v > 1}
        if leveled:
            levels_str = ", ".join(f"{k}:{v}" for k, v in sorted(leveled.items(), key=lambda x: -x[1])[:5])
            lines.append(f"  Hand levels: {levels_str}")

        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "victory": self.victory,
            "ante_reached": self.ante_reached,
            "blind_reached": self.blind_reached,
            "blinds_beaten": self.blinds_beaten,
            "final_money": self.final_money,
            "jokers_collected": self.jokers_collected,
            "planets_used": self.planets_used,
            "hand_levels": self.hand_levels,
            "preset_used": self.preset_used,
        }


@dataclass
class BatchResult:
    """Results from multiple simulation runs."""
    runs: int
    wins: int
    win_rate: float
    avg_blinds: float
    avg_ante: float
    max_ante: int
    avg_money: float
    avg_jokers: float
    avg_planets: float
    ante_distribution: dict[int, int]
    preset_used: str

    def __str__(self):
        lines = [
            f"{'='*50}",
            f"  BATCH RESULTS ({self.runs} runs)",
            f"  Preset: {self.preset_used}",
            f"{'='*50}",
            f"  Win rate: {self.wins}/{self.runs} ({self.win_rate:.1f}%)",
            f"  Avg blinds beaten: {self.avg_blinds:.1f}",
            f"  Avg ante reached: {self.avg_ante:.1f}",
            f"  Max ante reached: {self.max_ante}",
            f"  Avg final money: ${self.avg_money:.0f}",
            f"  Avg jokers collected: {self.avg_jokers:.1f}",
            f"  Avg planets used: {self.avg_planets:.1f}",
            "",
            "  Ante distribution:",
        ]

        for ante in sorted(self.ante_distribution.keys()):
            count = self.ante_distribution[ante]
            pct = count / self.runs * 100
            bar = "█" * int(pct / 2)
            lines.append(f"    Ante {ante}: {count:>3} ({pct:>5.1f}%) {bar}")

        lines.append(f"{'='*50}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "runs": self.runs,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "avg_blinds": self.avg_blinds,
            "avg_ante": self.avg_ante,
            "max_ante": self.max_ante,
            "avg_money": self.avg_money,
            "avg_jokers": self.avg_jokers,
            "avg_planets": self.avg_planets,
            "ante_distribution": self.ante_distribution,
            "preset_used": self.preset_used,
        }


class Simulator:
    """
    Main simulator class.

    Usage:
        sim = Simulator()
        result = sim.run("flush_build")
        print(result)

        # Or run many:
        batch = sim.run_batch("standard", runs=100)
        print(batch)
    """

    def __init__(self, jokers_path: str = None):
        """Initialize simulator with joker data."""
        if jokers_path is None:
            jokers_path = Path(__file__).parent.parent / "jokers_parsed.json"

        with open(jokers_path) as f:
            self.all_jokers = json.load(f)

        self._joker_lookup = {j["name"]: j for j in self.all_jokers}

    def _get_strategy(self, strategy_type: StrategyType):
        """Get strategy instance from type."""
        strategies = {
            StrategyType.BASIC: BasicStrategy,
            StrategyType.SMART: SmartStrategy,
            StrategyType.OPTIMIZED: OptimizedStrategy,
            StrategyType.AGGRESSIVE: AggressiveStrategy,
            StrategyType.COACH: CoachStrategy,
        }
        return strategies.get(strategy_type, SmartStrategy)()

    def _get_jokers(self, names: list[str]) -> list[dict]:
        """Get joker dicts from names."""
        return [self._joker_lookup[n] for n in names if n in self._joker_lookup]

    def get_available_presets(self) -> list[dict]:
        """Get list of available presets with info."""
        return [
            {
                "id": key,
                "name": p.name,
                "description": p.description,
                "strategy": p.strategy.value,
                "starting_jokers": p.starting_jokers,
            }
            for key, p in PRESETS.items()
        ]

    def get_available_jokers(self) -> list[str]:
        """Get list of all joker names."""
        return [j["name"] for j in self.all_jokers]

    def run(self, preset: Union[str, Preset] = "standard",
            verbose: bool = False,
            strategy_override=None) -> RunSummary:
        """
        Run a single simulation with optional strategy override.

        Args:
            preset: Preset name (string) or Preset object
            verbose: Print detailed output during run
            strategy_override: StrategyType to override preset's default strategy

        Returns:
            RunSummary with results
        """
        # Get preset
        if isinstance(preset, str):
            p = get_preset(preset)
            if p is None:
                raise ValueError(f"Unknown preset: {preset}. Available: {list_presets()}")
            preset_name = preset
        else:
            p = preset
            preset_name = p.name

        # Build config
        config = GameConfig()
        for key, value in p.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Get strategy (use override if provided)
        strategy_type = strategy_override if strategy_override else p.strategy
        strategy = self._get_strategy(strategy_type)

        # Get starting jokers
        starting_jokers = self._get_jokers(p.starting_jokers)

        # Create game state with history tracking
        game = GameState(
            config=config,
            preset_name=preset_name,
            deck_type=p.deck_type.value
        )

        # Apply preset hand levels
        for hand_name, level in p.hand_levels.items():
            try:
                ht = HandType[hand_name]
                game.hand_levels[ht] = level
            except KeyError:
                pass
        game.hand_detector.hand_levels = game.hand_levels

        # Add starting jokers
        for joker in starting_jokers:
            game.add_joker(joker, source="starting")

        # Log run start
        game.history.add_run_start(
            money=game.money,
            jokers=[j.get("name", "?") for j in game.jokers],
            hand_levels={ht.name: lv for ht, lv in game.hand_levels.items() if lv > 1}
        )

        # Run simulation
        blinds_beaten = 0
        victory = False

        while True:
            result = simulate_blind(game, strategy)

            if verbose:
                status = "WIN" if result.success else "LOSS"
                print(f"Ante {game.ante} {game.current_blind.name}: "
                      f"{result.score_achieved:,}/{result.score_required:,} - {status}")

            if not result.success:
                break

            blinds_beaten += 1

            # Shop visit
            if config.enable_shop:
                shop_result = simulate_shop(game, self.all_jokers)
                if verbose and (shop_result["jokers_bought"] or shop_result["planets_used"]):
                    if shop_result["jokers_bought"]:
                        print(f"  Shop: Bought {shop_result['jokers_bought']}")
                    for planet_msg in shop_result["planets_used"]:
                        print(f"  {planet_msg}")

            if not game.advance_blind():
                victory = True
                blinds_beaten = 24  # Full run
                break

        # Log run end
        game.history.add_run_end(
            success=victory,
            final_ante=game.ante,
            final_blind=game.current_blind.name,
            blinds_beaten=blinds_beaten,
            final_money=game.money,
            jokers=[j.get("name", "?") for j in game.jokers],
            hand_levels={ht.name: lv for ht, lv in game.hand_levels.items() if lv > 1}
        )

        # Save run history
        log_dir = Path(__file__).parent.parent / "run_logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"run_{timestamp}_{preset_name}.json"
        game.history.save(str(log_path))

        # Build detailed history from events
        blind_history = []
        shop_history = []
        vouchers_acquired = []
        bosses_encountered = []

        for event in game.history.events:
            if event.event_type == "blind_result":
                data = event.data
                blind_history.append(BlindDetail(
                    ante=event.ante,
                    blind_type=event.blind_type,
                    boss_name=data.get("boss_name"),
                    score=data.get("score", 0),
                    required=data.get("required", 0),
                    success=data.get("success", False),
                    hands_played=data.get("hands_played", []),
                    hands_used=data.get("hands_used", 0),
                    discards_used=data.get("discards_used", 0),
                ))
                if data.get("boss_name"):
                    bosses_encountered.append(data["boss_name"])

            elif event.event_type == "shop_visit":
                data = event.data
                shop_history.append(ShopDetail(
                    ante=event.ante,
                    jokers_bought=data.get("jokers_bought", []),
                    vouchers_bought=data.get("vouchers_bought", []) or [],
                    planets_used=data.get("planets_used", []),
                    money_spent=data.get("money_spent", 0),
                    packs_opened=data.get("packs_opened", []),
                ))

            elif event.event_type == "voucher_acquired":
                vouchers_acquired.append(event.data.get("voucher", ""))

        # Build summary
        return RunSummary(
            victory=victory,
            ante_reached=game.ante,
            blind_reached=game.current_blind.name,
            blinds_beaten=blinds_beaten,
            final_money=game.money,
            jokers_collected=[j.get("name", "?") for j in game.jokers],
            planets_used=game.planets_used,
            hand_levels={ht.name: lv for ht, lv in game.hand_levels.items()},
            preset_used=preset_name,
            blind_history=blind_history,
            shop_history=shop_history,
            vouchers_acquired=vouchers_acquired,
            bosses_encountered=bosses_encountered,
        )

    def run_batch(self, preset: Union[str, Preset] = "standard",
                  runs: int = 100, verbose: bool = False) -> BatchResult:
        """
        Run multiple simulations and aggregate results.

        Args:
            preset: Preset name or Preset object
            runs: Number of runs
            verbose: Print progress

        Returns:
            BatchResult with aggregated stats
        """
        if isinstance(preset, str):
            preset_name = preset
        else:
            preset_name = preset.name

        wins = 0
        total_blinds = 0
        total_ante = 0
        max_ante = 0
        total_money = 0
        total_jokers = 0
        total_planets = 0
        ante_distribution = {}

        for i in range(runs):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Run {i + 1}/{runs}...")

            summary = self.run(preset, verbose=False)

            if summary.victory:
                wins += 1
            total_blinds += summary.blinds_beaten
            total_ante += summary.ante_reached
            max_ante = max(max_ante, summary.ante_reached)
            total_money += summary.final_money
            total_jokers += len(summary.jokers_collected)
            total_planets += summary.planets_used

            ante_distribution[summary.ante_reached] = ante_distribution.get(summary.ante_reached, 0) + 1

        return BatchResult(
            runs=runs,
            wins=wins,
            win_rate=wins / runs * 100,
            avg_blinds=total_blinds / runs,
            avg_ante=total_ante / runs,
            max_ante=max_ante,
            avg_money=total_money / runs,
            avg_jokers=total_jokers / runs,
            avg_planets=total_planets / runs,
            ante_distribution=ante_distribution,
            preset_used=preset_name,
        )


# Convenience functions
def run(preset: str = "standard", verbose: bool = False) -> RunSummary:
    """Quick run with default simulator."""
    sim = Simulator()
    return sim.run(preset, verbose)


def run_batch(preset: str = "standard", runs: int = 100, verbose: bool = False) -> BatchResult:
    """Quick batch run with default simulator."""
    sim = Simulator()
    return sim.run_batch(preset, runs, verbose)
