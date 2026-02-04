"""
Adaptive Configuration for Simons Strategy Engine

Instead of hardcoding TA thresholds, this module provides:
1. Baseline parameter values
2. Automatic parameter optimization based on market conditions
3. Hot-reload support for live parameter updates

Key insight: The best z-score threshold for mean reversion depends on:
- Current market volatility
- Asset-specific behavior
- Recent performance of the strategy
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimonsAdaptiveConfig:
    """
    Tunable parameters for the Simons Strategy Engine.

    All thresholds start with baseline values and get optimized
    through the rolling performance window.
    """

    # === Identifier ===
    config_name: str = "global"  # Can be "global" or asset-specific

    # === Mean Reversion Parameters (Baseline) ===
    # These will be tuned based on market conditions
    entry_z_score: float = 2.0           # Baseline: enter at z > 2.0
    exit_z_score: float = 0.3            # Baseline: exit at z < 0.3
    extreme_z_score: float = 3.0         # Baseline: very high conviction

    # Asymmetric thresholds (crypto has upward bias)
    long_entry_z: float = 2.0            # Can be lower (buy dips)
    short_entry_z: float = 3.0           # Must be higher (risky)

    # === Pairs Trading Parameters ===
    pairs_entry_z: float = 1.8           # Lower for hedged positions
    pairs_exit_z: float = 0.4
    min_correlation: float = 0.70        # Minimum pair correlation
    spread_lookback: int = 60            # Candles for spread calculation

    # === RSI Filters ===
    rsi_oversold: float = 35.0           # Buy when RSI below this
    rsi_overbought: float = 80.0         # Short when RSI above this

    # === Volume Filters ===
    min_volume_ratio: float = 0.8        # Skip if volume too low
    volume_surge_threshold: float = 1.5  # Extra conviction on high volume

    # === Regime Detection ===
    trending_adx: float = 25.0           # ADX > this = trending
    high_vol_atr_ratio: float = 1.5      # ATR ratio > this = high vol

    # === Position Sizing ===
    base_position_pct: float = 0.06      # 6% of portfolio per trade
    max_portfolio_exposure: float = 0.50 # 50% max total exposure
    pairs_size_multiplier: float = 1.5   # Pairs get larger sizes
    mr_size_multiplier: float = 0.9      # Mean reversion slightly smaller

    # === Risk Management ===
    stop_loss_pct: float = 0.015         # 1.5% stop for singles
    take_profit_pct: float = 0.025       # 2.5% take profit
    pairs_stop_pct: float = 0.02         # 2% stop for pairs (wider)
    pairs_tp_pct: float = 0.03           # 3% TP for pairs
    max_holding_hours: int = 24          # Time-based exit

    # === Conviction Thresholds ===
    min_conviction: float = 50.0         # Minimum to trade
    high_conviction: float = 80.0        # Full position size

    # === Tuning Metadata ===
    last_tuned: str | None = None
    tune_count: int = 0
    performance_window_days: int = 14    # Days to evaluate
    min_trades_for_tuning: int = 20      # Need enough data

    # === Performance Tracking (for auto-tuning) ===
    recent_win_rate: float = 0.5
    recent_profit_factor: float = 1.0
    recent_sharpe: float = 0.0

    def __post_init__(self):
        if self.last_tuned is None:
            self.last_tuned = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimonsAdaptiveConfig":
        # Filter out unknown fields for backwards compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def mark_tuned(self):
        self.last_tuned = datetime.now(UTC).isoformat()
        self.tune_count += 1


class SimonsParameterOptimizer:
    """
    Automatic parameter optimization for Simons Strategy.

    Uses rolling performance metrics to adjust parameters:
    - If win rate drops: tighten entry thresholds
    - If profit factor high: can relax thresholds for more trades
    - If drawdown spikes: reduce position sizes
    """

    def __init__(
        self,
        config: SimonsAdaptiveConfig,
        config_path: Path | None = None,
    ):
        self.config = config
        self.config_path = config_path or Path("data/adaptive/simons_config.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Trade history for tuning
        self.trade_history: list[dict] = []

        # Parameter bounds (won't tune outside these)
        self.bounds = {
            "entry_z_score": (1.5, 3.5),
            "exit_z_score": (0.1, 0.8),
            "pairs_entry_z": (1.2, 2.5),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (70, 90),
            "stop_loss_pct": (0.01, 0.03),
            "take_profit_pct": (0.015, 0.05),
            "min_conviction": (40, 70),
        }

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade for performance tracking."""
        self.trade_history.append({
            **trade,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Keep only recent trades
        max_trades = 200
        if len(self.trade_history) > max_trades:
            self.trade_history = self.trade_history[-max_trades:]

    def should_tune(self) -> bool:
        """Check if we have enough data to tune."""
        return len(self.trade_history) >= self.config.min_trades_for_tuning

    def calculate_metrics(self, trades: list[dict]) -> dict:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "avg_win": 0,
                "avg_loss": 0,
                "total_pnl": 0,
            }

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]

        win_rate = len(wins) / len(trades) if trades else 0.5

        total_wins = sum(t.get("pnl", 0) for t in wins) if wins else 0
        total_losses = abs(sum(t.get("pnl", 0) for t in losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 1.0

        avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.get("pnl", 0)) for t in losses]) if losses else 0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_pnl": sum(t.get("pnl", 0) for t in trades),
        }

    def optimize(self) -> SimonsAdaptiveConfig:
        """
        Optimize parameters based on recent performance.

        Strategy:
        - Low win rate -> tighten entry thresholds (higher z-score)
        - High win rate + good profit factor -> relax thresholds
        - High losses -> widen stops, reduce position sizes
        """
        if not self.should_tune():
            logger.info(f"Not enough trades for tuning ({len(self.trade_history)}/{self.config.min_trades_for_tuning})")
            return self.config

        metrics = self.calculate_metrics(self.trade_history)

        # Update config metrics
        self.config.recent_win_rate = metrics["win_rate"]
        self.config.recent_profit_factor = metrics["profit_factor"]

        logger.info(f"Tuning with metrics: WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}")

        # === Rule-based optimization ===

        # 1. Win rate too low -> tighten entry (require higher z-score)
        if metrics["win_rate"] < 0.40:
            self.config.entry_z_score = min(
                self.config.entry_z_score + 0.2,
                self.bounds["entry_z_score"][1]
            )
            self.config.min_conviction = min(
                self.config.min_conviction + 5,
                self.bounds["min_conviction"][1]
            )
            logger.info(f"Low win rate: tightened entry_z to {self.config.entry_z_score:.2f}")

        # 2. Win rate good + profit factor good -> can relax
        elif metrics["win_rate"] > 0.55 and metrics["profit_factor"] > 1.5:
            self.config.entry_z_score = max(
                self.config.entry_z_score - 0.1,
                self.bounds["entry_z_score"][0]
            )
            self.config.min_conviction = max(
                self.config.min_conviction - 3,
                self.bounds["min_conviction"][0]
            )
            logger.info(f"Good performance: relaxed entry_z to {self.config.entry_z_score:.2f}")

        # 3. Average loss > average win -> widen stops
        if metrics["avg_loss"] > metrics["avg_win"] * 1.5:
            self.config.stop_loss_pct = min(
                self.config.stop_loss_pct + 0.003,
                self.bounds["stop_loss_pct"][1]
            )
            logger.info(f"High avg loss: widened stop to {self.config.stop_loss_pct:.1%}")

        # 4. Profit factor very high -> can take more risk
        if metrics["profit_factor"] > 2.0:
            self.config.base_position_pct = min(0.08, self.config.base_position_pct + 0.01)
            logger.info(f"High PF: increased position size to {self.config.base_position_pct:.1%}")

        # 5. Profit factor poor -> reduce risk
        elif metrics["profit_factor"] < 0.8:
            self.config.base_position_pct = max(0.03, self.config.base_position_pct - 0.01)
            logger.info(f"Low PF: reduced position size to {self.config.base_position_pct:.1%}")

        # Tune based on strategy-specific performance
        self._tune_by_strategy()

        self.config.mark_tuned()
        self.save()

        return self.config

    def _tune_by_strategy(self) -> None:
        """Tune parameters based on strategy-specific performance."""
        pairs_trades = [t for t in self.trade_history if t.get("strategy") == "pairs"]
        mr_trades = [t for t in self.trade_history if t.get("strategy") == "mean_reversion"]

        # Tune pairs parameters
        if len(pairs_trades) >= 10:
            pairs_metrics = self.calculate_metrics(pairs_trades)
            if pairs_metrics["win_rate"] > 0.50:
                # Pairs working well, can be more aggressive
                self.config.pairs_entry_z = max(
                    self.config.pairs_entry_z - 0.1,
                    self.bounds["pairs_entry_z"][0]
                )
            elif pairs_metrics["win_rate"] < 0.40:
                # Pairs underperforming, be more selective
                self.config.pairs_entry_z = min(
                    self.config.pairs_entry_z + 0.2,
                    self.bounds["pairs_entry_z"][1]
                )

        # Tune mean reversion parameters
        if len(mr_trades) >= 10:
            mr_metrics = self.calculate_metrics(mr_trades)
            if mr_metrics["win_rate"] > 0.50:
                self.config.rsi_oversold = min(
                    self.config.rsi_oversold + 2,
                    self.bounds["rsi_oversold"][1]
                )
            elif mr_metrics["win_rate"] < 0.40:
                self.config.rsi_oversold = max(
                    self.config.rsi_oversold - 3,
                    self.bounds["rsi_oversold"][0]
                )

    def save(self) -> None:
        """Save config to file for hot-reload."""
        with open(self.config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved optimized config to {self.config_path}")

    def load(self) -> SimonsAdaptiveConfig:
        """Load config from file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                data = json.load(f)
                self.config = SimonsAdaptiveConfig.from_dict(data)
                logger.info(f"Loaded config from {self.config_path}")
        return self.config


def get_default_config() -> SimonsAdaptiveConfig:
    """Get default baseline configuration."""
    return SimonsAdaptiveConfig()


def load_or_create_config(
    config_path: Path | None = None,
) -> tuple[SimonsAdaptiveConfig, SimonsParameterOptimizer]:
    """
    Load existing config or create new one with baseline values.

    Returns:
        (config, optimizer) tuple
    """
    config_path = config_path or Path("data/adaptive/simons_config.json")

    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
            config = SimonsAdaptiveConfig.from_dict(data)
    else:
        config = get_default_config()

    optimizer = SimonsParameterOptimizer(config, config_path)
    return config, optimizer
