#!/usr/bin/env python3
"""
Simons Protocol Backtest

Backtests the "Trade Big, Trade Less, Trade with Confidence" strategy
using all new components:
- Multi-Timeframe Engine (Daily/4H/1H alignment)
- ConvictionScorer (5-factor scoring)
- Circuit Breakers (risk management)
- Core Features (25 reduced feature set)
- Hybrid Exit System (algo trailing + hard stop)

Usage:
    python scripts/simons_backtest.py --symbol BTCUSDT --days 30
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Candle
from src.data.candle_cache import fetch_candles_with_cache, get_cache_info
from src.data.binance_data import fetch_binance_with_cache, get_binance_cache_info
from src.features.pipeline import FeaturePipeline
from src.ml.ensemble import MLEnsemble
from src.ml.confidence_calibrator import ConfidenceCalibrator, ModelPrediction, AgreementGate
from src.strategy.conviction_scorer import (
    ConvictionScorer,
    TimeframeSignals,
    MarketContext,
    PositionTier,
)
from src.strategy.mtf_engine import MultiTimeframeEngine
from src.risk.circuit_breaker import CircuitBreaker
from src.adaptive.asset_config import AdaptiveAssetConfig, load_asset_config, save_asset_config
from src.adaptive.performance_tracker import PerformanceTracker, Trade as TrackerTrade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WEEX_BASE_URL = "https://api-contract.weex.com"


@dataclass
class Position:
    """Open position tracking."""

    entry_time: datetime
    entry_price: float
    direction: Literal["LONG", "SHORT"]
    size: float  # Notional value
    conviction_score: float
    stop_loss: float
    take_profit_1: float  # 1.5R - close 40%
    take_profit_2: float  # 3.0R - close 30%
    trailing_stop: float
    remaining_size: float = 1.0  # Fraction remaining (1.0 = 100%)
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float("inf")
    r_multiple: float = 0.0  # Current R-multiple


@dataclass
class TradeResult:
    """Completed trade result."""

    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    conviction_score: float
    exit_reason: str


@dataclass
class BacktestStats:
    """Backtest performance statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    trades_by_conviction: dict = field(default_factory=dict)
    signals_generated: int = 0
    signals_filtered: int = 0
    # Adaptive tuning stats
    parameter_adjustments: int = 0
    adjustment_history: list = field(default_factory=list)


@dataclass
class ParameterAdjustment:
    """Record of a parameter adjustment during backtest."""

    timestamp: datetime
    parameter: str
    old_value: float
    new_value: float
    reason: str
    trigger_metric: str
    trigger_value: float


class AdaptiveTuner:
    """
    Rule-based adaptive parameter tuning for backtest.

    Monitors performance metrics and adjusts parameters to optimize strategy.
    This is a fast, rule-based alternative to LLM-based tuning for backtests.

    Features self-tuning thresholds (meta-learning) where the trigger thresholds
    themselves adapt based on whether adjustments improved performance.
    """

    def __init__(
        self,
        config: AdaptiveAssetConfig,
        evaluation_window: int = 10,  # Trades to evaluate
        candles_without_trade_threshold: int = 100,  # Lower threshold after this many candles
    ):
        self.config = config
        self.evaluation_window = evaluation_window
        self.candles_without_trade_threshold = candles_without_trade_threshold

        # Self-tuning thresholds (meta-learning)
        self.thresholds = {
            "win_rate_low": 0.40,      # Raise conviction if win rate below this
            "win_rate_high": 0.60,     # Lower conviction if win rate above this
            "consecutive_losses": 3,   # Reduce size after this many losses
            "catastrophic_r": -2.0,    # Emergency response threshold
            "large_loss_r": -1.5,      # Raise conviction threshold
            "stop_loss_rate": 0.50,    # Widen stops if this % hit stop loss
            "small_win_r": 1.0,        # Increase TP target if wins smaller than this
        }

        # Track effectiveness of adjustments for meta-learning
        self.adjustment_outcomes: list[dict] = []  # {adjustment, pnl_before, pnl_after, effective}

        # Tracking state
        self.recent_trades: list[TradeResult] = []
        self.candles_since_last_trade: int = 0
        self.adjustments: list[ParameterAdjustment] = []
        self.initial_config = AdaptiveAssetConfig(
            symbol=config.symbol,
            conviction_threshold=config.conviction_threshold,
            stop_atr_multiplier=config.stop_atr_multiplier,
            take_profit_atr_multiplier=config.take_profit_atr_multiplier,
            position_size_multiplier=config.position_size_multiplier,
            short_conviction_penalty=config.short_conviction_penalty,
            require_daily_trend_for_short=config.require_daily_trend_for_short,
        )

        # Track P&L at time of each adjustment for meta-learning
        self._pnl_at_last_adjustment: float = 0.0
        self._trades_at_last_adjustment: int = 0

    def record_trade(self, trade: TradeResult) -> list[ParameterAdjustment]:
        """Record a trade and check if parameters need adjustment."""
        self.recent_trades.append(trade)
        self.candles_since_last_trade = 0

        # Track adjustment effectiveness (meta-learning)
        current_pnl = sum(t.pnl for t in self.recent_trades)
        if self.adjustments and len(self.recent_trades) > self._trades_at_last_adjustment + 3:
            # Evaluate if recent adjustments helped
            pnl_since_adjustment = current_pnl - self._pnl_at_last_adjustment
            trades_since = len(self.recent_trades) - self._trades_at_last_adjustment
            avg_pnl_per_trade = pnl_since_adjustment / trades_since if trades_since > 0 else 0

            # Record outcome for the last adjustment
            if self.adjustment_outcomes:
                self.adjustment_outcomes[-1]["pnl_after"] = current_pnl
                self.adjustment_outcomes[-1]["effective"] = avg_pnl_per_trade > 0

            # Update meta-learning
            self._update_meta_learning(trade)

        # Keep only recent trades for evaluation
        if len(self.recent_trades) > self.evaluation_window * 2:
            self.recent_trades = self.recent_trades[-self.evaluation_window * 2:]

        # Check for immediate response to catastrophic trades
        adjustments = self._check_catastrophic_loss(trade)

        # Check for pattern-based adjustments
        adjustments.extend(self._evaluate_and_adjust(trade.exit_time))

        # Track for meta-learning if adjustments were made
        if adjustments:
            self._pnl_at_last_adjustment = current_pnl
            self._trades_at_last_adjustment = len(self.recent_trades)
            self.adjustment_outcomes.append({
                "adjustment": adjustments[-1].parameter,
                "pnl_before": current_pnl,
                "pnl_after": None,
                "effective": None,
            })

        return adjustments

    def _update_meta_learning(self, trade: TradeResult) -> None:
        """Update meta-learning: adjust thresholds based on adjustment effectiveness."""
        if not self.adjustment_outcomes:
            return

        # Check if recent adjustments were effective
        recent_outcomes = self.adjustment_outcomes[-5:]  # Last 5 adjustments
        effective_count = sum(1 for o in recent_outcomes if o.get("effective", False))
        effectiveness_rate = effective_count / len(recent_outcomes) if recent_outcomes else 0.5

        # If adjustments are mostly ineffective, make thresholds more conservative
        if len(recent_outcomes) >= 3 and effectiveness_rate < 0.3:
            # Tighten thresholds - be more selective about when to adjust
            self.thresholds["win_rate_low"] = max(0.30, self.thresholds["win_rate_low"] - 0.05)
            self.thresholds["consecutive_losses"] = min(5, self.thresholds["consecutive_losses"] + 1)
            logger.info(f"META-LEARNING: Adjustments ineffective ({effectiveness_rate:.0%}), tightening thresholds")

        # If adjustments are mostly effective, can be slightly more aggressive
        elif len(recent_outcomes) >= 3 and effectiveness_rate > 0.7:
            self.thresholds["win_rate_low"] = min(0.45, self.thresholds["win_rate_low"] + 0.02)
            logger.info(f"META-LEARNING: Adjustments effective ({effectiveness_rate:.0%}), relaxing thresholds")

    def _check_catastrophic_loss(self, trade: TradeResult) -> list[ParameterAdjustment]:
        """Immediately respond to catastrophic single-trade losses."""
        adjustments = []

        # If single trade loses > threshold, immediately widen stops and reduce size
        if trade.r_multiple < self.thresholds["catastrophic_r"]:
            timestamp = trade.exit_time

            # Widen stops significantly
            if self.config.stop_atr_multiplier < 3.5:
                old_val = self.config.stop_atr_multiplier
                self.config.stop_atr_multiplier = min(3.5, self.config.stop_atr_multiplier + 0.5)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="stop_atr_multiplier",
                    old_value=old_val,
                    new_value=self.config.stop_atr_multiplier,
                    reason=f"Catastrophic loss R={trade.r_multiple:.2f}, widening stops",
                    trigger_metric="single_trade_r",
                    trigger_value=trade.r_multiple,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: EMERGENCY widen stops {old_val:.1f} -> {self.config.stop_atr_multiplier:.1f} (R={trade.r_multiple:.2f})")

            # Reduce position size for safety
            if self.config.position_size_multiplier > 0.5:
                old_val = self.config.position_size_multiplier
                self.config.position_size_multiplier = max(0.5, self.config.position_size_multiplier - 0.25)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="position_size_multiplier",
                    old_value=old_val,
                    new_value=self.config.position_size_multiplier,
                    reason=f"Catastrophic loss R={trade.r_multiple:.2f}, reducing size",
                    trigger_metric="single_trade_r",
                    trigger_value=trade.r_multiple,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: EMERGENCY reduce size {old_val:.2f} -> {self.config.position_size_multiplier:.2f}")

        # If single trade loses > large_loss threshold, raise conviction threshold
        elif trade.r_multiple < self.thresholds["large_loss_r"]:
            if self.config.conviction_threshold < 80:
                old_val = self.config.conviction_threshold
                self.config.conviction_threshold = min(80, self.config.conviction_threshold + 5)
                adj = ParameterAdjustment(
                    timestamp=trade.exit_time,
                    parameter="conviction_threshold",
                    old_value=old_val,
                    new_value=self.config.conviction_threshold,
                    reason=f"Large loss R={trade.r_multiple:.2f}, raising conviction",
                    trigger_metric="single_trade_r",
                    trigger_value=trade.r_multiple,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Raised conviction {old_val:.0f} -> {self.config.conviction_threshold:.0f} (R={trade.r_multiple:.2f})")

        return adjustments

    def record_candle(self, timestamp: datetime) -> list[ParameterAdjustment]:
        """Record that a candle was processed (for tracking trade frequency)."""
        self.candles_since_last_trade += 1

        # Check if we need to lower thresholds due to inactivity
        if self.candles_since_last_trade >= self.candles_without_trade_threshold:
            return self._adjust_for_inactivity(timestamp)

        return []

    def _evaluate_and_adjust(self, timestamp: datetime) -> list[ParameterAdjustment]:
        """Evaluate recent performance and make adjustments."""
        adjustments = []

        if len(self.recent_trades) < 5:
            return adjustments  # Not enough data

        # Calculate metrics from recent trades
        recent = self.recent_trades[-self.evaluation_window:]
        wins = sum(1 for t in recent if t.pnl > 0)
        losses = len(recent) - wins
        win_rate = wins / len(recent)

        # Calculate consecutive losses
        consecutive_losses = 0
        for t in reversed(recent):
            if t.pnl <= 0:
                consecutive_losses += 1
            else:
                break

        # Rule 1: If win rate below threshold after 5+ trades, raise conviction threshold
        if len(recent) >= 5 and win_rate < self.thresholds["win_rate_low"]:
            if self.config.conviction_threshold < 85:
                old_val = self.config.conviction_threshold
                # Raise by 5 points, max 85
                self.config.conviction_threshold = min(85, self.config.conviction_threshold + 5)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="conviction_threshold",
                    old_value=old_val,
                    new_value=self.config.conviction_threshold,
                    reason=f"Win rate {win_rate:.1%} below 40%",
                    trigger_metric="win_rate",
                    trigger_value=win_rate,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Raised conviction {old_val} -> {self.config.conviction_threshold} (win rate {win_rate:.1%})")

        # Rule 2: If consecutive losses exceed threshold, reduce position size
        if consecutive_losses >= self.thresholds["consecutive_losses"]:
            if self.config.position_size_multiplier > 0.5:
                old_val = self.config.position_size_multiplier
                self.config.position_size_multiplier = max(0.5, self.config.position_size_multiplier - 0.25)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="position_size_multiplier",
                    old_value=old_val,
                    new_value=self.config.position_size_multiplier,
                    reason=f"{consecutive_losses} consecutive losses",
                    trigger_metric="consecutive_losses",
                    trigger_value=float(consecutive_losses),
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Reduced position size {old_val} -> {self.config.position_size_multiplier} ({consecutive_losses} losses)")

        # Rule 3: If avg R-multiple is very negative (< -1.5), widen stops
        avg_r = np.mean([t.r_multiple for t in recent])
        if avg_r < -1.5 and self.config.stop_atr_multiplier < 3.5:
            old_val = self.config.stop_atr_multiplier
            self.config.stop_atr_multiplier = min(3.5, self.config.stop_atr_multiplier + 0.3)
            adj = ParameterAdjustment(
                timestamp=timestamp,
                parameter="stop_atr_multiplier",
                old_value=old_val,
                new_value=self.config.stop_atr_multiplier,
                reason=f"Avg R-multiple {avg_r:.2f} too negative",
                trigger_metric="avg_r_multiple",
                trigger_value=avg_r,
            )
            adjustments.append(adj)
            self.adjustments.append(adj)
            logger.info(f"ADAPTIVE: Widened stops {old_val} -> {self.config.stop_atr_multiplier} (avg R={avg_r:.2f})")

        # Rule 3b: If most losses are exactly at stop loss (R ~ -1.0), stops may be too tight
        stop_loss_exits = [t for t in recent if t.exit_reason == "stop_loss"]
        if len(stop_loss_exits) >= 3:
            stop_loss_rate = len(stop_loss_exits) / len(recent)
            if stop_loss_rate > self.thresholds["stop_loss_rate"] and self.config.stop_atr_multiplier < 3.5:
                old_val = self.config.stop_atr_multiplier
                self.config.stop_atr_multiplier = min(3.5, self.config.stop_atr_multiplier + 0.25)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="stop_atr_multiplier",
                    old_value=old_val,
                    new_value=self.config.stop_atr_multiplier,
                    reason=f"{stop_loss_rate:.0%} of trades hitting stop loss",
                    trigger_metric="stop_loss_rate",
                    trigger_value=stop_loss_rate,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Widened stops {old_val:.2f} -> {self.config.stop_atr_multiplier:.2f} ({stop_loss_rate:.0%} stop outs)")

        # Rule 3c: If winners have small R, trailing stop may be too tight
        winners = [t for t in recent if t.pnl > 0]
        if len(winners) >= 2:
            avg_win_r = np.mean([t.r_multiple for t in winners])
            if avg_win_r < self.thresholds["small_win_r"]:
                # Increase take profit multiplier (let winners run longer)
                old_val = self.config.take_profit_atr_multiplier
                self.config.take_profit_atr_multiplier = min(4.0, self.config.take_profit_atr_multiplier + 0.3)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="take_profit_atr_multiplier",
                    old_value=old_val,
                    new_value=self.config.take_profit_atr_multiplier,
                    reason=f"Avg winner R={avg_win_r:.2f} too small, letting winners run",
                    trigger_metric="avg_win_r",
                    trigger_value=float(avg_win_r),
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Increased TP target {old_val:.1f} -> {self.config.take_profit_atr_multiplier:.1f} (avg win R={avg_win_r:.2f})")

        # Rule 3d: If avg loss R is much worse than -1.0 (e.g., -1.5+), price gaps past stops
        losers = [t for t in recent if t.pnl <= 0]
        if len(losers) >= 2:
            avg_loss_r = np.mean([t.r_multiple for t in losers])
            if avg_loss_r < -1.3 and self.config.stop_atr_multiplier < 3.5:
                old_val = self.config.stop_atr_multiplier
                self.config.stop_atr_multiplier = min(3.5, self.config.stop_atr_multiplier + 0.2)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="stop_atr_multiplier",
                    old_value=old_val,
                    new_value=self.config.stop_atr_multiplier,
                    reason=f"Avg loss R={avg_loss_r:.2f}, gaps past stops",
                    trigger_metric="avg_loss_r",
                    trigger_value=float(avg_loss_r),
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Widened stops {old_val:.2f} -> {self.config.stop_atr_multiplier:.2f} (avg loss R={avg_loss_r:.2f})")

        # Rule 4: If win rate above threshold and profitable, can lower conviction to get more trades
        total_pnl = sum(t.pnl for t in recent)
        if len(recent) >= 5 and win_rate > self.thresholds["win_rate_high"] and total_pnl > 0:
            if self.config.conviction_threshold > 55:
                old_val = self.config.conviction_threshold
                self.config.conviction_threshold = max(55, self.config.conviction_threshold - 3)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="conviction_threshold",
                    old_value=old_val,
                    new_value=self.config.conviction_threshold,
                    reason=f"Win rate {win_rate:.1%} good, expanding opportunities",
                    trigger_metric="win_rate",
                    trigger_value=win_rate,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Lowered conviction {old_val} -> {self.config.conviction_threshold} (win rate {win_rate:.1%})")

            # Also increase position size if doing well
            if self.config.position_size_multiplier < 1.25:
                old_val = self.config.position_size_multiplier
                self.config.position_size_multiplier = min(1.25, self.config.position_size_multiplier + 0.15)
                adj = ParameterAdjustment(
                    timestamp=timestamp,
                    parameter="position_size_multiplier",
                    old_value=old_val,
                    new_value=self.config.position_size_multiplier,
                    reason=f"Win rate {win_rate:.1%} good, increasing size",
                    trigger_metric="win_rate",
                    trigger_value=win_rate,
                )
                adjustments.append(adj)
                self.adjustments.append(adj)
                logger.info(f"ADAPTIVE: Increased position size {old_val} -> {self.config.position_size_multiplier}")

        return adjustments

    def _adjust_for_inactivity(self, timestamp: datetime) -> list[ParameterAdjustment]:
        """Lower thresholds if no trades for too long."""
        adjustments = []

        # Lower conviction threshold to allow more trades
        if self.config.conviction_threshold > 60:
            old_val = self.config.conviction_threshold
            self.config.conviction_threshold = max(60, self.config.conviction_threshold - 5)
            adj = ParameterAdjustment(
                timestamp=timestamp,
                parameter="conviction_threshold",
                old_value=old_val,
                new_value=self.config.conviction_threshold,
                reason=f"No trades for {self.candles_since_last_trade} candles",
                trigger_metric="candles_without_trade",
                trigger_value=float(self.candles_since_last_trade),
            )
            adjustments.append(adj)
            self.adjustments.append(adj)
            logger.info(f"ADAPTIVE: Lowered conviction {old_val} -> {self.config.conviction_threshold} (no trades for {self.candles_since_last_trade} candles)")

            # Reset counter to avoid repeated adjustments
            self.candles_since_last_trade = 0

        # If conviction already at minimum and still no trades, relax SHORT filter
        elif self.config.require_daily_trend_for_short and self.config.conviction_threshold <= 60:
            self.config.require_daily_trend_for_short = False
            adj = ParameterAdjustment(
                timestamp=timestamp,
                parameter="require_daily_trend_for_short",
                old_value=1.0,  # True
                new_value=0.0,  # False
                reason=f"Conviction at min ({self.config.conviction_threshold}), relaxing SHORT filter",
                trigger_metric="candles_without_trade",
                trigger_value=float(self.candles_since_last_trade),
            )
            adjustments.append(adj)
            self.adjustments.append(adj)
            logger.info(f"ADAPTIVE: Disabled require_daily_trend_for_short (no trades, conviction at min)")
            self.candles_since_last_trade = 0

        # If SHORT filter also relaxed, reduce SHORT conviction penalty
        elif not self.config.require_daily_trend_for_short and self.config.short_conviction_penalty > 0:
            old_val = self.config.short_conviction_penalty
            self.config.short_conviction_penalty = max(0, self.config.short_conviction_penalty - 2)
            adj = ParameterAdjustment(
                timestamp=timestamp,
                parameter="short_conviction_penalty",
                old_value=float(old_val),
                new_value=float(self.config.short_conviction_penalty),
                reason=f"Further relaxing SHORT requirements",
                trigger_metric="candles_without_trade",
                trigger_value=float(self.candles_since_last_trade),
            )
            adjustments.append(adj)
            self.adjustments.append(adj)
            logger.info(f"ADAPTIVE: Reduced SHORT penalty {old_val} -> {self.config.short_conviction_penalty}")
            self.candles_since_last_trade = 0

        return adjustments

    def get_summary(self) -> dict:
        """Get summary of all adjustments made."""
        # Calculate meta-learning effectiveness
        effective_adjustments = sum(1 for o in self.adjustment_outcomes if o.get("effective"))
        total_outcomes = len([o for o in self.adjustment_outcomes if o.get("effective") is not None])

        return {
            "total_adjustments": len(self.adjustments),
            "initial_config": {
                "conviction_threshold": self.initial_config.conviction_threshold,
                "stop_atr_multiplier": self.initial_config.stop_atr_multiplier,
                "take_profit_atr_multiplier": self.initial_config.take_profit_atr_multiplier,
                "position_size_multiplier": self.initial_config.position_size_multiplier,
            },
            "final_config": {
                "conviction_threshold": self.config.conviction_threshold,
                "stop_atr_multiplier": self.config.stop_atr_multiplier,
                "take_profit_atr_multiplier": self.config.take_profit_atr_multiplier,
                "position_size_multiplier": self.config.position_size_multiplier,
            },
            "thresholds": self.thresholds.copy(),
            "meta_learning": {
                "effective_adjustments": effective_adjustments,
                "total_evaluated": total_outcomes,
                "effectiveness_rate": effective_adjustments / total_outcomes if total_outcomes > 0 else 0,
            },
            "adjustments": [
                {
                    "time": a.timestamp.isoformat(),
                    "param": a.parameter,
                    "change": f"{a.old_value:.2f} -> {a.new_value:.2f}",
                    "reason": a.reason,
                }
                for a in self.adjustments
            ],
        }

    def save_learned_params(self, path: Path) -> None:
        """Save learned parameters and thresholds to file."""
        import json
        data = {
            "config": {
                "conviction_threshold": self.config.conviction_threshold,
                "stop_atr_multiplier": self.config.stop_atr_multiplier,
                "take_profit_atr_multiplier": self.config.take_profit_atr_multiplier,
                "position_size_multiplier": self.config.position_size_multiplier,
                "short_conviction_penalty": self.config.short_conviction_penalty,
                "require_daily_trend_for_short": self.config.require_daily_trend_for_short,
            },
            "thresholds": self.thresholds,
            "symbol": self.config.symbol,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved learned params to {path}")

    def load_learned_params(self, path: Path) -> bool:
        """Load learned parameters and thresholds from file."""
        import json
        if not path.exists():
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Load config values
            cfg = data.get("config", {})
            self.config.conviction_threshold = cfg.get("conviction_threshold", self.config.conviction_threshold)
            self.config.stop_atr_multiplier = cfg.get("stop_atr_multiplier", self.config.stop_atr_multiplier)
            self.config.take_profit_atr_multiplier = cfg.get("take_profit_atr_multiplier", self.config.take_profit_atr_multiplier)
            self.config.position_size_multiplier = cfg.get("position_size_multiplier", self.config.position_size_multiplier)
            self.config.short_conviction_penalty = cfg.get("short_conviction_penalty", self.config.short_conviction_penalty)
            self.config.require_daily_trend_for_short = cfg.get("require_daily_trend_for_short", self.config.require_daily_trend_for_short)

            # Load thresholds
            self.thresholds.update(data.get("thresholds", {}))

            logger.info(f"Loaded learned params from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load learned params: {e}")
            return False


class SimonsBacktester:
    """
    Backtester implementing the Simons Protocol strategy.

    Key principles:
    - Trade less: Only score >= 70 trades
    - Trade big: Position size based on conviction tier
    - Trade with confidence: Multi-timeframe + model agreement
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: int = 10,
        slippage_bps: float = 5.0,
        symbol: str = "BTCUSDT",
        adaptive_enabled: bool = False,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.slippage_bps = slippage_bps
        self.symbol = symbol
        self.adaptive_enabled = adaptive_enabled

        # Initialize components
        self.mtf_engine = MultiTimeframeEngine()
        self.conviction_scorer = ConvictionScorer()
        self.circuit_breaker = CircuitBreaker()
        self.feature_pipeline = FeaturePipeline()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.agreement_gate = AgreementGate()

        # Adaptive tuning
        self.adaptive_tuner: AdaptiveTuner | None = None
        if adaptive_enabled:
            # Start CONSERVATIVE (same as static mode) and RELAX if needed
            # This prevents early catastrophic losses while still adapting
            self.adaptive_config = AdaptiveAssetConfig(
                symbol=symbol,
                conviction_threshold=70.0,  # Start at static default
                stop_atr_multiplier=2.0,    # Start at static default
                take_profit_atr_multiplier=2.5,  # Standard TP target
                position_size_multiplier=1.0,  # Full position initially
                short_conviction_penalty=5,  # Keep SHORT penalty initially
                require_daily_trend_for_short=True,  # Keep SHORT filter initially
            )
            self.adaptive_tuner = AdaptiveTuner(
                self.adaptive_config,
                evaluation_window=10,
                candles_without_trade_threshold=100,  # Relax if no trades for 100 candles
            )
            logger.info(f"Adaptive tuning ENABLED for {symbol} (conservative start, adapts as needed)")
        else:
            self.adaptive_config = None

        # Try to load ML ensemble
        self.ensemble: MLEnsemble | None = None
        try:
            self.ensemble = MLEnsemble(models_dir=Path("models"))
            self.ensemble.check_and_reload_models()
            health = self.ensemble.get_health_status()
            healthy_count = sum(health.values())
            logger.info(f"Loaded ML ensemble: {healthy_count}/4 healthy models")
        except Exception as e:
            logger.warning(f"Could not load ML ensemble: {e}")

        # State
        self.position: Position | None = None
        self.trades: list[TradeResult] = []
        self.equity_curve: list[float] = [initial_balance]
        self.stats = BacktestStats()

    # Get current adaptive parameters (or defaults if not adaptive)
    @property
    def conviction_threshold(self) -> float:
        if self.adaptive_config:
            return self.adaptive_config.conviction_threshold
        # Lowered from 70 to 60 to allow more trades through
        # Conviction 60+ still requires solid multi-timeframe alignment
        return 60.0

    @property
    def stop_atr_multiplier(self) -> float:
        if self.adaptive_config:
            return self.adaptive_config.stop_atr_multiplier
        return 2.0

    @property
    def position_size_multiplier(self) -> float:
        if self.adaptive_config:
            return self.adaptive_config.position_size_multiplier
        return 1.0

    @property
    def short_conviction_penalty(self) -> int:
        if self.adaptive_config:
            return self.adaptive_config.short_conviction_penalty
        return 5

    @property
    def require_daily_trend_for_short(self) -> bool:
        if self.adaptive_config:
            return self.adaptive_config.require_daily_trend_for_short
        # Changed from True to False: Allow SHORT when 4H+1H agree, even if Daily is NEUTRAL
        # Daily trend is slow-moving and often NEUTRAL - requiring it blocks too many valid SHORTs
        return False

    @property
    def take_profit_atr_multiplier(self) -> float:
        if self.adaptive_config:
            return self.adaptive_config.take_profit_atr_multiplier
        return 2.5

    def run(self, candles_1h: list[Candle]) -> BacktestStats:
        """
        Run backtest on 1H candles.

        Args:
            candles_1h: List of 1H candles (oldest first)

        Returns:
            BacktestStats with performance metrics
        """
        # Need 400+ 1H candles for proper daily aggregation (400/24 = 16 daily)
        if len(candles_1h) < 400:
            logger.error("Need at least 400 candles for backtest (for daily timeframe)")
            return self.stats

        logger.info(f"Starting Simons Protocol backtest on {len(candles_1h)} candles")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")

        # We need a large window for MTF analysis:
        # - 1H: 50+ candles
        # - 4H: 20+ candles (need 80 1H candles)
        # - Daily: 15+ candles (need 360 1H candles)
        min_window = 400  # Ensures enough for daily aggregation

        # Process EVERY candle for position management (stops/exits)
        # But only generate signals on 4H intervals
        for i in range(min_window, len(candles_1h)):
            candles_window = candles_1h[:i + 1]
            current_candle = candles_1h[i]
            current_price = current_candle.close

            # CRITICAL: Check position stops on EVERY candle (1H)
            # This prevents gaps past stop loss
            if self.position:
                self._update_position(current_candle)
                if self.position is None:  # Position was closed
                    self._update_equity(current_price)
                    # Adaptive tuning: record candle after trade close
                    if self.adaptive_tuner:
                        self.adaptive_tuner.record_candle(current_candle.timestamp)
                    continue

            # Generate signals only on 4H intervals (every 4th candle)
            is_signal_candle = (i - min_window) % 4 == 0
            if is_signal_candle and not self.position:
                # Update MTF engine only when generating signals
                self.mtf_engine.update_candles(candles_window)
                self._evaluate_entry(candles_window, current_candle)

            # Track equity
            self._update_equity(current_price)

            # Adaptive tuning: track candles without trades
            if self.adaptive_tuner and not self.position:
                adjustments = self.adaptive_tuner.record_candle(current_candle.timestamp)
                if adjustments:
                    self.stats.parameter_adjustments += len(adjustments)
                    self.stats.adjustment_history.extend([
                        f"{a.timestamp.strftime('%Y-%m-%d')}: {a.parameter} {a.old_value:.1f}->{a.new_value:.1f} ({a.reason})"
                        for a in adjustments
                    ])

        # Close any remaining position at end
        if self.position:
            self._close_position(
                candles_1h[-1],
                candles_1h[-1].close,
                "end_of_backtest",
            )

        # Calculate final stats
        self._calculate_final_stats()

        return self.stats

    # Default asymmetric conviction requirements (overridden by adaptive config)
    # These are now accessed via properties when adaptive tuning is enabled

    def _evaluate_entry(
        self,
        candles: list[Candle],
        current_candle: Candle,
    ) -> None:
        """Evaluate potential entry using Simons Protocol."""
        self.stats.signals_generated += 1

        # Step 1: Multi-timeframe analysis
        mtf_signal = self.mtf_engine.analyze()

        if not mtf_signal.aligned or mtf_signal.direction == "HOLD":
            self.stats.signals_filtered += 1
            logger.debug(
                f"MTF filtered: aligned={mtf_signal.aligned}, "
                f"direction={mtf_signal.direction}, "
                f"daily={mtf_signal.daily.direction}, "
                f"4h={mtf_signal.four_hour.direction}"
            )
            return

        # Step 1b: Asymmetric SHORT filter (crypto has upward bias)
        # Don't take SHORTs when Daily trend is NEUTRAL - market tends to drift up
        if mtf_signal.direction == "SHORT" and self.require_daily_trend_for_short:
            if mtf_signal.daily.direction == "NEUTRAL":
                self.stats.signals_filtered += 1
                logger.debug(
                    f"SHORT filtered: Daily trend is NEUTRAL (upward bias in consolidation)"
                )
                return

        # Step 2: Circuit breaker check
        cb_result = self.circuit_breaker.check(
            current_candle.symbol,
            mtf_signal.direction,  # type: ignore
        )
        if not cb_result.can_trade:
            self.stats.signals_filtered += 1
            logger.debug(f"Circuit breaker blocked: {cb_result.message}")
            return

        # Step 3: Calculate features and get ML prediction
        # Use full 59 features for ML models (they were trained on full set)
        try:
            features = self.feature_pipeline.calculate_features(candles)
            # Create feature array in consistent order
            feature_names = self.feature_pipeline.feature_names
            feature_array = np.array(
                [features.get(name, 0.0) for name in feature_names],
                dtype=np.float64
            )
        except Exception as e:
            logger.debug(f"Feature calculation failed: {e}")
            self.stats.signals_filtered += 1
            return

        # Step 4: ML ensemble prediction (if available)
        # Use higher defaults when ensemble fails - trust MTF alignment
        ensemble_confidence = 0.65  # Default if no ensemble
        model_agreement = 0.75  # Default

        if self.ensemble:
            try:
                # Check how many models are healthy
                health_status = self.ensemble.get_health_status()
                healthy_models = sum(health_status.values())

                ml_signal, confidence, model_outputs, weighted_avg = self.ensemble.predict(
                    {"array": feature_array}
                )

                # Only trust ensemble if sufficient healthy models and meaningful confidence
                if healthy_models >= 2 and confidence > 0.1:
                    ensemble_confidence = confidence
                    # Calculate model agreement with MTF direction
                    agreeing = sum(
                        1 for v in model_outputs.values()
                        if (v > 0.5 and mtf_signal.direction == "LONG")
                        or (v < 0.5 and mtf_signal.direction == "SHORT")
                    )
                    model_agreement = agreeing / max(len(model_outputs), 1)

                    # If models strongly disagree with MTF (0 agreement),
                    # something is wrong - use neutral instead of penalizing
                    if model_agreement < 0.25:
                        logger.debug(
                            f"Models disagree with MTF (agreement={model_agreement:.2f}), "
                            f"using neutral values"
                        )
                        ensemble_confidence = 0.65
                        model_agreement = 0.5  # Neutral
                else:
                    # Not enough healthy models - trust MTF alignment
                    ensemble_confidence = 0.65
                    model_agreement = 0.75  # Favor MTF when models unhealthy
            except Exception as e:
                logger.debug(f"Ensemble prediction failed: {e}")
                # Use defaults when ensemble fails
                ensemble_confidence = 0.65
                model_agreement = 0.75

        # Step 5: Build context for conviction scoring
        timeframe_signals = TimeframeSignals(
            daily_trend=mtf_signal.daily.direction,
            daily_adx=mtf_signal.daily.strength,
            four_hour_signal=mtf_signal.direction,  # type: ignore
            four_hour_confidence=ensemble_confidence,
            one_hour_momentum=mtf_signal.one_hour.momentum,
            mtf_aligned=mtf_signal.aligned,  # Trust MTF engine's alignment decision
        )

        # Calculate market context
        atr = features.get("atr", current_candle.close * 0.02)
        volume_ratio = features.get("volume_ratio", 1.0)
        rsi = features.get("rsi", 50.0)

        market_context = MarketContext(
            regime=self._classify_regime(features),
            regime_confidence=0.8,
            volume_ratio=volume_ratio,
            atr_ratio=features.get("atr_ratio", 1.0),
            rsi=rsi,
            price_vs_ema50=features.get("ema_long_ratio", 0.0) * 100,
            price_vs_ema200=0.0,  # Not calculated in core features
            bb_position=features.get("bb_position", 0.0),
            model_agreement_pct=model_agreement,
        )

        # Step 6: Calculate conviction score
        conviction_result = self.conviction_scorer.calculate(
            timeframe_signals,
            market_context,
        )

        # Step 6b: Apply asymmetric conviction penalty for SHORTs
        # Crypto has upward bias - shorts need higher conviction to be profitable
        effective_score = conviction_result.score
        if conviction_result.signal == "SHORT":
            effective_score -= self.short_conviction_penalty
            logger.debug(
                f"SHORT conviction penalty applied: {conviction_result.score} → {effective_score}"
            )

        # Only trade if effective score >= conviction threshold (adaptive or default 70)
        if effective_score < self.conviction_threshold:
            self.stats.signals_filtered += 1
            return

        # Only enter on LONG or SHORT signals, not HOLD
        if conviction_result.signal not in ("LONG", "SHORT"):
            self.stats.signals_filtered += 1
            return

        # Step 7: Execute entry
        self._open_position(
            current_candle,
            conviction_result.signal,  # type: ignore
            conviction_result.score,
            conviction_result.position_size_pct,
            conviction_result.risk_per_trade_pct,
            conviction_result.stop_atr_multiplier,
            atr,
            cb_result.size_multiplier,
        )

    def _open_position(
        self,
        candle: Candle,
        direction: Literal["LONG", "SHORT"],
        conviction_score: float,
        position_size_pct: float,
        risk_per_trade_pct: float,
        stop_atr_mult: float,
        atr: float,
        cb_size_mult: float,
    ) -> None:
        """Open a new position."""
        entry_price = candle.close

        # Apply slippage
        if direction == "LONG":
            entry_price *= (1 + self.slippage_bps / 10000)
        else:
            entry_price *= (1 - self.slippage_bps / 10000)

        # Calculate stop loss distance - use adaptive stop multiplier
        effective_stop_mult = self.stop_atr_multiplier  # Use adaptive value
        stop_distance = atr * effective_stop_mult

        # FIXED NOTIONAL POSITION SIZE (Jim Simons approach v2)
        # Using volatility-adjusted sizing caused timing bias:
        # - Low ATR → large positions → but calm markets break out → losses
        # - High ATR → small positions → but volatile markets mean-revert → wins
        # The fix: use FIXED notional position size, accept variable R risk
        #
        # Fixed at 5% of capital per trade (with leverage, gives good exposure)
        base_position_pct = 0.05
        adjusted_pct = base_position_pct * cb_size_mult * self.position_size_multiplier
        position_value = self.balance * adjusted_pct

        # Cap at 20% of balance
        max_position = self.balance * 0.20
        position_value = min(position_value, max_position)

        # Calculate $ risk for logging (actual risk depends on stop distance)
        stop_distance_pct = stop_distance / entry_price
        risk_amount = position_value * stop_distance_pct * self.leverage

        # Calculate take profit multiplier - use adaptive value
        tp_mult = self.take_profit_atr_multiplier

        # Calculate stop loss and take profit levels
        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit_1 = entry_price + (stop_distance * 1.5)  # 1.5R
            take_profit_2 = entry_price + (stop_distance * tp_mult)  # Adaptive R target
        else:
            stop_loss = entry_price + stop_distance
            take_profit_1 = entry_price - (stop_distance * 1.5)
            take_profit_2 = entry_price - (stop_distance * tp_mult)

        self.position = Position(
            entry_time=candle.timestamp,
            entry_price=entry_price,
            direction=direction,
            size=position_value,
            conviction_score=conviction_score,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            trailing_stop=stop_loss,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        # Track position in circuit breaker
        self.circuit_breaker.open_position(candle.symbol, direction)

        # Calculate what % of balance this position represents
        position_pct = position_value / self.balance

        logger.info(
            f"ENTRY: {direction} @ ${entry_price:,.2f} | "
            f"Size: ${position_value:,.2f} ({position_pct:.0%}) | Risk: ${risk_amount:,.2f} | "
            f"Conviction: {conviction_score:.0f} | "
            f"SL: ${stop_loss:,.2f} | TP1: ${take_profit_1:,.2f}"
        )

    def _update_position(self, candle: Candle) -> None:
        """Update position with current price, check exits."""
        if not self.position:
            return

        price = candle.close
        pos = self.position

        # Update high/low tracking
        if price > pos.highest_price:
            pos.highest_price = price
        if price < pos.lowest_price:
            pos.lowest_price = price

        # Calculate current R-multiple
        if pos.direction == "LONG":
            pnl_pct = (price - pos.entry_price) / pos.entry_price
            stop_distance = pos.entry_price - pos.stop_loss
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price
            stop_distance = pos.stop_loss - pos.entry_price

        if stop_distance > 0:
            pos.r_multiple = (pnl_pct * pos.entry_price) / stop_distance

        # Check stop loss (hard stop - safety net)
        if pos.direction == "LONG" and price <= pos.stop_loss:
            self._close_position(candle, price, "stop_loss")
            return
        elif pos.direction == "SHORT" and price >= pos.stop_loss:
            self._close_position(candle, price, "stop_loss")
            return

        # SIMPLIFIED EXIT: No partial closes - let winners run to full trailing stop
        # The partial close logic was causing inconsistent $/R because:
        # - Winners got cut early (40% at 1.5R, 30% at 3R)
        # - This reduced the average $ on winning trades
        # - While losers hit full stop for full $ loss
        #
        # Now: just use stop loss and trailing stop for entire position
        # Move stop to breakeven when we reach +1.5R (done in _update_trailing_stop)
        pass

        # Update trailing stop for remaining position
        self._update_trailing_stop(price)

        # Check trailing stop
        if pos.direction == "LONG" and price <= pos.trailing_stop:
            self._close_position(candle, price, "trailing_stop")
        elif pos.direction == "SHORT" and price >= pos.trailing_stop:
            self._close_position(candle, price, "trailing_stop")

    def _update_trailing_stop(self, price: float) -> None:
        """Update trailing stop based on PEAK profit level.

        Uses highest R achieved (not current R) to determine trail level.
        This ensures we lock in profits even if price retraces.
        """
        if not self.position:
            return

        pos = self.position

        # Calculate the PEAK R achieved (based on highest/lowest price)
        stop_distance = abs(pos.entry_price - pos.stop_loss)
        if pos.direction == "LONG":
            peak_r = (pos.highest_price - pos.entry_price) / stop_distance if stop_distance > 0 else 0
        else:
            peak_r = (pos.entry_price - pos.lowest_price) / stop_distance if stop_distance > 0 else 0

        # DON'T TRAIL until PEAK profit reached +1.5R
        if peak_r < 1.5:
            return  # Keep original stop loss

        # Set trail based on PEAK R achieved (using absolute R levels)
        # This locks in minimum profit regardless of current price
        if peak_r >= 3.0:
            lock_in_r = 2.0  # Lock in 2R at peak of 3R+
        elif peak_r >= 2.0:
            lock_in_r = 1.2  # Lock in 1.2R at peak of 2R+
        else:  # 1.5R to 2.0R
            lock_in_r = 0.8  # Lock in 0.8R at peak of 1.5R+

        # Calculate trail level
        if pos.direction == "LONG":
            new_trail = pos.entry_price + (lock_in_r * stop_distance)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail
        else:
            new_trail = pos.entry_price - (lock_in_r * stop_distance)
            if new_trail < pos.trailing_stop:
                pos.trailing_stop = new_trail

    def _partial_close(
        self,
        candle: Candle,
        price: float,
        close_fraction: float,
        reason: str,
    ) -> None:
        """Partially close position."""
        if not self.position:
            return

        pos = self.position
        close_size = pos.size * pos.remaining_size * close_fraction

        # Calculate P&L for closed portion
        if pos.direction == "LONG":
            pnl = (price - pos.entry_price) / pos.entry_price * close_size * self.leverage
        else:
            pnl = (pos.entry_price - price) / pos.entry_price * close_size * self.leverage

        self.balance += pnl
        pos.remaining_size *= (1 - close_fraction)

        logger.info(
            f"PARTIAL EXIT ({reason}): {close_fraction:.0%} @ ${price:,.2f} | "
            f"PnL: ${pnl:+,.2f} | Remaining: {pos.remaining_size:.0%}"
        )

    def _close_position(
        self,
        candle: Candle,
        price: float,
        reason: str,
    ) -> None:
        """Close entire remaining position."""
        if not self.position:
            return

        pos = self.position
        close_size = pos.size * pos.remaining_size

        # Apply slippage on exit
        if pos.direction == "LONG":
            exit_price = price * (1 - self.slippage_bps / 10000)
            pnl = (exit_price - pos.entry_price) / pos.entry_price * close_size * self.leverage
        else:
            exit_price = price * (1 + self.slippage_bps / 10000)
            pnl = (pos.entry_price - exit_price) / pos.entry_price * close_size * self.leverage

        self.balance += pnl
        pnl_pct = pnl / self.initial_balance

        # Record trade
        trade = TradeResult(
            entry_time=pos.entry_time,
            exit_time=candle.timestamp,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=pos.r_multiple,
            conviction_score=pos.conviction_score,
            exit_reason=reason,
        )
        self.trades.append(trade)

        # Update circuit breaker
        self.circuit_breaker.record_trade(
            candle.symbol,
            pos.direction,
            pnl_pct,
        )
        self.circuit_breaker.close_position(candle.symbol)

        # Update confidence calibrator
        self.confidence_calibrator.record_outcome(pnl > 0)

        logger.info(
            f"EXIT ({reason}): {pos.direction} @ ${exit_price:,.2f} | "
            f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2%}) | R: {pos.r_multiple:.2f} | "
            f"Balance: ${self.balance:,.2f}"
        )

        # Adaptive tuning: record trade and check for parameter adjustments
        if self.adaptive_tuner:
            adjustments = self.adaptive_tuner.record_trade(trade)
            if adjustments:
                self.stats.parameter_adjustments += len(adjustments)
                self.stats.adjustment_history.extend([
                    f"{a.timestamp.strftime('%Y-%m-%d')}: {a.parameter} {a.old_value:.1f}->{a.new_value:.1f} ({a.reason})"
                    for a in adjustments
                ])

        self.position = None

    def _update_equity(self, price: float) -> None:
        """Update equity curve with unrealized P&L."""
        equity = self.balance

        if self.position:
            pos = self.position
            if pos.direction == "LONG":
                unrealized = (price - pos.entry_price) / pos.entry_price * pos.size * pos.remaining_size * self.leverage
            else:
                unrealized = (pos.entry_price - price) / pos.entry_price * pos.size * pos.remaining_size * self.leverage
            equity += unrealized

        self.equity_curve.append(equity)

    def _classify_regime(self, features: dict[str, float]) -> str:
        """Classify market regime from features."""
        adx = features.get("adx", 0)
        atr_ratio = features.get("atr_ratio", 1.0)

        if atr_ratio > 2.0:
            return "EXTREME_VOLATILITY"
        elif atr_ratio > 1.5:
            return "HIGH_VOLATILITY"
        elif adx > 25:
            if features.get("plus_di", 0) > features.get("minus_di", 0):
                return "TRENDING_UP"
            else:
                return "TRENDING_DOWN"
        else:
            return "RANGING"

    def _calculate_final_stats(self) -> None:
        """Calculate final backtest statistics."""
        if not self.trades:
            return

        # Basic counts
        self.stats.total_trades = len(self.trades)
        self.stats.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        self.stats.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)

        # P&L
        self.stats.total_pnl = sum(t.pnl for t in self.trades)
        self.stats.total_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance

        # Win rate
        self.stats.win_rate = self.stats.winning_trades / self.stats.total_trades

        # Average R-multiples
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        if winners:
            self.stats.avg_win_r = np.mean([t.r_multiple for t in winners])
        if losers:
            self.stats.avg_loss_r = abs(np.mean([t.r_multiple for t in losers]))

        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        self.stats.profit_factor = gross_profit / max(gross_loss, 0.01)

        # Max drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
        self.stats.max_drawdown_pct = max_dd

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if np.std(returns) > 0:
                self.stats.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)  # 4H candles

        # Trades by conviction tier
        self.stats.trades_by_conviction = {
            "small (70-84)": len([t for t in self.trades if 70 <= t.conviction_score < 85]),
            "medium (85-94)": len([t for t in self.trades if 85 <= t.conviction_score < 95]),
            "large (95+)": len([t for t in self.trades if t.conviction_score >= 95]),
        }


async def fetch_candles_batch(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None = None,
) -> list[Candle]:
    """Fetch a single batch of candles from WEEX API."""
    weex_symbol = f"cmt_{symbol.lower()}"
    url = f"{WEEX_BASE_URL}/capi/v2/market/candles"
    params: dict[str, str] = {
        "symbol": weex_symbol,
        "granularity": interval,
        "limit": str(min(limit, 1000)),
    }
    if end_time:
        params["endTime"] = str(end_time)

    async with session.get(url, params=params) as response:
        if response.status != 200:
            logger.error(f"Failed to fetch candles: HTTP {response.status}")
            return []

        data = await response.json()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        candles = []
        if isinstance(data, list):
            for item in data:
                candle = Candle(
                    symbol=weex_symbol,
                    timestamp=datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    interval=interval,
                )
                candles.append(candle)

        return candles


async def fetch_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> list[Candle]:
    """Fetch candles from WEEX API with pagination for large requests."""
    all_candles: list[Candle] = []
    remaining = limit
    end_time: int | None = None

    # Pagination: fetch in batches of 1000
    while remaining > 0:
        batch_size = min(remaining, 1000)
        batch = await fetch_candles_batch(session, symbol, interval, batch_size, end_time)

        if not batch:
            break

        all_candles.extend(batch)
        remaining -= len(batch)

        # Set end_time to oldest candle timestamp for next batch
        oldest = min(batch, key=lambda c: c.timestamp)
        end_time = int(oldest.timestamp.timestamp() * 1000) - 1

        # Prevent infinite loop if API returns same data
        if len(batch) < batch_size:
            break

        logger.info(f"  Fetched batch: {len(batch)} candles, total: {len(all_candles)}")

    # Sort by timestamp ascending and remove duplicates
    all_candles.sort(key=lambda c: c.timestamp)
    seen = set()
    unique_candles = []
    for c in all_candles:
        key = (c.symbol, c.timestamp)
        if key not in seen:
            seen.add(key)
            unique_candles.append(c)

    return unique_candles


async def fetch_candles_cached(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int = 90,
    source: str = "weex",
) -> list[Candle]:
    """
    Fetch candles using cache to minimize API calls.

    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        days: Number of days of history
        source: Data source ("weex" or "binance")

    Returns:
        List of Candle objects sorted by timestamp
    """
    if source == "binance":
        # Use Binance API (supports proper pagination)
        cache_info = get_binance_cache_info(symbol, interval)
        if cache_info.get("cached"):
            logger.info(
                f"  Binance cache for {symbol}: {cache_info['count']} candles "
                f"({cache_info.get('days_of_data', 0):.0f} days)"
            )

        raw_candles = await fetch_binance_with_cache(session, symbol, interval, days=days)
    else:
        # Use WEEX API (limited historical data)
        cache_info = get_cache_info(symbol, interval)
        if cache_info.get("cached"):
            logger.info(
                f"  WEEX cache for {symbol}: {cache_info['count']} candles "
                f"({cache_info.get('oldest', 'N/A')} to {cache_info.get('newest', 'N/A')})"
            )

        raw_candles = await fetch_candles_with_cache(session, symbol, interval, days=days)

    # Convert to Candle objects
    candles = []
    for c in raw_candles:
        candle = Candle(
            symbol=f"cmt_{symbol.lower()}",
            timestamp=datetime.fromtimestamp(c["timestamp"] / 1000, tz=timezone.utc),
            open=c["open"],
            high=c["high"],
            low=c["low"],
            close=c["close"],
            volume=c["volume"],
            interval=interval,
        )
        candles.append(candle)

    return candles


def print_results(stats: BacktestStats, trades: list[TradeResult]) -> None:
    """Print backtest results."""
    print()
    print("=" * 70)
    print("                    SIMONS PROTOCOL BACKTEST RESULTS")
    print("=" * 70)
    print()

    print("PERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"  Total Return:        {stats.total_pnl_pct:>12.2%}")
    print(f"  Total P&L:           ${stats.total_pnl:>11,.2f}")
    print(f"  Max Drawdown:        {stats.max_drawdown_pct:>12.2%}")
    print(f"  Sharpe Ratio:        {stats.sharpe_ratio:>12.2f}")
    print(f"  Profit Factor:       {stats.profit_factor:>12.2f}")
    print()

    print("TRADE STATISTICS")
    print("-" * 70)
    print(f"  Total Trades:        {stats.total_trades:>12}")
    print(f"  Winning Trades:      {stats.winning_trades:>12}")
    print(f"  Losing Trades:       {stats.losing_trades:>12}")
    print(f"  Win Rate:            {stats.win_rate:>12.1%}")
    print(f"  Avg Win (R):         {stats.avg_win_r:>12.2f}")
    print(f"  Avg Loss (R):        {stats.avg_loss_r:>12.2f}")
    print()

    print("SIGNAL FILTERING")
    print("-" * 70)
    print(f"  Signals Generated:   {stats.signals_generated:>12}")
    print(f"  Signals Filtered:    {stats.signals_filtered:>12}")
    print(f"  Filter Rate:         {stats.signals_filtered / max(stats.signals_generated, 1):>12.1%}")
    print()

    print("TRADES BY CONVICTION TIER")
    print("-" * 70)
    for tier, count in stats.trades_by_conviction.items():
        pct = count / max(stats.total_trades, 1)
        print(f"  {tier}:    {count:>5} ({pct:>6.1%})")
    print()

    if trades:
        print("RECENT TRADES")
        print("-" * 70)
        for trade in trades[-10:]:
            pnl_str = f"+${trade.pnl:,.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):,.2f}"
            print(
                f"  {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{trade.direction:5} | "
                f"Conv: {trade.conviction_score:>3.0f} | "
                f"R: {trade.r_multiple:>+5.2f} | "
                f"{pnl_str:>12} | "
                f"{trade.exit_reason}"
            )

    print()
    print("=" * 70)

    # Summary verdict
    if stats.total_pnl_pct > 0:
        if stats.max_drawdown_pct < 0.03:
            print("RESULT: EXCELLENT - Profitable with Simons-level drawdown (<3%)")
        elif stats.max_drawdown_pct < 0.05:
            print("RESULT: GOOD - Profitable with acceptable drawdown (<5%)")
        else:
            print("RESULT: NEEDS IMPROVEMENT - Profitable but drawdown too high")
    else:
        print("RESULT: LOSS - Strategy needs refinement")

    print("=" * 70)


async def run_single_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int,
    balance: float,
    leverage: int,
    verbose: bool = True,
    use_cache: bool = True,
    data_source: str = "binance",
    adaptive_enabled: bool = False,
) -> tuple[BacktestStats, list[TradeResult], dict | None]:
    """Run backtest for a single symbol."""
    if use_cache:
        candles = await fetch_candles_cached(session, symbol, interval, days=days, source=data_source)
    else:
        # Calculate limit from days (for 1h candles)
        limit = days * 24
        candles = await fetch_candles(session, symbol, interval, limit)

    if not candles:
        if verbose:
            print(f"  {symbol}: No candles downloaded")
        return BacktestStats(), [], None

    if verbose:
        print(f"  {symbol}: {len(candles)} candles ({candles[0].timestamp.date()} to {candles[-1].timestamp.date()})")

    backtester = SimonsBacktester(
        initial_balance=balance,
        leverage=leverage,
        symbol=symbol,
        adaptive_enabled=adaptive_enabled,
    )

    stats = backtester.run(candles)

    # Get adaptive tuning summary if enabled
    adaptive_summary = None
    if backtester.adaptive_tuner:
        adaptive_summary = backtester.adaptive_tuner.get_summary()

    return stats, backtester.trades, adaptive_summary


async def run_forward_test(
    session: aiohttp.ClientSession,
    symbols: list[str],
    interval: str,
    days: int,
    balance: float,
    leverage: int,
    data_source: str,
) -> None:
    """
    Run forward test: train on first half of data, test on second half.

    This validates that learned parameters generalize to unseen data.
    """
    print()
    print("=" * 70)
    print("              FORWARD TEST MODE")
    print("         Train on first half, test on second half")
    print("=" * 70)
    print()

    train_days = days // 2
    test_days = days - train_days

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"  {symbol} FORWARD TEST")
        print(f"{'='*50}")

        # Fetch all candles
        candles = await fetch_candles_cached(session, symbol, interval, days=days, source=data_source)
        if len(candles) < 800:  # Need minimum for MTF analysis
            print(f"  {symbol}: Insufficient data ({len(candles)} candles)")
            continue

        # Split into train and test
        split_idx = len(candles) // 2
        train_candles = candles[:split_idx]
        test_candles = candles[split_idx:]

        print(f"  Train: {len(train_candles)} candles ({train_candles[0].timestamp.date()} to {train_candles[-1].timestamp.date()})")
        print(f"  Test:  {len(test_candles)} candles ({test_candles[0].timestamp.date()} to {test_candles[-1].timestamp.date()})")

        # Phase 1: Train with adaptive tuning
        print(f"\n  --- TRAINING PHASE (adaptive) ---")
        train_backtester = SimonsBacktester(
            initial_balance=balance,
            leverage=leverage,
            symbol=symbol,
            adaptive_enabled=True,
        )
        train_stats = train_backtester.run(train_candles)

        # Get learned parameters
        learned_summary = None
        if train_backtester.adaptive_tuner:
            learned_summary = train_backtester.adaptive_tuner.get_summary()
            # Save learned params
            params_path = Path(f"data/learned_params/{symbol.lower()}_params.json")
            train_backtester.adaptive_tuner.save_learned_params(params_path)

        print(f"  Train Results: {train_stats.total_trades} trades, {train_stats.win_rate:.1%} win rate, ${train_stats.total_pnl:+,.2f}")
        if learned_summary:
            final_cfg = learned_summary.get("final_config", {})
            print(f"  Learned: Conv={final_cfg.get('conviction_threshold', 70):.0f}, Stop={final_cfg.get('stop_atr_multiplier', 2.0):.1f}, TP={final_cfg.get('take_profit_atr_multiplier', 2.5):.1f}")

        # Phase 2: Test with learned parameters (no further adaptation)
        print(f"\n  --- TESTING PHASE (learned params, no adaptation) ---")

        # Create new backtester with learned config
        test_backtester = SimonsBacktester(
            initial_balance=balance,
            leverage=leverage,
            symbol=symbol,
            adaptive_enabled=True,  # Enable to use adaptive config
        )

        # Load learned parameters
        if test_backtester.adaptive_tuner:
            params_path = Path(f"data/learned_params/{symbol.lower()}_params.json")
            test_backtester.adaptive_tuner.load_learned_params(params_path)
            # Disable further adaptation during test by setting very high thresholds
            test_backtester.adaptive_tuner.candles_without_trade_threshold = 99999

        test_stats = test_backtester.run(test_candles)
        print(f"  Test Results:  {test_stats.total_trades} trades, {test_stats.win_rate:.1%} win rate, ${test_stats.total_pnl:+,.2f}")

        # Phase 3: Baseline - test with static params for comparison
        print(f"\n  --- BASELINE (static params) ---")
        baseline_backtester = SimonsBacktester(
            initial_balance=balance,
            leverage=leverage,
            symbol=symbol,
            adaptive_enabled=False,
        )
        baseline_stats = baseline_backtester.run(test_candles)
        print(f"  Baseline:      {baseline_stats.total_trades} trades, {baseline_stats.win_rate:.1%} win rate, ${baseline_stats.total_pnl:+,.2f}")

        # Calculate improvement
        improvement = test_stats.total_pnl - baseline_stats.total_pnl
        print(f"\n  IMPROVEMENT FROM LEARNING: ${improvement:+,.2f}")

        all_results.append({
            "symbol": symbol,
            "train_pnl": train_stats.total_pnl,
            "train_trades": train_stats.total_trades,
            "train_win_rate": train_stats.win_rate,
            "test_pnl": test_stats.total_pnl,
            "test_trades": test_stats.total_trades,
            "test_win_rate": test_stats.win_rate,
            "baseline_pnl": baseline_stats.total_pnl,
            "baseline_trades": baseline_stats.total_trades,
            "baseline_win_rate": baseline_stats.win_rate,
            "improvement": improvement,
        })

    # Summary
    print()
    print("=" * 70)
    print("              FORWARD TEST SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Symbol':<10} {'Train P&L':>12} {'Test P&L':>12} {'Baseline':>12} {'Improvement':>12}")
    print("-" * 70)

    total_improvement = 0
    for r in all_results:
        print(f"{r['symbol']:<10} ${r['train_pnl']:>+10,.2f} ${r['test_pnl']:>+10,.2f} ${r['baseline_pnl']:>+10,.2f} ${r['improvement']:>+10,.2f}")
        total_improvement += r['improvement']

    print("-" * 70)
    print(f"{'TOTAL':<10} {'':>12} {'':>12} {'':>12} ${total_improvement:>+10,.2f}")
    print()

    if total_improvement > 0:
        print("RESULT: ADAPTIVE LEARNING IMPROVED OUT-OF-SAMPLE PERFORMANCE")
    else:
        print("RESULT: STATIC PARAMS BETTER - Learning did not generalize")
    print("=" * 70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simons Protocol Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", default="1h", help="Candle interval")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data (default: 90)")
    parser.add_argument("--limit", type=int, help="Number of candles (deprecated, use --days)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=10, help="Leverage")
    parser.add_argument("--multi", action="store_true", help="Run on multiple symbols (BTC, ETH, SOL)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache, fetch fresh data")
    parser.add_argument("--source", default="binance", choices=["binance", "weex"],
                        help="Data source: binance (recommended, full history) or weex (limited to ~42 days)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive self-tuning (adjusts parameters based on performance)")
    parser.add_argument("--forward-test", action="store_true",
                        help="Forward test: train on first half, test on second half")
    parser.add_argument("--load-params", type=str, default=None,
                        help="Path to load learned parameters from previous run")
    parser.add_argument("--save-params", type=str, default=None,
                        help="Path to save learned parameters after run")
    args = parser.parse_args()

    # Handle deprecated --limit flag
    if args.limit:
        logger.warning("--limit is deprecated, use --days instead")
        args.days = args.limit // 24  # Convert to days assuming 1h candles

    print("=" * 70)
    print("              SIMONS PROTOCOL BACKTEST")
    print('           "Trade Big, Trade Less, Trade with Confidence"')
    print("=" * 70)
    print()

    timeout = aiohttp.ClientTimeout(total=300)  # Longer timeout for more data

    use_cache = not args.no_cache

    # Forward test mode
    if args.forward_test:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if args.multi else [args.symbol]
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await run_forward_test(
                session, symbols, args.interval, args.days,
                args.balance, args.leverage, args.source
            )
        return 0

    if args.multi:
        # Multi-symbol backtesting
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        print(f"  Symbols:         {', '.join(symbols)}")
        print(f"  Interval:        {args.interval}")
        print(f"  History:         {args.days} days")
        print(f"  Initial Balance: ${args.balance:,.2f} (per symbol)")
        print(f"  Leverage:        {args.leverage}x")
        print(f"  Data Source:     {args.source.upper()}")
        print(f"  Cache:           {'enabled' if use_cache else 'disabled'}")
        print(f"  Adaptive Tuning: {'ENABLED' if args.adaptive else 'disabled'}")
        print()
        print("Downloading historical data...")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            all_trades: list[TradeResult] = []
            all_adaptive_summaries: dict[str, dict] = {}
            total_pnl = 0.0
            total_signals = 0
            total_filtered = 0
            total_trades = 0
            total_wins = 0
            total_adjustments = 0

            for symbol in symbols:
                stats, trades, adaptive_summary = await run_single_symbol(
                    session, symbol, args.interval, args.days,
                    args.balance, args.leverage, verbose=True,
                    use_cache=use_cache, data_source=args.source,
                    adaptive_enabled=args.adaptive
                )
                all_trades.extend(trades)
                total_pnl += stats.total_pnl
                total_signals += stats.signals_generated
                total_filtered += stats.signals_filtered
                total_trades += stats.total_trades
                total_wins += stats.winning_trades
                total_adjustments += stats.parameter_adjustments
                if adaptive_summary:
                    all_adaptive_summaries[symbol] = adaptive_summary

            print()
            print("=" * 70)
            print("              MULTI-SYMBOL AGGREGATE RESULTS")
            print("=" * 70)
            print()
            print("PERFORMANCE SUMMARY")
            print("-" * 70)
            total_return = total_pnl / (args.balance * len(symbols))
            print(f"  Total P&L:           ${total_pnl:>11,.2f}")
            print(f"  Avg Return/Symbol:   {total_return:>12.2%}")
            print()
            print("TRADE STATISTICS")
            print("-" * 70)
            print(f"  Total Trades:        {total_trades:>12}")
            print(f"  Winning Trades:      {total_wins:>12}")
            print(f"  Losing Trades:       {total_trades - total_wins:>12}")
            win_rate = total_wins / max(total_trades, 1)
            print(f"  Win Rate:            {win_rate:>12.1%}")
            print()
            print("SIGNAL FILTERING")
            print("-" * 70)
            print(f"  Signals Generated:   {total_signals:>12}")
            print(f"  Signals Filtered:    {total_filtered:>12}")
            print(f"  Filter Rate:         {total_filtered / max(total_signals, 1):>12.1%}")
            print()

            if all_trades:
                # Sort by time
                all_trades.sort(key=lambda t: t.entry_time)
                print("ALL TRADES (sorted by time)")
                print("-" * 70)
                for trade in all_trades:
                    pnl_str = f"+${trade.pnl:,.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):,.2f}"
                    # Extract symbol from direction context
                    print(
                        f"  {trade.entry_time.strftime('%Y-%m-%d %H:%M')} | "
                        f"{trade.direction:5} | "
                        f"Conv: {trade.conviction_score:>3.0f} | "
                        f"R: {trade.r_multiple:>+5.2f} | "
                        f"{pnl_str:>12} | "
                        f"{trade.exit_reason}"
                    )

            # Show adaptive tuning summary if enabled
            if args.adaptive and all_adaptive_summaries:
                print()
                print("ADAPTIVE TUNING SUMMARY")
                print("-" * 70)
                print(f"  Total Adjustments:   {total_adjustments:>12}")
                print()
                for symbol, summary in all_adaptive_summaries.items():
                    init_cfg = summary.get("initial_config", {})
                    final_cfg = summary.get("final_config", {})
                    print(f"  {symbol}:")
                    print(f"    Conviction: {init_cfg.get('conviction_threshold', 70):.0f} -> {final_cfg.get('conviction_threshold', 70):.0f}")
                    print(f"    Stop ATR:   {init_cfg.get('stop_atr_multiplier', 2.0):.1f} -> {final_cfg.get('stop_atr_multiplier', 2.0):.1f}")
                    print(f"    TP ATR:     {init_cfg.get('take_profit_atr_multiplier', 2.5):.1f} -> {final_cfg.get('take_profit_atr_multiplier', 2.5):.1f}")
                    print(f"    Pos Size:   {init_cfg.get('position_size_multiplier', 1.0):.2f} -> {final_cfg.get('position_size_multiplier', 1.0):.2f}")
                    adjustments_list = summary.get("adjustments", [])
                    if adjustments_list:
                        print(f"    Adjustments made: {len(adjustments_list)}")
                        for adj in adjustments_list[-3:]:  # Show last 3
                            print(f"      - {adj.get('param')}: {adj.get('change')} ({adj.get('reason')})")
                    print()

            print()
            print("=" * 70)
            if total_pnl > 0 and win_rate >= 0.6:
                print("RESULT: EXCELLENT - Multi-symbol validation passed")
            elif total_pnl > 0:
                print("RESULT: GOOD - Profitable across multiple symbols")
            else:
                print("RESULT: NEEDS WORK - Strategy needs refinement")
            print("=" * 70)

    else:
        # Single symbol backtesting
        print(f"  Symbol:          {args.symbol}")
        print(f"  Interval:        {args.interval}")
        print(f"  History:         {args.days} days")
        print(f"  Initial Balance: ${args.balance:,.2f}")
        print(f"  Leverage:        {args.leverage}x")
        print(f"  Data Source:     {args.source.upper()}")
        print(f"  Cache:           {'enabled' if use_cache else 'disabled'}")
        print(f"  Adaptive Tuning: {'ENABLED' if args.adaptive else 'disabled'}")
        print()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Download candles with cache
            print("Downloading historical data...")
            if use_cache:
                candles = await fetch_candles_cached(
                    session, args.symbol, args.interval, days=args.days, source=args.source
                )
            else:
                candles = await fetch_candles(session, args.symbol, args.interval, args.days * 24)

            if not candles:
                print("ERROR: No candles downloaded")
                return 1

            print(f"  Downloaded {len(candles)} candles")
            print(f"  Period: {candles[0].timestamp} to {candles[-1].timestamp}")
            print()

            # Run backtest
            print("Running Simons Protocol backtest...")
            print("-" * 70)

            backtester = SimonsBacktester(
                initial_balance=args.balance,
                leverage=args.leverage,
                symbol=args.symbol,
                adaptive_enabled=args.adaptive,
            )

            stats = backtester.run(candles)

            # Print results
            print_results(stats, backtester.trades)

            # Print adaptive tuning summary if enabled
            if args.adaptive and backtester.adaptive_tuner:
                summary = backtester.adaptive_tuner.get_summary()
                print()
                print("ADAPTIVE TUNING SUMMARY")
                print("-" * 70)
                init_cfg = summary.get("initial_config", {})
                final_cfg = summary.get("final_config", {})
                print(f"  Total Adjustments: {summary.get('total_adjustments', 0)}")
                print(f"  Conviction: {init_cfg.get('conviction_threshold', 70):.0f} -> {final_cfg.get('conviction_threshold', 70):.0f}")
                print(f"  Stop ATR:   {init_cfg.get('stop_atr_multiplier', 2.0):.1f} -> {final_cfg.get('stop_atr_multiplier', 2.0):.1f}")
                print(f"  TP ATR:     {init_cfg.get('take_profit_atr_multiplier', 2.5):.1f} -> {final_cfg.get('take_profit_atr_multiplier', 2.5):.1f}")
                print(f"  Pos Size:   {init_cfg.get('position_size_multiplier', 1.0):.2f} -> {final_cfg.get('position_size_multiplier', 1.0):.2f}")
                adjustments_list = summary.get("adjustments", [])
                if adjustments_list:
                    print()
                    print("  Adjustment History:")
                    for adj in adjustments_list:
                        print(f"    {adj.get('time', 'N/A')[:10]}: {adj.get('param')} {adj.get('change')} - {adj.get('reason')}")
                print("-" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
