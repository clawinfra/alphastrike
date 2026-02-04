"""
Per-Asset Performance Tracker

Monitors rolling performance metrics for each trading pair and
generates retune signals when performance degrades.
"""

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal


@dataclass
class Trade:
    """Record of a single completed trade."""

    trade_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float  # In quote currency
    pnl_r: float  # In R-multiples (risk units)
    fees: float
    conviction_score: float
    regime: str
    exit_reason: Literal["STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "SIGNAL", "MANUAL"]


@dataclass
class AssetPerformance:
    """Rolling performance metrics for a single asset."""

    symbol: str
    window_days: int = 30

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L metrics
    total_pnl: float = 0.0
    total_pnl_r: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0  # Expected R per trade

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0

    # Peak tracking
    peak_equity: float = 0.0
    equity: float = 0.0

    # Timestamps
    last_trade_time: datetime | None = None
    last_updated: datetime | None = None

    # Status
    needs_retune: bool = False
    retune_reason: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "symbol": self.symbol,
            "window_days": self.window_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_pnl_r": self.total_pnl_r,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "profit_factor": self.profit_factor,
            "expectancy_r": self.expectancy_r,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "max_consecutive_losses": self.max_consecutive_losses,
            "current_consecutive_losses": self.current_consecutive_losses,
            "peak_equity": self.peak_equity,
            "equity": self.equity,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "needs_retune": self.needs_retune,
            "retune_reason": self.retune_reason,
        }


@dataclass
class RetuneTrigger:
    """Signal that retuning is needed for an asset."""

    symbol: str
    trigger_type: Literal[
        "PERFORMANCE_DROP",
        "DRAWDOWN_BREACH",
        "REGIME_CHANGE",
        "SCHEDULED",
        "CONSECUTIVE_LOSSES",
    ]
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    reason: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "trigger_type": self.trigger_type,
            "severity": self.severity,
            "reason": self.reason,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceTracker:
    """
    Tracks per-asset performance and generates retune triggers.

    Uses a rolling window approach to evaluate recent performance
    and detect when parameters need adjustment.
    """

    def __init__(
        self,
        window_days: int = 30,
        win_rate_threshold: float = 0.5,
        max_drawdown_threshold: float = 0.05,
        max_consecutive_losses: int = 5,
        min_trades_for_evaluation: int = 10,
    ):
        self.window_days = window_days
        self.win_rate_threshold = win_rate_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self.min_trades_for_evaluation = min_trades_for_evaluation

        # Per-symbol tracking
        self._trades: dict[str, deque[Trade]] = {}
        self._performance: dict[str, AssetPerformance] = {}
        self._triggers: list[RetuneTrigger] = []

    def record_trade(self, trade: Trade) -> RetuneTrigger | None:
        """
        Record a completed trade and update performance metrics.

        Returns a RetuneTrigger if performance thresholds are breached.
        """
        symbol = trade.symbol

        # Initialize if needed
        if symbol not in self._trades:
            self._trades[symbol] = deque(maxlen=1000)  # Keep last 1000 trades
            self._performance[symbol] = AssetPerformance(
                symbol=symbol, window_days=self.window_days
            )

        # Add trade
        self._trades[symbol].append(trade)

        # Recalculate metrics
        self._recalculate_metrics(symbol)

        # Check for triggers
        trigger = self._check_triggers(symbol)
        if trigger:
            self._triggers.append(trigger)
            self._performance[symbol].needs_retune = True
            self._performance[symbol].retune_reason = trigger.reason

        return trigger

    def _recalculate_metrics(self, symbol: str) -> None:
        """Recalculate all metrics for a symbol based on rolling window."""
        perf = self._performance[symbol]
        trades = self._get_window_trades(symbol)

        if not trades:
            return

        # Basic counts
        perf.total_trades = len(trades)
        perf.winning_trades = sum(1 for t in trades if t.pnl > 0)
        perf.losing_trades = sum(1 for t in trades if t.pnl <= 0)
        perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0

        # P&L calculations
        perf.total_pnl = sum(t.pnl for t in trades)
        perf.total_pnl_r = sum(t.pnl_r for t in trades)

        wins = [t.pnl_r for t in trades if t.pnl_r > 0]
        losses = [abs(t.pnl_r) for t in trades if t.pnl_r <= 0]

        perf.avg_win_r = sum(wins) / len(wins) if wins else 0
        perf.avg_loss_r = sum(losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        perf.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy (expected R per trade)
        perf.expectancy_r = perf.total_pnl_r / perf.total_trades if perf.total_trades > 0 else 0

        # Consecutive losses
        current_streak = 0
        max_streak = 0
        for t in trades:
            if t.pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        perf.max_consecutive_losses = max_streak
        perf.current_consecutive_losses = current_streak

        # Drawdown calculation (simplified - based on cumulative R)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            cumulative += t.pnl_r
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        perf.max_drawdown = max_dd
        perf.current_drawdown = (peak - cumulative) / peak if peak > 0 else 0

        perf.last_trade_time = trades[-1].exit_time
        perf.last_updated = datetime.now(UTC)

    def _get_window_trades(self, symbol: str) -> list[Trade]:
        """Get trades within the rolling window."""
        cutoff = datetime.now(UTC) - timedelta(days=self.window_days)
        return [t for t in self._trades.get(symbol, []) if t.exit_time >= cutoff]

    def _check_triggers(self, symbol: str) -> RetuneTrigger | None:
        """Check if any retune triggers are breached."""
        perf = self._performance[symbol]

        # Need minimum trades for evaluation
        if perf.total_trades < self.min_trades_for_evaluation:
            return None

        # Check win rate
        if perf.win_rate < self.win_rate_threshold:
            return RetuneTrigger(
                symbol=symbol,
                trigger_type="PERFORMANCE_DROP",
                severity="WARNING",
                reason=f"Win rate {perf.win_rate:.1%} below threshold {self.win_rate_threshold:.1%}",
                current_value=perf.win_rate,
                threshold_value=self.win_rate_threshold,
            )

        # Check drawdown
        if perf.max_drawdown > self.max_drawdown_threshold:
            return RetuneTrigger(
                symbol=symbol,
                trigger_type="DRAWDOWN_BREACH",
                severity="CRITICAL",
                reason=f"Max drawdown {perf.max_drawdown:.1%} exceeds threshold {self.max_drawdown_threshold:.1%}",
                current_value=perf.max_drawdown,
                threshold_value=self.max_drawdown_threshold,
            )

        # Check consecutive losses
        if perf.current_consecutive_losses >= self.max_consecutive_losses:
            return RetuneTrigger(
                symbol=symbol,
                trigger_type="CONSECUTIVE_LOSSES",
                severity="WARNING",
                reason=f"{perf.current_consecutive_losses} consecutive losses",
                current_value=float(perf.current_consecutive_losses),
                threshold_value=float(self.max_consecutive_losses),
            )

        return None

    def get_performance(self, symbol: str) -> AssetPerformance | None:
        """Get current performance metrics for a symbol."""
        return self._performance.get(symbol)

    def get_all_performance(self) -> dict[str, AssetPerformance]:
        """Get performance metrics for all tracked symbols."""
        return self._performance.copy()

    def get_pending_triggers(self) -> list[RetuneTrigger]:
        """Get all pending retune triggers."""
        return self._triggers.copy()

    def clear_trigger(self, symbol: str) -> None:
        """Clear retune trigger for a symbol (after retuning)."""
        self._triggers = [t for t in self._triggers if t.symbol != symbol]
        if symbol in self._performance:
            self._performance[symbol].needs_retune = False
            self._performance[symbol].retune_reason = ""

    def generate_report(self) -> str:
        """Generate a human-readable performance report."""
        lines = ["=" * 70, "ASSET PERFORMANCE REPORT", "=" * 70, ""]

        for symbol, perf in sorted(self._performance.items()):
            status = "⚠️ NEEDS RETUNE" if perf.needs_retune else "✓ OK"
            lines.extend([
                f"{symbol}: {status}",
                "-" * 40,
                f"  Trades: {perf.total_trades} ({perf.winning_trades}W / {perf.losing_trades}L)",
                f"  Win Rate: {perf.win_rate:.1%}",
                f"  Total P&L: {perf.total_pnl_r:.2f}R (${perf.total_pnl:.2f})",
                f"  Expectancy: {perf.expectancy_r:.2f}R/trade",
                f"  Profit Factor: {perf.profit_factor:.2f}",
                f"  Max Drawdown: {perf.max_drawdown:.1%}",
                f"  Consecutive Losses: {perf.current_consecutive_losses}",
            ])
            if perf.retune_reason:
                lines.append(f"  Retune Reason: {perf.retune_reason}")
            lines.append("")

        return "\n".join(lines)

    def save_state(self, path: Path) -> None:
        """Save tracker state to file."""
        state = {
            "performance": {s: p.to_dict() for s, p in self._performance.items()},
            "triggers": [t.to_dict() for t in self._triggers],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Path) -> None:
        """Load tracker state from file."""
        if not path.exists():
            return
        with open(path) as f:
            _state = json.load(f)  # TODO: Implement state restoration
        # Note: This only loads performance summaries, not individual trades
        # Full trade history would need separate persistence
