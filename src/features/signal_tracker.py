"""
AlphaStrike Trading Bot - Signal Decay Tracker (Simons-Inspired)

Tracks the performance of each signal over time and adjusts weights accordingly.
This implements Jim Simons' principle: "Retire signals when they stop working."

Key Features:
1. Per-signal accuracy tracking over rolling windows
2. Automatic weight adjustment based on recent performance
3. Signal retirement when accuracy drops below threshold
4. Regime-aware signal performance (signals may work in some regimes but not others)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """Record of a single signal prediction."""
    signal_name: str
    prediction: float  # -1 to +1 (direction and strength)
    actual_outcome: float  # Actual price change (normalized)
    was_correct: bool  # Did prediction match outcome direction?
    timestamp: datetime
    regime: str = "unknown"
    symbol: str = ""


@dataclass
class SignalStats:
    """Statistics for a single signal."""
    signal_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5  # Default to 50% (random)
    avg_magnitude: float = 0.0  # Average prediction magnitude
    weight: float = 1.0  # Current weight (0 = retired)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_retired: bool = False

    @property
    def edge(self) -> float:
        """Edge over random (accuracy - 0.5)."""
        return self.accuracy - 0.5

    @property
    def scaled_weight(self) -> float:
        """Weight scaled by edge (signals with more edge get more weight)."""
        if self.is_retired or self.accuracy < 0.5:
            return 0.0
        return self.edge * 2  # Scale edge to 0-1 range


class SignalTracker:
    """
    Tracks signal performance over time and adjusts weights.

    Usage:
        tracker = SignalTracker()

        # Record predictions before each trade
        tracker.record_prediction("funding_signal", prediction=-0.5, symbol="BTCUSDT")

        # After trade outcome is known, record result
        tracker.record_outcome("BTCUSDT", actual_return=-0.02)

        # Get current signal weights
        weights = tracker.get_signal_weights()
    """

    # Thresholds
    MIN_SAMPLES_FOR_EVALUATION = 20  # Minimum samples before adjusting weight
    RETIREMENT_ACCURACY = 0.48  # Retire signals below this accuracy
    LOOKBACK_WINDOW = 100  # Rolling window for accuracy calculation

    def __init__(
        self,
        signals_dir: Path | None = None,
        lookback_window: int = 100,
    ):
        """
        Initialize the signal tracker.

        Args:
            signals_dir: Directory to persist signal stats
            lookback_window: Rolling window for accuracy calculation
        """
        self.signals_dir = signals_dir or Path("data/signal_stats")
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_window = lookback_window

        # Per-signal history (rolling window)
        self._history: dict[str, deque[SignalRecord]] = {}

        # Per-signal statistics
        self._stats: dict[str, SignalStats] = {}

        # Pending predictions (waiting for outcome)
        self._pending: dict[str, list[tuple[str, float, datetime]]] = {}  # symbol -> [(signal_name, prediction, time)]

        # Load persisted stats
        self._load_stats()

        logger.info(f"SignalTracker initialized with {len(self._stats)} tracked signals")

    def _load_stats(self):
        """Load persisted signal stats."""
        stats_file = self.signals_dir / "signal_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                for name, stats_dict in data.items():
                    self._stats[name] = SignalStats(
                        signal_name=name,
                        total_predictions=stats_dict.get("total_predictions", 0),
                        correct_predictions=stats_dict.get("correct_predictions", 0),
                        accuracy=stats_dict.get("accuracy", 0.5),
                        weight=stats_dict.get("weight", 1.0),
                        is_retired=stats_dict.get("is_retired", False),
                    )
                logger.info(f"Loaded stats for {len(self._stats)} signals")
            except Exception as e:
                logger.warning(f"Failed to load signal stats: {e}")

    def _save_stats(self):
        """Persist signal stats to disk."""
        stats_file = self.signals_dir / "signal_stats.json"
        data = {}
        for name, stats in self._stats.items():
            data[name] = {
                "total_predictions": stats.total_predictions,
                "correct_predictions": stats.correct_predictions,
                "accuracy": stats.accuracy,
                "weight": stats.weight,
                "is_retired": stats.is_retired,
            }
        try:
            with open(stats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save signal stats: {e}")

    def record_prediction(
        self,
        signal_name: str,
        prediction: float,
        symbol: str,
        regime: str = "unknown",
    ) -> None:
        """
        Record a signal prediction before trade execution.

        Args:
            signal_name: Name of the signal
            prediction: Predicted direction/magnitude (-1 to +1)
            symbol: Trading symbol
            regime: Current market regime
        """
        # Initialize if needed
        if signal_name not in self._stats:
            self._stats[signal_name] = SignalStats(signal_name=signal_name)
        if signal_name not in self._history:
            self._history[signal_name] = deque(maxlen=self.lookback_window)
        if symbol not in self._pending:
            self._pending[symbol] = []

        # Store pending prediction
        self._pending[symbol].append((signal_name, prediction, datetime.now(UTC)))

    def record_outcome(
        self,
        symbol: str,
        actual_return: float,
    ) -> None:
        """
        Record the actual outcome of a trade.

        Args:
            symbol: Trading symbol
            actual_return: Actual return from the trade (positive = profit)
        """
        if symbol not in self._pending:
            return

        # Process all pending predictions for this symbol
        for signal_name, prediction, timestamp in self._pending[symbol]:
            # Determine if prediction was correct
            # Correct if prediction and outcome have same sign
            was_correct = (prediction > 0 and actual_return > 0) or (prediction < 0 and actual_return < 0)

            # Record in history
            record = SignalRecord(
                signal_name=signal_name,
                prediction=prediction,
                actual_outcome=actual_return,
                was_correct=was_correct,
                timestamp=timestamp,
                symbol=symbol,
            )
            self._history[signal_name].append(record)

            # Update stats
            self._update_signal_stats(signal_name)

        # Clear pending
        self._pending[symbol] = []

        # Persist stats periodically
        self._save_stats()

    def _update_signal_stats(self, signal_name: str) -> None:
        """Update statistics for a signal based on recent history."""
        if signal_name not in self._history:
            return

        history = list(self._history[signal_name])
        if len(history) < self.MIN_SAMPLES_FOR_EVALUATION:
            return

        # Calculate accuracy from recent history
        correct = sum(1 for r in history if r.was_correct)
        accuracy = correct / len(history)

        # Update stats
        stats = self._stats[signal_name]
        stats.total_predictions = len(history)
        stats.correct_predictions = correct
        stats.accuracy = accuracy
        stats.last_updated = datetime.now(UTC)

        # Calculate weight based on accuracy
        if accuracy < self.RETIREMENT_ACCURACY:
            stats.is_retired = True
            stats.weight = 0.0
            logger.warning(f"Signal {signal_name} RETIRED: accuracy={accuracy:.1%}")
        else:
            stats.is_retired = False
            # Weight proportional to edge squared (emphasize strong signals)
            edge = accuracy - 0.5
            stats.weight = (edge * 2) ** 2  # 0 to 1

    def get_signal_weights(self) -> dict[str, float]:
        """
        Get current weights for all signals.

        Returns:
            Dictionary mapping signal names to weights (0-1)
        """
        return {
            name: stats.scaled_weight
            for name, stats in self._stats.items()
            if not stats.is_retired
        }

    def get_signal_stats(self, signal_name: str) -> SignalStats | None:
        """Get stats for a specific signal."""
        return self._stats.get(signal_name)

    def get_active_signals(self) -> list[str]:
        """Get list of non-retired signals."""
        return [name for name, stats in self._stats.items() if not stats.is_retired]

    def get_retired_signals(self) -> list[str]:
        """Get list of retired signals."""
        return [name for name, stats in self._stats.items() if stats.is_retired]

    def generate_report(self) -> str:
        """Generate a human-readable report of signal performance."""
        lines = [
            "=" * 60,
            "SIGNAL PERFORMANCE REPORT",
            "=" * 60,
            "",
        ]

        if not self._stats:
            lines.append("No signals tracked yet.")
            return "\n".join(lines)

        # Sort by accuracy
        sorted_signals = sorted(
            self._stats.values(),
            key=lambda s: s.accuracy,
            reverse=True,
        )

        lines.append(f"{'Signal Name':<25} {'Accuracy':<10} {'Edge':<8} {'Weight':<8} {'Status':<10}")
        lines.append("-" * 60)

        for stats in sorted_signals:
            status = "RETIRED" if stats.is_retired else "Active"
            lines.append(
                f"{stats.signal_name:<25} {stats.accuracy:>8.1%} {stats.edge:>+7.1%} "
                f"{stats.weight:>7.2f} {status:<10}"
            )

        lines.extend([
            "",
            "-" * 60,
            f"Active Signals: {len(self.get_active_signals())}",
            f"Retired Signals: {len(self.get_retired_signals())}",
            "=" * 60,
        ])

        return "\n".join(lines)


# Singleton instance
_tracker: SignalTracker | None = None


def get_signal_tracker() -> SignalTracker:
    """Get or create the global signal tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SignalTracker()
    return _tracker
