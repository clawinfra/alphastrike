"""
Walk-Forward Optimization Framework

Prevents overfitting by using rolling in-sample/out-of-sample windows.
This is how Renaissance Technologies validates their models.

Key Concepts:
- In-Sample (IS): Data used for parameter optimization
- Out-of-Sample (OOS): Data held back for validation
- Walk-Forward: Roll the windows forward through history
- Anchored: IS always starts from same date (more data over time)
- Rolling: IS window slides forward (fixed size)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward optimization window."""

    window_id: int
    is_start: datetime  # In-sample start
    is_end: datetime  # In-sample end
    oos_start: datetime  # Out-of-sample start
    oos_end: datetime  # Out-of-sample end

    # Results
    best_params: dict[str, Any] = field(default_factory=dict)
    is_sharpe: float = 0.0
    is_return: float = 0.0
    is_trades: int = 0
    oos_sharpe: float = 0.0
    oos_return: float = 0.0
    oos_trades: int = 0

    # Efficiency ratio: OOS / IS performance
    efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "is_start": self.is_start.isoformat(),
            "is_end": self.is_end.isoformat(),
            "oos_start": self.oos_start.isoformat(),
            "oos_end": self.oos_end.isoformat(),
            "best_params": self.best_params,
            "is_sharpe": self.is_sharpe,
            "is_return": self.is_return,
            "is_trades": self.is_trades,
            "oos_sharpe": self.oos_sharpe,
            "oos_return": self.oos_return,
            "oos_trades": self.oos_trades,
            "efficiency": self.efficiency,
        }


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""

    symbol: str
    n_windows: int
    window_type: Literal["rolling", "anchored"]

    # Aggregate metrics
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_efficiency: float = 0.0
    total_is_return: float = 0.0
    total_oos_return: float = 0.0

    # Individual windows
    windows: list[WalkForwardWindow] = field(default_factory=list)

    # Final recommended params (from most recent window)
    recommended_params: dict[str, Any] = field(default_factory=dict)

    # Validation
    passed_validation: bool = False
    validation_message: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_windows": self.n_windows,
            "window_type": self.window_type,
            "avg_is_sharpe": self.avg_is_sharpe,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "avg_efficiency": self.avg_efficiency,
            "total_is_return": self.total_is_return,
            "total_oos_return": self.total_oos_return,
            "windows": [w.to_dict() for w in self.windows],
            "recommended_params": self.recommended_params,
            "passed_validation": self.passed_validation,
            "validation_message": self.validation_message,
        }


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.

    Rolling Window Example (3 windows):
    ┌──────────────────────────────────────────────────────────────┐
    │ Window 1: [IS: Jan-Mar] [OOS: Apr]                           │
    │ Window 2:      [IS: Feb-Apr] [OOS: May]                      │
    │ Window 3:           [IS: Mar-May] [OOS: Jun]                 │
    └──────────────────────────────────────────────────────────────┘

    Anchored Window Example (3 windows):
    ┌──────────────────────────────────────────────────────────────┐
    │ Window 1: [IS: Jan-Mar] [OOS: Apr]                           │
    │ Window 2: [IS: Jan-Apr] [OOS: May]                           │
    │ Window 3: [IS: Jan-May] [OOS: Jun]                           │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        optimize_func: Callable[[str, datetime, datetime], tuple[dict, float]],
        backtest_func: Callable[[str, dict, datetime, datetime], dict],
        is_days: int = 90,  # In-sample window size
        oos_days: int = 30,  # Out-of-sample window size
        n_windows: int = 4,  # Number of walk-forward windows
        window_type: Literal["rolling", "anchored"] = "rolling",
        min_efficiency: float = 0.5,  # OOS must be >= 50% of IS
        min_oos_sharpe: float = 0.3,  # Minimum acceptable OOS Sharpe
    ):
        """
        Initialize walk-forward validator.

        Args:
            optimize_func: Function(symbol, start, end) -> (best_params, sharpe)
            backtest_func: Function(symbol, params, start, end) -> {sharpe, return, trades}
            is_days: In-sample period length
            oos_days: Out-of-sample period length
            n_windows: Number of rolling windows
            window_type: "rolling" or "anchored"
            min_efficiency: Minimum OOS/IS ratio to pass
            min_oos_sharpe: Minimum OOS Sharpe to pass
        """
        self.optimize_func = optimize_func
        self.backtest_func = backtest_func
        self.is_days = is_days
        self.oos_days = oos_days
        self.n_windows = n_windows
        self.window_type = window_type
        self.min_efficiency = min_efficiency
        self.min_oos_sharpe = min_oos_sharpe

    def run(
        self,
        symbol: str,
        end_date: datetime | None = None,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward analysis.

        Args:
            symbol: Trading pair
            end_date: End of analysis period (default: now)

        Returns:
            WalkForwardResult with all windows and validation status
        """
        if end_date is None:
            end_date = datetime.now(UTC)

        # Generate window dates
        windows = self._generate_windows(end_date)

        logger.info(
            f"Running walk-forward for {symbol}: {len(windows)} windows, "
            f"IS={self.is_days}d, OOS={self.oos_days}d"
        )

        # Run each window
        for window in windows:
            self._run_window(symbol, window)

        # Calculate aggregate metrics
        result = self._aggregate_results(symbol, windows)

        # Validate
        result.passed_validation, result.validation_message = self._validate(result)

        return result

    def _generate_windows(self, end_date: datetime) -> list[WalkForwardWindow]:
        """Generate walk-forward window dates."""
        windows = []

        # Work backwards from end date
        current_oos_end = end_date

        for i in range(self.n_windows - 1, -1, -1):  # Reverse order
            oos_end = current_oos_end
            oos_start = oos_end - timedelta(days=self.oos_days)
            is_end = oos_start

            if self.window_type == "rolling":
                is_start = is_end - timedelta(days=self.is_days)
            else:  # anchored
                # All windows start from same point
                total_needed = self.is_days + (self.n_windows - 1 - i) * self.oos_days
                is_start = end_date - timedelta(days=total_needed + self.oos_days)

            windows.append(WalkForwardWindow(
                window_id=i,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            ))

            # Move to previous window
            current_oos_end = oos_start

        # Reverse to chronological order
        windows.reverse()
        return windows

    def _run_window(self, symbol: str, window: WalkForwardWindow) -> None:
        """Run optimization and validation for a single window."""
        logger.info(f"Window {window.window_id}: IS={window.is_start.date()} to {window.is_end.date()}")

        try:
            # Optimize on in-sample
            best_params, is_sharpe = self.optimize_func(
                symbol, window.is_start, window.is_end
            )
            window.best_params = best_params
            window.is_sharpe = is_sharpe

            # Get full IS metrics
            is_result = self.backtest_func(
                symbol, best_params, window.is_start, window.is_end
            )
            window.is_return = is_result.get("return", 0)
            window.is_trades = is_result.get("n_trades", 0)

            # Validate on out-of-sample
            oos_result = self.backtest_func(
                symbol, best_params, window.oos_start, window.oos_end
            )
            window.oos_sharpe = oos_result.get("sharpe", 0)
            window.oos_return = oos_result.get("return", 0)
            window.oos_trades = oos_result.get("n_trades", 0)

            # Calculate efficiency
            if window.is_sharpe > 0:
                window.efficiency = window.oos_sharpe / window.is_sharpe
            else:
                window.efficiency = 0

            logger.info(
                f"  IS: Sharpe={window.is_sharpe:.2f}, Return={window.is_return:.2%}"
            )
            logger.info(
                f"  OOS: Sharpe={window.oos_sharpe:.2f}, Return={window.oos_return:.2%}, "
                f"Efficiency={window.efficiency:.2%}"
            )

        except Exception as e:
            logger.error(f"Window {window.window_id} failed: {e}")
            window.efficiency = 0

    def _aggregate_results(
        self, symbol: str, windows: list[WalkForwardWindow]
    ) -> WalkForwardResult:
        """Aggregate results from all windows."""
        valid_windows = [w for w in windows if w.is_sharpe != 0]

        if not valid_windows:
            return WalkForwardResult(
                symbol=symbol,
                n_windows=len(windows),
                window_type=self.window_type,
                windows=windows,
            )

        avg_is_sharpe = np.mean([w.is_sharpe for w in valid_windows])
        avg_oos_sharpe = np.mean([w.oos_sharpe for w in valid_windows])
        avg_efficiency = np.mean([w.efficiency for w in valid_windows])

        # Compound returns
        total_is_return = np.prod([1 + w.is_return for w in valid_windows]) - 1
        total_oos_return = np.prod([1 + w.oos_return for w in valid_windows]) - 1

        # Use most recent window's params as recommendation
        recommended = valid_windows[-1].best_params if valid_windows else {}

        return WalkForwardResult(
            symbol=symbol,
            n_windows=len(windows),
            window_type=self.window_type,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_efficiency=avg_efficiency,
            total_is_return=total_is_return,
            total_oos_return=total_oos_return,
            windows=windows,
            recommended_params=recommended,
        )

    def _validate(self, result: WalkForwardResult) -> tuple[bool, str]:
        """Validate walk-forward results."""
        # Check efficiency
        if result.avg_efficiency < self.min_efficiency:
            return False, (
                f"Efficiency {result.avg_efficiency:.2%} below threshold "
                f"{self.min_efficiency:.2%} - likely overfitting"
            )

        # Check OOS Sharpe
        if result.avg_oos_sharpe < self.min_oos_sharpe:
            return False, (
                f"OOS Sharpe {result.avg_oos_sharpe:.2f} below threshold "
                f"{self.min_oos_sharpe:.2f}"
            )

        # Check consistency (variance of OOS results)
        oos_sharpes = [w.oos_sharpe for w in result.windows if w.oos_sharpe != 0]
        if len(oos_sharpes) >= 2:
            oos_std = np.std(oos_sharpes)
            if oos_std > abs(result.avg_oos_sharpe):
                return False, (
                    f"OOS results inconsistent: std={oos_std:.2f} > mean={result.avg_oos_sharpe:.2f}"
                )

        return True, "Walk-forward validation passed"


def quick_walk_forward_split(
    data: list,
    is_ratio: float = 0.7,
) -> tuple[list, list]:
    """
    Simple helper to split data into in-sample and out-of-sample.

    Args:
        data: List of candles or trades
        is_ratio: Ratio of data for in-sample (default 70%)

    Returns:
        (in_sample_data, out_of_sample_data)
    """
    split_idx = int(len(data) * is_ratio)
    return data[:split_idx], data[split_idx:]
