"""
Backtest Performance Metrics

Calculates comprehensive performance metrics from backtest results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.engine import BacktestTrade


@dataclass
class BacktestMetrics:
    """Complete metrics for a backtest run."""

    # Returns
    total_return: float  # Total return %
    cagr: float  # Compound annual growth rate

    # Risk metrics
    sharpe_ratio: float  # Annualized Sharpe (risk-free = 0)
    sortino_ratio: float  # Downside deviation variant
    max_drawdown: float  # Maximum drawdown %
    max_drawdown_duration: int  # Duration in periods
    calmar_ratio: float  # CAGR / Max DD

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # Winning trades %
    profit_factor: float  # Gross profit / Gross loss
    avg_win: float  # Average winning trade %
    avg_loss: float  # Average losing trade %
    largest_win: float  # Largest single win
    largest_loss: float  # Largest single loss
    avg_trade_duration_minutes: float

    # Exposure
    exposure_time: float  # % of time in market

    # Balances
    initial_balance: float
    final_balance: float


class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    @staticmethod
    def calculate(
        trades: list[BacktestTrade],
        equity_curve: list[tuple[datetime, float]],
        initial_balance: float,
        interval_minutes: int = 1,
    ) -> BacktestMetrics:
        """
        Calculate all metrics from trades and equity curve.

        Args:
            trades: List of completed trades
            equity_curve: List of (timestamp, equity) tuples
            initial_balance: Starting balance
            interval_minutes: Candle interval in minutes (for annualization)

        Returns:
            BacktestMetrics with all calculated values
        """
        if not equity_curve:
            return MetricsCalculator._empty_metrics(initial_balance)

        final_balance = equity_curve[-1][1]
        total_return = ((final_balance - initial_balance) / initial_balance) * 100

        # Calculate returns series
        returns = MetricsCalculator._calculate_returns(equity_curve)

        # Time-based calculations
        periods_per_year = (365 * 24 * 60) / interval_minutes
        duration_days = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds() / 86400
        duration_years = duration_days / 365.0 if duration_days > 0 else 1.0

        # CAGR
        if duration_years > 0 and final_balance > 0:
            cagr = ((final_balance / initial_balance) ** (1 / duration_years) - 1) * 100
        else:
            cagr = 0.0

        # Sharpe ratio (annualized)
        sharpe = MetricsCalculator._calculate_sharpe(returns, periods_per_year)

        # Sortino ratio (annualized)
        sortino = MetricsCalculator._calculate_sortino(returns, periods_per_year)

        # Max drawdown
        max_dd, max_dd_duration = MetricsCalculator._calculate_max_drawdown(equity_curve)

        # Calmar ratio
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

        # Trade statistics
        trade_stats = MetricsCalculator._calculate_trade_stats(trades)

        # Exposure time
        exposure = MetricsCalculator._calculate_exposure(trades, equity_curve)

        return BacktestMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            total_trades=trade_stats["total"],
            winning_trades=trade_stats["winning"],
            losing_trades=trade_stats["losing"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            largest_win=trade_stats["largest_win"],
            largest_loss=trade_stats["largest_loss"],
            avg_trade_duration_minutes=trade_stats["avg_duration"],
            exposure_time=exposure,
            initial_balance=initial_balance,
            final_balance=final_balance,
        )

    @staticmethod
    def _empty_metrics(initial_balance: float) -> BacktestMetrics:
        """Return empty metrics when no data available."""
        return BacktestMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_duration_minutes=0.0,
            exposure_time=0.0,
            initial_balance=initial_balance,
            final_balance=initial_balance,
        )

    @staticmethod
    def _calculate_returns(equity_curve: list[tuple[datetime, float]]) -> list[float]:
        """Calculate period-over-period returns."""
        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i - 1][1]
            curr_equity = equity_curve[i][1]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        return returns

    @staticmethod
    def _calculate_sharpe(returns: list[float], periods_per_year: float) -> float:
        """Calculate annualized Sharpe ratio (risk-free rate = 0)."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_dev * math.sqrt(periods_per_year)

        return annualized_return / annualized_std

    @staticmethod
    def _calculate_sortino(returns: list[float], periods_per_year: float) -> float:
        """Calculate annualized Sortino ratio (downside deviation)."""
        if not returns or len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float("inf") if mean_return > 0 else 0.0

        downside_variance = sum(r**2 for r in negative_returns) / len(returns)
        downside_std = math.sqrt(downside_variance)

        if downside_std == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_downside = downside_std * math.sqrt(periods_per_year)

        return annualized_return / annualized_downside

    @staticmethod
    def _calculate_max_drawdown(
        equity_curve: list[tuple[datetime, float]],
    ) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if not equity_curve:
            return 0.0, 0

        peak = equity_curve[0][1]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0

        for i, (_, equity) in enumerate(equity_curve):
            if equity > peak:
                peak = equity
                current_dd_start = i
            else:
                dd = ((peak - equity) / peak) * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = i - current_dd_start

        return max_dd, max_dd_duration

    @staticmethod
    def _calculate_trade_stats(trades: list[BacktestTrade]) -> dict:
        """Calculate trade-level statistics."""
        if not trades:
            return {
                "total": 0,
                "winning": 0,
                "losing": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_duration": 0.0,
            }

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate average trade duration
        durations = []
        for t in trades:
            duration = (t.exit_time - t.entry_time).total_seconds() / 60
            durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total": len(trades),
            "winning": len(winning),
            "losing": len(losing),
            "win_rate": (len(winning) / len(trades)) * 100 if trades else 0.0,
            "profit_factor": profit_factor,
            "avg_win": (sum(t.pnl for t in winning) / len(winning)) if winning else 0.0,
            "avg_loss": (sum(t.pnl for t in losing) / len(losing)) if losing else 0.0,
            "largest_win": max((t.pnl for t in winning), default=0.0),
            "largest_loss": min((t.pnl for t in losing), default=0.0),
            "avg_duration": avg_duration,
        }

    @staticmethod
    def _calculate_exposure(
        trades: list[BacktestTrade],
        equity_curve: list[tuple[datetime, float]],
    ) -> float:
        """Calculate percentage of time in market."""
        if not trades or not equity_curve:
            return 0.0

        total_duration = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds()
        if total_duration <= 0:
            return 0.0

        # Sum up time in positions
        time_in_market = 0.0
        for trade in trades:
            trade_duration = (trade.exit_time - trade.entry_time).total_seconds()
            time_in_market += trade_duration

        return (time_in_market / total_duration) * 100
