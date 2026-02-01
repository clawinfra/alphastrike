"""
Backtest Report Generator

Generates human-readable reports from backtest results.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.engine import BacktestConfig, BacktestTrade
    from src.backtest.metrics import BacktestMetrics


class ReportGenerator:
    """Generate backtest reports in various formats."""

    def __init__(
        self,
        metrics: BacktestMetrics,
        trades: list[BacktestTrade],
        config: BacktestConfig,
        equity_curve: list[tuple[datetime, float]],
    ):
        """
        Initialize report generator.

        Args:
            metrics: Calculated backtest metrics
            trades: List of completed trades
            config: Backtest configuration
            equity_curve: List of (timestamp, equity) tuples
        """
        self.metrics = metrics
        self.trades = trades
        self.config = config
        self.equity_curve = equity_curve

    def generate_summary(self) -> str:
        """Generate text summary for CLI output."""
        m = self.metrics
        c = self.config

        duration_days = (c.end_date - c.start_date).days

        lines = [
            "=" * 70,
            "                    ALPHASTRIKE BACKTEST REPORT",
            "=" * 70,
            "",
            "CONFIGURATION",
            "-" * 40,
            f"Symbol:             {c.symbol}",
            f"Period:             {c.start_date.strftime('%Y-%m-%d')} to {c.end_date.strftime('%Y-%m-%d')} ({duration_days} days)",
            f"Interval:           {c.interval}",
            f"Initial Balance:    ${c.initial_balance:,.2f}",
            f"Leverage:           {c.leverage}x",
            f"Slippage:           {c.slippage_bps} bps",
            "",
            "PERFORMANCE",
            "-" * 40,
            f"Final Balance:      ${m.final_balance:,.2f}",
            f"Total Return:       {m.total_return:+.2f}%",
            f"CAGR:               {m.cagr:+.2f}%",
            f"Sharpe Ratio:       {m.sharpe_ratio:.2f}",
            f"Sortino Ratio:      {m.sortino_ratio:.2f}",
            f"Max Drawdown:       {m.max_drawdown:.2f}%",
            f"Calmar Ratio:       {m.calmar_ratio:.2f}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:       {m.total_trades}",
            f"Winning Trades:     {m.winning_trades}",
            f"Losing Trades:      {m.losing_trades}",
            f"Win Rate:           {m.win_rate:.1f}%",
            f"Profit Factor:      {m.profit_factor:.2f}",
            f"Avg Win:            ${m.avg_win:,.2f}",
            f"Avg Loss:           ${m.avg_loss:,.2f}",
            f"Largest Win:        ${m.largest_win:,.2f}",
            f"Largest Loss:       ${m.largest_loss:,.2f}",
            f"Avg Duration:       {m.avg_trade_duration_minutes:.1f} min",
            "",
            "EXPOSURE",
            "-" * 40,
            f"Time in Market:     {m.exposure_time:.1f}%",
            "",
            "=" * 70,
        ]

        return "\n".join(lines)

    def generate_trade_log(self) -> str:
        """Generate trade-by-trade CSV breakdown."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "trade_id",
                "symbol",
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "pnl_pct",
                "fees",
                "confidence",
                "regime",
                "duration_min",
            ]
        )

        # Data rows
        for trade in self.trades:
            duration_min = (trade.exit_time - trade.entry_time).total_seconds() / 60
            pnl_pct = (trade.pnl / (trade.entry_price * trade.size)) * 100 if trade.size > 0 else 0

            writer.writerow(
                [
                    trade.id,
                    trade.symbol,
                    trade.side,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    f"{trade.entry_price:.4f}",
                    f"{trade.exit_price:.4f}",
                    f"{trade.size:.6f}",
                    f"{trade.pnl:.2f}",
                    f"{pnl_pct:.2f}",
                    f"{trade.fees:.4f}",
                    f"{trade.confidence:.2f}",
                    trade.regime,
                    f"{duration_min:.1f}",
                ]
            )

        return output.getvalue()

    def generate_equity_csv(self) -> str:
        """Generate equity curve CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["timestamp", "equity", "drawdown_pct"])

        peak = self.equity_curve[0][1] if self.equity_curve else 0

        for timestamp, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd_pct = ((peak - equity) / peak) * 100 if peak > 0 else 0

            writer.writerow(
                [
                    timestamp.isoformat(),
                    f"{equity:.2f}",
                    f"{dd_pct:.2f}",
                ]
            )

        return output.getvalue()

    def save_report(self, output_dir: Path) -> None:
        """
        Save full report package to directory.

        Creates:
        - summary.txt: Human-readable summary
        - trades.csv: Trade-by-trade breakdown
        - equity.csv: Equity curve data

        Args:
            output_dir: Directory to save reports
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.config.symbol}_{timestamp}"

        # Save summary
        summary_path = output_dir / f"{prefix}_summary.txt"
        summary_path.write_text(self.generate_summary())

        # Save trades
        trades_path = output_dir / f"{prefix}_trades.csv"
        trades_path.write_text(self.generate_trade_log())

        # Save equity curve
        equity_path = output_dir / f"{prefix}_equity.csv"
        equity_path.write_text(self.generate_equity_csv())

        # Also save latest versions without timestamp
        (output_dir / "latest_summary.txt").write_text(self.generate_summary())
        (output_dir / "latest_trades.csv").write_text(self.generate_trade_log())
        (output_dir / "latest_equity.csv").write_text(self.generate_equity_csv())
