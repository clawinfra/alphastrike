"""
Trade Logger for Live Trading

Persists all trades to JSONL files or SQLite for analysis and audit trail.
Generates daily reports and performance summaries.

Supports two backends:
- jsonl: Simple append-only JSONL files (default)
- sqlite: Full SQLite database with query support
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.trading.position_tracker import CompletedTrade, LivePosition

logger = logging.getLogger(__name__)


@runtime_checkable
class TradeLoggerBackend(Protocol):
    """Protocol for trade logger backends."""

    async def log_entry(self, position: LivePosition) -> None: ...
    async def log_exit(self, trade: CompletedTrade) -> None: ...
    async def log_signal(
        self,
        symbol: str,
        direction: str,
        conviction: float,
        action: str,
        reason: str | None = None,
        regime: str | None = None,
        tier: str | None = None,
    ) -> None: ...
    async def log_error(self, error_type: str, message: str, context: dict | None = None) -> None: ...
    def generate_daily_report(self) -> str: ...
    def get_performance_metrics(self) -> dict: ...


def create_trade_logger(
    backend: Literal["jsonl", "sqlite"] = "jsonl",
    log_dir: Path | str = "paper_trades",
) -> TradeLoggerBackend:
    """
    Factory function to create a trade logger with the specified backend.

    Args:
        backend: Either "jsonl" (default) or "sqlite"
        log_dir: Directory for log files / database

    Returns:
        TradeLoggerBackend implementation
    """
    if backend == "sqlite":
        from src.trading.trade_db import TradeDatabase
        db_path = Path(log_dir) / "trades.db"
        return TradeDatabase(db_path)
    else:
        return TradeLogger(log_dir)


class TradeLogger:
    """
    Logs trades and signals to JSONL files for persistence and analysis.

    File structure:
        paper_trades/
            2026-02-03_trades.jsonl      # All trade events
            2026-02-03_signals.jsonl     # Signal decisions
            daily_summary.json           # Running summary
    """

    def __init__(self, log_dir: Path | str = "paper_trades"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize files for today
        self.today = datetime.now(UTC).strftime("%Y-%m-%d")
        self.trades_file = self.log_dir / f"{self.today}_trades.jsonl"
        self.signals_file = self.log_dir / f"{self.today}_signals.jsonl"
        self.summary_file = self.log_dir / "daily_summary.json"

        logger.info(f"Trade logger initialized: {self.log_dir}")

    def _ensure_current_date(self) -> None:
        """Update file paths if date has changed."""
        current_date = datetime.now(UTC).strftime("%Y-%m-%d")
        if current_date != self.today:
            self.today = current_date
            self.trades_file = self.log_dir / f"{self.today}_trades.jsonl"
            self.signals_file = self.log_dir / f"{self.today}_signals.jsonl"

    async def log_entry(self, position: LivePosition) -> None:
        """Log a new position entry."""
        self._ensure_current_date()

        record = {
            "type": "entry",
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "size": position.size,
            "leverage": position.leverage,
            "strategy": position.strategy,
            "conviction": round(position.conviction, 1),
            "order_id": position.order_id,
        }

        self._append_jsonl(self.trades_file, record)
        logger.debug(f"Logged entry: {position.symbol} {position.direction}")

    async def log_exit(self, trade: CompletedTrade) -> None:
        """Log a position exit."""
        self._ensure_current_date()

        record = {
            "type": "exit",
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "size": trade.size,
            "leverage": trade.leverage,
            "pnl": round(trade.pnl, 2),
            "pnl_pct": round(trade.pnl_pct * 100, 2),
            "exit_reason": trade.exit_reason,
            "strategy": trade.strategy,
            "conviction": round(trade.conviction, 1),
            "holding_hours": round(
                (trade.exit_time - trade.entry_time).total_seconds() / 3600, 1
            ),
            "order_id": trade.order_id,
        }

        self._append_jsonl(self.trades_file, record)
        logger.debug(f"Logged exit: {trade.symbol} P&L=${trade.pnl:+.2f}")

    async def log_signal(
        self,
        symbol: str,
        direction: str,
        conviction: float,
        action: str,  # "ENTER", "SKIP", "EXIT"
        reason: str | None = None,
        regime: str | None = None,
        tier: str | None = None,
    ) -> None:
        """Log a signal decision (whether taken or not)."""
        self._ensure_current_date()

        record = {
            "type": "signal",
            "timestamp": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "conviction": round(conviction, 1),
            "action": action,
            "reason": reason,
            "regime": regime,
            "tier": tier,
        }

        self._append_jsonl(self.signals_file, record)

    async def log_error(self, error_type: str, message: str, context: dict | None = None) -> None:
        """Log an error event."""
        self._ensure_current_date()

        record = {
            "type": "error",
            "timestamp": datetime.now(UTC).isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {},
        }

        self._append_jsonl(self.trades_file, record)
        logger.error(f"Logged error: {error_type} - {message}")

    def _append_jsonl(self, file_path: Path, record: dict) -> None:
        """Append a record to a JSONL file."""
        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to {file_path}: {e}")

    def generate_daily_report(self) -> str:
        """Generate a summary report for today's trading."""
        self._ensure_current_date()

        if not self.trades_file.exists():
            return "No trades recorded today."

        trades = []
        with open(self.trades_file) as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        entries = [t for t in trades if t.get("type") == "entry"]
        exits = [t for t in trades if t.get("type") == "exit"]

        total_pnl = sum(t.get("pnl", 0) for t in exits)
        winning = [t for t in exits if t.get("pnl", 0) > 0]
        losing = [t for t in exits if t.get("pnl", 0) <= 0]

        report = f"""
================================================================================
                    DAILY TRADING REPORT - {self.today}
================================================================================

SUMMARY
-------
  Entries:           {len(entries)}
  Exits:             {len(exits)}
  Total P&L:         ${total_pnl:+.2f}

PERFORMANCE
-----------
  Winning Trades:    {len(winning)}
  Losing Trades:     {len(losing)}
  Win Rate:          {len(winning)/len(exits)*100:.1f}% (of closed)

EXIT REASONS
------------"""

        # Count exit reasons
        reasons = {}
        for t in exits:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        for reason, count in sorted(reasons.items()):
            report += f"\n  {reason}: {count}"

        report += "\n\nRECENT TRADES\n-------------"
        for t in exits[-10:]:
            report += f"\n  {t.get('symbol'):12} {t.get('direction'):5} ${t.get('pnl', 0):+8.2f} ({t.get('exit_reason')})"

        report += "\n\n================================================================================"

        return report

    def get_all_trades(self) -> list[dict]:
        """Load all trades from today's file."""
        self._ensure_current_date()

        if not self.trades_file.exists():
            return []

        trades = []
        with open(self.trades_file) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("type") in ("entry", "exit"):
                        trades.append(record)
                except json.JSONDecodeError:
                    continue

        return trades

    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics from logged trades."""
        trades = self.get_all_trades()
        exits = [t for t in trades if t.get("type") == "exit"]

        if not exits:
            return {
                "total_pnl": 0,
                "trade_count": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
            }

        pnls = [t.get("pnl", 0) for t in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return {
            "total_pnl": sum(pnls),
            "trade_count": len(exits),
            "win_rate": len(wins) / len(exits) if exits else 0,
            "avg_win": total_wins / len(wins) if wins else 0,
            "avg_loss": total_losses / len(losses) if losses else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        }
