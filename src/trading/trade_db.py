"""
SQLite Trade Database

Persistent storage for trades, signals, and performance metrics.
Provides better querying capabilities than JSONL for analysis.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trading.position_tracker import CompletedTrade, LivePosition

logger = logging.getLogger(__name__)


class TradeDatabase:
    """
    SQLite-based trade storage with full query support.

    Tables:
        - trades: Entry and exit records
        - signals: All signal decisions (taken or skipped)
        - errors: Error events for debugging
    """

    def __init__(self, db_path: Path | str = "paper_trades/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"Trade database initialized: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,  -- 'entry' or 'exit'
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    leverage INTEGER,
                    pnl REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    strategy TEXT,
                    conviction REAL,
                    holding_hours REAL,
                    order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    conviction REAL,
                    action TEXT NOT NULL,  -- 'ENTER', 'SKIP', 'EXIT'
                    reason TEXT,
                    regime TEXT,
                    tier TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_type ON trades(type);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def log_entry(self, position: LivePosition) -> None:
        """Log a new position entry."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (timestamp, type, symbol, direction, entry_price,
                                   size, leverage, strategy, conviction, order_id)
                VALUES (?, 'entry', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    position.symbol,
                    position.direction,
                    position.entry_price,
                    position.size,
                    position.leverage,
                    position.strategy,
                    round(position.conviction, 1),
                    position.order_id,
                ),
            )
        logger.debug(f"DB logged entry: {position.symbol} {position.direction}")

    async def log_exit(self, trade: CompletedTrade) -> None:
        """Log a position exit."""
        holding_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (timestamp, type, symbol, direction, entry_price,
                                   exit_price, size, leverage, pnl, pnl_pct,
                                   exit_reason, strategy, conviction, holding_hours, order_id)
                VALUES (?, 'exit', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    trade.symbol,
                    trade.direction,
                    trade.entry_price,
                    trade.exit_price,
                    trade.size,
                    trade.leverage,
                    round(trade.pnl, 2),
                    round(trade.pnl_pct * 100, 2),
                    trade.exit_reason,
                    trade.strategy,
                    round(trade.conviction, 1),
                    round(holding_hours, 1),
                    trade.order_id,
                ),
            )
        logger.debug(f"DB logged exit: {trade.symbol} P&L=${trade.pnl:+.2f}")

    async def log_signal(
        self,
        symbol: str,
        direction: str,
        conviction: float,
        action: str,
        reason: str | None = None,
        regime: str | None = None,
        tier: str | None = None,
    ) -> None:
        """Log a signal decision."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (timestamp, symbol, direction, conviction,
                                    action, reason, regime, tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    symbol,
                    direction,
                    round(conviction, 1),
                    action,
                    reason,
                    regime,
                    tier,
                ),
            )

    async def log_error(
        self, error_type: str, message: str, context: dict | None = None
    ) -> None:
        """Log an error event."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO errors (timestamp, error_type, message, context)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    error_type,
                    message,
                    json.dumps(context) if context else None,
                ),
            )
        logger.error(f"DB logged error: {error_type} - {message}")

    def get_trades_today(self) -> list[dict]:
        """Get all trades from today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM trades
                WHERE date(timestamp) = ?
                ORDER BY timestamp DESC
                """,
                (today,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_exits_today(self) -> list[dict]:
        """Get completed trades (exits) from today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM trades
                WHERE date(timestamp) = ? AND type = 'exit'
                ORDER BY timestamp DESC
                """,
                (today,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_metrics(self, days: int = 1) -> dict:
        """Calculate performance metrics for the last N days."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl <= 0 THEN ABS(pnl) END) as avg_loss,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_wins,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as total_losses
                FROM trades
                WHERE type = 'exit'
                  AND timestamp >= datetime('now', ?)
                """,
                (f"-{days} days",),
            )
            row = cursor.fetchone()

            if not row or row["trade_count"] == 0:
                return {
                    "total_pnl": 0,
                    "trade_count": 0,
                    "win_rate": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0,
                }

            total_wins = row["total_wins"] or 0
            total_losses = row["total_losses"] or 0

            return {
                "total_pnl": row["total_pnl"] or 0,
                "trade_count": row["trade_count"],
                "win_rate": (row["wins"] or 0) / row["trade_count"],
                "avg_win": row["avg_win"] or 0,
                "avg_loss": row["avg_loss"] or 0,
                "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
            }

    def get_signals_summary(self, hours: int = 24) -> dict:
        """Get signal summary for the last N hours."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    action,
                    COUNT(*) as count
                FROM signals
                WHERE timestamp >= datetime('now', ?)
                GROUP BY action
                """,
                (f"-{hours} hours",),
            )
            return {row["action"]: row["count"] for row in cursor.fetchall()}

    def generate_daily_report(self) -> str:
        """Generate a summary report for today's trading."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        exits = self.get_exits_today()

        if not exits:
            return "No trades recorded today."

        total_pnl = sum(t["pnl"] or 0 for t in exits)
        wins = [t for t in exits if (t["pnl"] or 0) > 0]
        losses = [t for t in exits if (t["pnl"] or 0) <= 0]

        # Count exit reasons
        reasons: dict[str, int] = {}
        for t in exits:
            r = t.get("exit_reason") or "unknown"
            reasons[r] = reasons.get(r, 0) + 1

        report = f"""
================================================================================
                    DAILY TRADING REPORT - {today}
================================================================================

SUMMARY
-------
  Exits:             {len(exits)}
  Total P&L:         ${total_pnl:+.2f}

PERFORMANCE
-----------
  Winning Trades:    {len(wins)}
  Losing Trades:     {len(losses)}
  Win Rate:          {len(wins)/len(exits)*100:.1f}%

EXIT REASONS
------------"""

        for reason, count in sorted(reasons.items()):
            report += f"\n  {reason}: {count}"

        report += "\n\nRECENT TRADES\n-------------"
        for t in exits[:10]:
            pnl = t.get("pnl") or 0
            report += f"\n  {t.get('symbol'):12} {t.get('direction'):5} ${pnl:+8.2f} ({t.get('exit_reason')})"

        report += "\n\n================================================================================"

        return report

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a raw SQL query (for advanced analysis)."""
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
