"""
AlphaStrike Trading Bot - Database Layer

Async SQLite database for persistent storage of candles, trades, and model metadata.
Uses aiosqlite for non-blocking database operations.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class TradeSide(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    """Trade lifecycle status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Candle:
    """OHLCV candle data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "1m"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "interval": self.interval,
        }

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Candle:
        """Create from database row."""
        return cls(
            symbol=row["symbol"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            interval=row["interval"],
        )


@dataclass
class Trade:
    """Trade record."""
    id: str
    symbol: str
    side: TradeSide
    entry_price: float
    quantity: float
    leverage: int
    status: TradeStatus = TradeStatus.PENDING
    exit_price: float | None = None
    realized_pnl: float | None = None
    fees: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: datetime | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    ai_explanation: str | None = None
    order_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "status": self.status.value,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "fees": self.fees,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "ai_explanation": self.ai_explanation,
            "order_id": self.order_id,
        }

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Trade:
        """Create from database row."""
        return cls(
            id=row["id"],
            symbol=row["symbol"],
            side=TradeSide(row["side"]),
            entry_price=row["entry_price"],
            quantity=row["quantity"],
            leverage=row["leverage"],
            status=TradeStatus(row["status"]),
            exit_price=row["exit_price"],
            realized_pnl=row["realized_pnl"],
            fees=row["fees"],
            entry_time=datetime.fromisoformat(row["entry_time"]),
            exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
            stop_loss_price=row["stop_loss_price"],
            take_profit_price=row["take_profit_price"],
            ai_explanation=row["ai_explanation"],
            order_id=row["order_id"],
        )


@dataclass(frozen=True)
class AILogEntry:
    """AI explanation log entry."""
    id: str
    order_id: str
    symbol: str
    signal: str
    confidence: float
    weighted_average: float
    model_outputs: str  # JSON string
    regime: str
    risk_checks: str  # JSON string
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uploaded: bool = False


@dataclass(frozen=True)
class PerformanceMetric:
    """Rolling performance metric."""
    timestamp: datetime
    metric_name: str
    metric_value: float
    window_size: int
    symbol: str | None = None


# SQL Schema definitions
SCHEMA_SQL = """
-- Candles table for OHLCV data
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    interval TEXT NOT NULL DEFAULT '1m',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, interval)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_timestamp
ON candles(symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_candles_timestamp
ON candles(timestamp DESC);

-- Trades table for trade records
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    leverage INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    exit_price REAL,
    realized_pnl REAL,
    fees REAL DEFAULT 0,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    stop_loss_price REAL,
    take_profit_price REAL,
    ai_explanation TEXT,
    order_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);

-- AI log uploads for competition compliance
CREATE TABLE IF NOT EXISTS ai_log_uploads (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal TEXT NOT NULL,
    confidence REAL NOT NULL,
    weighted_average REAL NOT NULL,
    model_outputs TEXT NOT NULL,
    regime TEXT NOT NULL,
    risk_checks TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    uploaded INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ai_logs_order_id ON ai_log_uploads(order_id);
CREATE INDEX IF NOT EXISTS idx_ai_logs_uploaded ON ai_log_uploads(uploaded);

-- Training cache for model training data
CREATE TABLE IF NOT EXISTS training_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    features TEXT NOT NULL,
    label TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_symbol_timestamp
ON training_cache(symbol, timestamp DESC);

-- Performance metrics for monitoring
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    window_size INTEGER NOT NULL,
    symbol TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
ON performance_metrics(metric_name, timestamp DESC);

-- Triggers for updated_at
CREATE TRIGGER IF NOT EXISTS trades_updated_at
AFTER UPDATE ON trades
BEGIN
    UPDATE trades SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""


class Database:
    """
    Async database manager for AlphaStrike.

    Provides non-blocking database operations for candles, trades,
    AI logs, and performance metrics.

    Usage:
        db = Database()
        await db.initialize()
        await db.save_candle(candle)
        candles = await db.get_candles("cmt_btcusdt", limit=100)
        await db.close()
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file. Uses config default if None.
        """
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize database connection and create schema.

        Creates the database file and tables if they don't exist.
        """
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent access
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA cache_size=10000")

        # Create schema
        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()

        logger.info(f"Database initialized at {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get database connection with lock for thread safety."""
        if not self._connection:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        async with self._lock:
            yield self._connection

    # ==================== Candle Operations ====================

    async def save_candle(self, candle: Candle) -> None:
        """
        Save a single candle to database.

        Uses INSERT OR REPLACE to handle duplicates.
        """
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO candles
                (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candle.symbol,
                    candle.timestamp.isoformat(),
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.interval,
                ),
            )
            await conn.commit()

    async def save_candles(self, candles: list[Candle]) -> int:
        """
        Save multiple candles in a batch.

        Args:
            candles: List of candles to save

        Returns:
            Number of candles saved
        """
        if not candles:
            return 0

        async with self._get_connection() as conn:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.symbol,
                        c.timestamp.isoformat(),
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                        c.interval,
                    )
                    for c in candles
                ],
            )
            await conn.commit()
            return len(candles)

    async def get_candles(
        self,
        symbol: str,
        limit: int = 100,
        interval: str = "1m",
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Candle]:
        """
        Get candles for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Maximum number of candles to return
            interval: Candle interval (default "1m")
            since: Start timestamp (inclusive)
            until: End timestamp (inclusive)

        Returns:
            List of candles ordered by timestamp descending
        """
        async with self._get_connection() as conn:
            query = """
                SELECT * FROM candles
                WHERE symbol = ? AND interval = ?
            """
            params: list = [symbol, interval]

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            if until:
                query += " AND timestamp <= ?"
                params.append(until.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [Candle.from_row(row) for row in rows]

    async def get_latest_candle(self, symbol: str, interval: str = "1m") -> Candle | None:
        """Get the most recent candle for a symbol."""
        candles = await self.get_candles(symbol, limit=1, interval=interval)
        return candles[0] if candles else None

    async def get_candle_count(self, symbol: str, interval: str = "1m") -> int:
        """Get total candle count for a symbol."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ==================== Trade Operations ====================

    async def save_trade(self, trade: Trade) -> None:
        """Save or update a trade record."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO trades
                (id, symbol, side, entry_price, quantity, leverage, status,
                 exit_price, realized_pnl, fees, entry_time, exit_time,
                 stop_loss_price, take_profit_price, ai_explanation, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    trade.symbol,
                    trade.side.value,
                    trade.entry_price,
                    trade.quantity,
                    trade.leverage,
                    trade.status.value,
                    trade.exit_price,
                    trade.realized_pnl,
                    trade.fees,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.stop_loss_price,
                    trade.take_profit_price,
                    trade.ai_explanation,
                    trade.order_id,
                ),
            )
            await conn.commit()

    async def get_trade(self, trade_id: str) -> Trade | None:
        """Get a trade by ID."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            )
            row = await cursor.fetchone()
            return Trade.from_row(row) if row else None

    async def get_trades(
        self,
        symbol: str | None = None,
        status: TradeStatus | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Trade]:
        """
        Get trades with optional filters.

        Args:
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum results
            since: Filter by entry time

        Returns:
            List of trades ordered by entry_time descending
        """
        async with self._get_connection() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params: list = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            if since:
                query += " AND entry_time >= ?"
                params.append(since.isoformat())

            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

            return [Trade.from_row(row) for row in rows]

    async def get_open_trades(self, symbol: str | None = None) -> list[Trade]:
        """Get all open trades, optionally filtered by symbol."""
        return await self.get_trades(
            symbol=symbol,
            status=TradeStatus.OPEN,
            limit=100,
        )

    async def update_trade_status(
        self,
        trade_id: str,
        status: TradeStatus,
        exit_price: float | None = None,
        realized_pnl: float | None = None,
        exit_time: datetime | None = None,
    ) -> None:
        """Update trade status and exit details."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                UPDATE trades
                SET status = ?, exit_price = ?, realized_pnl = ?, exit_time = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    exit_price,
                    realized_pnl,
                    exit_time.isoformat() if exit_time else None,
                    trade_id,
                ),
            )
            await conn.commit()

    async def get_trade_count_today(self, symbol: str | None = None) -> int:
        """Get number of trades opened today."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        trades = await self.get_trades(symbol=symbol, since=today_start, limit=1000)
        return len(trades)

    # ==================== AI Log Operations ====================

    async def save_ai_log(self, log: AILogEntry) -> None:
        """Save AI explanation log."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO ai_log_uploads
                (id, order_id, symbol, signal, confidence, weighted_average,
                 model_outputs, regime, risk_checks, reasoning, timestamp, uploaded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.id,
                    log.order_id,
                    log.symbol,
                    log.signal,
                    log.confidence,
                    log.weighted_average,
                    log.model_outputs,
                    log.regime,
                    log.risk_checks,
                    log.reasoning,
                    log.timestamp.isoformat(),
                    1 if log.uploaded else 0,
                ),
            )
            await conn.commit()

    async def get_pending_ai_logs(self, limit: int = 100) -> list[dict]:
        """Get AI logs not yet uploaded."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM ai_log_uploads
                WHERE uploaded = 0
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def mark_ai_log_uploaded(self, log_id: str) -> None:
        """Mark AI log as uploaded."""
        async with self._get_connection() as conn:
            await conn.execute(
                "UPDATE ai_log_uploads SET uploaded = 1 WHERE id = ?",
                (log_id,),
            )
            await conn.commit()

    # ==================== Performance Metrics ====================

    async def save_metric(self, metric: PerformanceMetric) -> None:
        """Save a performance metric."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO performance_metrics
                (timestamp, metric_name, metric_value, window_size, symbol)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.metric_value,
                    metric.window_size,
                    metric.symbol,
                ),
            )
            await conn.commit()

    async def get_metrics(
        self,
        metric_name: str,
        limit: int = 100,
        symbol: str | None = None,
    ) -> list[dict]:
        """Get performance metrics by name."""
        async with self._get_connection() as conn:
            if symbol:
                cursor = await conn.execute(
                    """
                    SELECT * FROM performance_metrics
                    WHERE metric_name = ? AND symbol = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (metric_name, symbol, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM performance_metrics
                    WHERE metric_name = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (metric_name, limit),
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ==================== Training Cache ====================

    async def save_training_sample(
        self,
        symbol: str,
        features: str,
        label: str,
        timestamp: datetime,
    ) -> None:
        """Save a training sample."""
        async with self._get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO training_cache
                (symbol, features, label, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, features, label, timestamp.isoformat()),
            )
            await conn.commit()

    async def get_training_samples(
        self,
        symbol: str,
        limit: int = 1000,
    ) -> list[dict]:
        """Get training samples for a symbol."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM training_cache
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def clear_old_training_samples(self, days: int = 30) -> int:
        """Clear training samples older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM training_cache WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
            await conn.commit()
            return cursor.rowcount

    # ==================== Utility Methods ====================

    async def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        async with self._get_connection() as conn:
            await conn.execute("VACUUM")
            logger.info("Database vacuumed")

    async def get_db_stats(self) -> dict:
        """Get database statistics."""
        async with self._get_connection() as conn:
            stats = {}

            for table in ["candles", "trades", "ai_log_uploads", "training_cache", "performance_metrics"]:
                cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                row = await cursor.fetchone()
                stats[f"{table}_count"] = row[0] if row else 0

            # Get database file size
            stats["file_size_mb"] = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

            return stats


# Module-level singleton
_db_instance: Database | None = None


async def get_database() -> Database:
    """
    Get the database singleton instance.

    Initializes on first call.
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        await _db_instance.initialize()
    return _db_instance


async def close_database() -> None:
    """Close the database singleton."""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None
