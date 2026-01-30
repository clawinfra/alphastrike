"""
Unit tests for the database module.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from src.data.database import (
    AILogEntry,
    Candle,
    Database,
    PerformanceMetric,
    Trade,
    TradeSide,
    TradeStatus,
)


@pytest.fixture
async def db(tmp_path: Path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    database = Database(db_path)
    await database.initialize()
    yield database
    await database.close()


class TestCandles:
    """Tests for candle operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_candle(self, db: Database):
        """Test saving and retrieving a single candle."""
        candle = Candle(
            symbol="cmt_btcusdt",
            timestamp=datetime(2026, 1, 30, 12, 0, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            interval="1m",
        )

        await db.save_candle(candle)
        candles = await db.get_candles("cmt_btcusdt", limit=1)

        assert len(candles) == 1
        assert candles[0].symbol == "cmt_btcusdt"
        assert candles[0].open == 50000.0
        assert candles[0].close == 50050.0

    @pytest.mark.asyncio
    async def test_save_candles_batch(self, db: Database):
        """Test saving multiple candles in a batch."""
        candles = [
            Candle(
                symbol="cmt_btcusdt",
                timestamp=datetime(2026, 1, 30, 12, i, 0),
                open=50000.0 + i,
                high=50100.0 + i,
                low=49900.0 + i,
                close=50050.0 + i,
                volume=100.0,
            )
            for i in range(10)
        ]

        count = await db.save_candles(candles)
        assert count == 10

        retrieved = await db.get_candles("cmt_btcusdt", limit=10)
        assert len(retrieved) == 10

    @pytest.mark.asyncio
    async def test_get_candles_with_time_filter(self, db: Database):
        """Test filtering candles by time range."""
        base_time = datetime(2026, 1, 30, 12, 0, 0)
        candles = [
            Candle(
                symbol="cmt_btcusdt",
                timestamp=base_time + timedelta(minutes=i),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=100.0,
            )
            for i in range(10)
        ]
        await db.save_candles(candles)

        # Get candles in specific range
        since = base_time + timedelta(minutes=3)
        until = base_time + timedelta(minutes=7)
        filtered = await db.get_candles(
            "cmt_btcusdt",
            since=since,
            until=until,
            limit=100,
        )

        assert len(filtered) == 5  # minutes 3, 4, 5, 6, 7

    @pytest.mark.asyncio
    async def test_get_latest_candle(self, db: Database):
        """Test getting the most recent candle."""
        candles = [
            Candle(
                symbol="cmt_btcusdt",
                timestamp=datetime(2026, 1, 30, 12, i, 0),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0 + i,  # Different close for each
                volume=100.0,
            )
            for i in range(5)
        ]
        await db.save_candles(candles)

        latest = await db.get_latest_candle("cmt_btcusdt")
        assert latest is not None
        assert latest.close == 50054.0  # Last candle

    @pytest.mark.asyncio
    async def test_get_candle_count(self, db: Database):
        """Test counting candles."""
        candles = [
            Candle(
                symbol="cmt_btcusdt",
                timestamp=datetime(2026, 1, 30, 12, i, 0),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=100.0,
            )
            for i in range(25)
        ]
        await db.save_candles(candles)

        count = await db.get_candle_count("cmt_btcusdt")
        assert count == 25

    @pytest.mark.asyncio
    async def test_candle_upsert(self, db: Database):
        """Test that duplicate candles are updated, not duplicated."""
        candle1 = Candle(
            symbol="cmt_btcusdt",
            timestamp=datetime(2026, 1, 30, 12, 0, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )
        await db.save_candle(candle1)

        # Save same timestamp with different close
        candle2 = Candle(
            symbol="cmt_btcusdt",
            timestamp=datetime(2026, 1, 30, 12, 0, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50100.0,  # Different close
            volume=100.0,
        )
        await db.save_candle(candle2)

        # Should still be only 1 candle
        count = await db.get_candle_count("cmt_btcusdt")
        assert count == 1

        # Should have updated close
        candles = await db.get_candles("cmt_btcusdt", limit=1)
        assert candles[0].close == 50100.0


class TestTrades:
    """Tests for trade operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, db: Database):
        """Test saving and retrieving a trade."""
        trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            status=TradeStatus.OPEN,
        )

        await db.save_trade(trade)
        retrieved = await db.get_trade(trade.id)

        assert retrieved is not None
        assert retrieved.symbol == "cmt_btcusdt"
        assert retrieved.side == TradeSide.LONG
        assert retrieved.entry_price == 50000.0
        assert retrieved.leverage == 5

    @pytest.mark.asyncio
    async def test_get_trades_filtered(self, db: Database):
        """Test filtering trades by symbol and status."""
        trades = [
            Trade(
                id=str(uuid4()),
                symbol="cmt_btcusdt",
                side=TradeSide.LONG,
                entry_price=50000.0,
                quantity=0.1,
                leverage=5,
                status=TradeStatus.OPEN,
            ),
            Trade(
                id=str(uuid4()),
                symbol="cmt_btcusdt",
                side=TradeSide.SHORT,
                entry_price=50000.0,
                quantity=0.1,
                leverage=5,
                status=TradeStatus.CLOSED,
            ),
            Trade(
                id=str(uuid4()),
                symbol="cmt_ethusdt",
                side=TradeSide.LONG,
                entry_price=3000.0,
                quantity=1.0,
                leverage=3,
                status=TradeStatus.OPEN,
            ),
        ]

        for trade in trades:
            await db.save_trade(trade)

        # Filter by symbol
        btc_trades = await db.get_trades(symbol="cmt_btcusdt")
        assert len(btc_trades) == 2

        # Filter by status
        open_trades = await db.get_trades(status=TradeStatus.OPEN)
        assert len(open_trades) == 2

        # Filter by both
        btc_open = await db.get_trades(symbol="cmt_btcusdt", status=TradeStatus.OPEN)
        assert len(btc_open) == 1

    @pytest.mark.asyncio
    async def test_get_open_trades(self, db: Database):
        """Test getting open trades."""
        open_trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            status=TradeStatus.OPEN,
        )
        closed_trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            status=TradeStatus.CLOSED,
        )

        await db.save_trade(open_trade)
        await db.save_trade(closed_trade)

        open_trades = await db.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].status == TradeStatus.OPEN

    @pytest.mark.asyncio
    async def test_update_trade_status(self, db: Database):
        """Test updating trade status."""
        trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            status=TradeStatus.OPEN,
        )
        await db.save_trade(trade)

        # Close the trade
        exit_time = datetime.utcnow()
        await db.update_trade_status(
            trade_id=trade.id,
            status=TradeStatus.CLOSED,
            exit_price=51000.0,
            realized_pnl=100.0,
            exit_time=exit_time,
        )

        updated = await db.get_trade(trade.id)
        assert updated is not None
        assert updated.status == TradeStatus.CLOSED
        assert updated.exit_price == 51000.0
        assert updated.realized_pnl == 100.0

    @pytest.mark.asyncio
    async def test_trade_count_today(self, db: Database):
        """Test counting trades from today."""
        # Create trades with different times
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)

        today_trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            entry_time=today,
        )
        yesterday_trade = Trade(
            id=str(uuid4()),
            symbol="cmt_btcusdt",
            side=TradeSide.SHORT,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
            entry_time=yesterday,
        )

        await db.save_trade(today_trade)
        await db.save_trade(yesterday_trade)

        # Should only count today's trade
        count = await db.get_trade_count_today()
        assert count == 1


class TestAILogs:
    """Tests for AI log operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_ai_log(self, db: Database):
        """Test saving and retrieving AI logs."""
        log = AILogEntry(
            id=str(uuid4()),
            order_id="order123",
            symbol="cmt_btcusdt",
            signal="LONG",
            confidence=0.85,
            weighted_average=0.82,
            model_outputs='{"xgboost": 0.85, "lightgbm": 0.80}',
            regime="trending_up",
            risk_checks='["exposure_ok", "drawdown_ok"]',
            reasoning="Strong bullish momentum detected",
            uploaded=False,
        )

        await db.save_ai_log(log)

        pending = await db.get_pending_ai_logs()
        assert len(pending) == 1
        assert pending[0]["signal"] == "LONG"
        assert pending[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_mark_ai_log_uploaded(self, db: Database):
        """Test marking AI log as uploaded."""
        log = AILogEntry(
            id=str(uuid4()),
            order_id="order123",
            symbol="cmt_btcusdt",
            signal="LONG",
            confidence=0.85,
            weighted_average=0.82,
            model_outputs="{}",
            regime="trending_up",
            risk_checks="[]",
            reasoning="Test",
            uploaded=False,
        )

        await db.save_ai_log(log)

        # Mark as uploaded
        await db.mark_ai_log_uploaded(log.id)

        # Should no longer be in pending
        pending = await db.get_pending_ai_logs()
        assert len(pending) == 0


class TestPerformanceMetrics:
    """Tests for performance metric operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_metrics(self, db: Database):
        """Test saving and retrieving metrics."""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_name="win_rate",
            metric_value=0.65,
            window_size=100,
        )

        await db.save_metric(metric)

        metrics = await db.get_metrics("win_rate")
        assert len(metrics) == 1
        assert metrics[0]["metric_value"] == 0.65

    @pytest.mark.asyncio
    async def test_get_metrics_by_symbol(self, db: Database):
        """Test filtering metrics by symbol."""
        metrics = [
            PerformanceMetric(
                timestamp=datetime.utcnow(),
                metric_name="pnl",
                metric_value=100.0,
                window_size=50,
                symbol="cmt_btcusdt",
            ),
            PerformanceMetric(
                timestamp=datetime.utcnow(),
                metric_name="pnl",
                metric_value=50.0,
                window_size=50,
                symbol="cmt_ethusdt",
            ),
        ]

        for m in metrics:
            await db.save_metric(m)

        btc_metrics = await db.get_metrics("pnl", symbol="cmt_btcusdt")
        assert len(btc_metrics) == 1
        assert btc_metrics[0]["metric_value"] == 100.0


class TestTrainingCache:
    """Tests for training cache operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_training_samples(self, db: Database):
        """Test saving and retrieving training samples."""
        for i in range(5):
            await db.save_training_sample(
                symbol="cmt_btcusdt",
                features='{"rsi": 50, "adx": 30}',
                label="LONG",
                timestamp=datetime.utcnow() - timedelta(hours=i),
            )

        samples = await db.get_training_samples("cmt_btcusdt", limit=10)
        assert len(samples) == 5

    @pytest.mark.asyncio
    async def test_clear_old_training_samples(self, db: Database):
        """Test clearing old training samples."""
        # Add old and new samples
        old_time = datetime.utcnow() - timedelta(days=60)
        new_time = datetime.utcnow()

        await db.save_training_sample(
            symbol="cmt_btcusdt",
            features="{}",
            label="LONG",
            timestamp=old_time,
        )
        await db.save_training_sample(
            symbol="cmt_btcusdt",
            features="{}",
            label="SHORT",
            timestamp=new_time,
        )

        # Clear samples older than 30 days
        deleted = await db.clear_old_training_samples(days=30)
        assert deleted == 1

        samples = await db.get_training_samples("cmt_btcusdt")
        assert len(samples) == 1


class TestDatabaseUtilities:
    """Tests for database utility methods."""

    @pytest.mark.asyncio
    async def test_get_db_stats(self, db: Database):
        """Test getting database statistics."""
        # Add some data
        await db.save_candle(
            Candle(
                symbol="cmt_btcusdt",
                timestamp=datetime.utcnow(),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=100.0,
            )
        )

        stats = await db.get_db_stats()

        assert "candles_count" in stats
        assert stats["candles_count"] == 1
        assert "trades_count" in stats
        assert "file_size_mb" in stats

    @pytest.mark.asyncio
    async def test_vacuum(self, db: Database):
        """Test database vacuum operation."""
        # Just verify it doesn't raise
        await db.vacuum()


class TestCandleDataclass:
    """Tests for Candle dataclass."""

    def test_to_dict(self):
        """Test converting candle to dictionary."""
        candle = Candle(
            symbol="cmt_btcusdt",
            timestamp=datetime(2026, 1, 30, 12, 0, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
        )

        d = candle.to_dict()
        assert d["symbol"] == "cmt_btcusdt"
        assert d["open"] == 50000.0
        assert "timestamp" in d


class TestTradeDataclass:
    """Tests for Trade dataclass."""

    def test_to_dict(self):
        """Test converting trade to dictionary."""
        trade = Trade(
            id="test123",
            symbol="cmt_btcusdt",
            side=TradeSide.LONG,
            entry_price=50000.0,
            quantity=0.1,
            leverage=5,
        )

        d = trade.to_dict()
        assert d["id"] == "test123"
        assert d["side"] == "LONG"
        assert d["entry_price"] == 50000.0
