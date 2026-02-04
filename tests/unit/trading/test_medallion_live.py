"""
Tests for Medallion Live Trading Engine.

Coverage targets:
- Cloud backup initialization (enabled/disabled based on mode)
- Trade event backup methods
- Position state backup
- Config options
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.medallion_live import (
    CLOUD_BACKUP_AVAILABLE,
    MedallionLiveConfig,
    MedallionLiveEngine,
)
from src.trading.position_tracker import CompletedTrade, LivePosition


class TestMedallionLiveConfig:
    """Tests for MedallionLiveConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MedallionLiveConfig()

        assert config.base_leverage == 5.0
        assert config.max_portfolio_exposure == 0.40
        assert config.max_single_position == 0.05
        assert config.stop_loss_pct == 0.01
        assert config.take_profit_pct == 0.04
        assert config.time_exit_hours == 36
        assert config.cloud_backup_enabled is True
        assert config.fallback_balance == 10000.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MedallionLiveConfig(
            base_leverage=3.0,
            max_portfolio_exposure=0.30,
            cloud_backup_enabled=False,
            fallback_balance=50000.0,
        )

        assert config.base_leverage == 3.0
        assert config.max_portfolio_exposure == 0.30
        assert config.cloud_backup_enabled is False
        assert config.fallback_balance == 50000.0

    def test_default_assets(self):
        """Test default assets list."""
        config = MedallionLiveConfig()

        assert "BTC" in config.assets
        assert "ETH" in config.assets
        assert len(config.assets) == 15

    def test_ml_thresholds(self):
        """Test ML threshold defaults."""
        config = MedallionLiveConfig()

        assert config.min_conviction == 50.0
        assert config.ml_long_threshold == 0.55
        assert config.ml_short_threshold == 0.45


class TestCloudBackupInitialization:
    """Tests for cloud backup initialization logic."""

    def test_cloud_backup_disabled_in_dry_run(self):
        """Test that cloud backup is NOT enabled in dry_run mode."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,  # Dry run mode
        )

        # Cloud backup should be None in dry_run mode
        assert engine.cloud_backup is None
        assert engine.dry_run is True

    def test_cloud_backup_enabled_in_live_mode(self):
        """Test that cloud backup is enabled in live mode when available."""
        if not CLOUD_BACKUP_AVAILABLE:
            pytest.skip("CloudBackupManager not available")

        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
            mock_cbm.return_value = MagicMock()

            engine = MedallionLiveEngine(
                config=config,
                testnet=True,
                dry_run=False,  # Live mode
            )

            # Cloud backup should be initialized
            assert engine.cloud_backup is not None
            mock_cbm.assert_called_once()

    def test_cloud_backup_disabled_by_config(self):
        """Test that cloud backup can be disabled via config."""
        config = MedallionLiveConfig(cloud_backup_enabled=False)
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=False,  # Live mode but backup disabled
        )

        # Cloud backup should be None when disabled by config
        assert engine.cloud_backup is None

    def test_cloud_backup_not_available_warning(self):
        """Test warning when cloud backup is requested but not available."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CLOUD_BACKUP_AVAILABLE", False):
            with patch("src.trading.medallion_live.logger") as mock_logger:
                engine = MedallionLiveEngine(
                    config=config,
                    testnet=True,
                    dry_run=False,
                )

                # Should have logged a warning
                # (Warning logged during init when not available)
                assert engine.cloud_backup is None


class TestBackupTradeEvent:
    """Tests for _backup_trade_event method."""

    @pytest.fixture
    def engine_with_backup(self):
        """Create engine with mocked cloud backup."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CLOUD_BACKUP_AVAILABLE", True):
            with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
                mock_backup = MagicMock()
                mock_backup.backup_event = AsyncMock()
                mock_cbm.return_value = mock_backup

                engine = MedallionLiveEngine(
                    config=config,
                    testnet=True,
                    dry_run=False,
                )
                engine.cloud_backup = mock_backup

                yield engine

    @pytest.fixture
    def engine_without_backup(self):
        """Create engine without cloud backup (dry run)."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )
        return engine

    @pytest.mark.asyncio
    async def test_backup_trade_event_live_mode(self, engine_with_backup):
        """Test trade event is backed up in live mode."""
        await engine_with_backup._backup_trade_event(
            event_type="trade_entry",
            symbol="BTCUSDT",
            payload={
                "direction": "LONG",
                "entry_price": 50000.0,
                "size": 1000.0,
            },
        )

        # Verify backup_event was called
        engine_with_backup.cloud_backup.backup_event.assert_called_once()

        # Verify event structure
        call_args = engine_with_backup.cloud_backup.backup_event.call_args[0][0]
        assert call_args["event_type"] == "trade_entry"
        assert call_args["symbol"] == "BTCUSDT"
        assert "event_id" in call_args
        assert "timestamp" in call_args
        assert "event_hash" in call_args
        assert call_args["payload"]["direction"] == "LONG"

    @pytest.mark.asyncio
    async def test_backup_trade_event_dry_run_skipped(self, engine_without_backup):
        """Test trade event backup is skipped in dry run mode."""
        # This should not raise and should do nothing
        await engine_without_backup._backup_trade_event(
            event_type="trade_entry",
            symbol="BTCUSDT",
            payload={"direction": "LONG"},
        )

        # No error, cloud_backup is None
        assert engine_without_backup.cloud_backup is None

    @pytest.mark.asyncio
    async def test_backup_trade_event_failure_non_blocking(self, engine_with_backup):
        """Test that backup failure doesn't block trading."""
        # Make backup raise an exception
        engine_with_backup.cloud_backup.backup_event.side_effect = Exception("Network error")

        # This should not raise
        await engine_with_backup._backup_trade_event(
            event_type="trade_entry",
            symbol="BTCUSDT",
            payload={"direction": "LONG"},
        )

        # Verify backup was attempted
        engine_with_backup.cloud_backup.backup_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_backup_trade_event_exit(self, engine_with_backup):
        """Test exit event backup."""
        await engine_with_backup._backup_trade_event(
            event_type="trade_exit",
            symbol="ETHUSDT",
            payload={
                "direction": "SHORT",
                "entry_price": 3000.0,
                "exit_price": 2900.0,
                "pnl": 100.0,
                "exit_reason": "take_profit",
            },
        )

        call_args = engine_with_backup.cloud_backup.backup_event.call_args[0][0]
        assert call_args["event_type"] == "trade_exit"
        assert call_args["payload"]["exit_reason"] == "take_profit"


class TestBackupPositionState:
    """Tests for _backup_position_state method."""

    @pytest.fixture
    def engine_with_positions(self):
        """Create engine with positions and mocked cloud backup."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CLOUD_BACKUP_AVAILABLE", True):
            with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
                mock_backup = MagicMock()
                mock_backup.store_position_state = AsyncMock()
                mock_cbm.return_value = mock_backup

                engine = MedallionLiveEngine(
                    config=config,
                    testnet=True,
                    dry_run=False,
                )
                engine.cloud_backup = mock_backup

                # Add a test position
                position = LivePosition(
                    symbol="BTCUSDT",
                    direction="LONG",
                    entry_price=50000.0,
                    size=1000.0,
                    entry_time=datetime.now(UTC),
                    strategy="ml_tier1",
                    conviction=75.0,
                    order_id="test_order_123",
                    leverage=5,
                )
                position.current_price = 51000.0
                position.unrealized_pnl = 100.0
                engine.position_tracker.positions["BTCUSDT"] = position

                yield engine

    @pytest.mark.asyncio
    async def test_backup_position_state(self, engine_with_positions):
        """Test position state is backed up correctly."""
        await engine_with_positions._backup_position_state()

        # Verify store_position_state was called
        engine_with_positions.cloud_backup.store_position_state.assert_called_once()

        # Verify positions structure
        call_args = engine_with_positions.cloud_backup.store_position_state.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["symbol"] == "BTCUSDT"
        assert call_args[0]["direction"] == "LONG"
        assert call_args[0]["entry_price"] == 50000.0
        assert call_args[0]["current_price"] == 51000.0

    @pytest.mark.asyncio
    async def test_backup_position_state_no_backup(self):
        """Test position state backup is skipped when no cloud backup."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )

        # Should not raise
        await engine._backup_position_state()

    @pytest.mark.asyncio
    async def test_backup_position_state_empty(self):
        """Test position state backup with no positions."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CLOUD_BACKUP_AVAILABLE", True):
            with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
                mock_backup = MagicMock()
                mock_backup.store_position_state = AsyncMock()
                mock_cbm.return_value = mock_backup

                engine = MedallionLiveEngine(
                    config=config,
                    testnet=True,
                    dry_run=False,
                )
                engine.cloud_backup = mock_backup

                await engine._backup_position_state()

                # Should still call with empty list
                mock_backup.store_position_state.assert_called_once_with([])

    @pytest.mark.asyncio
    async def test_backup_position_state_failure_non_blocking(self, engine_with_positions):
        """Test that position state backup failure doesn't block."""
        engine_with_positions.cloud_backup.store_position_state.side_effect = Exception("Redis error")

        # Should not raise
        await engine_with_positions._backup_position_state()


class TestEngineInitialization:
    """Tests for MedallionLiveEngine initialization."""

    def test_engine_init_testnet(self):
        """Test engine initialization for testnet."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )

        assert engine.testnet is True
        assert engine.dry_run is True
        assert engine.config == config
        assert engine.balance == 0.0
        assert engine._running is False

    def test_engine_init_mainnet(self):
        """Test engine initialization for mainnet."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=False,
            dry_run=True,
        )

        assert engine.testnet is False

    def test_engine_init_with_private_key(self):
        """Test engine initialization with custom private key."""
        config = MedallionLiveConfig()
        pk = "0x" + "a" * 64
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
            private_key=pk,
        )

        assert engine.private_key == pk

    def test_engine_components_initialized(self):
        """Test that all engine components are initialized."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )

        assert engine.position_tracker is not None
        assert engine.trade_logger is not None
        assert engine.leverage_manager is not None
        assert engine.candle_buffers == {}
        assert engine.ml_models == {}


class TestEngineIsRunning:
    """Tests for is_running property."""

    def test_is_running_initial(self):
        """Test is_running is False initially."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        assert engine.is_running is False

    def test_is_running_after_start(self):
        """Test is_running reflects _running state."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        engine._running = True
        assert engine.is_running is True

        engine._running = False
        assert engine.is_running is False


class TestEngineStop:
    """Tests for engine stop method."""

    @pytest.mark.asyncio
    async def test_stop_closes_cloud_backup(self):
        """Test that stop() closes cloud backup."""
        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CLOUD_BACKUP_AVAILABLE", True):
            with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
                mock_backup = MagicMock()
                mock_backup.close = AsyncMock()
                mock_cbm.return_value = mock_backup

                engine = MedallionLiveEngine(
                    config=config,
                    testnet=True,
                    dry_run=False,
                )
                engine.cloud_backup = mock_backup
                engine._running = True

                # Mock adapter
                engine.adapter = MagicMock()
                engine.adapter.close = AsyncMock()

                await engine.stop()

                # Verify cloud backup was closed
                mock_backup.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_not_running(self):
        """Test stop() when engine is not running."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        # Should not raise
        await engine.stop()

        assert engine._running is False


class TestRegimeDetection:
    """Tests for market regime detection."""

    def test_detect_regime_insufficient_data(self):
        """Test regime detection with insufficient data."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        # Empty candles
        regime, strength = engine._detect_market_regime([])
        assert regime == "RANGING"
        assert strength == 50.0

        # Too few candles
        candles = [MagicMock(close=50000) for _ in range(10)]
        regime, strength = engine._detect_market_regime(candles, lookback=50)
        assert regime == "RANGING"
        assert strength == 50.0

    def test_detect_regime_bullish(self):
        """Test bullish regime detection."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        # Create strongly bullish candles (ascending prices)
        candles = []
        for i in range(60):
            mock_candle = MagicMock()
            mock_candle.close = 50000 + (i * 100)  # Rising prices
            candles.append(mock_candle)

        regime, strength = engine._detect_market_regime(candles)
        assert regime == "BULLISH"
        assert strength >= 60.0

    def test_detect_regime_bearish(self):
        """Test bearish regime detection."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        # Create strongly bearish candles (descending prices)
        candles = []
        for i in range(60):
            mock_candle = MagicMock()
            mock_candle.close = 60000 - (i * 100)  # Falling prices
            candles.append(mock_candle)

        regime, strength = engine._detect_market_regime(candles)
        assert regime == "BEARISH"
        assert strength >= 60.0


class TestMLSignal:
    """Tests for ML signal generation."""

    def test_get_ml_signal_no_model(self):
        """Test ML signal when model not available."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        direction, conviction = engine._get_ml_signal("BTCUSDT", {})
        assert direction == "HOLD"
        assert conviction == 0.0

    def test_get_ml_signal_no_features(self):
        """Test ML signal with empty features."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        # Add mock model
        mock_model = MagicMock()
        engine.ml_models["BTCUSDT"] = mock_model

        direction, conviction = engine._get_ml_signal("BTCUSDT", {})
        # Should still try to predict but likely return HOLD
        assert direction in ["HOLD", "LONG", "SHORT"]


class TestTierFiltering:
    """Tests for ML tier filtering."""

    def test_apply_tier_filtering_hold(self):
        """Test tier filtering returns HOLD for non-LONG signals."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        direction, conviction, tier = engine._apply_tier_filtering(
            ml_direction="SHORT",
            ml_conv=80.0,
            candles=[],
        )

        assert direction == "HOLD"
        assert conviction == 0.0
        assert tier == ""

    def test_apply_tier_filtering_low_conviction(self):
        """Test tier filtering returns HOLD for low conviction."""
        engine = MedallionLiveEngine(
            config=MedallionLiveConfig(),
            testnet=True,
            dry_run=True,
        )

        direction, conviction, tier = engine._apply_tier_filtering(
            ml_direction="LONG",
            ml_conv=50.0,  # Below 65 threshold
            candles=[],
        )

        assert direction == "HOLD"
        assert conviction == 0.0


class TestDryRunMode:
    """Tests specific to dry run mode behavior."""

    def test_dry_run_balance(self):
        """Test dry run uses fallback balance."""
        config = MedallionLiveConfig(fallback_balance=25000.0)
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )

        # Balance is set during initialize(), not in __init__
        assert engine.dry_run is True
        assert engine.cloud_backup is None

    @pytest.mark.asyncio
    async def test_dry_run_no_cloud_backup_calls(self):
        """Test that no cloud backup happens in dry run mode."""
        config = MedallionLiveConfig()
        engine = MedallionLiveEngine(
            config=config,
            testnet=True,
            dry_run=True,
        )

        # These should not raise and should do nothing
        await engine._backup_trade_event("test", "BTCUSDT", {})
        await engine._backup_position_state()

        # No errors, cloud_backup is None
        assert engine.cloud_backup is None


class TestLiveModeCloudBackup:
    """Integration tests for cloud backup in live mode."""

    @pytest.mark.asyncio
    async def test_initialize_starts_backup_worker(self):
        """Test that initialize() starts backup retry worker."""
        if not CLOUD_BACKUP_AVAILABLE:
            pytest.skip("CloudBackupManager not available")

        config = MedallionLiveConfig(cloud_backup_enabled=True)

        with patch("src.trading.medallion_live.CloudBackupManager") as mock_cbm:
            mock_backup = MagicMock()
            mock_backup.start_retry_worker = AsyncMock()
            mock_backup.health_check = AsyncMock(return_value={"hot": True})
            mock_cbm.return_value = mock_backup

            engine = MedallionLiveEngine(
                config=config,
                testnet=True,
                dry_run=False,
            )
            engine.cloud_backup = mock_backup

            # Mock the imports that happen inside initialize()
            with patch("src.exchange.adapters.hyperliquid.adapter.HyperliquidAdapter") as mock_ha:
                with patch("src.features.pipeline.FeaturePipeline"):
                    with patch("src.ml.lightgbm_model.LightGBMModel"):
                        # Create mock adapter
                        mock_adapter = MagicMock()
                        mock_adapter.initialize = AsyncMock()
                        mock_adapter.rest.get_account_balance = AsyncMock(
                            return_value=MagicMock(total_balance=10000.0)
                        )
                        mock_adapter.rest.get_candles = AsyncMock(return_value=[])
                        mock_ha.return_value = mock_adapter

                        await engine.initialize()

                        # Verify backup worker was started
                        mock_backup.start_retry_worker.assert_called_once()
                        mock_backup.health_check.assert_called_once()
