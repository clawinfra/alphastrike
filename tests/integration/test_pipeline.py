"""
AlphaStrike Trading Bot - Integration Tests for Full Pipeline (US-029)

Comprehensive integration tests covering:
- Data flow: WebSocket -> Gateway -> Features -> ML -> Strategy
- Risk validation with various scenarios
- Order execution flow (mocked exchange)
- Position sync reconciliation
- Model health check and weight redistribution
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.data.data_gateway import (
    CircuitState,
    DataCircuitBreaker,
    DataGateway,
    GateResult,
    ValidationResult,
)
from src.data.database import Candle
from src.data.rest_client import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    PositionSide,
    RESTClient,
    TimeInForce,
)
from src.data.rest_client import Position as ExchangePosition
from src.execution.order_manager import (
    ExecutionResult,
    OrderManager,
    OrderTypeSelection,
    SignalResult,
)
from src.execution.position_sync import PositionSync, SyncResult
from src.features.pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
    OrderbookData,
    TickerData,
)
from src.ml.ensemble import MLEnsemble
from src.risk.portfolio import PortfolioManager, Position
from src.risk.risk_manager import RiskCheck, RiskManager
from src.strategy.exit_manager import ExitManager, ExitPrices


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_candles() -> list[Candle]:
    """Generate mock candle data for testing with valid OHLC relationships."""
    base_price = 50000.0
    candles: list[Candle] = []
    base_time = datetime.utcnow() - timedelta(minutes=150)

    for i in range(150):
        # Create realistic price movement
        noise = np.sin(i * 0.1) * 100 + np.random.uniform(-50, 50)
        price = base_price + noise

        # Generate OHLC with valid relationships: high >= max(open, close), low <= min(open, close)
        open_price = price
        close_price = price + np.random.uniform(-20, 20)
        high_price = max(open_price, close_price) + np.random.uniform(10, 50)
        low_price = min(open_price, close_price) - np.random.uniform(10, 50)

        candle = Candle(
            symbol="cmt_btcusdt",
            timestamp=base_time + timedelta(minutes=i),
            open=float(open_price),
            high=float(high_price),
            low=float(low_price),
            close=float(close_price),
            volume=float(np.random.uniform(100, 1000)),
            interval="1m",
        )
        candles.append(candle)

    return candles


@pytest.fixture
def mock_ticker() -> TickerData:
    """Create mock ticker data."""
    return TickerData(
        symbol="cmt_btcusdt",
        last_price=50000.0,
        bid_price=49995.0,
        ask_price=50005.0,
        volume_24h=10000.0,
        high_24h=51000.0,
        low_24h=49000.0,
        funding_rate=0.0001,
        timestamp=datetime.utcnow().timestamp(),
    )


@pytest.fixture
def mock_orderbook() -> OrderbookData:
    """Create mock orderbook data."""
    return OrderbookData(
        bids=[(49995.0, 1.0), (49990.0, 2.0), (49985.0, 3.0)],
        asks=[(50005.0, 1.0), (50010.0, 2.0), (50015.0, 3.0)],
        timestamp=datetime.utcnow().timestamp(),
    )


@pytest.fixture
def portfolio() -> PortfolioManager:
    """Create portfolio manager with initial balance."""
    return PortfolioManager(initial_balance=10000.0)


@pytest.fixture
def mock_rest_client() -> MagicMock:
    """Create mock REST client for exchange interactions."""
    client = MagicMock(spec=RESTClient)

    # Configure async methods
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.get_ticker = AsyncMock(
        return_value={"last": "50000.0", "bid": "49995.0", "ask": "50005.0"}
    )
    client.get_orderbook = AsyncMock(
        return_value={
            "bids": [["49995.0", "1.0"], ["49990.0", "2.0"]],
            "asks": [["50005.0", "1.0"], ["50010.0", "2.0"]],
        }
    )
    client.get_positions = AsyncMock(return_value=[])
    client.set_leverage = AsyncMock(return_value=True)
    client.place_order = AsyncMock(
        return_value=OrderResult(
            order_id="test_order_123",
            client_order_id="as_test_123",
            symbol="cmt_btcusdt",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.01,
            price=None,
            status="filled",
            filled_size=0.01,
            avg_fill_price=50000.0,
        )
    )
    client.get_order = AsyncMock(
        return_value={
            "state": "filled",
            "filledQty": "0.01",
            "priceAvg": "50000.0",
        }
    )

    return client


# =============================================================================
# Test: Data Flow (WebSocket -> Gateway -> Features -> ML -> Strategy)
# =============================================================================


class TestDataFlowIntegration:
    """Test the complete data flow pipeline."""

    @pytest.mark.asyncio
    async def test_data_flow_websocket_to_features(
        self, mock_candles: list[Candle], mock_ticker: TickerData, mock_orderbook: OrderbookData
    ) -> None:
        """
        Test data flow from WebSocket data through to feature computation.

        Validates:
        1. DataCircuitBreaker and validators work correctly
        2. FeaturePipeline computes features from candles
        3. Features are valid (no NaN, proper count)
        """
        # Step 1: Test DataCircuitBreaker and validation components directly
        # (avoiding full DataGateway initialization which requires complex settings)
        from src.data.data_gateway import (
            OHLCLogicValidator,
            StalenessChecker,
        )

        staleness_checker = StalenessChecker(threshold_seconds=60.0)
        ohlc_validator = OHLCLogicValidator()

        # Process a fresh candle
        fresh_candle = Candle(
            symbol="cmt_btcusdt",
            timestamp=datetime.utcnow(),
            open=50000.0,
            high=50050.0,
            low=49950.0,
            close=50025.0,
            volume=500.0,
            interval="1m",
        )

        # Test staleness check
        stale_result, age = staleness_checker.check(fresh_candle.timestamp)
        assert stale_result == GateResult.PASS, f"Fresh candle should pass staleness: {age}s old"

        # Test OHLC logic check
        ohlc_result, ohlc_error = ohlc_validator.check(fresh_candle)
        assert ohlc_result == GateResult.PASS, f"Valid candle should pass OHLC: {ohlc_error}"

        # Step 2: Compute features using FeaturePipeline
        pipeline = FeaturePipeline(config=FeaturePipelineConfig(min_candles=100))

        features = pipeline.calculate_features(
            candles=mock_candles,
            ticker_data=mock_ticker,
            orderbook_data=mock_orderbook,
            use_cache=False,
        )

        # Validate features
        assert len(features) > 0, "Should compute features"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Features should not contain NaN"
        assert all(
            not np.isinf(v) for v in features.values()
        ), "Features should not contain Inf"

        # Check key feature categories exist
        assert "rsi_14" in features, "Should have RSI_14 feature"
        assert "atr" in features, "Should have ATR feature"
        assert "session_indicator" in features, "Should have time feature"
        assert "macd_line" in features, "Should have MACD feature"

    @pytest.mark.asyncio
    async def test_gateway_circuit_breaker_integration(self) -> None:
        """Test circuit breaker opens after consecutive failures."""
        circuit_breaker = DataCircuitBreaker(
            failure_threshold=3,
            reset_timeout_seconds=1.0,
            half_open_successes=2,
        )

        # Initially closed
        assert circuit_breaker.state == CircuitState.CLOSED
        assert not circuit_breaker.is_open()

        # Record failures to trigger opening
        for _ in range(3):
            circuit_breaker.record_failure("test_gate")

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open()

        # Wait for reset timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Record successes to close
        circuit_breaker.record_success()
        circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitState.CLOSED


# =============================================================================
# Test: Risk Validation Scenarios
# =============================================================================


class TestRiskValidationScenarios:
    """Test various risk validation scenarios."""

    def test_exposure_limit_rejection(self, portfolio: PortfolioManager) -> None:
        """Test order rejected when exposure exceeds limit."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            # Create order that exceeds per-trade exposure
            large_order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=100.0,  # Very large size
                price=50000.0,  # $5M notional - exceeds 10% of $10k balance
            )

            result = risk_manager.validate_order(large_order, portfolio)

            assert not result.allowed
            assert "exposure" in (result.reason or "").lower()
            assert "exposure_limits" in result.checks_failed

    def test_drawdown_limit_rejection(self, portfolio: PortfolioManager) -> None:
        """Test order rejected when drawdown exceeds limit."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            # Simulate significant drawdown
            portfolio._balance = 8000.0  # 20% loss from 10k peak
            portfolio._peak_balance = 10000.0

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.01,
                price=50000.0,
            )

            result = risk_manager.validate_order(order, portfolio)

            assert not result.allowed
            assert "drawdown" in (result.reason or "").lower()
            assert "drawdown_limits" in result.checks_failed

    def test_add_to_loser_rejection(self, portfolio: PortfolioManager) -> None:
        """Test order rejected when adding to a losing position."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            # Add a losing LONG position with small size to not trigger exposure limits
            losing_position = Position(
                symbol="cmt_btcusdt",
                side="LONG",
                size=0.01,  # Small size to avoid exposure limits
                entry_price=51000.0,
                entry_time=datetime.utcnow(),
                leverage=5,
                unrealized_pnl=-50.0,  # Losing position
                current_price=46000.0,
            )
            portfolio.add_position(losing_position)

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            # Try to add to the losing position with small order
            add_order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,  # Buy adds to LONG
                order_type=OrderType.MARKET,
                size=0.001,  # Small size
                price=50000.0,  # $50 notional
                position_side=PositionSide.LONG,
            )

            result = risk_manager.validate_order(add_order, portfolio)

            assert not result.allowed
            assert "losing" in (result.reason or "").lower()
            assert "no_add_to_loser" in result.checks_failed

    def test_close_order_always_allowed(self, portfolio: PortfolioManager) -> None:
        """Test close/reduce-only orders bypass all risk checks."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            # Simulate extreme drawdown
            portfolio._balance = 5000.0  # 50% loss
            portfolio._peak_balance = 10000.0

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            # Close order should still be allowed
            close_order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                size=0.1,
                reduce_only=True,
            )

            result = risk_manager.validate_order(close_order, portfolio)

            assert result.allowed
            assert "close_order" in result.checks_passed[0]

    def test_leverage_limit_check(self, portfolio: PortfolioManager) -> None:
        """Test leverage limit validation."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            # Add position with excessive leverage and small profitable PnL
            # to pass add-to-loser check
            high_leverage_position = Position(
                symbol="cmt_btcusdt",
                side="LONG",
                size=0.001,  # Small size to avoid exposure limits
                entry_price=50000.0,
                entry_time=datetime.utcnow(),
                leverage=25,  # Exceeds 20x limit
                unrealized_pnl=10.0,  # Profitable to pass add-to-loser check
                current_price=50100.0,
            )
            portfolio.add_position(high_leverage_position)

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            # Order on position with excessive leverage - small order
            order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.0001,  # Very small size
                price=50000.0,  # $5 notional
                position_side=PositionSide.LONG,
            )

            result = risk_manager.validate_order(order, portfolio)

            assert not result.allowed
            assert "leverage" in (result.reason or "").lower()

    def test_all_checks_pass(self, portfolio: PortfolioManager) -> None:
        """Test order passes all risk checks."""
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            # Small, reasonable order
            order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.01,
                price=50000.0,  # $500 notional = 5% of $10k balance
            )

            result = risk_manager.validate_order(order, portfolio)

            assert result.allowed
            assert result.reason is None
            assert len(result.checks_failed) == 0
            assert "exposure_limits" in result.checks_passed
            assert "drawdown_limits" in result.checks_passed
            assert "leverage_limit" in result.checks_passed
            assert "no_add_to_loser" in result.checks_passed


# =============================================================================
# Test: Order Execution Flow
# =============================================================================


class TestOrderExecutionFlow:
    """Test order execution lifecycle with mocked exchange."""

    @pytest.mark.asyncio
    async def test_order_execution_market_order(
        self, mock_rest_client: MagicMock
    ) -> None:
        """Test market order execution flow."""
        order_manager = OrderManager(rest_client=mock_rest_client)

        signal = SignalResult(
            signal="LONG",
            confidence=0.85,
            weighted_avg=0.80,
            urgency=0.9,  # High urgency -> market order
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
        )

        result = await order_manager.execute_signal(
            signal=signal,
            symbol="cmt_btcusdt",
            balance=10000.0,
            position_size=500.0,
            leverage=5,
        )

        assert result.success
        assert result.order_id == "test_order_123"
        assert result.fill_price == 50000.0
        mock_rest_client.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_type_selection_logic(self) -> None:
        """Test order type selection based on market conditions."""
        mock_client = MagicMock(spec=RESTClient)
        order_manager = OrderManager(rest_client=mock_client)

        # High urgency -> Market
        order_type = order_manager.select_order_type(
            spread=0.0001,
            size=500.0,
            book_depth=100000.0,
            urgency=0.9,
        )
        assert order_type == OrderTypeSelection.MARKET

        # Wide spread -> Limit at mid
        order_type = order_manager.select_order_type(
            spread=0.001,  # 0.1% spread
            size=500.0,
            book_depth=100000.0,
            urgency=0.5,
        )
        assert order_type == OrderTypeSelection.LIMIT_MID

        # Large order relative to book -> TWAP
        order_type = order_manager.select_order_type(
            spread=0.0001,
            size=5000.0,  # 5% of book depth
            book_depth=100000.0,
            urgency=0.5,
        )
        assert order_type == OrderTypeSelection.SPLIT_TWAP

    @pytest.mark.asyncio
    async def test_slippage_estimation(self, mock_rest_client: MagicMock) -> None:
        """Test slippage estimation based on orderbook."""
        order_manager = OrderManager(rest_client=mock_rest_client)

        orderbook = {
            "bids": [["49995.0", "1.0"], ["49990.0", "2.0"], ["49985.0", "3.0"]],
            "asks": [["50005.0", "1.0"], ["50010.0", "2.0"], ["50015.0", "3.0"]],
        }

        # Small order - low slippage
        slippage_small = order_manager.estimate_slippage(
            symbol="cmt_btcusdt",
            size=0.1,
            orderbook=orderbook,
        )

        # Large order - higher slippage
        slippage_large = order_manager.estimate_slippage(
            symbol="cmt_btcusdt",
            size=10.0,
            orderbook=orderbook,
        )

        assert slippage_large > slippage_small
        assert slippage_small >= 0

    @pytest.mark.asyncio
    async def test_hold_signal_not_executed(self, mock_rest_client: MagicMock) -> None:
        """Test HOLD signal does not execute any order."""
        order_manager = OrderManager(rest_client=mock_rest_client)

        signal = SignalResult(
            signal="HOLD",
            confidence=0.5,
            weighted_avg=0.5,
        )

        result = await order_manager.execute_signal(
            signal=signal,
            symbol="cmt_btcusdt",
            balance=10000.0,
            leverage=5,
        )

        assert not result.success
        assert "HOLD" in (result.error_message or "")
        mock_rest_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self, mock_rest_client: MagicMock) -> None:
        """Test low confidence signal is rejected."""
        order_manager = OrderManager(rest_client=mock_rest_client)

        signal = SignalResult(
            signal="LONG",
            confidence=0.5,  # Below 0.75 threshold
            weighted_avg=0.76,
        )

        result = await order_manager.execute_signal(
            signal=signal,
            symbol="cmt_btcusdt",
            balance=10000.0,
            leverage=5,
        )

        assert not result.success
        assert "confidence" in (result.error_message or "").lower()
        mock_rest_client.place_order.assert_not_called()


# =============================================================================
# Test: Position Sync Reconciliation
# =============================================================================


class TestPositionSyncReconciliation:
    """Test position synchronization with exchange."""

    def test_detect_orphan_positions(self, portfolio: PortfolioManager) -> None:
        """Test detection of positions on exchange not tracked locally."""
        mock_client = MagicMock(spec=RESTClient)
        sync = PositionSync(
            rest_client=mock_client,
            portfolio_manager=portfolio,
        )

        # Exchange has position not tracked locally
        exchange_positions = [
            ExchangePosition(
                symbol="cmt_btcusdt",
                side=PositionSide.LONG,
                size=0.1,
                entry_price=50000.0,
                mark_price=50500.0,
                unrealized_pnl=50.0,
                leverage=5,
            ),
        ]

        orphans = sync.detect_orphan_positions(
            exchange_positions=exchange_positions,
            local_positions=[],
        )

        assert len(orphans) == 1
        assert orphans[0].symbol == "cmt_btcusdt"
        assert orphans[0].side == "LONG"

    def test_reconcile_size_discrepancy(self, portfolio: PortfolioManager) -> None:
        """Test reconciliation when local and exchange sizes differ."""
        mock_client = MagicMock(spec=RESTClient)
        sync = PositionSync(
            rest_client=mock_client,
            portfolio_manager=portfolio,
        )

        # Add local position
        local_position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )
        portfolio.add_position(local_position)

        # Exchange has different size
        exchange_positions = [
            ExchangePosition(
                symbol="cmt_btcusdt",
                side=PositionSide.LONG,
                size=0.15,  # Different from local 0.1
                entry_price=50000.0,
                mark_price=50500.0,
                unrealized_pnl=75.0,
                leverage=5,
            ),
        ]

        discrepancies = sync.reconcile_discrepancies(
            exchange_positions=exchange_positions,
            local_positions=portfolio.get_all_positions(),
        )

        assert len(discrepancies) == 1
        assert discrepancies[0].local_size == 0.1
        assert discrepancies[0].exchange_size == 0.15
        assert discrepancies[0].action_taken == "updated_size_to_exchange_value"

    def test_reconcile_position_removed(self, portfolio: PortfolioManager) -> None:
        """Test reconciliation when position closed on exchange."""
        mock_client = MagicMock(spec=RESTClient)
        sync = PositionSync(
            rest_client=mock_client,
            portfolio_manager=portfolio,
        )

        # Add local position
        local_position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )
        portfolio.add_position(local_position)

        # Exchange has no positions (position was closed)
        exchange_positions: list[ExchangePosition] = []

        discrepancies = sync.reconcile_discrepancies(
            exchange_positions=exchange_positions,
            local_positions=portfolio.get_all_positions(),
        )

        assert len(discrepancies) == 1
        assert discrepancies[0].exchange_size == 0.0
        assert discrepancies[0].action_taken == "removed_local_position"

    @pytest.mark.asyncio
    async def test_full_sync_operation(self, portfolio: PortfolioManager) -> None:
        """Test complete sync operation with exchange."""
        mock_client = MagicMock(spec=RESTClient)
        mock_client.get_positions = AsyncMock(
            return_value=[
                ExchangePosition(
                    symbol="cmt_btcusdt",
                    side=PositionSide.LONG,
                    size=0.1,
                    entry_price=50000.0,
                    mark_price=50500.0,
                    unrealized_pnl=50.0,
                    leverage=5,
                ),
            ]
        )

        sync = PositionSync(
            rest_client=mock_client,
            portfolio_manager=portfolio,
        )

        result = await sync.sync_positions()

        assert result.success
        assert result.positions_synced == 1
        assert portfolio.get_position_count() == 1

        # Verify portfolio was updated
        synced_pos = portfolio.get_position("cmt_btcusdt", "LONG")
        assert synced_pos is not None
        assert synced_pos.size == 0.1


# =============================================================================
# Test: Model Health Check and Weight Redistribution
# =============================================================================


class TestModelHealthCheck:
    """Test ML ensemble health checks and weight redistribution."""

    def test_ensemble_weight_redistribution(self) -> None:
        """Test weight redistribution when models are unhealthy."""
        ensemble = MLEnsemble()

        # Test redistribution with one unhealthy model
        new_weights = ensemble._redistribute_weights(["xgboost"])

        assert new_weights["xgboost"] == 0.0
        assert new_weights["lightgbm"] > 0.25  # Should get redistributed weight
        assert new_weights["lstm"] > 0.25
        assert new_weights["random_forest"] > 0.20

        # Weights should still sum to 1.0
        total = sum(new_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_ensemble_multiple_unhealthy(self) -> None:
        """Test redistribution with multiple unhealthy models."""
        ensemble = MLEnsemble()

        # Two unhealthy models
        new_weights = ensemble._redistribute_weights(["xgboost", "lstm"])

        assert new_weights["xgboost"] == 0.0
        assert new_weights["lstm"] == 0.0
        assert new_weights["lightgbm"] > 0
        assert new_weights["random_forest"] > 0

        # Remaining weights should sum to 1.0
        total = sum(new_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_ensemble_health_status(self) -> None:
        """Test health status reporting."""
        ensemble = MLEnsemble()

        # Initially all models are unhealthy (not loaded)
        health = ensemble.get_health_status()

        assert not health["xgboost"]
        assert not health["lightgbm"]
        assert not health["lstm"]
        assert not health["random_forest"]

    def test_ensemble_minimum_models_required(self) -> None:
        """Test ensemble requires minimum healthy models."""
        ensemble = MLEnsemble()

        # With no healthy models, prediction should return HOLD
        features = {"array": np.random.randn(59)}

        signal, confidence, outputs, weighted_avg = ensemble.predict(features)

        assert signal == "HOLD"
        assert confidence == 0.0
        assert len(outputs) == 0
        assert weighted_avg == 0.5

    def test_ensemble_signal_thresholds(self) -> None:
        """Test signal generation thresholds."""
        ensemble = MLEnsemble()

        # Test LONG threshold
        signal = ensemble._determine_signal(0.80)
        assert signal == "LONG"

        # Test SHORT threshold
        signal = ensemble._determine_signal(0.20)
        assert signal == "SHORT"

        # Test HOLD (between thresholds)
        signal = ensemble._determine_signal(0.50)
        assert signal == "HOLD"

    def test_ensemble_confidence_calculation(self) -> None:
        """Test confidence score calculation."""
        ensemble = MLEnsemble()

        # LONG confidence
        confidence = ensemble._calculate_confidence("LONG", 0.90)
        assert 0 <= confidence <= 1

        # SHORT confidence
        confidence = ensemble._calculate_confidence("SHORT", 0.10)
        assert 0 <= confidence <= 1

        # HOLD confidence should be 0
        confidence = ensemble._calculate_confidence("HOLD", 0.50)
        assert confidence == 0.0


# =============================================================================
# Test: Exit Manager Integration
# =============================================================================


class TestExitManagerIntegration:
    """Test exit manager with position lifecycle."""

    def test_exit_prices_calculation(self) -> None:
        """Test exit price calculation for different regimes."""
        exit_manager = ExitManager()

        # Trending regime - wider stops
        exit_prices_trending = exit_manager.calculate_exit_prices(
            entry_price=50000.0,
            side="LONG",
            atr=500.0,
            regime="trending_up",
        )

        # Ranging regime - tighter stops
        exit_prices_ranging = exit_manager.calculate_exit_prices(
            entry_price=50000.0,
            side="LONG",
            atr=500.0,
            regime="ranging",
        )

        # Trending should have wider stop distance
        trending_sl_distance = 50000.0 - exit_prices_trending.stop_loss
        ranging_sl_distance = 50000.0 - exit_prices_ranging.stop_loss

        assert trending_sl_distance > ranging_sl_distance

    def test_position_registration_and_exit_check(self) -> None:
        """Test position registration and exit condition checking."""
        exit_manager = ExitManager()

        position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )

        # Register position
        exit_prices = exit_manager.register_position(
            position=position,
            atr=500.0,
            regime="ranging",
        )

        assert exit_prices.stop_loss < 50000.0  # SL below entry for LONG
        assert exit_prices.tp1_price > 50000.0  # TP above entry for LONG

        # Check no exit at current price
        should_exit, reason = exit_manager.should_exit(
            position=position,
            current_price=50100.0,  # Small profit
            atr=500.0,
            regime="ranging",
        )

        assert not should_exit

    def test_stop_loss_triggered(self) -> None:
        """Test stop loss trigger."""
        exit_manager = ExitManager()

        position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )

        exit_manager.register_position(
            position=position,
            atr=500.0,
            regime="ranging",
        )

        # Price drops below stop loss
        should_exit, reason = exit_manager.should_exit(
            position=position,
            current_price=48000.0,  # Well below SL
            atr=500.0,
            regime="ranging",
        )

        assert should_exit
        assert reason == "stop_loss"

    def test_take_profit_triggered(self) -> None:
        """Test take profit trigger and breakeven adjustment."""
        exit_manager = ExitManager()

        position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )

        exit_prices = exit_manager.register_position(
            position=position,
            atr=500.0,
            regime="ranging",
        )

        # Price reaches TP1
        should_exit, reason = exit_manager.should_exit(
            position=position,
            current_price=exit_prices.tp1_price + 10,
            atr=500.0,
            regime="ranging",
        )

        assert should_exit
        assert reason == "take_profit_1"

        # Verify stop moved to breakeven
        exit_state = exit_manager.get_exit_state("cmt_btcusdt", "LONG")
        assert exit_state is not None
        assert exit_state.tp1_hit
        assert exit_state.exit_prices.stop_loss > 50000.0 - 100  # Near breakeven


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================


class TestFullPipelineIntegration:
    """Test complete pipeline from data to execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow(
        self,
        mock_candles: list[Candle],
        mock_ticker: TickerData,
        mock_orderbook: OrderbookData,
        mock_rest_client: MagicMock,
        portfolio: PortfolioManager,
    ) -> None:
        """
        Test complete pipeline: Data -> Features -> (Mock) ML -> Risk -> Order.

        This test validates the integration of all major components.
        """
        # Step 1: Compute features
        pipeline = FeaturePipeline(config=FeaturePipelineConfig(min_candles=100))
        features = pipeline.calculate_features(
            candles=mock_candles,
            ticker_data=mock_ticker,
            orderbook_data=mock_orderbook,
        )

        assert len(features) > 0

        # Step 2: Create mock signal (simulating ML ensemble output)
        signal = SignalResult(
            signal="LONG",
            confidence=0.85,
            weighted_avg=0.80,
            urgency=0.5,
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
        )

        # Step 3: Risk validation
        with patch("src.risk.risk_manager.get_settings") as mock_settings:
            mock_settings.return_value.risk.max_leverage = 20

            risk_manager = RiskManager(volatility=1.0, regime="ranging")

            order = OrderRequest(
                symbol="cmt_btcusdt",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=0.01,
                price=50000.0,
            )

            risk_check = risk_manager.validate_order(order, portfolio)
            assert risk_check.allowed

        # Step 4: Execute order
        order_manager = OrderManager(rest_client=mock_rest_client)
        result = await order_manager.execute_signal(
            signal=signal,
            symbol="cmt_btcusdt",
            balance=portfolio.balance,
            position_size=500.0,
            leverage=5,
        )

        assert result.success
        assert result.order_id is not None

        # Step 5: Register with exit manager
        exit_manager = ExitManager()
        position = Position(
            symbol="cmt_btcusdt",
            side="LONG",
            size=result.fill_size or 0.01,
            entry_price=result.fill_price or 50000.0,
            entry_time=datetime.utcnow(),
            leverage=5,
        )

        exit_prices = exit_manager.register_position(
            position=position,
            atr=500.0,
            regime="ranging",
        )

        assert exit_prices.stop_loss < position.entry_price
        assert exit_prices.tp1_price > position.entry_price
