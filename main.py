#!/usr/bin/env python3
"""
AlphaStrike Trading Bot - Main Entry Point (US-028)

Main entry point with trading loop for the AlphaStrike trading bot.
Orchestrates all components for automated trading on WEEX exchange.

Trading Loop:
    data -> features -> ML -> strategy -> risk -> execution

Features:
- Component initialization with dependency injection
- WebSocket candle event handling
- Periodic tasks (position sync, model reload check)
- Graceful shutdown on SIGINT/SIGTERM
- Comprehensive logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.config import MarketRegime, Settings, get_settings
from src.data.data_gateway import DataGateway, ValidationResult, get_data_gateway
from src.data.database import (
    AILogEntry,
    Candle,
    Database,
    Trade,
    TradeSide,
    TradeStatus,
    close_database,
    get_database,
)
from src.data.rest_client import (
    OrderRequest,
    OrderSide,
    OrderType,
    PositionSide,
    RESTClient,
    close_rest_client,
    get_rest_client,
)
from src.data.websocket_client import (
    WebSocketClient,
    close_websocket_client,
    get_websocket_client,
)
from src.execution.order_manager import ExecutionResult, OrderManager
from src.execution.order_manager import SignalResult as OrderSignalResult
from src.execution.signal_processor import SignalProcessor, SignalResult
from src.features.pipeline import FeaturePipeline, TickerData, get_feature_pipeline
from src.ml.confidence_filter import ConfidenceFilter
from src.ml.ensemble import MLEnsemble
from src.risk.portfolio import PortfolioManager, Position
from src.risk.risk_manager import RiskCheck, RiskManager
from src.strategy.regime_detector import RegimeDetector, RegimeState

# Adaptive trading system (optional, gracefully degrades if not available)
try:
    from src.adaptive import AdaptiveManager, AdaptiveState, OPTIMIZER_AVAILABLE
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveManager = None  # type: ignore
    AdaptiveState = None  # type: ignore
    OPTIMIZER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Position Sync Component
# ============================================================================


class PositionSync:
    """
    Synchronizes local portfolio state with exchange positions.

    Periodically fetches positions from the exchange and updates
    the local PortfolioManager to ensure consistency.
    """

    def __init__(
        self,
        rest_client: RESTClient,
        portfolio: PortfolioManager,
        database: Database,
    ) -> None:
        """
        Initialize PositionSync.

        Args:
            rest_client: REST client for exchange API.
            portfolio: Portfolio manager to update.
            database: Database for trade records.
        """
        self.rest_client = rest_client
        self.portfolio = portfolio
        self.database = database
        self._last_sync_time: float = 0.0

    async def sync_positions(self) -> None:
        """
        Sync local positions with exchange state.

        Fetches current positions from exchange and updates
        the local portfolio manager.
        """
        try:
            # Fetch account balance
            balance = await self.rest_client.get_account_balance()
            self.portfolio.balance = balance.total_balance

            # Fetch all positions
            exchange_positions = await self.rest_client.get_positions()

            # Convert to local Position objects
            local_positions: list[Position] = []
            for pos in exchange_positions:
                side = "LONG" if pos.side.value == "long" else "SHORT"
                local_pos = Position(
                    symbol=pos.symbol,
                    side=side,  # type: ignore[arg-type]
                    size=pos.size,
                    entry_price=pos.entry_price,
                    entry_time=pos.timestamp,
                    leverage=pos.leverage,
                    unrealized_pnl=pos.unrealized_pnl,
                    current_price=pos.mark_price,
                )
                local_positions.append(local_pos)

            # Sync with portfolio
            self.portfolio.sync_with_exchange(local_positions)
            self._last_sync_time = time.time()

            logger.info(
                "Position sync completed",
                extra={
                    "position_count": len(local_positions),
                    "balance": balance.total_balance,
                    "unrealized_pnl": balance.unrealized_pnl,
                },
            )

        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    @property
    def last_sync_time(self) -> float:
        """Get timestamp of last successful sync."""
        return self._last_sync_time


# ============================================================================
# AI Logger Component
# ============================================================================


@dataclass
class AILogData:
    """Data for AI decision logging."""

    order_id: str
    symbol: str
    signal: str
    confidence: float
    weighted_average: float
    model_outputs: dict[str, float]
    regime: str
    risk_checks: dict[str, Any]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AILogger:
    """
    Logs AI trading decisions for compliance and analysis.

    Records detailed information about each trading decision
    including model outputs, confidence scores, and risk checks.
    """

    def __init__(self, database: Database, logs_dir: Path | None = None) -> None:
        """
        Initialize AILogger.

        Args:
            database: Database for storing logs.
            logs_dir: Directory for file-based logs.
        """
        self.database = database
        self.logs_dir = logs_dir or Path("ai_logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def log_decision(self, data: AILogData) -> None:
        """
        Log an AI trading decision.

        Args:
            data: AI log data to record.
        """
        try:
            # Create database entry
            log_entry = AILogEntry(
                id=f"ailog_{uuid.uuid4().hex[:16]}",
                order_id=data.order_id,
                symbol=data.symbol,
                signal=data.signal,
                confidence=data.confidence,
                weighted_average=data.weighted_average,
                model_outputs=json.dumps(data.model_outputs),
                regime=data.regime,
                risk_checks=json.dumps(data.risk_checks),
                reasoning=data.reasoning,
                timestamp=data.timestamp,
                uploaded=False,
            )

            await self.database.save_ai_log(log_entry)

            # Also write to file for immediate access
            log_file = self.logs_dir / f"{data.timestamp.strftime('%Y%m%d')}_decisions.jsonl"
            log_dict = {
                "id": log_entry.id,
                "order_id": data.order_id,
                "symbol": data.symbol,
                "signal": data.signal,
                "confidence": data.confidence,
                "weighted_average": data.weighted_average,
                "model_outputs": data.model_outputs,
                "regime": data.regime,
                "risk_checks": data.risk_checks,
                "reasoning": data.reasoning,
                "timestamp": data.timestamp.isoformat(),
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_dict) + "\n")

            logger.debug(f"AI decision logged: {log_entry.id}")

        except Exception as e:
            logger.error(f"Failed to log AI decision: {e}")


# ============================================================================
# Trading Bot Core
# ============================================================================


class TradingBot:
    """
    Main trading bot orchestrator.

    Coordinates all components and manages the trading lifecycle.
    """

    # Periodic task intervals (seconds)
    POSITION_SYNC_INTERVAL = 60  # 60 seconds
    MODEL_RELOAD_CHECK_INTERVAL = 300  # 5 minutes

    def __init__(self) -> None:
        """Initialize trading bot with all components."""
        self.settings: Settings = get_settings()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized in start())
        self.database: Database | None = None
        self.rest_client: RESTClient | None = None
        self.ws_client: WebSocketClient | None = None
        self.data_gateway: DataGateway | None = None
        self.feature_pipeline: FeaturePipeline | None = None
        self.ml_ensemble: MLEnsemble | None = None
        self.confidence_filter: ConfidenceFilter | None = None
        self.regime_detector: RegimeDetector | None = None
        self.portfolio: PortfolioManager | None = None
        self.risk_manager: RiskManager | None = None
        self.signal_processor: SignalProcessor | None = None
        self.order_manager: OrderManager | None = None
        self.position_sync: PositionSync | None = None
        self.ai_logger: AILogger | None = None
        self.adaptive_manager: "AdaptiveManager | None" = None  # type: ignore

        # Candle buffers for feature calculation
        self._candle_buffers: dict[str, list[Candle]] = {}
        self._max_candle_buffer_size = 500

        # Last tick times
        self._last_position_sync = 0.0
        self._last_model_reload_check = 0.0

        logger.info("TradingBot initialized")

    async def initialize_components(self) -> None:
        """Initialize all trading components with dependency injection."""
        logger.info("Initializing components...")

        # 1. Database
        self.database = await get_database()
        logger.info("Database initialized")

        # 2. REST client
        self.rest_client = await get_rest_client()
        logger.info("REST client initialized")

        # 3. WebSocket client
        self.ws_client = await get_websocket_client()
        logger.info("WebSocket client initialized")

        # 4. Data gateway
        self.data_gateway = get_data_gateway()
        logger.info("Data gateway initialized")

        # 5. Feature pipeline
        self.feature_pipeline = get_feature_pipeline()
        logger.info(f"Feature pipeline initialized ({self.feature_pipeline.feature_count} features)")

        # 6. ML ensemble
        self.ml_ensemble = MLEnsemble(
            models_dir=self.settings.models_dir,
            lstm_input_size=self.feature_pipeline.feature_count,
        )
        # Try to load existing models
        self.ml_ensemble.check_and_reload_models()
        logger.info("ML ensemble initialized")

        # 7. Confidence filter
        self.confidence_filter = ConfidenceFilter(config=self.settings.confidence)
        logger.info("Confidence filter initialized")

        # 8. Regime detector
        self.regime_detector = RegimeDetector()
        logger.info("Regime detector initialized")

        # 9. Portfolio manager
        # Get initial balance from exchange
        try:
            balance = await self.rest_client.get_account_balance()
            initial_balance = balance.total_balance
        except Exception:
            initial_balance = 10000.0  # Default fallback

        self.portfolio = PortfolioManager(initial_balance=initial_balance)
        logger.info(f"Portfolio manager initialized with balance: {initial_balance}")

        # 10. Risk manager
        self.risk_manager = RiskManager()
        logger.info("Risk manager initialized")

        # 11. Signal processor
        self.signal_processor = SignalProcessor()
        logger.info("Signal processor initialized")

        # 12. Order manager
        self.order_manager = OrderManager(rest_client=self.rest_client)
        logger.info("Order manager initialized")

        # 13. Position sync
        self.position_sync = PositionSync(
            rest_client=self.rest_client,
            portfolio=self.portfolio,
            database=self.database,
        )
        logger.info("Position sync initialized")

        # 14. AI logger
        self.ai_logger = AILogger(
            database=self.database,
            logs_dir=self.settings.logging.ai_logs_dir,
        )
        logger.info("AI logger initialized")

        # 15. Adaptive trading manager (loads learned parameters per symbol)
        if ADAPTIVE_AVAILABLE and AdaptiveManager is not None:
            try:
                self.adaptive_manager = AdaptiveManager(
                    backtest_func=None,  # Live trading, no backtest
                    auto_optimize=False,  # Don't auto-optimize in production
                    auto_save=True,
                )
                self.adaptive_manager.initialize(self.settings.trading_pairs)
                logger.info(
                    f"Adaptive manager initialized for {len(self.settings.trading_pairs)} symbols"
                )
                logger.info(self.adaptive_manager.get_summary())
            except Exception as e:
                logger.warning(f"Adaptive manager initialization failed: {e}")
                self.adaptive_manager = None
        else:
            logger.info("Adaptive trading system not available, using default parameters")

        # Initialize candle buffers for each trading pair
        for symbol in self.settings.trading_pairs:
            self._candle_buffers[symbol] = []

        logger.info("All components initialized successfully")

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting AlphaStrike Trading Bot...")

        # Initialize all components
        await self.initialize_components()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Initial position sync
        if self.position_sync:
            await self.position_sync.sync_positions()
            self._last_position_sync = time.time()

        # Set running flag
        self._running = True

        # Start WebSocket connection and subscribe to candles
        if self.ws_client:
            self.ws_client.on_candle = self._on_candle_received
            await self.ws_client.connect()
            await self.ws_client.subscribe_candles(
                symbols=self.settings.trading_pairs,
                interval="1m",
            )
            logger.info(f"Subscribed to candles for {len(self.settings.trading_pairs)} pairs")

        logger.info("Trading bot started successfully")

    async def run(self) -> None:
        """Run the main trading loop."""
        logger.info("Entering main trading loop")

        # Start WebSocket message handler and periodic tasks concurrently
        ws_task = asyncio.create_task(self._run_websocket())
        periodic_task = asyncio.create_task(self._run_periodic_tasks())

        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")

        finally:
            # Cancel tasks
            ws_task.cancel()
            periodic_task.cancel()

            try:
                await asyncio.gather(ws_task, periodic_task, return_exceptions=True)
            except Exception:
                pass

            logger.info("Main trading loop exited")

    async def _run_websocket(self) -> None:
        """Run WebSocket message handler."""
        if self.ws_client:
            try:
                await self.ws_client.run_forever()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

    async def _run_periodic_tasks(self) -> None:
        """Run periodic tasks (position sync, model reload check)."""
        while self._running:
            try:
                current_time = time.time()

                # Position sync every 60 seconds
                if current_time - self._last_position_sync >= self.POSITION_SYNC_INTERVAL:
                    if self.position_sync:
                        await self.position_sync.sync_positions()
                    self._last_position_sync = current_time

                # Model reload check every 5 minutes
                if current_time - self._last_model_reload_check >= self.MODEL_RELOAD_CHECK_INTERVAL:
                    await self._check_model_health()
                    self._last_model_reload_check = current_time

                # Small sleep to prevent busy loop
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic task error: {e}")
                await asyncio.sleep(5.0)

    async def _check_model_health(self) -> None:
        """Check model health and reload if necessary."""
        if not self.ml_ensemble:
            return

        health_status = self.ml_ensemble.get_health_status()
        healthy_count = sum(health_status.values())

        logger.info(
            f"Model health check: {healthy_count}/4 healthy",
            extra={"health_status": health_status},
        )

        # Reload models if health is degraded
        if healthy_count < self.settings.ml.min_healthy_models:
            logger.warning("Unhealthy models detected, attempting reload...")
            self.ml_ensemble.check_and_reload_models()

    def _on_candle_received(self, candle: Candle) -> None:
        """
        Handle incoming candle from WebSocket.

        This is a synchronous callback that schedules async processing.
        """
        # Schedule async processing
        asyncio.create_task(self._process_candle(candle))

    async def _process_candle(self, candle: Candle) -> None:
        """
        Process a received candle through the full trading pipeline.

        Pipeline:
        1. Validate data through gateway
        2. Update candle buffer
        3. Calculate features
        4. Generate ML prediction
        5. Apply confidence filter
        6. Detect market regime
        7. Process signal
        8. Validate risk
        9. Execute order if approved
        """
        if not self._running:
            return

        # Check trading enabled
        if not self.settings.trading_enabled:
            return

        symbol = candle.symbol

        try:
            # 1. Validate data through gateway
            if self.data_gateway:
                validation: ValidationResult = await self.data_gateway.process_candle(candle)
                if not validation.passed:
                    logger.debug(
                        f"Candle rejected by gateway: {validation.rejection_reason}"
                    )
                    return

            # 2. Update candle buffer
            self._update_candle_buffer(candle)
            candles = self._candle_buffers.get(symbol, [])

            # Need minimum candles for feature calculation
            if len(candles) < 100:
                logger.debug(f"Insufficient candles for {symbol}: {len(candles)}")
                return

            # 3. Calculate features
            if not self.feature_pipeline:
                return

            features = self.feature_pipeline.calculate_features(candles=candles)

            if not features:
                logger.warning(f"Feature calculation failed for {symbol}")
                return

            # 4. Generate ML prediction
            if not self.ml_ensemble:
                return

            signal, confidence, model_outputs, weighted_avg = self.ml_ensemble.predict(features)

            # 5. Detect market regime
            if not self.regime_detector:
                return

            # Add price info for regime detection
            features["price"] = candle.close
            features["ema_20"] = features.get("ema_20", candle.close)

            regime_state: RegimeState = self.regime_detector.detect_regime(features)
            regime = regime_state.regime.value

            # 5b. Update adaptive manager with regime and get adjusted parameters
            adaptive_params = None
            if self.adaptive_manager:
                try:
                    self.adaptive_manager.update_regime(
                        symbol=symbol,
                        regime=regime_state.regime,
                        confidence=regime_state.confidence,
                    )
                    adaptive_params = self.adaptive_manager.get_adjusted_params(symbol)
                except Exception as e:
                    logger.debug(f"Adaptive params lookup failed: {e}")

            # 6. Apply confidence filter
            if not self.confidence_filter:
                return

            should_reject, rejection_reason, score_breakdown = self.confidence_filter.should_reject(
                signal=signal,  # type: ignore[arg-type]
                raw_confidence=confidence,
                weighted_avg=weighted_avg,
                model_outputs=model_outputs,
                regime=regime,
                symbol=symbol,
            )

            if should_reject:
                logger.debug(
                    f"Signal rejected by confidence filter: {rejection_reason}"
                )
                signal = "HOLD"
                confidence = 0.0

            # Skip HOLD signals
            if signal == "HOLD":
                return

            # 7. Process signal through signal processor
            if not self.signal_processor:
                return

            # Create signal processing features dict with regime added
            signal_features: dict[str, Any] = dict(features)
            signal_features["regime"] = regime

            model_agreement = len([v for v in model_outputs.values() if (v > 0.5) == (weighted_avg > 0.5)]) / max(len(model_outputs), 1)

            signal_result: SignalResult = self.signal_processor.process_signal(
                raw_signal=signal,  # type: ignore[arg-type]
                confidence=confidence,
                features=signal_features,
                model_agreement=model_agreement,
            )

            # Skip if signal was filtered to HOLD
            if signal_result.signal == "HOLD":
                logger.debug("Signal filtered to HOLD by signal processor")
                return

            # 7b. Check adaptive short blocking (learned from optimization)
            if signal_result.signal == "SHORT" and adaptive_params:
                if not adaptive_params.short_enabled:
                    logger.info(
                        f"SHORT signal blocked for {symbol}: adaptive system disabled shorts"
                    )
                    return

            # 8. Validate risk
            if not self.risk_manager or not self.portfolio:
                return

            # Update risk manager with current conditions
            atr_ratio = features.get("atr_ratio", 1.0)
            self.risk_manager.update_market_conditions(
                volatility=atr_ratio,
                regime=regime,
            )

            # Build order request for risk validation
            order_side = OrderSide.BUY if signal_result.signal == "LONG" else OrderSide.SELL
            position_side = PositionSide.LONG if signal_result.signal == "LONG" else PositionSide.SHORT

            # Calculate position size with adaptive multiplier
            adaptive_size_mult = 1.0
            adaptive_stop_mult = 1.0
            if adaptive_params:
                adaptive_size_mult = adaptive_params.position_size_multiplier
                adaptive_stop_mult = adaptive_params.stop_atr_multiplier
                # Log adaptive adjustments for high-conviction trades
                if signal_result.position_scale > 0.5:
                    logger.info(
                        f"Adaptive params for {symbol}: size={adaptive_size_mult:.2f}x, "
                        f"stop={adaptive_stop_mult:.2f}x ATR, regime={regime}"
                    )

            position_size = (
                self.portfolio.balance
                * self.settings.position.per_trade_exposure
                * signal_result.position_scale
                * adaptive_size_mult
            )

            # Create order request for risk check
            order_request = OrderRequest(
                symbol=symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                size=position_size / candle.close,  # Convert to base currency
                price=candle.close,
                position_side=position_side,
            )

            risk_check: RiskCheck = self.risk_manager.validate_order(
                order=order_request,
                portfolio=self.portfolio,
            )

            if not risk_check.allowed:
                logger.info(
                    f"Order blocked by risk manager: {risk_check.reason}",
                    extra={
                        "symbol": symbol,
                        "signal": signal_result.signal,
                        "checks_passed": risk_check.checks_passed,
                        "checks_failed": risk_check.checks_failed,
                    },
                )
                return

            # 9. Execute order
            if not self.order_manager:
                return

            # Log AI decision before execution
            order_id = f"ord_{uuid.uuid4().hex[:16]}"
            reasoning = self._generate_reasoning(
                signal=signal_result.signal,
                confidence=confidence,
                regime=regime,
                features=features,
                model_outputs=model_outputs,
            )

            if self.ai_logger:
                ai_log_data = AILogData(
                    order_id=order_id,
                    symbol=symbol,
                    signal=signal_result.signal,
                    confidence=confidence,
                    weighted_average=weighted_avg,
                    model_outputs=model_outputs,
                    regime=regime,
                    risk_checks={
                        "allowed": risk_check.allowed,
                        "checks_passed": risk_check.checks_passed,
                        "checks_failed": risk_check.checks_failed,
                    },
                    reasoning=reasoning,
                )
                await self.ai_logger.log_decision(ai_log_data)

            # Skip actual execution in paper trading mode
            if self.settings.paper_trading:
                logger.info(
                    f"[PAPER] Would execute {signal_result.signal} for {symbol}",
                    extra={
                        "position_size": position_size,
                        "price": candle.close,
                        "confidence": confidence,
                    },
                )
                return

            # Execute the order
            execution_signal = OrderSignalResult(
                signal=signal_result.signal,  # type: ignore[arg-type]
                confidence=confidence,
                weighted_avg=weighted_avg,
            )

            execution_result: ExecutionResult = await self.order_manager.execute_signal(
                signal=execution_signal,
                symbol=symbol,
                balance=self.portfolio.balance,
                position_size=position_size,
                leverage=self.settings.risk.default_leverage,
            )

            if execution_result.success:
                logger.info(
                    f"Order executed successfully",
                    extra={
                        "symbol": symbol,
                        "signal": signal_result.signal,
                        "order_id": execution_result.order_id,
                        "fill_price": execution_result.fill_price,
                        "fill_size": execution_result.fill_size,
                    },
                )

                # Save trade to database
                if self.database:
                    trade = Trade(
                        id=order_id,
                        symbol=symbol,
                        side=TradeSide.LONG if signal_result.signal == "LONG" else TradeSide.SHORT,
                        entry_price=execution_result.fill_price or candle.close,
                        quantity=execution_result.fill_size or (position_size / candle.close),
                        leverage=self.settings.risk.default_leverage,
                        status=TradeStatus.OPEN,
                        order_id=execution_result.order_id,
                        ai_explanation=reasoning,
                    )
                    await self.database.save_trade(trade)
            else:
                logger.warning(
                    f"Order execution failed: {execution_result.error_message}",
                    extra={"symbol": symbol, "signal": signal_result.signal},
                )

        except Exception as e:
            logger.exception(f"Error processing candle for {symbol}: {e}")

    def _update_candle_buffer(self, candle: Candle) -> None:
        """Update candle buffer for a symbol."""
        symbol = candle.symbol

        if symbol not in self._candle_buffers:
            self._candle_buffers[symbol] = []

        buffer = self._candle_buffers[symbol]

        # Check if this is a duplicate timestamp
        if buffer and buffer[-1].timestamp == candle.timestamp:
            # Update the last candle instead of appending
            buffer[-1] = candle
        else:
            buffer.append(candle)

        # Trim buffer if too large
        if len(buffer) > self._max_candle_buffer_size:
            self._candle_buffers[symbol] = buffer[-self._max_candle_buffer_size:]

    def _generate_reasoning(
        self,
        signal: str,
        confidence: float,
        regime: str,
        features: dict[str, float],
        model_outputs: dict[str, float],
    ) -> str:
        """Generate human-readable reasoning for the trading decision."""
        parts = [
            f"Signal: {signal} (confidence: {confidence:.2%})",
            f"Regime: {regime}",
            f"Model outputs: {', '.join(f'{k}={v:.3f}' for k, v in model_outputs.items())}",
        ]

        # Add key technical indicators
        key_features = ["rsi", "adx", "macd", "bb_position", "atr_ratio"]
        feature_strs = []
        for feat in key_features:
            if feat in features:
                feature_strs.append(f"{feat}={features[feat]:.2f}")

        if feature_strs:
            parts.append(f"Key indicators: {', '.join(feature_strs)}")

        return " | ".join(parts)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown_signal)

        logger.info("Signal handlers configured")

    def _handle_shutdown_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()

    async def shutdown(self) -> None:
        """Gracefully shutdown the trading bot."""
        logger.info("Shutting down trading bot...")
        self._running = False

        # Disconnect WebSocket
        await close_websocket_client()
        logger.info("WebSocket client closed")

        # Close REST client
        await close_rest_client()
        logger.info("REST client closed")

        # Close database
        await close_database()
        logger.info("Database closed")

        logger.info("Trading bot shutdown complete")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main() -> None:
    """Main entry point for the trading bot."""
    logger.info("=" * 60)
    logger.info("AlphaStrike Trading Bot Starting")
    logger.info("=" * 60)

    bot = TradingBot()

    try:
        # Start the bot
        await bot.start()

        # Run the main trading loop
        await bot.run()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")

    finally:
        # Graceful shutdown
        await bot.shutdown()

    logger.info("=" * 60)
    logger.info("AlphaStrike Trading Bot Stopped")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
