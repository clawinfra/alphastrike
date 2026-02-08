"""
Medallion Live Trading Engine

Implements the Medallion V2 strategy for live trading on Hyperliquid.
Adapts the proven backtest logic (67.5% CAGR, 3.9% DD, 3.67 Sharpe) for real-time execution.

Key strategy components:
- LightGBM-only predictions (proven superior to ensemble)
- Strict regime detection (BULLISH 60%+ only)
- ML tier filtering (65-70+ conviction)
- Conservative position sizing (5% max, 40% total)
- Tight risk management (1% SL, 4% TP, 36h time exit)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from src.adaptive.dynamic_leverage import DynamicLeverageManager
from src.trading.position_tracker import (
    LivePosition,
    PositionTracker,
    PositionTrackerConfig,
)
from src.trading.trade_logger import create_trade_logger

# Cloud backup imports (only used in live mode)
try:
    from src.resilience.cloud_backup import CloudBackupManager
    CLOUD_BACKUP_AVAILABLE = True
except ImportError:
    CloudBackupManager = None  # type: ignore
    CLOUD_BACKUP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MedallionLiveConfig:
    """Configuration for Medallion V2 live trading."""

    assets: list[str] = field(
        default_factory=lambda: [
            "BTC", "ETH", "BNB", "XRP", "SOL", "AVAX", "NEAR", "APT",
            "AAVE", "UNI", "LINK", "FET", "DOGE", "PAXG", "SPX",
        ]
    )
    candle_interval: str = "1h"

    # Dynamic leverage (reduces during emergencies, otherwise uses base)
    base_leverage: float = 5.0
    min_leverage: float = 1.0
    max_leverage: float = 10.0

    # Position sizing (from Medallion v2 backtest)
    max_portfolio_exposure: float = 0.40  # 40% max total
    max_single_position: float = 0.05     # 5% per position

    # Risk management
    stop_loss_pct: float = 0.01           # 1% stop loss
    take_profit_pct: float = 0.04         # 4% take profit
    time_exit_hours: int = 36             # 36 hour time exit

    # ML thresholds
    min_conviction: float = 50.0
    ml_long_threshold: float = 0.55
    ml_short_threshold: float = 0.45

    # Regime detection
    regime_required: str = "BULLISH"
    regime_confidence_min: float = 60.0

    # Buffer sizes
    min_candles_for_features: int = 150
    max_candle_buffer: int = 500

    # Sync intervals
    position_sync_seconds: int = 60
    exit_check_seconds: int = 30

    # Paper trading (dry_run mode)
    fallback_balance: float = 10000.0  # Simulated balance for dry_run

    # Cloud backup (live mode only, configured via .env)
    cloud_backup_enabled: bool = True  # Auto-disabled in dry_run mode


class MedallionLiveEngine:
    """
    Live trading engine implementing the Medallion V2 strategy.

    Features:
    - Real-time WebSocket candle feeds
    - LightGBM prediction per asset
    - Strict regime detection (BULLISH only)
    - Multi-tier ML signal filtering
    - Position tracking and risk management
    - Trade logging and persistence
    """

    def __init__(
        self,
        config: MedallionLiveConfig | None = None,
        testnet: bool = True,
        dry_run: bool = False,
        db_backend: Literal["jsonl", "sqlite"] = "jsonl",
        private_key: str | None = None,
    ):
        self.config = config or MedallionLiveConfig()
        self.testnet = testnet
        self.dry_run = dry_run
        self.private_key = private_key  # Wallet-specific key for multi-wallet

        # Components (lazy init)
        self.adapter = None
        self.feature_pipeline = None
        self.ml_models: dict = {}

        # Position tracking
        tracker_config = PositionTrackerConfig(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
            time_exit_hours=self.config.time_exit_hours,
            max_portfolio_exposure=self.config.max_portfolio_exposure,
            max_single_position=self.config.max_single_position,
        )
        self.position_tracker = PositionTracker(tracker_config)
        self.trade_logger = create_trade_logger(backend=db_backend)

        # Dynamic leverage manager (self-tuning based on market conditions)
        self.leverage_manager = DynamicLeverageManager(
            base_leverage=self.config.base_leverage,
            min_leverage=self.config.min_leverage,
            max_leverage=self.config.max_leverage,
        )

        # Cloud backup (live mode only)
        self.cloud_backup = None  # type: CloudBackupManager | None
        config = self.config  # Ensure config is available
        if not dry_run and config.cloud_backup_enabled and CLOUD_BACKUP_AVAILABLE:
            self.cloud_backup = CloudBackupManager()
            logger.info("Cloud backup enabled (live mode)")
        elif not dry_run and config.cloud_backup_enabled and not CLOUD_BACKUP_AVAILABLE:
            logger.warning("Cloud backup requested but CloudBackupManager not available")

        # State
        self.candle_buffers: dict[str, list] = {}
        self.balance: float = 0.0
        self._running: bool = False
        self._shutdown_event = asyncio.Event()

        # Tasks
        self._ws_task: asyncio.Task | None = None
        self._periodic_task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Check if the engine is currently running."""
        return self._running
        self._periodic_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Medallion Live Engine...")

        # Import here to avoid circular imports
        from src.exchange.adapters.hyperliquid.adapter import HyperliquidAdapter
        from src.features.pipeline import FeaturePipeline
        from src.ml.lightgbm_model import LightGBMModel

        # Initialize exchange adapter (use wallet-specific key if provided)
        self.adapter = HyperliquidAdapter(
            testnet=self.testnet,
            private_key=self.private_key,
        )
        await self.adapter.initialize()
        logger.info(f"Hyperliquid adapter initialized (testnet={self.testnet})")

        # Get balance based on mode
        if self.dry_run:
            # Paper trading: use fallback balance (configurable)
            self.balance = getattr(self.config, 'fallback_balance', 10000.0)
            logger.info(f"DRY RUN: Using simulated balance ${self.balance:,.2f}")
        else:
            # Live trading: must use real balance from API
            try:
                balance_info = await self.adapter.rest.get_account_balance()
                self.balance = balance_info.total_balance
                logger.info(f"API balance: ${self.balance:,.2f}")

                # Validate balance is tradable
                min_balance = 100.0  # Minimum required to trade
                if self.balance < min_balance:
                    raise ValueError(
                        f"Insufficient balance: ${self.balance:,.2f}. "
                        f"Minimum required: ${min_balance:,.2f}. "
                        f"Fund your wallet or use --dry-run for paper trading."
                    )

            except Exception as e:
                logger.error(f"Failed to get account balance: {e}")
                logger.error("Ensure EXCHANGE_WALLET_PRIVATE_KEY is set correctly")
                raise

        # Initialize feature pipeline
        self.feature_pipeline = FeaturePipeline()
        logger.info(f"Feature pipeline initialized ({len(self.feature_pipeline.feature_names)} features)")

        # Load ML models (LightGBM only - proven superior)
        models_dir = Path("models")
        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            lgb_path = models_dir / f"lightgbm_hyperliquid_{asset.lower()}.txt"
            if lgb_path.exists():
                try:
                    model = LightGBMModel()
                    model.load(lgb_path)
                    self.ml_models[symbol] = model
                    logger.debug(f"Loaded model for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")

        logger.info(f"Loaded {len(self.ml_models)} LightGBM models")

        # Initialize candle buffers
        await self._initialize_candle_buffers()

        # Start cloud backup worker (live mode only)
        if self.cloud_backup:
            await self.cloud_backup.start_retry_worker()
            health = await self.cloud_backup.health_check()
            logger.info(f"Cloud backup health: {health}")

        logger.info("Medallion Live Engine initialized successfully")

    async def _initialize_candle_buffers(self) -> None:
        """Fetch historical candles to warm up buffers."""
        logger.info("Fetching historical candles for warmup...")

        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            try:
                candles = await self.adapter.rest.get_candles(
                    symbol=symbol,
                    interval=self.config.candle_interval,
                    limit=200,
                )
                self.candle_buffers[symbol] = list(candles)
                logger.debug(f"Loaded {len(candles)} candles for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch candles for {symbol}: {e}")
                self.candle_buffers[symbol] = []

        logger.info(f"Candle buffers initialized for {len(self.candle_buffers)} assets")

    async def start(self) -> None:
        """Start the trading engine."""
        if self._running:
            logger.warning("Engine already running")
            return

        self._running = True
        logger.info("Starting Medallion Live Engine...")

        # Register candle callback and subscribe
        symbols = [f"{asset}USDT" for asset in self.config.assets]
        self.adapter.websocket.on_candle(self._on_candle)
        await self.adapter.websocket.subscribe_candles(
            symbols,
            interval=self.config.candle_interval,
        )
        logger.info(f"Subscribed to {len(symbols)} candle feeds")

        # Start background tasks
        self._ws_task = asyncio.create_task(self._run_websocket())
        self._periodic_task = asyncio.create_task(self._run_periodic_tasks())

        logger.info("Medallion Live Engine started")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Gracefully stop the trading engine."""
        if not self._running:
            return

        logger.info("Stopping Medallion Live Engine...")
        self._running = False
        self._shutdown_event.set()

        # Cancel tasks
        if self._ws_task:
            self._ws_task.cancel()
        if self._periodic_task:
            self._periodic_task.cancel()

        # Close adapter
        if self.adapter:
            await self.adapter.close()

        # Close cloud backup
        if self.cloud_backup:
            await self.cloud_backup.close()
            logger.info("Cloud backup closed")

        # Print final report
        report = self.trade_logger.generate_daily_report()
        print(report)

        logger.info("Medallion Live Engine stopped")

    async def _run_websocket(self) -> None:
        """Run WebSocket connection."""
        try:
            await self.adapter.websocket.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _run_periodic_tasks(self) -> None:
        """Run periodic background tasks."""
        last_sync = 0
        last_exit_check = 0
        last_leverage_update = 0
        leverage_update_interval = 300  # Update leverage every 5 minutes

        try:
            while self._running:
                now = asyncio.get_event_loop().time()

                # Position sync
                if now - last_sync >= self.config.position_sync_seconds:
                    await self._sync_positions()
                    last_sync = now

                # Exit condition check
                if now - last_exit_check >= self.config.exit_check_seconds:
                    await self._check_exits()
                    last_exit_check = now

                # Dynamic leverage update
                if now - last_leverage_update >= leverage_update_interval:
                    await self._update_dynamic_leverage()
                    last_leverage_update = now

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    async def _update_dynamic_leverage(self) -> None:
        """Update dynamic leverage based on market conditions."""
        try:
            # Calculate current volatility from BTC candles (market leader)
            btc_candles = self.candle_buffers.get("BTCUSDT", [])
            if len(btc_candles) >= 20:
                # Calculate ATR-based volatility
                closes = [c.close for c in btc_candles[-20:]]
                highs = [c.high for c in btc_candles[-20:]]
                lows = [c.low for c in btc_candles[-20:]]

                tr_values = []
                for i in range(1, len(closes)):
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                    tr_values.append(tr)

                atr = sum(tr_values) / len(tr_values)
                current_volatility = atr / closes[-1]
            else:
                current_volatility = 0.02  # Default

            # Get current drawdown from position tracker
            summary = self.position_tracker.get_summary()
            total_pnl = summary["total_realized_pnl"] + summary["total_unrealized_pnl"]
            current_drawdown = abs(min(0, total_pnl) / max(self.balance, 1))

            # Get rolling win rate
            metrics = self.trade_logger.get_performance_metrics()
            rolling_win_rate = metrics.get("win_rate", 0.5)

            # Update leverage manager
            new_leverage, changed, reason = self.leverage_manager.update_conditions(
                current_volatility=current_volatility,
                current_drawdown=current_drawdown,
                rolling_win_rate=rolling_win_rate,
            )

            if changed:
                logger.info(f"Dynamic leverage updated: {new_leverage:.1f}x ({reason})")

        except Exception as e:
            logger.error(f"Failed to update dynamic leverage: {e}")

    def _on_candle(self, candle) -> None:
        """Handle incoming candle from WebSocket."""
        symbol = candle.symbol
        if symbol not in self.candle_buffers:
            self.candle_buffers[symbol] = []

        buffer = self.candle_buffers[symbol]

        # Add new candle or update last one
        if buffer and buffer[-1].timestamp == candle.timestamp:
            buffer[-1] = candle  # Update existing
        else:
            buffer.append(candle)  # Add new

        # Trim buffer
        if len(buffer) > self.config.max_candle_buffer:
            self.candle_buffers[symbol] = buffer[-self.config.max_candle_buffer:]

        # Schedule async processing
        asyncio.create_task(self._process_candle(symbol))

    async def _process_candle(self, symbol: str) -> None:
        """Process a candle update - full trading pipeline."""
        try:
            candles = self.candle_buffers.get(symbol, [])
            if len(candles) < self.config.min_candles_for_features:
                return  # Not enough data

            # Skip if we have an open position for this symbol
            if symbol in self.position_tracker.positions:
                return

            # Check exposure limit
            current_exposure = self.position_tracker.calculate_exposure(self.balance)
            if current_exposure >= self.config.max_portfolio_exposure:
                return

            # Get BTC candles for regime detection
            btc_candles = self.candle_buffers.get("BTCUSDT", [])
            regime, regime_conf = self._detect_market_regime(btc_candles)

            # Only trade in BULLISH regime with strong confidence
            if regime != self.config.regime_required or regime_conf < self.config.regime_confidence_min:
                return

            # Calculate features
            features = self.feature_pipeline.calculate_features(candles)
            if not features:
                return

            # Get ML signal
            ml_direction, ml_conv = self._get_ml_signal(symbol, features)

            # Calculate momentum indicators
            direction, conviction, tier = self._apply_tier_filtering(
                ml_direction, ml_conv, candles
            )

            if direction == "HOLD" or conviction < self.config.min_conviction:
                await self.trade_logger.log_signal(
                    symbol, ml_direction, ml_conv,
                    action="SKIP",
                    reason="below_threshold" if direction == "HOLD" else "low_conviction",
                    regime=regime,
                )
                return

            # Calculate position size
            size = min(
                self.balance * self.config.max_single_position,
                self.balance * (1 - current_exposure) * 0.5,
            ) * (conviction / 100)

            if size < 50:  # Minimum position size
                return

            price = candles[-1].close

            # Execute trade
            await self._execute_entry(symbol, direction, size, price, conviction, tier, regime)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            await self.trade_logger.log_error("process_candle", str(e), {"symbol": symbol})

    def _detect_market_regime(self, btc_candles: list, lookback: int = 50) -> tuple[str, float]:
        """
        Detect market regime using BTC as the market leader.

        WARNING: This is one of THREE separate regime detectors in the codebase:
        1. src/strategy/regime_detector.py (ADX/ATR, 6 regimes) — formal module
        2. src/trading/medallion_live.py (this one, SMA/slope, 3 regimes)
        3. src/strategy/simons_engine.py (simplified ADX, 3 regimes)
        TODO: Unify into a single implementation to avoid divergent behavior.

        Returns:
            (regime, strength): regime is BULLISH/BEARISH/RANGING, strength 0-100
        """
        if len(btc_candles) < lookback:
            return "RANGING", 50.0

        closes = np.array([c.close for c in btc_candles[-lookback:]])

        # Calculate trend indicators
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes)
        current_price = closes[-1]

        # Trend direction (linear regression slope)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        slope_pct = slope / closes[0] * 100

        # Recent momentum
        recent_avg = np.mean(closes[-10:])
        prev_avg = np.mean(closes[-20:-10])
        momentum = (recent_avg - prev_avg) / prev_avg * 100

        # Count signals
        bullish_signals = 0
        bearish_signals = 0

        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if sma_20 > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if slope_pct > 0.1:
            bullish_signals += 2
        elif slope_pct < -0.1:
            bearish_signals += 2

        if momentum > 1:
            bullish_signals += 1
        elif momentum < -1:
            bearish_signals += 1

        # Determine regime
        total_signals = bullish_signals + bearish_signals
        if bullish_signals >= 4:
            regime = "BULLISH"
            strength = min(100, bullish_signals / total_signals * 100 + 20)
        elif bearish_signals >= 4:
            regime = "BEARISH"
            strength = min(100, bearish_signals / total_signals * 100 + 20)
        else:
            regime = "RANGING"
            strength = 50.0

        return regime, strength

    def _get_ml_signal(self, symbol: str, features: dict) -> tuple[str, float]:
        """Get LightGBM prediction with feature name validation."""
        if symbol not in self.ml_models or not features:
            return "HOLD", 0.0

        try:
            model = self.ml_models[symbol]
            feature_names = self.feature_pipeline.feature_names

            # Validate feature alignment: if the model has named features,
            # use its feature order to avoid silent misalignment
            model_feature_names = model.feature_names
            if model_feature_names and model_feature_names[0] != "feature_0":
                # Model has real feature names — use model's ordering
                X = np.array([[features.get(name, 0.0) for name in model_feature_names]])
            else:
                # Model has generic names — use pipeline ordering (legacy)
                X = np.array([[features.get(name, 0.0) for name in feature_names]])

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model.predict(X)[0]

            if pred > self.config.ml_long_threshold:
                confidence = min(100, (pred - 0.5) * 250)
                return "LONG", confidence
            elif pred < self.config.ml_short_threshold:
                confidence = min(100, (0.5 - pred) * 250)
                return "SHORT", confidence

            return "HOLD", 0.0
        except Exception:
            return "HOLD", 0.0

    def _apply_tier_filtering(
        self,
        ml_direction: str,
        ml_conv: float,
        candles: list,
    ) -> tuple[str, float, str]:
        """
        Apply multi-tier ML signal filtering.

        Returns:
            (direction, conviction, tier)
        """
        if ml_direction != "LONG" or ml_conv < 65:
            return "HOLD", 0.0, ""

        # Calculate momentum
        mom_12h = 0.0
        mom_24h = 0.0
        volatility = 0.0

        if len(candles) >= 48:
            recent_returns = []
            for j in range(1, min(48, len(candles))):
                ret = (candles[-j].close - candles[-j-1].close) / candles[-j-1].close
                recent_returns.append(ret)

            mom_12h = sum(recent_returns[:12]) if len(recent_returns) >= 12 else 0
            mom_24h = sum(recent_returns[:24]) if len(recent_returns) >= 24 else 0
            volatility = np.std(recent_returns[:24]) if len(recent_returns) >= 24 else 0

        # Tier 1: Very high ML confidence (70+) with positive momentum
        if ml_conv >= 70 and mom_12h > 0.01:
            return "LONG", min(100, ml_conv + 10), "ml_tier1"

        # Tier 2: High ML confidence (65+) with strong momentum AND low volatility
        if ml_conv >= 65 and mom_12h > 0.015 and volatility < 0.03:
            return "LONG", ml_conv, "ml_tier2"

        # Tier 3: High ML confidence with consistent uptrend
        if ml_conv >= 65 and mom_12h > 0.005 and mom_24h > 0.01:
            return "LONG", ml_conv - 5, "ml_tier3"

        return "HOLD", 0.0, ""

    async def _execute_entry(
        self,
        symbol: str,
        direction: Literal["LONG", "SHORT"],
        size: float,
        price: float,
        conviction: float,
        tier: str,
        regime: str,
    ) -> bool:
        """Execute a trade entry."""
        # Get current dynamic leverage
        current_leverage = int(self.leverage_manager.get_leverage())

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}Entering {direction} {symbol} "
            f"@ {price:.2f}, size=${size:.2f}, leverage={current_leverage}x, "
            f"conviction={conviction:.1f}, tier={tier}"
        )

        await self.trade_logger.log_signal(
            symbol, direction, conviction,
            action="ENTER",
            regime=regime,
            tier=tier,
        )

        if self.dry_run:
            # Simulate entry for dry run
            position = LivePosition(
                symbol=symbol,
                direction=direction,
                entry_price=price,
                size=size,
                entry_time=datetime.now(UTC),
                strategy=tier,
                conviction=conviction,
                order_id=f"dry_run_{symbol}_{int(datetime.now().timestamp())}",
                leverage=current_leverage,
            )
            self.position_tracker.add_position(position)
            await self.trade_logger.log_entry(position)
            return True

        try:
            # Set dynamic leverage on exchange
            await self.adapter.rest.set_leverage(symbol, current_leverage)

            # Create and execute order
            from src.exchange.models import OrderSide, OrderType, PositionSide, UnifiedOrder

            side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL
            position_side = PositionSide.LONG if direction == "LONG" else PositionSide.SHORT
            quantity = size / price

            order = UnifiedOrder(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price,
                position_side=position_side,
            )

            result = await self.adapter.rest.place_order(order)

            if result and result.order_id:
                position = LivePosition(
                    symbol=symbol,
                    direction=direction,
                    entry_price=result.avg_fill_price or price,
                    size=size,
                    entry_time=datetime.now(UTC),
                    strategy=tier,
                    conviction=conviction,
                    order_id=result.order_id,
                    leverage=current_leverage,
                )
                self.position_tracker.add_position(position)
                await self.trade_logger.log_entry(position)

                # Cloud backup (live mode only)
                await self._backup_trade_event(
                    event_type="trade_entry",
                    symbol=symbol,
                    payload={
                        "direction": direction,
                        "entry_price": position.entry_price,
                        "size": size,
                        "leverage": current_leverage,
                        "conviction": conviction,
                        "strategy": tier,
                        "regime": regime,
                        "order_id": result.order_id,
                    },
                )
                await self._backup_position_state()

                return True

            logger.error(f"Order placement failed for {symbol}")
            return False

        except Exception as e:
            logger.error(f"Failed to execute entry for {symbol}: {e}")
            await self.trade_logger.log_error("execute_entry", str(e), {"symbol": symbol})
            return False

    async def _execute_exit(self, symbol: str, reason: str) -> bool:
        """Execute a trade exit."""
        if symbol not in self.position_tracker.positions:
            return False

        position = self.position_tracker.positions[symbol]
        price = position.current_price or position.entry_price

        logger.info(
            f"{'[DRY RUN] ' if self.dry_run else ''}Exiting {position.direction} {symbol} "
            f"@ {price:.2f}, reason={reason}"
        )

        if self.dry_run:
            trade = self.position_tracker.close_position(symbol, price, reason)
            if trade:
                await self.trade_logger.log_exit(trade)
            return True

        try:
            from src.exchange.models import OrderSide, OrderType, PositionSide, UnifiedOrder

            # Close position with opposite order
            side = OrderSide.SELL if position.direction == "LONG" else OrderSide.BUY
            position_side = PositionSide.LONG if position.direction == "LONG" else PositionSide.SHORT
            quantity = position.size / position.entry_price

            order = UnifiedOrder(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price,
                position_side=position_side,
                reduce_only=True,
            )

            result = await self.adapter.rest.place_order(order)

            if result:
                exit_price = result.avg_fill_price or price
                trade = self.position_tracker.close_position(symbol, exit_price, reason)
                if trade:
                    await self.trade_logger.log_exit(trade)

                    # Cloud backup (live mode only)
                    await self._backup_trade_event(
                        event_type="trade_exit",
                        symbol=symbol,
                        payload={
                            "direction": position.direction,
                            "entry_price": position.entry_price,
                            "exit_price": exit_price,
                            "size": position.size,
                            "leverage": position.leverage,
                            "pnl": trade.pnl,
                            "pnl_pct": trade.pnl_pct,
                            "exit_reason": reason,
                            "strategy": position.strategy,
                            "order_id": position.order_id,
                        },
                    )
                    await self._backup_position_state()

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to execute exit for {symbol}: {e}")
            await self.trade_logger.log_error("execute_exit", str(e), {"symbol": symbol, "reason": reason})
            return False

    async def _sync_positions(self) -> None:
        """Sync local state with exchange positions."""
        try:
            if self.dry_run:
                # In dry-run mode, update prices from candle buffers
                prices = {}
                for symbol, candles in self.candle_buffers.items():
                    if candles and symbol in self.position_tracker.positions:
                        prices[symbol] = candles[-1].close

                self.position_tracker.update_prices(prices)

            else:
                # In live mode, sync with exchange
                exchange_positions = await self.adapter.rest.get_positions()

                # Get account balance
                balance_info = await self.adapter.rest.get_account_balance()
                self.balance = balance_info.total_balance

                # Update prices for tracked positions
                prices = {}
                for pos in exchange_positions:
                    if pos.symbol in self.position_tracker.positions:
                        prices[pos.symbol] = pos.mark_price

                self.position_tracker.update_prices(prices)

            # Log sync
            summary = self.position_tracker.get_summary()
            logger.debug(
                f"Position sync: {summary['open_positions']} open, "
                f"balance=${self.balance:,.2f}, unrealized=${summary['total_unrealized_pnl']:+.2f}"
            )

            # Periodic position state backup (live mode only)
            if not self.dry_run:
                await self._backup_position_state()

        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    async def _check_exits(self) -> None:
        """Check all positions for exit conditions."""
        exits = self.position_tracker.get_positions_to_exit()
        for symbol, reason in exits:
            await self._execute_exit(symbol, reason)

    async def _backup_trade_event(
        self,
        event_type: str,
        symbol: str,
        payload: dict,
    ) -> None:
        """
        Backup trade event to cloud (live mode only).

        Events are backed up asynchronously and don't block trading.
        """
        if not self.cloud_backup:
            return

        import hashlib
        import os
        import uuid

        try:
            timestamp = datetime.now(UTC).isoformat()
            event_id = str(uuid.uuid4())

            # Create event hash for integrity verification
            event_data = f"{event_type}:{symbol}:{timestamp}:{str(payload)}"
            event_hash = hashlib.sha256(event_data.encode()).hexdigest()[:16]

            event = {
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": timestamp,
                "symbol": symbol,
                "payload": payload,
                "sequence_number": int(datetime.now(UTC).timestamp() * 1000),
                "event_hash": event_hash,
                "instance_id": os.getenv("INSTANCE_ID", "default"),
            }

            # Non-blocking backup to all tiers
            await self.cloud_backup.backup_event(event)
            logger.debug(f"Backed up {event_type} event for {symbol}")

        except Exception as e:
            # Never let backup failures affect trading
            logger.warning(f"Cloud backup failed (non-critical): {e}")

    async def _backup_position_state(self) -> None:
        """Backup current position state to hot tier (live mode only)."""
        if not self.cloud_backup:
            return

        try:
            positions = []
            for symbol, pos in self.position_tracker.positions.items():
                positions.append({
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "size": pos.size,
                    "entry_time": pos.entry_time.isoformat(),
                    "strategy": pos.strategy,
                    "conviction": pos.conviction,
                    "order_id": pos.order_id,
                    "leverage": pos.leverage,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                })

            await self.cloud_backup.store_position_state(positions)
            logger.debug(f"Backed up {len(positions)} positions to cloud")

        except Exception as e:
            logger.warning(f"Position state backup failed (non-critical): {e}")
