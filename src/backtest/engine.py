"""
Backtest Engine

Core backtesting engine that orchestrates signal generation and simulated execution.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.simulator import ExecutionSimulator, SimulatedFill
from src.data.database import Candle
from src.exchange.models import UnifiedCandle

if TYPE_CHECKING:
    from src.execution.signal_processor import SignalProcessor
    from src.features.pipeline import FeaturePipeline
    from src.ml.ensemble import MLEnsemble

logger = logging.getLogger(__name__)


def unified_to_db_candle(unified: UnifiedCandle) -> Candle:
    """Convert UnifiedCandle to database Candle format."""
    return Candle(
        symbol=unified.symbol,
        timestamp=unified.timestamp,
        open=unified.open,
        high=unified.high,
        low=unified.low,
        close=unified.close,
        volume=unified.volume,
        interval=unified.interval,
    )


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    leverage: int = 5
    slippage_bps: float = 5.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    interval: str = "1m"
    warmup_candles: int = 100

    # Position sizing
    per_trade_exposure: float = 0.10  # 10% of balance per trade
    max_position_size: float = 0.20  # 20% max

    # Exit parameters (ATR multiples)
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 1.5


@dataclass
class BacktestTrade:
    """Record of a completed backtest trade."""

    id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float
    confidence: float
    regime: str


@dataclass
class OpenPosition:
    """Currently open position during backtest."""

    id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    entry_fees: float
    confidence: float
    regime: str
    atr: float  # ATR at entry for exit calculations


@dataclass
class BacktestResult:
    """Complete backtest results."""

    config: BacktestConfig
    trades: list[BacktestTrade]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics


class BacktestEngine:
    """
    Core backtesting engine.

    Mirrors the live trading pipeline from main.py:
    1. Calculate features from candle buffer
    2. Generate ML prediction
    3. Process signal through filters
    4. Simulate execution if signal is actionable
    5. Track open positions and check exits
    6. Record equity curve
    """

    def __init__(
        self,
        config: BacktestConfig,
        feature_pipeline: FeaturePipeline,
        ml_ensemble: MLEnsemble,
        signal_processor: SignalProcessor,
        execution_simulator: ExecutionSimulator | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            feature_pipeline: Feature calculation pipeline
            ml_ensemble: ML ensemble for predictions
            signal_processor: Signal filter and processor
            execution_simulator: Order execution simulator
        """
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.ml_ensemble = ml_ensemble
        self.signal_processor = signal_processor
        self.simulator = execution_simulator or ExecutionSimulator(
            slippage_bps=config.slippage_bps,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )

        # State
        self.balance = config.initial_balance
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.open_position: OpenPosition | None = None

        # Candle buffer for feature calculation
        self.candle_buffer: list[Candle] = []

    def run(self, candles: list[UnifiedCandle]) -> BacktestResult:
        """
        Run backtest over historical candles.

        Args:
            candles: List of candles ordered by timestamp ascending

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        logger.info(f"Starting backtest: {len(candles)} candles, {self.config.symbol}")
        logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")

        # Reset state
        self.balance = self.config.initial_balance
        self.trades = []
        self.equity_curve = []
        self.open_position = None
        self.candle_buffer = []

        # Process each candle
        for i, unified_candle in enumerate(candles):
            candle = unified_to_db_candle(unified_candle)
            self._process_candle(candle)

            # Progress logging
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(candles)} candles")

        # Close any remaining position at last price
        if self.open_position and candles:
            last_candle = unified_to_db_candle(candles[-1])
            self._close_position(last_candle.close, last_candle.timestamp, "end_of_backtest")

        # Calculate metrics
        interval_minutes = self._interval_to_minutes(self.config.interval)
        metrics = MetricsCalculator.calculate(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.config.initial_balance,
            interval_minutes=interval_minutes,
        )

        logger.info(f"Backtest complete: {len(self.trades)} trades")
        logger.info(f"Final balance: ${self.balance:,.2f} ({metrics.total_return:+.2f}%)")

        return BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics=metrics,
        )

    def _process_candle(self, candle: Candle) -> None:
        """Process a single candle through the trading pipeline."""
        # Add to buffer
        self.candle_buffer.append(candle)

        # Keep buffer size manageable
        max_buffer = self.config.warmup_candles + 50
        if len(self.candle_buffer) > max_buffer:
            self.candle_buffer = self.candle_buffer[-max_buffer:]

        # Check exits first (before generating new signals)
        if self.open_position:
            self._check_exits(candle)

        # Need warmup period before generating signals
        if len(self.candle_buffer) < self.config.warmup_candles:
            self._update_equity(candle.timestamp)
            return

        # Calculate features
        try:
            features = self.feature_pipeline.calculate_features(
                candles=self.candle_buffer,
                use_cache=False,
            )
        except Exception as e:
            logger.debug(f"Feature calculation error: {e}")
            self._update_equity(candle.timestamp)
            return

        # Get regime from features
        regime = self._get_regime(features)

        # Get ML prediction
        try:
            signal, confidence, model_outputs, weighted_avg = self.ml_ensemble.predict(
                features=features,
                current_regime=regime,
            )
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
            self._update_equity(candle.timestamp)
            return

        # Skip if HOLD or low confidence
        if signal == "HOLD" or confidence < 0.5:
            self._update_equity(candle.timestamp)
            return

        # Calculate model agreement
        model_agreement = self._calculate_agreement(model_outputs)

        # Process signal through filters
        try:
            signal_result = self.signal_processor.process_signal(
                raw_signal=signal,
                confidence=confidence,
                features=features,
                model_agreement=model_agreement,
            )
        except Exception as e:
            logger.debug(f"Signal processing error: {e}")
            self._update_equity(candle.timestamp)
            return

        # Execute if actionable and no open position
        if (
            signal_result.signal != "HOLD"
            and signal_result.position_scale > 0
            and self.open_position is None
        ):
            self._execute_signal(
                side=signal_result.signal,
                confidence=signal_result.confidence,
                position_scale=signal_result.position_scale,
                candle=candle,
                features=features,
                regime=regime,
            )

        self._update_equity(candle.timestamp)

    def _execute_signal(
        self,
        side: str,
        confidence: float,
        position_scale: float,
        candle: Candle,
        features: dict,
        regime: str,
    ) -> None:
        """Execute a trading signal."""
        # Calculate position size
        exposure = self.config.per_trade_exposure * position_scale
        exposure = min(exposure, self.config.max_position_size)
        notional = self.balance * exposure * self.config.leverage
        size = notional / candle.close

        # Get ATR for stop/take profit
        atr = features.get("atr", candle.close * 0.01)  # Default 1% if missing
        atr_ratio = atr / candle.close if candle.close > 0 else 0.01

        # Simulate execution
        fill = self.simulator.execute_market_order(
            symbol=self.config.symbol,
            side=side,
            size=size,
            current_price=candle.close,
            timestamp=candle.timestamp,
            atr_ratio=atr_ratio,
        )

        # Calculate stop loss and take profit
        if side == "LONG":
            stop_loss = fill.entry_price * (1 - self.config.stop_loss_atr * atr_ratio)
            take_profit = fill.entry_price * (1 + self.config.take_profit_atr * atr_ratio)
        else:
            stop_loss = fill.entry_price * (1 + self.config.stop_loss_atr * atr_ratio)
            take_profit = fill.entry_price * (1 - self.config.take_profit_atr * atr_ratio)

        # Create open position
        self.open_position = OpenPosition(
            id=fill.order_id,
            symbol=fill.symbol,
            side=side,
            entry_price=fill.entry_price,
            size=fill.size,
            entry_time=fill.timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_fees=fill.fees,
            confidence=confidence,
            regime=regime,
            atr=atr,
        )

        # Deduct fees from balance
        self.balance -= fill.fees

        logger.debug(
            f"Opened {side} position: {fill.size:.6f} @ {fill.entry_price:.2f}, "
            f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
        )

    def _check_exits(self, candle: Candle) -> None:
        """Check if open position should be closed."""
        if not self.open_position:
            return

        pos = self.open_position

        # Check stop loss
        if pos.side == "LONG":
            if candle.low <= pos.stop_loss:
                self._close_position(pos.stop_loss, candle.timestamp, "stop_loss")
                return
            if candle.high >= pos.take_profit:
                self._close_position(pos.take_profit, candle.timestamp, "take_profit")
                return
        else:  # SHORT
            if candle.high >= pos.stop_loss:
                self._close_position(pos.stop_loss, candle.timestamp, "stop_loss")
                return
            if candle.low <= pos.take_profit:
                self._close_position(pos.take_profit, candle.timestamp, "take_profit")
                return

    def _close_position(
        self,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> None:
        """Close the open position."""
        if not self.open_position:
            return

        pos = self.open_position

        # Simulate exit
        atr_ratio = pos.atr / exit_price if exit_price > 0 else 0.01
        adjusted_exit, exit_fees = self.simulator.simulate_exit(
            side=pos.side,
            size=pos.size,
            exit_price=exit_price,
            atr_ratio=atr_ratio,
        )

        # Calculate P&L
        pnl = self.simulator.calculate_pnl(
            entry_price=pos.entry_price,
            exit_price=adjusted_exit,
            size=pos.size,
            side=pos.side,
            entry_fees=pos.entry_fees,
            exit_fees=exit_fees,
        )

        # Update balance
        self.balance += pnl + (pos.entry_price * pos.size)  # Return notional + pnl
        self.balance -= exit_fees

        # Record trade
        trade = BacktestTrade(
            id=pos.id,
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=adjusted_exit,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            fees=pos.entry_fees + exit_fees,
            confidence=pos.confidence,
            regime=pos.regime,
        )
        self.trades.append(trade)

        logger.debug(
            f"Closed {pos.side} position ({reason}): "
            f"Exit @ {adjusted_exit:.2f}, PnL: ${pnl:.2f}"
        )

        self.open_position = None

    def _update_equity(self, timestamp: datetime) -> None:
        """Update equity curve with current balance + unrealized P&L."""
        equity = self.balance

        # Add unrealized P&L if position is open
        if self.open_position:
            # Use the latest candle's close for mark-to-market
            if self.candle_buffer:
                current_price = self.candle_buffer[-1].close
                pos = self.open_position
                if pos.side == "LONG":
                    unrealized = (current_price - pos.entry_price) * pos.size
                else:
                    unrealized = (pos.entry_price - current_price) * pos.size
                equity += unrealized

        self.equity_curve.append((timestamp, equity))

    def _get_regime(self, features: dict) -> str:
        """Extract market regime from features."""
        # Use volatility regime if available
        vol_regime = features.get("vol_regime", "")
        if vol_regime:
            return vol_regime

        # Fallback: derive from ADX
        adx = features.get("adx", 25)
        if adx > 40:
            return "trending"
        elif adx < 20:
            return "ranging"
        else:
            return "unknown"

    def _calculate_agreement(self, model_outputs: dict[str, float]) -> float:
        """Calculate model agreement percentage."""
        if not model_outputs:
            return 0.0

        values = list(model_outputs.values())
        if not values:
            return 0.0

        # Count how many models agree on direction
        bullish = sum(1 for v in values if v > 0.5)
        bearish = sum(1 for v in values if v < 0.5)

        # Agreement is the proportion of models that agree with majority
        majority = max(bullish, bearish)
        return majority / len(values) if values else 0.0

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        return mapping.get(interval, 1)
