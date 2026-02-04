"""
Multi-Asset Backtest Engine

Extends the production backtest engine with:
- Multiple assets trading simultaneously
- Correlation-aware position sizing
- Portfolio heat management
- Cross-asset signals

Uses PRODUCTION components (FeaturePipeline, MLEnsemble, SignalProcessor)
to ensure backtest-production parity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from src.backtest.engine import BacktestTrade, OpenPosition
from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.simulator import ExecutionSimulator
from src.data.database import Candle
from src.exchange.models import UnifiedCandle

if TYPE_CHECKING:
    from src.execution.signal_processor import SignalProcessor
    from src.features.pipeline import FeaturePipeline
    from src.ml.ensemble import MLEnsemble

logger = logging.getLogger(__name__)


# =============================================================================
# ASSET CONFIGURATION
# =============================================================================

@dataclass
class AssetConfig:
    """Configuration for each tradeable asset."""
    symbol: str
    asset_class: str  # "crypto_major", "crypto_l1", "crypto_defi", "gold_proxy"
    volatility_mult: float = 1.0
    typical_spread_bps: float = 2.0


# Default diversified crypto portfolio
DEFAULT_ASSETS = [
    AssetConfig("BTC", "crypto_major", 1.0, 1.0),
    AssetConfig("ETH", "crypto_major", 1.0, 1.5),
    AssetConfig("PAXG", "gold_proxy", 0.3, 2.0),  # Key diversifier
    AssetConfig("SOL", "crypto_l1", 1.2, 2.0),
    AssetConfig("AAVE", "crypto_defi", 1.0, 3.0),
]

# Correlation matrix for portfolio heat calculation
CORRELATIONS = {
    ("BTC", "ETH"): 0.85,
    ("BTC", "SOL"): 0.75,
    ("BTC", "AAVE"): 0.65,
    ("BTC", "PAXG"): 0.08,
    ("ETH", "SOL"): 0.80,
    ("ETH", "AAVE"): 0.72,
    ("ETH", "PAXG"): 0.06,
    ("SOL", "AAVE"): 0.60,
    ("SOL", "PAXG"): 0.05,
    ("AAVE", "PAXG"): 0.10,
}


def get_correlation(asset1: str, asset2: str) -> float:
    """Get correlation between two assets."""
    if asset1 == asset2:
        return 1.0
    key = (asset1, asset2)
    rev_key = (asset2, asset1)
    if key in CORRELATIONS:
        return CORRELATIONS[key]
    if rev_key in CORRELATIONS:
        return CORRELATIONS[rev_key]
    return 0.30  # Default moderate correlation


# =============================================================================
# MULTI-ASSET CONFIG
# =============================================================================

@dataclass
class MultiAssetBacktestConfig:
    """Configuration for multi-asset backtest."""
    assets: list[AssetConfig] = field(default_factory=lambda: DEFAULT_ASSETS.copy())
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_balance: float = 10000.0
    leverage: int = 5
    slippage_bps: float = 5.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    interval: str = "1h"
    warmup_candles: int = 100

    # Portfolio management
    max_portfolio_heat: float = 0.40  # Max 40% correlated exposure
    max_single_position: float = 0.20  # Max 20% per position
    per_trade_exposure: float = 0.10  # Base 10% per trade

    # Risk management
    stop_loss_atr: float = 1.5
    take_profit_atr: float = 2.5

    # Drawdown controls
    drawdown_reduce_10: float = 0.75
    drawdown_reduce_20: float = 0.50
    drawdown_reduce_30: float = 0.30


# =============================================================================
# MULTI-ASSET RESULT
# =============================================================================

@dataclass
class MultiAssetBacktestResult:
    """Complete multi-asset backtest results."""
    config: MultiAssetBacktestConfig
    trades: list[BacktestTrade]
    equity_curve: list[tuple[datetime, float]]
    metrics: BacktestMetrics
    per_asset_metrics: dict[str, dict]
    long_short_breakdown: dict


# =============================================================================
# MULTI-ASSET BACKTEST ENGINE
# =============================================================================

class MultiAssetBacktestEngine:
    """
    Multi-asset backtesting engine using PRODUCTION components.

    Key features:
    1. Uses FeaturePipeline.calculate_features() for each asset
    2. Uses MLEnsemble.predict() for signal generation
    3. Uses SignalProcessor.process_signal() for filtering
    4. Adds portfolio-level risk management
    """

    def __init__(
        self,
        config: MultiAssetBacktestConfig,
        feature_pipeline: FeaturePipeline,
        ml_ensemble: MLEnsemble,
        signal_processor: SignalProcessor,
    ):
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.ml_ensemble = ml_ensemble
        self.signal_processor = signal_processor
        self.simulator = ExecutionSimulator(
            slippage_bps=config.slippage_bps,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )

        # State
        self.balance = config.initial_balance
        self.peak_balance = config.initial_balance
        self.current_drawdown = 0.0
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.open_positions: dict[str, OpenPosition] = {}

        # Per-asset state
        self.candle_buffers: dict[str, list[Candle]] = {}
        self.asset_performance: dict[str, dict] = {}

    def run(
        self,
        candles_by_asset: dict[str, list[UnifiedCandle]],
    ) -> MultiAssetBacktestResult:
        """
        Run multi-asset backtest.

        Args:
            candles_by_asset: Dict of symbol -> list of UnifiedCandle

        Returns:
            MultiAssetBacktestResult
        """
        logger.info(f"Starting multi-asset backtest: {len(candles_by_asset)} assets")

        # Reset state
        self._reset_state()

        # Initialize per-asset tracking
        for symbol in candles_by_asset:
            self.candle_buffers[symbol] = []
            self.asset_performance[symbol] = {"trades": 0, "wins": 0, "pnl": 0.0}

        # Find common time range
        min_len = min(len(c) for c in candles_by_asset.values())
        logger.info(f"Processing {min_len} candles across {len(candles_by_asset)} assets")

        # Main loop
        for idx in range(min_len):
            timestamp = None

            for asset_cfg in self.config.assets:
                symbol = asset_cfg.symbol
                if symbol not in candles_by_asset:
                    continue

                candles = candles_by_asset[symbol]
                if idx >= len(candles):
                    continue

                unified_candle = candles[idx]
                timestamp = unified_candle.timestamp
                candle = self._to_db_candle(unified_candle)

                # Process this asset
                self._process_asset_candle(asset_cfg, candle, idx)

            # Update equity curve
            if timestamp:
                self._update_equity(timestamp)

        # Close remaining positions
        for symbol in list(self.open_positions.keys()):
            if symbol in candles_by_asset and candles_by_asset[symbol]:
                last_candle = candles_by_asset[symbol][-1]
                self._close_position(
                    symbol,
                    last_candle.close,
                    last_candle.timestamp,
                    "end_of_backtest",
                )

        # Calculate metrics
        metrics = MetricsCalculator.calculate(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.config.initial_balance,
            interval_minutes=self._interval_to_minutes(self.config.interval),
        )

        # Per-asset breakdown
        per_asset = {}
        for symbol, perf in self.asset_performance.items():
            if perf["trades"] > 0:
                per_asset[symbol] = {
                    "trades": perf["trades"],
                    "win_rate": perf["wins"] / perf["trades"],
                    "pnl": perf["pnl"],
                }

        # Long/short breakdown
        long_trades = [t for t in self.trades if t.side == "LONG"]
        short_trades = [t for t in self.trades if t.side == "SHORT"]
        long_short = {
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": sum(t.pnl for t in long_trades),
            "short_pnl": sum(t.pnl for t in short_trades),
            "long_win_rate": sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) if long_trades else 0,
            "short_win_rate": sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) if short_trades else 0,
        }

        logger.info(f"Backtest complete: {len(self.trades)} trades, final balance ${self.balance:,.2f}")

        return MultiAssetBacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics=metrics,
            per_asset_metrics=per_asset,
            long_short_breakdown=long_short,
        )

    def _process_asset_candle(
        self,
        asset_cfg: AssetConfig,
        candle: Candle,
        idx: int,
    ) -> None:
        """Process a single candle for one asset."""
        symbol = asset_cfg.symbol

        # Update buffer
        self.candle_buffers[symbol].append(candle)
        max_buffer = self.config.warmup_candles + 50
        if len(self.candle_buffers[symbol]) > max_buffer:
            self.candle_buffers[symbol] = self.candle_buffers[symbol][-max_buffer:]

        # Check exits for this asset
        if symbol in self.open_positions:
            self._check_exits(symbol, candle)

        # Need warmup
        if len(self.candle_buffers[symbol]) < self.config.warmup_candles:
            return

        # Calculate features using PRODUCTION pipeline
        try:
            features = self.feature_pipeline.calculate_features(
                candles=self.candle_buffers[symbol],
                use_cache=False,
            )
        except Exception as e:
            logger.debug(f"Feature calculation error for {symbol}: {e}")
            return

        # Get regime
        regime = self._get_regime(features)

        # Get ML prediction using PRODUCTION ensemble
        try:
            signal, confidence, model_outputs, _ = self.ml_ensemble.predict(
                features=features,
                current_regime=regime,
            )
        except Exception as e:
            logger.debug(f"ML prediction error for {symbol}: {e}")
            return

        # Skip if HOLD or low confidence
        if signal == "HOLD" or confidence < 0.5:
            return

        # Process signal using PRODUCTION processor
        try:
            model_agreement = self._calculate_agreement(model_outputs)
            signal_result = self.signal_processor.process_signal(
                raw_signal=signal,
                confidence=confidence,
                features=features,
                model_agreement=model_agreement,
            )
        except Exception as e:
            logger.debug(f"Signal processing error for {symbol}: {e}")
            return

        # Execute if actionable and no position in this asset
        if (
            signal_result.signal != "HOLD"
            and signal_result.position_scale > 0
            and symbol not in self.open_positions
        ):
            self._execute_signal(
                asset_cfg=asset_cfg,
                side=signal_result.signal,
                confidence=signal_result.confidence,
                position_scale=signal_result.position_scale,
                candle=candle,
                features=features,
                regime=regime,
            )

    def _execute_signal(
        self,
        asset_cfg: AssetConfig,
        side: str,
        confidence: float,
        position_scale: float,
        candle: Candle,
        features: dict,
        regime: str,
    ) -> None:
        """Execute a trading signal with portfolio-aware sizing."""
        symbol = asset_cfg.symbol

        # Check portfolio heat
        current_heat = self._calculate_portfolio_heat()
        if current_heat >= self.config.max_portfolio_heat:
            logger.debug(f"Portfolio heat {current_heat:.2%} exceeds max, skipping {symbol}")
            return

        # Drawdown adjustment
        dd_mult = self._get_drawdown_multiplier()

        # Base position size
        exposure = self.config.per_trade_exposure * position_scale * dd_mult
        exposure = min(exposure, self.config.max_single_position)

        # Volatility adjustment
        exposure /= asset_cfg.volatility_mult

        # Correlation adjustment
        for existing_symbol in self.open_positions:
            corr = get_correlation(symbol, existing_symbol)
            if corr > 0.6:
                exposure *= (1 - corr * 0.3)

        # Calculate notional and size
        notional = self.balance * exposure * self.config.leverage
        size = notional / candle.close

        # Get ATR
        atr = features.get("atr", candle.close * 0.02)
        atr_ratio = atr / candle.close

        # Simulate execution
        fill = self.simulator.execute_market_order(
            symbol=symbol,
            side=side,
            size=size,
            current_price=candle.close,
            timestamp=candle.timestamp,
            atr_ratio=atr_ratio,
        )

        # Calculate stops
        if side == "LONG":
            stop_loss = fill.entry_price * (1 - self.config.stop_loss_atr * atr_ratio)
            take_profit = fill.entry_price * (1 + self.config.take_profit_atr * atr_ratio)
        else:
            stop_loss = fill.entry_price * (1 + self.config.stop_loss_atr * atr_ratio)
            take_profit = fill.entry_price * (1 - self.config.take_profit_atr * atr_ratio)

        # Create position
        self.open_positions[symbol] = OpenPosition(
            id=fill.order_id,
            symbol=symbol,
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

        self.balance -= fill.fees
        logger.debug(f"Opened {side} {symbol} @ {fill.entry_price:.2f}, size={fill.size:.4f}")

    def _check_exits(self, symbol: str, candle: Candle) -> None:
        """Check if position should exit."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]

        if pos.side == "LONG":
            if candle.low <= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, candle.timestamp, "stop_loss")
            elif candle.high >= pos.take_profit:
                self._close_position(symbol, pos.take_profit, candle.timestamp, "take_profit")
        else:
            if candle.high >= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, candle.timestamp, "stop_loss")
            elif candle.low <= pos.take_profit:
                self._close_position(symbol, pos.take_profit, candle.timestamp, "take_profit")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> None:
        """Close a position."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        atr_ratio = pos.atr / exit_price if exit_price > 0 else 0.01

        adjusted_exit, exit_fees = self.simulator.simulate_exit(
            side=pos.side,
            size=pos.size,
            exit_price=exit_price,
            atr_ratio=atr_ratio,
        )

        # Calculate gross PnL (price difference only)
        if pos.side == "LONG":
            gross_pnl = (adjusted_exit - pos.entry_price) * pos.size
        else:
            gross_pnl = (pos.entry_price - adjusted_exit) * pos.size

        # Entry fees already deducted when opening, so only subtract exit fees
        self.balance += gross_pnl - exit_fees

        # Net PnL for trade record (includes both fees)
        pnl = gross_pnl - pos.entry_fees - exit_fees

        # Record trade
        trade = BacktestTrade(
            id=pos.id,
            symbol=symbol,
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

        # Update per-asset stats
        self.asset_performance[symbol]["trades"] += 1
        self.asset_performance[symbol]["pnl"] += pnl
        if pnl > 0:
            self.asset_performance[symbol]["wins"] += 1

        del self.open_positions[symbol]
        logger.debug(f"Closed {pos.side} {symbol} ({reason}): PnL=${pnl:.2f}")

    def _calculate_portfolio_heat(self) -> float:
        """Calculate correlation-adjusted portfolio exposure."""
        if not self.open_positions:
            return 0.0

        positions = list(self.open_positions.values())
        if len(positions) == 1:
            return positions[0].size * positions[0].entry_price / self.balance

        total_heat = 0.0
        for i, pos1 in enumerate(positions):
            for pos2 in positions:
                corr = get_correlation(pos1.symbol, pos2.symbol)
                exp1 = (pos1.size * pos1.entry_price) / self.balance
                exp2 = (pos2.size * pos2.entry_price) / self.balance

                if pos1.side == pos2.side:
                    total_heat += corr * exp1 * exp2
                else:
                    total_heat -= corr * exp1 * exp2

        return np.sqrt(max(0, total_heat))

    def _get_drawdown_multiplier(self) -> float:
        """Get position size multiplier based on drawdown."""
        if self.current_drawdown > 0.30:
            return self.config.drawdown_reduce_30
        elif self.current_drawdown > 0.20:
            return self.config.drawdown_reduce_20
        elif self.current_drawdown > 0.10:
            return self.config.drawdown_reduce_10
        return 1.0

    def _update_equity(self, timestamp: datetime) -> None:
        """Update equity curve and drawdown."""
        equity = self.balance

        for symbol, pos in self.open_positions.items():
            if symbol in self.candle_buffers and self.candle_buffers[symbol]:
                current_price = self.candle_buffers[symbol][-1].close
                if pos.side == "LONG":
                    equity += (current_price - pos.entry_price) * pos.size
                else:
                    equity += (pos.entry_price - current_price) * pos.size

        self.equity_curve.append((timestamp, equity))

        if equity > self.peak_balance:
            self.peak_balance = equity
        self.current_drawdown = (self.peak_balance - equity) / self.peak_balance

    def _reset_state(self) -> None:
        """Reset engine state for new run."""
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        self.current_drawdown = 0.0
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        self.candle_buffers = {}
        self.asset_performance = {}

    def _get_regime(self, features: dict) -> str:
        """Extract regime from features."""
        vol_regime = features.get("vol_regime", "")
        if vol_regime:
            return vol_regime
        adx = features.get("adx", 25)
        if adx > 40:
            return "trending"
        elif adx < 20:
            return "ranging"
        return "unknown"

    def _calculate_agreement(self, model_outputs: dict[str, float]) -> float:
        """Calculate model agreement."""
        if not model_outputs:
            return 0.0
        values = list(model_outputs.values())
        bullish = sum(1 for v in values if v > 0.5)
        bearish = sum(1 for v in values if v < 0.5)
        return max(bullish, bearish) / len(values)

    def _to_db_candle(self, unified: UnifiedCandle) -> Candle:
        """Convert UnifiedCandle to Candle."""
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

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval to minutes."""
        mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        return mapping.get(interval, 60)
