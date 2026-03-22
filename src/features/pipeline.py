"""
AlphaStrike Trading Bot - Feature Pipeline Orchestrator (US-008)

Orchestrates all feature calculations for ML models, combining:
- Technical indicators (35 features)
- Microstructure features (7 features)
- Fee features (5 features)
- Cross-asset features (3 features)
- Time features (5 features)
- Volatility features (4 features)

Total: 59 base features with multi-period variants for expanded feature set.

Features:
- Unified interface for all feature calculations
- Caching with configurable TTL (default 250ms)
- Graceful handling of missing data
- Standardized naming convention (lowercase with underscores)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.data.database import Candle
from src.features.microstructure import (
    MicrostructureData,
    MicrostructureFeatures,
    OrderbookCache,
    OrderbookSnapshot,
    calculate_top5_orderbook_imbalance,
    orderbook_from_unified,
)
from src.features.technical import TechnicalFeatures

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TickerData:
    """Ticker data for fee and correlation calculations."""

    symbol: str
    last_price: float
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    funding_rate: float = 0.0
    timestamp: float = 0.0


@dataclass
class OrderbookData:
    """Simplified orderbook data for pipeline."""

    bids: list[tuple[float, float]]  # List of (price, quantity)
    asks: list[tuple[float, float]]  # List of (price, quantity)
    timestamp: float = 0.0

    def to_snapshot(self) -> OrderbookSnapshot:
        """Convert to OrderbookSnapshot for microstructure calculations."""
        return OrderbookSnapshot(
            bids=self.bids,
            asks=self.asks,
            timestamp=self.timestamp,
        )


@dataclass
class FeeConfig:
    """Fee configuration for the trading venue."""

    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0005  # 0.05% taker fee


@dataclass
class CrossAssetData:
    """Cross-asset price data for correlation calculations."""

    btc_prices: NDArray[np.float64] | None = None
    eth_prices: NDArray[np.float64] | None = None
    market_index: NDArray[np.float64] | None = None  # Composite market index


@dataclass
class CachedFeatures:
    """Cached feature calculation result."""

    features: dict[str, float]
    timestamp: float
    ttl_ms: float

    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        age_ms = (time.time() - self.timestamp) * 1000
        return age_ms < self.ttl_ms


# =============================================================================
# Fee Feature Calculator
# =============================================================================


class FeeFeatureCalculator:
    """Calculate fee-related features."""

    def __init__(self, config: FeeConfig | None = None):
        """
        Initialize fee calculator.

        Args:
            config: Fee configuration (uses defaults if None)
        """
        self.config = config or FeeConfig()

    def calculate(
        self,
        ticker: TickerData | None,
        atr: float,
        close_price: float,
    ) -> dict[str, float]:
        """
        Calculate all fee-related features.

        Args:
            ticker: Current ticker data
            atr: Current ATR value
            close_price: Current close price

        Returns:
            Dictionary of fee features
        """
        features: dict[str, float] = {}

        # Base fees (normalized to bps)
        features["maker_fee"] = self.config.maker_fee * 10000  # Convert to bps
        features["taker_fee"] = self.config.taker_fee * 10000

        # Funding impact (if available)
        if ticker and ticker.funding_rate != 0:
            # Normalize funding rate: typical range is -0.01% to +0.01% per 8 hours
            # Scale to [-1, 1] range
            features["funding_impact"] = float(
                np.clip(ticker.funding_rate / 0.01, -1.0, 1.0)
            )
        else:
            features["funding_impact"] = 0.0

        # Fee drag: total round-trip cost as percentage of typical move
        round_trip_fee = self.config.maker_fee + self.config.taker_fee
        if atr > 0 and close_price > 0:
            # ATR ratio (volatility as % of price)
            atr_pct = atr / close_price
            # Fee drag = fees / expected move
            features["fee_drag"] = float(
                np.clip(round_trip_fee / max(atr_pct, 0.001), 0.0, 1.0)
            )
        else:
            features["fee_drag"] = 0.5  # Neutral value

        # Breakeven move: minimum price move to cover fees (as % of price)
        features["breakeven_move"] = round_trip_fee * 100  # As percentage

        return features


# =============================================================================
# Cross-Asset Feature Calculator
# =============================================================================


class CrossAssetFeatureCalculator:
    """Calculate cross-asset correlation features."""

    def __init__(self, correlation_window: int = 20):
        """
        Initialize cross-asset calculator.

        Args:
            correlation_window: Window size for correlation calculation
        """
        self.correlation_window = correlation_window

    def calculate(
        self,
        close_prices: NDArray[np.float64],
        cross_asset_data: CrossAssetData | None,
    ) -> dict[str, float]:
        """
        Calculate cross-asset correlation features.

        Args:
            close_prices: Current asset's close prices
            cross_asset_data: Price data for correlated assets

        Returns:
            Dictionary of correlation features
        """
        features: dict[str, float] = {}

        # Default to neutral correlations
        features["btc_correlation"] = 0.0
        features["eth_correlation"] = 0.0
        features["market_correlation"] = 0.0

        if cross_asset_data is None or len(close_prices) < self.correlation_window:
            return features

        # Calculate returns for correlation
        if len(close_prices) < self.correlation_window + 1:
            return features

        returns = np.diff(close_prices[-self.correlation_window - 1 :])
        if len(returns) < self.correlation_window:
            return features

        returns = returns[-self.correlation_window :]

        # BTC correlation
        if (
            cross_asset_data.btc_prices is not None
            and len(cross_asset_data.btc_prices) >= self.correlation_window + 1
        ):
            btc_returns = np.diff(
                cross_asset_data.btc_prices[-self.correlation_window - 1 :]
            )
            btc_returns = btc_returns[-self.correlation_window :]
            if len(btc_returns) == len(returns):
                corr = self._calculate_correlation(returns, btc_returns)
                features["btc_correlation"] = corr

        # ETH correlation
        if (
            cross_asset_data.eth_prices is not None
            and len(cross_asset_data.eth_prices) >= self.correlation_window + 1
        ):
            eth_returns = np.diff(
                cross_asset_data.eth_prices[-self.correlation_window - 1 :]
            )
            eth_returns = eth_returns[-self.correlation_window :]
            if len(eth_returns) == len(returns):
                corr = self._calculate_correlation(returns, eth_returns)
                features["eth_correlation"] = corr

        # Market index correlation
        if (
            cross_asset_data.market_index is not None
            and len(cross_asset_data.market_index) >= self.correlation_window + 1
        ):
            market_returns = np.diff(
                cross_asset_data.market_index[-self.correlation_window - 1 :]
            )
            market_returns = market_returns[-self.correlation_window :]
            if len(market_returns) == len(returns):
                corr = self._calculate_correlation(returns, market_returns)
                features["market_correlation"] = corr

        return features

    def _calculate_correlation(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            x: First array
            y: Second array

        Returns:
            Correlation coefficient in [-1, 1]
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x_std = np.std(x)
        y_std = np.std(y)

        if x_std == 0 or y_std == 0:
            return 0.0

        covariance = np.mean((x - x_mean) * (y - y_mean))
        correlation = covariance / (x_std * y_std)

        return float(np.clip(correlation, -1.0, 1.0))


# =============================================================================
# Time Feature Calculator
# =============================================================================


class TimeFeatureCalculator:
    """Calculate time-based features."""

    # Trading session definitions (UTC)
    SESSION_ASIA = (0, 8)  # 00:00 - 08:00 UTC
    SESSION_EUROPE = (8, 16)  # 08:00 - 16:00 UTC
    SESSION_US = (16, 24)  # 16:00 - 24:00 UTC

    def calculate(self, timestamp: datetime | None = None) -> dict[str, float]:
        """
        Calculate time-based features.

        Args:
            timestamp: Timestamp for features (uses current time if None)

        Returns:
            Dictionary of time features
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        features: dict[str, float] = {}

        # Session indicator: 0 = Asia, 1 = Europe, 2 = US
        hour = timestamp.hour
        if self.SESSION_ASIA[0] <= hour < self.SESSION_ASIA[1]:
            features["session_indicator"] = 0.0
        elif self.SESSION_EUROPE[0] <= hour < self.SESSION_EUROPE[1]:
            features["session_indicator"] = 1.0
        else:
            features["session_indicator"] = 2.0

        # Cyclical encoding of hour (0-23)
        hour_rad = (hour / 24.0) * 2 * math.pi
        features["hour_sin"] = math.sin(hour_rad)
        features["hour_cos"] = math.cos(hour_rad)

        # Cyclical encoding of day of week (0-6, Monday=0)
        day_of_week = timestamp.weekday()
        day_rad = (day_of_week / 7.0) * 2 * math.pi
        features["day_sin"] = math.sin(day_rad)
        features["day_cos"] = math.cos(day_rad)

        return features


# =============================================================================
# Volatility Feature Calculator
# =============================================================================


class VolatilityFeatureCalculator:
    """Calculate volatility-related features beyond basic ATR."""

    def __init__(
        self,
        atr_period: int = 14,
        vol_lookback: int = 20,
        percentile_lookback: int = 100,
    ):
        """
        Initialize volatility calculator.

        Args:
            atr_period: Period for ATR calculation
            vol_lookback: Lookback for realized volatility
            percentile_lookback: Lookback for percentile ranking
        """
        self.atr_period = atr_period
        self.vol_lookback = vol_lookback
        self.percentile_lookback = percentile_lookback

    def calculate(
        self,
        close_prices: NDArray[np.float64],
        high_prices: NDArray[np.float64],
        low_prices: NDArray[np.float64],
        atr_values: NDArray[np.float64],
    ) -> dict[str, float]:
        """
        Calculate volatility features.

        Args:
            close_prices: Close prices array
            high_prices: High prices array
            low_prices: Low prices array
            atr_values: Pre-calculated ATR values

        Returns:
            Dictionary of volatility features
        """
        features: dict[str, float] = {}

        n = len(close_prices)

        # ATR relative: current ATR / historical average ATR
        # (Different from technical atr_ratio which is ATR/price)
        if n >= self.percentile_lookback and len(atr_values) >= self.percentile_lookback:
            current_atr = atr_values[-1]
            avg_atr = np.mean(atr_values[-self.percentile_lookback :])
            if avg_atr > 0:
                features["atr_relative"] = float(
                    np.clip(current_atr / avg_atr, 0.0, 3.0)
                )
            else:
                features["atr_relative"] = 1.0
        else:
            features["atr_relative"] = 1.0

        # Realized volatility (annualized standard deviation of returns)
        if n >= self.vol_lookback + 1:
            returns = np.diff(np.log(close_prices[-self.vol_lookback - 1 :]))
            if len(returns) >= self.vol_lookback:
                returns = returns[-self.vol_lookback :]
                # Annualize assuming 1-minute candles
                realized_vol = np.std(returns) * np.sqrt(525600)  # Minutes in a year
                features["realized_vol"] = float(np.clip(realized_vol, 0.0, 5.0))
            else:
                features["realized_vol"] = 0.0
        else:
            features["realized_vol"] = 0.0

        # Volatility regime: 0 = low, 1 = medium, 2 = high
        # Based on ATR percentile ranking
        if n >= self.percentile_lookback and len(atr_values) >= self.percentile_lookback:
            current_atr = atr_values[-1]
            atr_window = atr_values[-self.percentile_lookback :]
            percentile = (
                np.sum(atr_window < current_atr) / len(atr_window)
            ) * 100

            if percentile < 33:
                features["vol_regime"] = 0.0
            elif percentile < 67:
                features["vol_regime"] = 1.0
            else:
                features["vol_regime"] = 2.0

            features["vol_percentile"] = float(percentile / 100.0)
        else:
            features["vol_regime"] = 1.0  # Default to medium
            features["vol_percentile"] = 0.5  # Default to median

        return features


# =============================================================================
# Main Feature Pipeline
# =============================================================================


@dataclass
class FeaturePipelineConfig:
    """Configuration for feature pipeline."""

    cache_ttl_ms: float = 250.0  # Cache TTL in milliseconds
    min_candles: int = 100  # Minimum candles for feature calculation
    correlation_window: int = 20  # Window for cross-asset correlations
    vol_lookback: int = 20  # Lookback for realized volatility
    percentile_lookback: int = 100  # Lookback for percentile calculations


class FeaturePipeline:
    """
    Orchestrates all feature calculations for ML models.

    Combines:
    - Technical indicators (35 features)
    - Microstructure features (7 features)
    - Fee features (5 features)
    - Cross-asset features (3 features)
    - Time features (5 features)
    - Volatility features (4 features)

    The Simons Protocol uses a reduced feature set (25 features) for
    signal generation, accessible via `get_core_features()`.

    Usage:
        pipeline = FeaturePipeline()

        # Calculate all features (59)
        features = pipeline.calculate_features(candles=candle_list)

        # Get core features only (25) for ML signal generation
        core_features = pipeline.get_core_features(candles=candle_list)

        # Get timing features for entry optimization
        timing_features = pipeline.get_timing_features(candles=candle_list)
    """

    # Tier 1+2 Core Features (25) for signal generation
    # These have the highest predictive power and should be used for ML models
    CORE_FEATURE_NAMES = [
        # Tier 1 - Signal Drivers
        "rsi",  # RSI momentum
        "rsi_slope",  # RSI direction
        "adx",  # Trend strength
        "plus_di",  # Positive directional indicator
        "minus_di",  # Negative directional indicator
        "ema_short_ratio",  # Price vs short EMA
        "ema_long_ratio",  # Price vs long EMA
        "obv_slope",  # On-balance volume trend
        "volume_ratio",  # Volume confirmation
        # Tier 2 - Confirmation Features
        "bb_position",  # Bollinger Band position
        "bb_bandwidth",  # Volatility via BB width
        "macd_histogram",  # MACD signal direction
        "atr",  # Average True Range
        "atr_ratio",  # ATR relative to price
        "atr_relative",  # Current vs historical ATR
        "vol_regime",  # Volatility regime (0/1/2)
        "funding_rate",  # Funding rate sentiment
        "orderbook_imbalance",  # Order flow imbalance
        "trade_flow_imbalance",  # Trade direction flow
        "btc_correlation",  # Market correlation
        # Trend confirmation
        "ema_cross",  # EMA crossover signal
        "trend_strength",  # Combined trend metric
        "momentum",  # Price momentum
        "price_position",  # Price in range (0-1)
        "stoch_k",  # Stochastic oscillator
    ]

    # Tier 3 - Timing Features for entry optimization (used separately)
    TIMING_FEATURE_NAMES = [
        "session_indicator",  # Trading session (Asia/EU/US)
        "hour_sin",  # Hour cyclical encoding
        "hour_cos",  # Hour cyclical encoding
        "vol_percentile",  # Volatility percentile
        "realized_vol",  # Realized volatility
    ]

    def __init__(
        self,
        config: FeaturePipelineConfig | None = None,
        fee_config: FeeConfig | None = None,
    ):
        """
        Initialize feature pipeline.

        Args:
            config: Pipeline configuration
            fee_config: Fee configuration for the venue
        """
        self.config = config or FeaturePipelineConfig()
        self.fee_config = fee_config or FeeConfig()

        # Initialize sub-calculators
        self._technical = TechnicalFeatures()
        self._microstructure = MicrostructureFeatures()
        self._fee_calculator = FeeFeatureCalculator(self.fee_config)
        self._cross_asset_calculator = CrossAssetFeatureCalculator(
            self.config.correlation_window
        )
        self._time_calculator = TimeFeatureCalculator()
        self._volatility_calculator = VolatilityFeatureCalculator(
            vol_lookback=self.config.vol_lookback,
            percentile_lookback=self.config.percentile_lookback,
        )

        # Cache
        self._cache: dict[str, CachedFeatures] = {}

        # Feature names (lazily computed)
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        """Get list of all feature names in consistent order."""
        if self._feature_names is None:
            self._feature_names = self._generate_feature_names()
        return self._feature_names

    @property
    def feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

    def _generate_feature_names(self) -> list[str]:
        """Generate complete list of feature names."""
        names: list[str] = []

        # Technical features (35)
        names.extend(self._technical.feature_names)

        # Microstructure features (8)
        names.extend([
            "orderbook_imbalance",
            "funding_rate",
            "open_interest",
            "open_interest_change",
            "volume_profile",
            "large_trade_ratio",
            "trade_flow_imbalance",
            "top5_orderbook_imbalance",
        ])

        # Fee features (5)
        names.extend([
            "maker_fee",
            "taker_fee",
            "funding_impact",
            "fee_drag",
            "breakeven_move",
        ])

        # Cross-asset features (3)
        names.extend([
            "btc_correlation",
            "eth_correlation",
            "market_correlation",
        ])

        # Time features (5)
        names.extend([
            "session_indicator",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ])

        # Volatility features (4)
        names.extend([
            "atr_relative",
            "realized_vol",
            "vol_regime",
            "vol_percentile",
        ])

        return names

    def calculate_features(
        self,
        candles: list[Candle],
        ticker_data: TickerData | None = None,
        orderbook_data: OrderbookData | None = None,
        cross_asset_data: CrossAssetData | None = None,
        microstructure_data: MicrostructureData | None = None,
        timestamp: datetime | None = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """
        Calculate all features from input data.

        Args:
            candles: List of OHLCV candles (most recent last)
            ticker_data: Current ticker data for fee calculations
            orderbook_data: Current orderbook snapshot
            cross_asset_data: Cross-asset price data for correlations
            microstructure_data: Pre-built microstructure data (optional)
            timestamp: Timestamp for time features (uses current time if None)
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary mapping feature names to float values
        """
        # Generate cache key
        cache_key = self._generate_cache_key(candles, ticker_data)

        # Check cache
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.is_valid():
                logger.debug("Using cached features")
                return cached.features

        # Calculate features
        features = self._calculate_all_features(
            candles=candles,
            ticker_data=ticker_data,
            orderbook_data=orderbook_data,
            cross_asset_data=cross_asset_data,
            microstructure_data=microstructure_data,
            timestamp=timestamp,
        )

        # Update cache
        self._cache[cache_key] = CachedFeatures(
            features=features,
            timestamp=time.time(),
            ttl_ms=self.config.cache_ttl_ms,
        )

        # Prune old cache entries
        self._prune_cache()

        return features

    def _calculate_all_features(
        self,
        candles: list[Candle],
        ticker_data: TickerData | None,
        orderbook_data: OrderbookData | None,
        cross_asset_data: CrossAssetData | None,
        microstructure_data: MicrostructureData | None,
        timestamp: datetime | None,
    ) -> dict[str, float]:
        """Calculate all features (internal implementation)."""
        features: dict[str, float] = {}

        # Convert candles to numpy arrays
        if len(candles) < self.config.min_candles:
            logger.warning(
                f"Insufficient candles: {len(candles)} < {self.config.min_candles}"
            )
            # Return default values for all features
            return self._get_default_features()

        ohlcv = self._candles_to_arrays(candles)

        # 1. Technical features (35)
        try:
            technical_features = self._technical.calculate_latest(
                open_prices=ohlcv["open"],
                high_prices=ohlcv["high"],
                low_prices=ohlcv["low"],
                close_prices=ohlcv["close"],
                volume=ohlcv["volume"],
            )
            features.update(technical_features)
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
            features.update(self._get_default_technical_features())

        # 2. Microstructure features (7)
        try:
            micro_features = self._calculate_microstructure_features(
                orderbook_data=orderbook_data,
                ticker_data=ticker_data,
                microstructure_data=microstructure_data,
            )
            features.update(micro_features)
        except Exception as e:
            logger.error(f"Error calculating microstructure features: {e}")
            features.update(self._get_default_microstructure_features())

        # 3. Fee features (5)
        try:
            # Get ATR from technical features
            atr = features.get("atr", 0.0)
            close_price = ohlcv["close"][-1]

            fee_features = self._fee_calculator.calculate(
                ticker=ticker_data,
                atr=atr,
                close_price=close_price,
            )
            features.update(fee_features)
        except Exception as e:
            logger.error(f"Error calculating fee features: {e}")
            features.update(self._get_default_fee_features())

        # 4. Cross-asset features (3)
        try:
            cross_asset_features = self._cross_asset_calculator.calculate(
                close_prices=ohlcv["close"],
                cross_asset_data=cross_asset_data,
            )
            features.update(cross_asset_features)
        except Exception as e:
            logger.error(f"Error calculating cross-asset features: {e}")
            features.update(self._get_default_cross_asset_features())

        # 5. Time features (5)
        try:
            time_features = self._time_calculator.calculate(timestamp)
            features.update(time_features)
        except Exception as e:
            logger.error(f"Error calculating time features: {e}")
            features.update(self._get_default_time_features())

        # 6. Volatility features (4)
        try:
            # Get ATR array from technical calculation
            technical_arrays = self._technical.calculate(
                open_prices=ohlcv["open"],
                high_prices=ohlcv["high"],
                low_prices=ohlcv["low"],
                close_prices=ohlcv["close"],
                volume=ohlcv["volume"],
            )
            atr_values = technical_arrays.get("atr", np.array([]))

            vol_features = self._volatility_calculator.calculate(
                close_prices=ohlcv["close"],
                high_prices=ohlcv["high"],
                low_prices=ohlcv["low"],
                atr_values=atr_values,
            )
            features.update(vol_features)
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
            features.update(self._get_default_volatility_features())

        # Ensure all values are valid floats
        features = self._validate_features(features)

        logger.debug(f"Calculated {len(features)} features")

        return features

    def _calculate_microstructure_features(
        self,
        orderbook_data: OrderbookData | None,
        ticker_data: TickerData | None,
        microstructure_data: MicrostructureData | None,
    ) -> dict[str, float]:
        """Calculate microstructure features."""
        # If pre-built microstructure data provided, use it
        if microstructure_data is not None:
            result = self._microstructure.calculate(microstructure_data)
            result["top5_orderbook_imbalance"] = calculate_top5_orderbook_imbalance(
                microstructure_data.orderbook
            )
            return result

        # Build MicrostructureData from inputs
        data = MicrostructureData()

        if orderbook_data is not None:
            data.orderbook = orderbook_data.to_snapshot()

        if ticker_data is not None and ticker_data.funding_rate != 0:
            from src.features.microstructure import FundingInfo

            data.funding = FundingInfo(funding_rate=ticker_data.funding_rate)

        result = self._microstructure.calculate(data)
        result["top5_orderbook_imbalance"] = calculate_top5_orderbook_imbalance(data.orderbook)
        return result

    def _candles_to_arrays(
        self,
        candles: list[Candle],
    ) -> dict[str, NDArray[np.float64]]:
        """Convert candle list to numpy arrays."""
        n = len(candles)

        arrays: dict[str, NDArray[np.float64]] = {
            "open": np.zeros(n, dtype=np.float64),
            "high": np.zeros(n, dtype=np.float64),
            "low": np.zeros(n, dtype=np.float64),
            "close": np.zeros(n, dtype=np.float64),
            "volume": np.zeros(n, dtype=np.float64),
        }

        for i, candle in enumerate(candles):
            arrays["open"][i] = candle.open
            arrays["high"][i] = candle.high
            arrays["low"][i] = candle.low
            arrays["close"][i] = candle.close
            arrays["volume"][i] = candle.volume

        return arrays

    def _generate_cache_key(
        self,
        candles: list[Candle],
        ticker_data: TickerData | None,
    ) -> str:
        """Generate cache key from input data."""
        if not candles:
            return "empty"

        latest_candle = candles[-1]
        key_parts = [
            latest_candle.symbol,
            latest_candle.timestamp.isoformat(),
            str(len(candles)),
        ]

        if ticker_data:
            key_parts.append(f"{ticker_data.last_price:.8f}")

        return "|".join(key_parts)

    def _prune_cache(self, max_entries: int = 100) -> None:
        """Remove old cache entries."""
        if len(self._cache) <= max_entries:
            return

        # Remove oldest entries
        entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp,
        )

        for key, _ in entries[: len(entries) - max_entries]:
            del self._cache[key]

    def _validate_features(self, features: dict[str, float]) -> dict[str, float]:
        """Validate and clean feature values."""
        validated: dict[str, float] = {}

        for name, value in features.items():
            if isinstance(value, (int, float)):
                # Handle NaN and Inf
                if np.isnan(value) or np.isinf(value):
                    validated[name] = 0.0
                else:
                    validated[name] = float(value)
            else:
                validated[name] = 0.0

        return validated

    def _get_default_features(self) -> dict[str, float]:
        """Get default values for all features."""
        defaults: dict[str, float] = {}
        defaults.update(self._get_default_technical_features())
        defaults.update(self._get_default_microstructure_features())
        defaults.update(self._get_default_fee_features())
        defaults.update(self._get_default_cross_asset_features())
        defaults.update(self._get_default_time_features())
        defaults.update(self._get_default_volatility_features())
        return defaults

    def _get_default_technical_features(self) -> dict[str, float]:
        """Get default values for technical features."""
        return {name: 0.0 for name in self._technical.feature_names}

    def _get_default_microstructure_features(self) -> dict[str, float]:
        """Get default values for microstructure features."""
        return {
            "orderbook_imbalance": 0.0,
            "funding_rate": 0.0,
            "open_interest": 0.5,
            "open_interest_change": 0.0,
            "volume_profile": 0.5,
            "large_trade_ratio": 0.0,
            "trade_flow_imbalance": 0.0,
            "top5_orderbook_imbalance": 0.5,
        }

    def _get_default_fee_features(self) -> dict[str, float]:
        """Get default values for fee features."""
        return {
            "maker_fee": self.fee_config.maker_fee * 10000,
            "taker_fee": self.fee_config.taker_fee * 10000,
            "funding_impact": 0.0,
            "fee_drag": 0.5,
            "breakeven_move": (self.fee_config.maker_fee + self.fee_config.taker_fee)
            * 100,
        }

    def _get_default_cross_asset_features(self) -> dict[str, float]:
        """Get default values for cross-asset features."""
        return {
            "btc_correlation": 0.0,
            "eth_correlation": 0.0,
            "market_correlation": 0.0,
        }

    def _get_default_time_features(self) -> dict[str, float]:
        """Get default values for time features."""
        return {
            "session_indicator": 1.0,  # Default to Europe session
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_sin": 0.0,
            "day_cos": 1.0,
        }

    def _get_default_volatility_features(self) -> dict[str, float]:
        """Get default values for volatility features."""
        return {
            "atr_relative": 1.0,
            "realized_vol": 0.0,
            "vol_regime": 1.0,
            "vol_percentile": 0.5,
        }

    def get_core_features(
        self,
        candles: list[Candle],
        ticker_data: TickerData | None = None,
        orderbook_data: OrderbookData | None = None,
        cross_asset_data: CrossAssetData | None = None,
        microstructure_data: MicrostructureData | None = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """
        Get only the 25 core features for ML signal generation.

        This is the reduced feature set for the Simons Protocol,
        containing only Tier 1 (signal drivers) and Tier 2 (confirmation) features.

        Args:
            candles: List of OHLCV candles
            ticker_data: Current ticker data
            orderbook_data: Current orderbook snapshot
            cross_asset_data: Cross-asset price data
            microstructure_data: Pre-built microstructure data
            use_cache: Whether to use cached results

        Returns:
            Dictionary of 25 core feature values
        """
        # Calculate all features first
        all_features = self.calculate_features(
            candles=candles,
            ticker_data=ticker_data,
            orderbook_data=orderbook_data,
            cross_asset_data=cross_asset_data,
            microstructure_data=microstructure_data,
            use_cache=use_cache,
        )

        # Extract only core features
        core_features: dict[str, float] = {}
        for name in self.CORE_FEATURE_NAMES:
            if name in all_features:
                core_features[name] = all_features[name]
            else:
                # Calculate derived features if not present
                core_features[name] = self._calculate_derived_feature(name, all_features)

        return core_features

    def get_timing_features(
        self,
        candles: list[Candle],
        timestamp: datetime | None = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """
        Get timing features for entry optimization.

        These are Tier 3 features used for fine-tuning entry points,
        not for signal generation.

        Args:
            candles: List of OHLCV candles
            timestamp: Timestamp for time features
            use_cache: Whether to use cached results

        Returns:
            Dictionary of timing feature values
        """
        all_features = self.calculate_features(
            candles=candles,
            timestamp=timestamp,
            use_cache=use_cache,
        )

        timing_features: dict[str, float] = {}
        for name in self.TIMING_FEATURE_NAMES:
            timing_features[name] = all_features.get(name, 0.0)

        return timing_features

    def _calculate_derived_feature(
        self,
        name: str,
        features: dict[str, float],
    ) -> float:
        """
        Calculate derived features not directly available.

        Some core features are combinations or transformations
        of raw technical indicators.
        """
        if name == "ema_short_ratio":
            # Price relative to short EMA (e.g., EMA9)
            ema = features.get("ema_9", 0.0)
            close = features.get("close", ema)
            if ema > 0:
                return (close - ema) / ema
            return 0.0

        if name == "ema_long_ratio":
            # Price relative to long EMA (e.g., EMA50)
            ema = features.get("ema_50", 0.0)
            close = features.get("close", ema)
            if ema > 0:
                return (close - ema) / ema
            return 0.0

        if name == "ema_cross":
            # EMA crossover signal: +1 if short > long, -1 if short < long
            ema_short = features.get("ema_21", 0.0)
            ema_long = features.get("ema_50", 0.0)
            if ema_long > 0:
                return 1.0 if ema_short > ema_long else -1.0
            return 0.0

        if name == "trend_strength":
            # Combined trend metric from ADX and DI difference
            adx = features.get("adx", 0.0)
            plus_di = features.get("plus_di", 0.0)
            minus_di = features.get("minus_di", 0.0)
            di_diff = plus_di - minus_di
            # Normalize: ADX gives strength (0-100), DI diff gives direction
            return (adx / 100.0) * (1.0 if di_diff > 0 else -1.0)

        if name == "momentum":
            # Simple price momentum (rate of change)
            roc = features.get("roc", 0.0)
            return float(np.clip(roc / 10.0, -1.0, 1.0))  # Normalize ROC

        if name == "price_position":
            # Price position in recent range (0 = low, 1 = high)
            bb_upper = features.get("bb_upper", 0.0)
            bb_lower = features.get("bb_lower", 0.0)
            close = features.get("close", 0.0)
            if bb_upper > bb_lower:
                return (close - bb_lower) / (bb_upper - bb_lower)
            return 0.5

        # Default: return 0 for unknown features
        return 0.0

    @property
    def core_feature_names(self) -> list[str]:
        """Get list of core feature names (25 features)."""
        return self.CORE_FEATURE_NAMES.copy()

    @property
    def core_feature_count(self) -> int:
        """Get count of core features."""
        return len(self.CORE_FEATURE_NAMES)

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        valid_count = sum(1 for c in self._cache.values() if c.is_valid())
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "ttl_ms": self.config.cache_ttl_ms,
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

_pipeline_instance: FeaturePipeline | None = None


def get_feature_pipeline() -> FeaturePipeline:
    """
    Get the feature pipeline singleton instance.

    Returns:
        FeaturePipeline instance with default configuration
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FeaturePipeline()
    return _pipeline_instance


def calculate_all_features(
    candles: list[Candle],
    ticker_data: TickerData | None = None,
    orderbook_data: OrderbookData | None = None,
) -> dict[str, float]:
    """
    Convenience function to calculate all features.

    Args:
        candles: List of OHLCV candles
        ticker_data: Current ticker data
        orderbook_data: Current orderbook snapshot

    Returns:
        Dictionary of all feature values
    """
    return get_feature_pipeline().calculate_features(
        candles=candles,
        ticker_data=ticker_data,
        orderbook_data=orderbook_data,
    )
