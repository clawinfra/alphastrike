"""
V2 Feature Pipeline — Quality Over Quantity

V1 had 59 features (architecture claimed 86). Many were correlated,
some were ghost features never used. This pipeline uses 25 carefully
selected features in 2 tiers:

Tier 1 — Signal Drivers (12 features):
  RSI, RSI slope, ADX, +DI, -DI, VWAP deviation, OBV slope,
  funding rate, BTC correlation, volume ratio, EMA cross, momentum

Tier 2 — Context (13 features):
  ATR ratio, BB %B, BB bandwidth, MACD histogram, orderbook imbalance,
  hour sin/cos, day sin/cos, realized vol, spread, stoch K, price position

All features are:
  - Named (not feature_0, feature_1)
  - Bounded (clipped to reasonable ranges)
  - NaN-safe (explicit defaults)
  - Documented with expected ranges
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Feature definitions with expected ranges for validation
FEATURE_SPEC: dict[str, tuple[float, float, float]] = {
    # name: (min, max, default)
    # Tier 1 — Signal Drivers
    "rsi": (0.0, 100.0, 50.0),
    "rsi_slope": (-10.0, 10.0, 0.0),
    "adx": (0.0, 100.0, 20.0),
    "plus_di": (0.0, 100.0, 20.0),
    "minus_di": (0.0, 100.0, 20.0),
    "vwap_deviation": (-0.1, 0.1, 0.0),
    "obv_slope": (-1.0, 1.0, 0.0),
    "funding_rate": (-0.001, 0.001, 0.0),
    "btc_correlation": (-1.0, 1.0, 0.0),
    "volume_ratio": (0.0, 10.0, 1.0),
    "ema_cross": (-1.0, 1.0, 0.0),
    "momentum": (-0.2, 0.2, 0.0),
    # Tier 2 — Context
    "atr_ratio": (0.0, 5.0, 1.0),
    "bb_pct_b": (-1.0, 2.0, 0.5),
    "bb_bandwidth": (0.0, 1.0, 0.1),
    "macd_histogram": (-1.0, 1.0, 0.0),
    "orderbook_imbalance": (-1.0, 1.0, 0.0),
    "hour_sin": (-1.0, 1.0, 0.0),
    "hour_cos": (-1.0, 1.0, 1.0),
    "day_sin": (-1.0, 1.0, 0.0),
    "day_cos": (-1.0, 1.0, 1.0),
    "realized_vol": (0.0, 5.0, 0.5),
    "spread_bps": (0.0, 100.0, 5.0),
    "stoch_k": (0.0, 100.0, 50.0),
    "price_position": (0.0, 1.0, 0.5),
}

FEATURE_NAMES = list(FEATURE_SPEC.keys())
N_FEATURES = len(FEATURE_NAMES)

# Tier 1 only (for lightweight signal generation)
TIER1_NAMES = FEATURE_NAMES[:12]


@dataclass
class FeatureConfig:
    """Configuration for feature calculations."""

    min_candles: int = 100
    rsi_period: int = 14
    adx_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 9
    ema_slow: int = 21
    obv_slope_period: int = 20
    vol_lookback: int = 20
    stoch_period: int = 14
    correlation_window: int = 20


class V2FeaturePipeline:
    """
    25-feature pipeline with named, bounded, documented features.

    Usage:
        pipeline = V2FeaturePipeline()
        features = pipeline.calculate(close, high, low, volume)
        X = pipeline.to_array(features)  # (1, 25) numpy array
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()

    @property
    def n_features(self) -> int:
        return N_FEATURES

    def calculate(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        volume: NDArray[np.float64],
        *,
        btc_close: NDArray[np.float64] | None = None,
        funding_rate: float = 0.0,
        orderbook_imbalance: float = 0.0,
        spread_bps: float = 5.0,
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """
        Calculate all 25 features.

        Args:
            close, high, low, volume: OHLCV arrays (oldest first)
            btc_close: BTC close prices for correlation (optional)
            funding_rate: Current funding rate (optional)
            orderbook_imbalance: Current orderbook imbalance [-1, 1] (optional)
            spread_bps: Current bid-ask spread in bps (optional)
            timestamp: Current timestamp for time features (optional)

        Returns:
            Dictionary of 25 named features, all bounded and NaN-safe
        """
        n = len(close)
        if n < self.config.min_candles:
            return self._defaults()

        features: dict[str, float] = {}

        # === Tier 1: Signal Drivers ===

        # RSI (14)
        rsi = self._rsi(close, self.config.rsi_period)
        features["rsi"] = rsi

        # RSI slope (5-period rate of change of RSI)
        rsi_arr = self._rsi_array(close, self.config.rsi_period)
        if len(rsi_arr) >= 5:
            features["rsi_slope"] = float(rsi_arr[-1] - rsi_arr[-5])
        else:
            features["rsi_slope"] = 0.0

        # ADX, +DI, -DI
        adx, plus_di, minus_di = self._adx(high, low, close, self.config.adx_period)
        features["adx"] = adx
        features["plus_di"] = plus_di
        features["minus_di"] = minus_di

        # VWAP deviation
        features["vwap_deviation"] = self._vwap_deviation(close, high, low, volume)

        # OBV slope
        features["obv_slope"] = self._obv_slope(close, volume, self.config.obv_slope_period)

        # Funding rate (passthrough)
        features["funding_rate"] = funding_rate

        # BTC correlation
        if btc_close is not None and len(btc_close) >= self.config.correlation_window + 1:
            features["btc_correlation"] = self._correlation(
                close, btc_close, self.config.correlation_window
            )
        else:
            features["btc_correlation"] = 0.0

        # Volume ratio (current / SMA20)
        vol_sma = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        features["volume_ratio"] = float(volume[-1] / vol_sma) if vol_sma > 0 else 1.0

        # EMA cross (fast - slow, normalized)
        ema_fast = self._ema(close, self.config.ema_fast)
        ema_slow = self._ema(close, self.config.ema_slow)
        if ema_slow > 0:
            features["ema_cross"] = float(np.clip((ema_fast - ema_slow) / ema_slow, -1, 1))
        else:
            features["ema_cross"] = 0.0

        # Momentum (12-bar rate of change, normalized)
        if n >= 13:
            mom = (close[-1] - close[-13]) / close[-13] if close[-13] > 0 else 0.0
            features["momentum"] = float(np.clip(mom, -0.2, 0.2))
        else:
            features["momentum"] = 0.0

        # === Tier 2: Context ===

        # ATR ratio
        features["atr_ratio"] = self._atr_ratio(high, low, close)

        # Bollinger Bands %B and bandwidth
        bb_upper, bb_mid, bb_lower = self._bollinger(
            close, self.config.bb_period, self.config.bb_std
        )
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            features["bb_pct_b"] = float((close[-1] - bb_lower) / bb_range)
            features["bb_bandwidth"] = float(bb_range / bb_mid) if bb_mid > 0 else 0.1
        else:
            features["bb_pct_b"] = 0.5
            features["bb_bandwidth"] = 0.0

        # MACD histogram (normalized by price)
        macd_hist = self._macd_histogram(close)
        features["macd_histogram"] = float(
            np.clip(macd_hist / close[-1] * 100 if close[-1] > 0 else 0, -1, 1)
        )

        # Orderbook imbalance (passthrough)
        features["orderbook_imbalance"] = orderbook_imbalance

        # Time features
        ts = timestamp or datetime.utcnow()
        hour_rad = (ts.hour / 24.0) * 2 * math.pi
        day_rad = (ts.weekday() / 7.0) * 2 * math.pi
        features["hour_sin"] = math.sin(hour_rad)
        features["hour_cos"] = math.cos(hour_rad)
        features["day_sin"] = math.sin(day_rad)
        features["day_cos"] = math.cos(day_rad)

        # Realized volatility (annualized, from log returns)
        if n >= self.config.vol_lookback + 1:
            log_rets = np.diff(np.log(close[-self.config.vol_lookback - 1 :]))
            features["realized_vol"] = float(np.std(log_rets) * np.sqrt(365 * 24))  # hourly bars
        else:
            features["realized_vol"] = 0.5

        # Spread (passthrough)
        features["spread_bps"] = spread_bps

        # Stochastic K
        features["stoch_k"] = self._stochastic_k(close, high, low, self.config.stoch_period)

        # Price position (where in recent range: 0 = bottom, 1 = top)
        recent_high = float(np.max(high[-20:]))
        recent_low = float(np.min(low[-20:]))
        if recent_high > recent_low:
            features["price_position"] = float(
                (close[-1] - recent_low) / (recent_high - recent_low)
            )
        else:
            features["price_position"] = 0.5

        # Clip all features to spec ranges
        return self._clip_and_validate(features)

    def to_array(self, features: dict[str, float]) -> NDArray[np.float64]:
        """Convert feature dict to ordered numpy array (1, N_FEATURES)."""
        arr = np.array(
            [features.get(name, FEATURE_SPEC[name][2]) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        return arr.reshape(1, -1)

    def _defaults(self) -> dict[str, float]:
        """Default feature values when insufficient data."""
        return {name: spec[2] for name, spec in FEATURE_SPEC.items()}

    def _clip_and_validate(self, features: dict[str, float]) -> dict[str, float]:
        """Clip features to spec ranges and replace NaN/Inf."""
        result: dict[str, float] = {}
        for name in FEATURE_NAMES:
            val = features.get(name, FEATURE_SPEC[name][2])
            if np.isnan(val) or np.isinf(val):
                val = FEATURE_SPEC[name][2]  # default
            lo, hi, _ = FEATURE_SPEC[name]
            result[name] = float(np.clip(val, lo, hi))
        return result

    # === Technical Indicator Implementations ===

    def _rsi(self, close: NDArray, period: int) -> float:
        """RSI using Wilder's smoothing."""
        arr = self._rsi_array(close, period)
        return float(arr[-1]) if len(arr) > 0 else 50.0

    def _rsi_array(self, close: NDArray, period: int) -> NDArray:
        """Full RSI array."""
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(gains) < period:
            return np.array([50.0])

        # Wilder's smoothing
        avg_gain = np.zeros(len(gains))
        avg_loss = np.zeros(len(gains))
        avg_gain[period - 1] = np.mean(gains[:period])
        avg_loss[period - 1] = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi[period - 1 :]

    def _adx(
        self,
        high: NDArray,
        low: NDArray,
        close: NDArray,
        period: int,
    ) -> tuple[float, float, float]:
        """ADX with +DI and -DI."""
        n = len(close)
        if n < period * 2:
            return 20.0, 20.0, 20.0

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's EMA
        alpha = 1.0 / period
        atr_s = float(np.mean(tr[:period]))
        pdm_s = float(np.mean(plus_dm[:period]))
        mdm_s = float(np.mean(minus_dm[:period]))

        for i in range(period, len(tr)):
            atr_s = atr_s * (1 - alpha) + float(tr[i]) * alpha
            pdm_s = pdm_s * (1 - alpha) + float(plus_dm[i]) * alpha
            mdm_s = mdm_s * (1 - alpha) + float(minus_dm[i]) * alpha

        plus_di = 100.0 * pdm_s / atr_s if atr_s > 0 else 0.0
        minus_di = 100.0 * mdm_s / atr_s if atr_s > 0 else 0.0

        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

        return dx, plus_di, minus_di  # Simplified ADX ≈ DX for latest bar

    def _atr_ratio(
        self, high: NDArray, low: NDArray, close: NDArray
    ) -> float:
        """Current ATR / long-term average ATR."""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        if len(tr) < 100:
            return 1.0
        current = float(np.mean(tr[-14:]))
        average = float(np.mean(tr[-100:]))
        return current / average if average > 0 else 1.0

    def _vwap_deviation(
        self,
        close: NDArray,
        high: NDArray,
        low: NDArray,
        volume: NDArray,
    ) -> float:
        """Price deviation from session VWAP (last 24 bars)."""
        lookback = min(24, len(close))
        typical = (close[-lookback:] + high[-lookback:] + low[-lookback:]) / 3.0
        vol = volume[-lookback:]
        total_vol = float(np.sum(vol))
        if total_vol <= 0:
            return 0.0
        vwap = float(np.sum(typical * vol) / total_vol)
        if vwap <= 0:
            return 0.0
        return float(np.clip((close[-1] - vwap) / vwap, -0.1, 0.1))

    def _obv_slope(self, close: NDArray, volume: NDArray, period: int) -> float:
        """Normalized OBV slope over period."""
        if len(close) < period + 1:
            return 0.0
        deltas = np.sign(np.diff(close[-period - 1 :]))
        obv = np.cumsum(deltas * volume[-period:])
        if len(obv) < 2:
            return 0.0
        x = np.arange(len(obv), dtype=np.float64)
        slope = np.polyfit(x, obv, 1)[0]
        # Normalize by average volume
        avg_vol = float(np.mean(volume[-period:]))
        return float(np.clip(slope / avg_vol if avg_vol > 0 else 0, -1, 1))

    def _correlation(
        self, a: NDArray, b: NDArray, window: int
    ) -> float:
        """Rolling Pearson correlation."""
        if len(a) < window + 1 or len(b) < window + 1:
            return 0.0
        ret_a = np.diff(a[-window - 1 :])
        ret_b = np.diff(b[-window - 1 :])
        if np.std(ret_a) == 0 or np.std(ret_b) == 0:
            return 0.0
        corr = np.corrcoef(ret_a[-window:], ret_b[-window:])[0, 1]
        return float(np.clip(corr, -1, 1)) if not np.isnan(corr) else 0.0

    def _ema(self, close: NDArray, period: int) -> float:
        """Exponential moving average (latest value)."""
        if len(close) < period:
            return float(close[-1])
        alpha = 2.0 / (period + 1)
        ema = float(close[-period])
        for i in range(-period + 1, 0):
            ema = ema * (1 - alpha) + float(close[i]) * alpha
        return ema

    def _bollinger(
        self, close: NDArray, period: int, num_std: float
    ) -> tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)."""
        if len(close) < period:
            return float(close[-1]), float(close[-1]), float(close[-1])
        window = close[-period:]
        mid = float(np.mean(window))
        std = float(np.std(window))
        return mid + num_std * std, mid, mid - num_std * std

    def _macd_histogram(self, close: NDArray) -> float:
        """MACD histogram value."""
        if len(close) < self.config.macd_slow + self.config.macd_signal:
            return 0.0
        fast_ema = self._ema(close, self.config.macd_fast)
        slow_ema = self._ema(close, self.config.macd_slow)
        macd = fast_ema - slow_ema
        return macd  # Simplified — full impl would use signal line EMA

    def _stochastic_k(
        self, close: NDArray, high: NDArray, low: NDArray, period: int
    ) -> float:
        """Stochastic %K."""
        if len(close) < period:
            return 50.0
        highest = float(np.max(high[-period:]))
        lowest = float(np.min(low[-period:]))
        if highest == lowest:
            return 50.0
        return float(np.clip((close[-1] - lowest) / (highest - lowest) * 100, 0, 100))
