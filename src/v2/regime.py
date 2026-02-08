"""
Unified Regime Detector — Single Source of Truth

Replaces the THREE separate regime detectors in V1:
  1. src/strategy/regime_detector.py (ADX/ATR, 6 regimes)
  2. src/trading/medallion_live.py (SMA/slope, 3 regimes)
  3. src/strategy/simons_engine.py (simplified ADX, 3 regimes)

Design principles:
  - 4 regimes: TREND_UP, TREND_DOWN, RANGE, CHAOS
  - Each maps to exactly one strategy mode
  - Confidence is continuous (0-1), not bucketed
  - Hysteresis prevents regime flapping (must sustain for N bars)
  - All inputs are standard TA features (no lookback surprises)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    """Market regime — exactly 4 states."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    CHAOS = "CHAOS"


# Strategy mode mapping
REGIME_STRATEGY: dict[Regime, str] = {
    Regime.TREND_UP: "trend_follow_long",
    Regime.TREND_DOWN: "reduce_exposure",  # not short — hedging only
    Regime.RANGE: "mean_reversion",
    Regime.CHAOS: "flat",
}


@dataclass
class RegimeState:
    """Current regime with full diagnostic context."""

    regime: Regime
    confidence: float  # 0.0 to 1.0
    strategy_mode: str
    adx: float
    atr_ratio: float  # current ATR / ATR_SMA100
    trend_slope: float  # linear regression slope (normalized)
    di_spread: float  # +DI - -DI
    bars_in_regime: int  # how long we've been in this regime
    raw_regime: Regime  # before hysteresis (for debugging)


@dataclass
class RegimeConfig:
    """Tunable thresholds — all in one place."""

    # ADX thresholds
    adx_trending: float = 25.0  # above this = trending
    adx_ranging: float = 18.0  # below this = ranging

    # Volatility thresholds (ATR ratio = current_ATR / SMA100_ATR)
    atr_chaos: float = 2.0  # above this = chaos (overrides all)

    # Hysteresis: must sustain new regime for N bars before switching
    hysteresis_bars: int = 3

    # Slope threshold for trend direction
    slope_threshold: float = 0.0  # positive = up, negative = down

    # Lookback periods
    adx_period: int = 14
    atr_period: int = 14
    slope_lookback: int = 20
    atr_ma_lookback: int = 100


class UnifiedRegimeDetector:
    """
    Single regime detector for the entire system.

    Usage:
        detector = UnifiedRegimeDetector()
        state = detector.update(close, high, low)
        if state.regime == Regime.RANGE:
            # use mean reversion strategy
        elif state.regime == Regime.TREND_UP:
            # use trend following
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._current_regime = Regime.RANGE
        self._bars_in_regime = 0
        self._pending_regime: Regime | None = None
        self._pending_bars = 0

        # History for diagnostics
        self._regime_history: deque[Regime] = deque(maxlen=500)

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    def update(
        self,
        close: NDArray[np.float64],
        high: NDArray[np.float64],
        low: NDArray[np.float64],
    ) -> RegimeState:
        """
        Update regime detection with new price data.

        Args:
            close: Close prices array (oldest first, most recent last)
            high: High prices array
            low: Low prices array

        Returns:
            RegimeState with current regime and diagnostics
        """
        n = len(close)
        if n < self.config.atr_ma_lookback:
            return self._default_state()

        # Calculate indicators
        adx, plus_di, minus_di = self._calculate_adx(high, low, close)
        atr_ratio = self._calculate_atr_ratio(high, low, close)
        slope = self._calculate_slope(close)
        di_spread = plus_di - minus_di

        # Determine raw regime (before hysteresis)
        raw_regime = self._classify_regime(adx, atr_ratio, slope, di_spread)

        # Apply hysteresis to prevent flapping
        confirmed_regime = self._apply_hysteresis(raw_regime)

        # Update state
        self._regime_history.append(confirmed_regime)

        state = RegimeState(
            regime=confirmed_regime,
            confidence=self._calculate_confidence(
                confirmed_regime, adx, atr_ratio, slope, di_spread
            ),
            strategy_mode=REGIME_STRATEGY[confirmed_regime],
            adx=adx,
            atr_ratio=atr_ratio,
            trend_slope=slope,
            di_spread=di_spread,
            bars_in_regime=self._bars_in_regime,
            raw_regime=raw_regime,
        )

        return state

    def _classify_regime(
        self,
        adx: float,
        atr_ratio: float,
        slope: float,
        di_spread: float,
    ) -> Regime:
        """Raw regime classification (no hysteresis)."""

        # CHAOS overrides everything — extreme volatility
        if atr_ratio > self.config.atr_chaos:
            return Regime.CHAOS

        # TRENDING — ADX above threshold + directional indicators agree
        if adx > self.config.adx_trending:
            if slope > self.config.slope_threshold and di_spread > 0:
                return Regime.TREND_UP
            elif slope < -self.config.slope_threshold and di_spread < 0:
                return Regime.TREND_DOWN
            # ADX high but slope/DI disagree — treat as ranging
            # (trend is losing direction, transition period)

        # RANGE — ADX below threshold
        if adx < self.config.adx_ranging:
            return Regime.RANGE

        # In-between zone (ADX 18-25): default to current regime for stability
        return self._current_regime

    def _apply_hysteresis(self, raw_regime: Regime) -> Regime:
        """Prevent regime flapping with confirmation bars."""
        if raw_regime == self._current_regime:
            # Same regime — reset pending and increment bars
            self._pending_regime = None
            self._pending_bars = 0
            self._bars_in_regime += 1
            return self._current_regime

        # Different regime detected
        if raw_regime == self._pending_regime:
            # Same new regime as pending — increment counter
            self._pending_bars += 1
            if self._pending_bars >= self.config.hysteresis_bars:
                # Confirmed! Switch regime
                old = self._current_regime
                self._current_regime = raw_regime
                self._bars_in_regime = self._pending_bars
                self._pending_regime = None
                self._pending_bars = 0
                logger.info(
                    f"Regime change: {old.value} → {raw_regime.value} "
                    f"(confirmed after {self.config.hysteresis_bars} bars)"
                )
                return self._current_regime
        else:
            # New pending regime
            self._pending_regime = raw_regime
            self._pending_bars = 1

        # Not yet confirmed — stay in current regime
        self._bars_in_regime += 1
        return self._current_regime

    def _calculate_confidence(
        self,
        regime: Regime,
        adx: float,
        atr_ratio: float,
        slope: float,
        di_spread: float,
    ) -> float:
        """Calculate confidence in current regime classification."""
        if regime == Regime.CHAOS:
            # Higher ATR ratio = more confident it's chaos
            excess = atr_ratio - self.config.atr_chaos
            return float(np.clip(0.6 + excess * 0.2, 0.5, 1.0))

        if regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            # Confidence from: ADX strength + DI spread + slope consistency
            adx_score = np.clip((adx - self.config.adx_trending) / 25.0, 0, 0.4)
            di_score = np.clip(abs(di_spread) / 30.0, 0, 0.3)
            slope_score = np.clip(abs(slope) * 10, 0, 0.3)
            return float(np.clip(adx_score + di_score + slope_score, 0.3, 1.0))

        if regime == Regime.RANGE:
            # Lower ADX = more confident it's ranging
            adx_score = np.clip(
                (self.config.adx_ranging - adx) / self.config.adx_ranging, 0, 0.5
            )
            # Low ATR ratio = more confident
            atr_score = np.clip(1.0 - atr_ratio, 0, 0.3)
            # Flat slope = more confident
            slope_score = np.clip(0.2 - abs(slope) * 5, 0, 0.2)
            return float(np.clip(adx_score + atr_score + slope_score, 0.3, 1.0))

        return 0.5

    def _calculate_adx(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Calculate ADX, +DI, -DI using Wilder's smoothing."""
        period = self.config.adx_period
        n = len(close)
        if n < period + 1:
            return 20.0, 0.0, 0.0

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder's smoothing (EMA with alpha = 1/period)
        def wilder_smooth(arr: NDArray, p: int) -> NDArray:
            result = np.zeros_like(arr)
            result[p - 1] = np.mean(arr[:p])
            for i in range(p, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / p + arr[i]
            return result

        atr_smoothed = wilder_smooth(tr, period)
        plus_dm_smoothed = wilder_smooth(plus_dm, period)
        minus_dm_smoothed = wilder_smooth(minus_dm, period)

        # Avoid division by zero
        atr_safe = np.where(atr_smoothed > 0, atr_smoothed, 1.0)

        plus_di_arr = 100.0 * plus_dm_smoothed / atr_safe
        minus_di_arr = 100.0 * minus_dm_smoothed / atr_safe

        # DX and ADX
        di_sum = plus_di_arr + minus_di_arr
        di_sum_safe = np.where(di_sum > 0, di_sum, 1.0)
        dx = 100.0 * np.abs(plus_di_arr - minus_di_arr) / di_sum_safe

        adx_arr = wilder_smooth(dx[period - 1 :], period)

        # Return latest values
        adx = float(adx_arr[-1]) if len(adx_arr) > 0 else 20.0
        plus_di = float(plus_di_arr[-1]) if len(plus_di_arr) > 0 else 0.0
        minus_di = float(minus_di_arr[-1]) if len(minus_di_arr) > 0 else 0.0

        return adx, plus_di, minus_di

    def _calculate_atr_ratio(
        self,
        high: NDArray[np.float64],
        low: NDArray[np.float64],
        close: NDArray[np.float64],
    ) -> float:
        """Calculate ATR ratio: current ATR / long-term average ATR."""
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        if len(tr) < self.config.atr_ma_lookback:
            return 1.0

        current_atr = float(np.mean(tr[-self.config.atr_period :]))
        long_term_atr = float(np.mean(tr[-self.config.atr_ma_lookback :]))

        if long_term_atr <= 0:
            return 1.0

        return current_atr / long_term_atr

    def _calculate_slope(self, close: NDArray[np.float64]) -> float:
        """Normalized linear regression slope over lookback period."""
        lookback = self.config.slope_lookback
        if len(close) < lookback:
            return 0.0

        window = close[-lookback:]
        x = np.arange(lookback, dtype=np.float64)
        slope = np.polyfit(x, window, 1)[0]

        # Normalize by price level (slope per bar as fraction of price)
        return float(slope / window[-1]) if window[-1] > 0 else 0.0

    def _default_state(self) -> RegimeState:
        """Default state when insufficient data."""
        return RegimeState(
            regime=Regime.RANGE,
            confidence=0.3,
            strategy_mode=REGIME_STRATEGY[Regime.RANGE],
            adx=20.0,
            atr_ratio=1.0,
            trend_slope=0.0,
            di_spread=0.0,
            bars_in_regime=0,
            raw_regime=Regime.RANGE,
        )

    def get_regime_distribution(self, lookback: int = 100) -> dict[str, float]:
        """Get distribution of regimes over recent history."""
        if not self._regime_history:
            return {r.value: 0.25 for r in Regime}

        recent = list(self._regime_history)[-lookback:]
        total = len(recent)
        dist = {}
        for regime in Regime:
            count = sum(1 for r in recent if r == regime)
            dist[regime.value] = count / total
        return dist
