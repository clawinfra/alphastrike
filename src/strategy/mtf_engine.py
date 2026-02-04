"""
AlphaStrike Trading Bot - Multi-Timeframe Engine

Coordinates signal generation across Daily, 4H, and 1H timeframes.

Architecture:
- Daily: Trend direction filter (EMA200 + ADX)
- 4H: Primary signal generation (ML Ensemble)
- 1H: Entry timing optimization

Only generates trade signals when timeframes align.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from src.data.database import Candle

logger = logging.getLogger(__name__)


@dataclass
class TimeframeTrend:
    """Trend analysis for a single timeframe."""

    direction: Literal["BULL", "BEAR", "NEUTRAL"]
    strength: float  # 0-100 (ADX or similar)
    ema_short: float  # Short-term EMA value
    ema_long: float  # Long-term EMA value
    price: float  # Current price
    confidence: float  # 0-1


@dataclass
class EntryTiming:
    """Entry timing signals from 1H timeframe."""

    action: Literal["NOW", "WAIT", "ABORT"]
    reason: str
    optimal_entry_price: float | None
    momentum: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    rsi: float
    distance_from_ema21: float  # Percentage


@dataclass
class MTFSignal:
    """Multi-timeframe signal result."""

    # Timeframe analyses
    daily: TimeframeTrend
    four_hour: TimeframeTrend
    one_hour: EntryTiming

    # Aggregated signal
    aligned: bool
    direction: Literal["LONG", "SHORT", "HOLD"]
    alignment_score: float  # 0-100
    entry_ready: bool

    # Timestamps
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MultiTimeframeEngine:
    """
    Coordinates multi-timeframe analysis for high-conviction trading.

    Implements the "trade less" philosophy by requiring alignment
    across Daily, 4H, and 1H timeframes before generating signals.
    """

    # EMA periods by timeframe
    EMA_PERIODS = {
        "daily": {"short": 21, "long": 200},
        "4h": {"short": 21, "long": 50},
        "1h": {"short": 9, "long": 21},
    }

    # ADX thresholds
    ADX_TRENDING = 25
    ADX_STRONG_TREND = 35
    ADX_RANGING = 20

    def __init__(self) -> None:
        """Initialize the multi-timeframe engine."""
        self._daily_candles: list[Candle] = []
        self._4h_candles: list[Candle] = []
        self._1h_candles: list[Candle] = []

        logger.info("MultiTimeframeEngine initialized")

    def update_candles(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle] | None = None,
        candles_daily: list[Candle] | None = None,
    ) -> None:
        """
        Update candle buffers for all timeframes.

        If 4H/Daily candles not provided, they will be aggregated from 1H.

        Args:
            candles_1h: 1-hour candles (required)
            candles_4h: 4-hour candles (optional, aggregated if not provided)
            candles_daily: Daily candles (optional, aggregated if not provided)
        """
        self._1h_candles = candles_1h

        if candles_4h is not None:
            self._4h_candles = candles_4h
        else:
            self._4h_candles = self._aggregate_candles(candles_1h, 4)

        if candles_daily is not None:
            self._daily_candles = candles_daily
        else:
            self._daily_candles = self._aggregate_candles(candles_1h, 24)

        logger.debug(
            f"MTF candles updated: 1H={len(self._1h_candles)}, "
            f"4H={len(self._4h_candles)}, Daily={len(self._daily_candles)}"
        )

    def analyze(self) -> MTFSignal:
        """
        Perform multi-timeframe analysis.

        Returns:
            MTFSignal with aligned direction or HOLD if not aligned
        """
        if not self._has_sufficient_data():
            return self._create_insufficient_data_signal()

        # Analyze each timeframe
        daily = self._analyze_daily()
        four_hour = self._analyze_4h()
        one_hour = self._analyze_1h_entry()

        # Check alignment
        aligned, direction, alignment_score = self._check_alignment(
            daily, four_hour, one_hour
        )

        # Determine if entry is ready
        entry_ready = aligned and one_hour.action == "NOW"

        signal = MTFSignal(
            daily=daily,
            four_hour=four_hour,
            one_hour=one_hour,
            aligned=aligned,
            direction=direction,
            alignment_score=alignment_score,
            entry_ready=entry_ready,
        )

        logger.info(
            f"MTF Analysis: Daily={daily.direction}({daily.strength:.0f}) "
            f"4H={four_hour.direction}({four_hour.strength:.0f}) "
            f"1H={one_hour.momentum} | Aligned={aligned} → {direction}"
        )

        return signal

    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data for analysis."""
        # Adjusted thresholds for realistic data availability:
        # - 1H: 50 candles (2 days)
        # - 4H: 20 candles (3+ days)
        # - Daily: 15 candles (2+ weeks)
        return (
            len(self._1h_candles) >= 50
            and len(self._4h_candles) >= 20
            and len(self._daily_candles) >= 15
        )

    def _create_insufficient_data_signal(self) -> MTFSignal:
        """Create a HOLD signal due to insufficient data."""
        empty_trend = TimeframeTrend(
            direction="NEUTRAL",
            strength=0.0,
            ema_short=0.0,
            ema_long=0.0,
            price=0.0,
            confidence=0.0,
        )
        empty_entry = EntryTiming(
            action="ABORT",
            reason="Insufficient data",
            optimal_entry_price=None,
            momentum="NEUTRAL",
            rsi=50.0,
            distance_from_ema21=0.0,
        )
        return MTFSignal(
            daily=empty_trend,
            four_hour=empty_trend,
            one_hour=empty_entry,
            aligned=False,
            direction="HOLD",
            alignment_score=0.0,
            entry_ready=False,
        )

    def _analyze_daily(self) -> TimeframeTrend:
        """Analyze daily timeframe for trend direction."""
        closes = np.array([c.close for c in self._daily_candles])
        highs = np.array([c.high for c in self._daily_candles])
        lows = np.array([c.low for c in self._daily_candles])

        # Calculate EMAs
        ema_short = self._calculate_ema(closes, self.EMA_PERIODS["daily"]["short"])
        ema_long = self._calculate_ema(closes, self.EMA_PERIODS["daily"]["long"])

        # Calculate ADX
        adx = self._calculate_adx(highs, lows, closes)

        # Current price
        price = closes[-1]

        # Determine direction
        if price > ema_long and price > ema_short:
            if adx >= self.ADX_TRENDING:
                direction = "BULL"
                confidence = min(1.0, adx / 50)
            else:
                direction = "NEUTRAL"
                confidence = 0.5
        elif price < ema_long and price < ema_short:
            if adx >= self.ADX_TRENDING:
                direction = "BEAR"
                confidence = min(1.0, adx / 50)
            else:
                direction = "NEUTRAL"
                confidence = 0.5
        else:
            direction = "NEUTRAL"
            confidence = 0.3

        return TimeframeTrend(
            direction=direction,
            strength=adx,
            ema_short=ema_short,
            ema_long=ema_long,
            price=price,
            confidence=confidence,
        )

    def _analyze_4h(self) -> TimeframeTrend:
        """Analyze 4H timeframe for signal generation."""
        closes = np.array([c.close for c in self._4h_candles])
        highs = np.array([c.high for c in self._4h_candles])
        lows = np.array([c.low for c in self._4h_candles])

        # Calculate EMAs
        ema_short = self._calculate_ema(closes, self.EMA_PERIODS["4h"]["short"])
        ema_long = self._calculate_ema(closes, self.EMA_PERIODS["4h"]["long"])

        # Calculate ADX
        adx = self._calculate_adx(highs, lows, closes)

        # Current price
        price = closes[-1]

        # Determine direction with more sensitivity than daily
        price_above_short = price > ema_short
        price_above_long = price > ema_long
        short_above_long = ema_short > ema_long

        if price_above_short and price_above_long and short_above_long:
            direction = "BULL"
            confidence = min(1.0, (adx / 40) * 0.7 + 0.3)
        elif not price_above_short and not price_above_long and not short_above_long:
            direction = "BEAR"
            confidence = min(1.0, (adx / 40) * 0.7 + 0.3)
        else:
            direction = "NEUTRAL"
            confidence = 0.4

        return TimeframeTrend(
            direction=direction,
            strength=adx,
            ema_short=ema_short,
            ema_long=ema_long,
            price=price,
            confidence=confidence,
        )

    def _analyze_1h_entry(self) -> EntryTiming:
        """Analyze 1H timeframe for entry timing."""
        closes = np.array([c.close for c in self._1h_candles])

        price = closes[-1]

        # Calculate EMA21 for entry zone
        ema_21 = self._calculate_ema(closes, 21)
        distance_from_ema21 = ((price - ema_21) / ema_21) * 100

        # Calculate RSI
        rsi = self._calculate_rsi(closes)

        # Determine momentum
        ema_9 = self._calculate_ema(closes, 9)
        if price > ema_9 and ema_9 > ema_21:
            momentum = "BULLISH"
        elif price < ema_9 and ema_9 < ema_21:
            momentum = "BEARISH"
        else:
            momentum = "NEUTRAL"

        # Determine entry action
        action, reason, optimal_price = self._determine_entry_action(
            price, ema_21, distance_from_ema21, rsi, momentum
        )

        return EntryTiming(
            action=action,
            reason=reason,
            optimal_entry_price=optimal_price,
            momentum=momentum,
            rsi=rsi,
            distance_from_ema21=distance_from_ema21,
        )

    def _determine_entry_action(
        self,
        price: float,
        ema_21: float,
        distance: float,
        rsi: float,
        momentum: str,
    ) -> tuple[Literal["NOW", "WAIT", "ABORT"], str, float | None]:
        """Determine entry action based on 1H conditions."""
        # Abort conditions
        if rsi > 80 or rsi < 20:
            return "ABORT", f"RSI extreme ({rsi:.0f})", None

        if abs(distance) > 3.0:
            return "ABORT", f"Too far from EMA21 ({distance:.1f}%)", None

        # NOW conditions - near EMA21 pullback
        if abs(distance) < 0.5:
            return "NOW", "Price at EMA21 zone", price

        # WAIT for pullback
        if distance > 1.0:
            return "WAIT", "Wait for pullback to EMA21", ema_21 * 1.003

        if distance < -1.0:
            return "WAIT", "Wait for pullback to EMA21", ema_21 * 0.997

        # Default: NOW if momentum aligns
        if momentum in ("BULLISH", "BEARISH"):
            return "NOW", f"Momentum aligned ({momentum})", price

        return "WAIT", "Waiting for momentum confirmation", ema_21

    def _check_alignment(
        self,
        daily: TimeframeTrend,
        four_hour: TimeframeTrend,
        one_hour: EntryTiming,
    ) -> tuple[bool, Literal["LONG", "SHORT", "HOLD"], float]:
        """
        Check if all timeframes are aligned.

        Returns:
            Tuple of (aligned, direction, alignment_score)
        """
        score = 0.0

        # Daily + 4H must agree for base alignment
        daily_bull = daily.direction == "BULL"
        daily_bear = daily.direction == "BEAR"
        four_hour_bull = four_hour.direction == "BULL"
        four_hour_bear = four_hour.direction == "BEAR"
        one_hour_bull = one_hour.momentum == "BULLISH"
        one_hour_bear = one_hour.momentum == "BEARISH"

        # Check bullish alignment
        if daily_bull and four_hour_bull:
            score += 60.0
            if one_hour_bull:
                score += 30.0
            elif one_hour.momentum == "NEUTRAL":
                score += 10.0

            if score >= 60:
                return True, "LONG", min(100.0, score + daily.strength * 0.2)

        # Check bearish alignment
        if daily_bear and four_hour_bear:
            score += 60.0
            if one_hour_bear:
                score += 30.0
            elif one_hour.momentum == "NEUTRAL":
                score += 10.0

            if score >= 60:
                return True, "SHORT", min(100.0, score + daily.strength * 0.2)

        # Neutral daily allows 4H to lead with reduced score
        if daily.direction == "NEUTRAL":
            if four_hour_bull and one_hour_bull:
                return True, "LONG", 50.0
            if four_hour_bear and one_hour_bear:
                return True, "SHORT", 50.0

        return False, "HOLD", score

    def _aggregate_candles(self, candles: list[Candle], hours: int) -> list[Candle]:
        """Aggregate 1H candles into higher timeframe candles."""
        if not candles:
            return []

        aggregated = []
        chunk_size = hours

        for i in range(0, len(candles) - chunk_size + 1, chunk_size):
            chunk = candles[i : i + chunk_size]
            if len(chunk) < chunk_size:
                continue

            agg_candle = Candle(
                symbol=chunk[0].symbol,
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
                interval=f"{hours}h" if hours < 24 else "1d",
            )
            aggregated.append(agg_candle)

        return aggregated

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0

        multiplier = 2 / (period + 1)
        ema = float(data[0])

        for price in data[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Calculate ADX indicator."""
        if len(closes) < period + 1:
            return 0.0

        # True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr = self._wilder_smooth(tr, period)
        plus_di = 100 * self._wilder_smooth(plus_dm, period) / atr if atr > 0 else 0
        minus_di = 100 * self._wilder_smooth(minus_dm, period) / atr if atr > 0 else 0

        # DX and ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / di_sum
        return float(dx)

    def _wilder_smooth(self, data: np.ndarray, period: int) -> float:
        """Wilder's smoothing method."""
        if len(data) < period:
            return float(np.mean(data)) if len(data) > 0 else 0.0

        smoothed: float = float(np.mean(data[:period]))
        for val in data[period:]:
            smoothed = (smoothed * (period - 1) + float(val)) / period

        return smoothed
