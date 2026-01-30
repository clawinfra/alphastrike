"""
AlphaStrike Trading Bot - Market Regime Detector

Detects current market regime based on technical indicators.
Used for adaptive strategy adjustments and confidence scaling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from src.core.config import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Current market regime state with supporting metrics."""

    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification
    adx: float
    atr_ratio: float
    trend_direction: Literal["up", "down", "neutral"]


class RegimeDetector:
    """
    Detects market regime from technical features.

    Regime Classification Priority:
    1. EXTREME_VOLATILITY: ATR_ratio > 2.0 (overrides others)
    2. HIGH_VOLATILITY: ATR_ratio > 1.5
    3. TREND_EXHAUSTION: ADX > 60 AND ADX_ROC < -2
    4. TRENDING_UP: ADX > 25, Price > EMA20, +DI > -DI
    5. TRENDING_DOWN: ADX > 25, Price < EMA20, -DI > +DI
    6. RANGING: ADX < 20 (default fallback)

    Feature Requirements:
    - adx: Average Directional Index (14-period)
    - plus_di: Positive Directional Indicator
    - minus_di: Negative Directional Indicator
    - adx_roc: ADX Rate of Change (5-period)
    - atr_ratio: ATR / ATR_20MA ratio
    - price: Current price
    - ema_20: 20-period EMA
    """

    # Thresholds for regime detection
    ADX_TRENDING_THRESHOLD = 25.0
    ADX_RANGING_THRESHOLD = 20.0
    ADX_EXHAUSTION_THRESHOLD = 60.0
    ADX_ROC_EXHAUSTION_THRESHOLD = -2.0
    ATR_HIGH_VOLATILITY_THRESHOLD = 1.5
    ATR_EXTREME_VOLATILITY_THRESHOLD = 2.0

    def detect_regime(self, features: dict) -> RegimeState:
        """
        Detect current market regime from features.

        Args:
            features: Dictionary containing required technical indicators:
                - adx: ADX value (0-100)
                - plus_di: +DI value
                - minus_di: -DI value
                - adx_roc: ADX rate of change
                - atr_ratio: ATR / ATR_20MA
                - price: Current price
                - ema_20: 20-period EMA

        Returns:
            RegimeState with detected regime and confidence
        """
        # Extract features with defaults
        adx = features.get("adx", 0.0)
        plus_di = features.get("plus_di", 0.0)
        minus_di = features.get("minus_di", 0.0)
        adx_roc = features.get("adx_roc", 0.0)
        atr_ratio = features.get("atr_ratio", 1.0)
        price = features.get("price", 0.0)
        ema_20 = features.get("ema_20", 0.0)

        # Determine trend direction
        trend_direction = self._determine_trend_direction(
            price=price,
            ema_20=ema_20,
            plus_di=plus_di,
            minus_di=minus_di,
        )

        # Priority 1: Check EXTREME_VOLATILITY (overrides all)
        if atr_ratio > self.ATR_EXTREME_VOLATILITY_THRESHOLD:
            confidence = self._calculate_volatility_confidence(
                atr_ratio, self.ATR_EXTREME_VOLATILITY_THRESHOLD
            )
            return RegimeState(
                regime=MarketRegime.EXTREME_VOLATILITY,
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                trend_direction=trend_direction,
            )

        # Priority 2: Check HIGH_VOLATILITY
        if atr_ratio > self.ATR_HIGH_VOLATILITY_THRESHOLD:
            confidence = self._calculate_volatility_confidence(
                atr_ratio, self.ATR_HIGH_VOLATILITY_THRESHOLD
            )
            return RegimeState(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                trend_direction=trend_direction,
            )

        # Priority 3: Check TREND_EXHAUSTION
        if adx > self.ADX_EXHAUSTION_THRESHOLD and adx_roc < self.ADX_ROC_EXHAUSTION_THRESHOLD:
            confidence = self._calculate_exhaustion_confidence(adx, adx_roc)
            return RegimeState(
                regime=MarketRegime.TREND_EXHAUSTION,
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                trend_direction=trend_direction,
            )

        # Priority 4: Check TRENDING_UP
        if (
            adx > self.ADX_TRENDING_THRESHOLD
            and price > ema_20
            and plus_di > minus_di
        ):
            confidence = self._calculate_trend_confidence(adx, plus_di, minus_di)
            return RegimeState(
                regime=MarketRegime.TRENDING_UP,
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                trend_direction="up",
            )

        # Priority 5: Check TRENDING_DOWN
        if (
            adx > self.ADX_TRENDING_THRESHOLD
            and price < ema_20
            and minus_di > plus_di
        ):
            confidence = self._calculate_trend_confidence(adx, minus_di, plus_di)
            return RegimeState(
                regime=MarketRegime.TRENDING_DOWN,
                confidence=confidence,
                adx=adx,
                atr_ratio=atr_ratio,
                trend_direction="down",
            )

        # Priority 6: Default to RANGING
        confidence = self._calculate_ranging_confidence(adx)
        return RegimeState(
            regime=MarketRegime.RANGING,
            confidence=confidence,
            adx=adx,
            atr_ratio=atr_ratio,
            trend_direction=trend_direction,
        )

    def _determine_trend_direction(
        self,
        price: float,
        ema_20: float,
        plus_di: float,
        minus_di: float,
    ) -> Literal["up", "down", "neutral"]:
        """
        Determine overall trend direction.

        Args:
            price: Current price
            ema_20: 20-period EMA
            plus_di: +DI value
            minus_di: -DI value

        Returns:
            "up", "down", or "neutral"
        """
        # Count bullish/bearish signals
        bullish_signals = 0
        bearish_signals = 0

        if ema_20 > 0:
            if price > ema_20:
                bullish_signals += 1
            elif price < ema_20:
                bearish_signals += 1

        if plus_di > minus_di:
            bullish_signals += 1
        elif minus_di > plus_di:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            return "up"
        elif bearish_signals > bullish_signals:
            return "down"
        return "neutral"

    def _calculate_volatility_confidence(
        self, atr_ratio: float, threshold: float
    ) -> float:
        """
        Calculate confidence for volatility regime.

        Higher ATR ratio = higher confidence.
        Confidence scales from 0.5 at threshold to 1.0 at 2x threshold.

        Args:
            atr_ratio: Current ATR ratio
            threshold: Volatility threshold

        Returns:
            Confidence score (0-1)
        """
        # Base confidence of 0.5 at threshold
        excess = atr_ratio - threshold
        # Scale to 1.0 at 2x threshold
        confidence = 0.5 + (excess / threshold) * 0.5
        return min(max(confidence, 0.0), 1.0)

    def _calculate_exhaustion_confidence(self, adx: float, adx_roc: float) -> float:
        """
        Calculate confidence for trend exhaustion regime.

        Higher ADX and more negative ADX_ROC = higher confidence.

        Args:
            adx: Current ADX value
            adx_roc: ADX rate of change

        Returns:
            Confidence score (0-1)
        """
        # ADX contribution: 60 = 0.5, 80 = 1.0
        adx_factor = min((adx - 60) / 20.0, 0.5)

        # ADX_ROC contribution: -2 = 0.5, -6 = 1.0
        roc_factor = min((abs(adx_roc) - 2) / 4.0, 0.5)

        confidence = 0.5 + adx_factor + roc_factor
        return min(max(confidence, 0.0), 1.0)

    def _calculate_trend_confidence(
        self, adx: float, dominant_di: float, other_di: float
    ) -> float:
        """
        Calculate confidence for trending regime.

        Higher ADX and larger DI spread = higher confidence.

        Args:
            adx: Current ADX value
            dominant_di: The dominant DI (+DI for up, -DI for down)
            other_di: The non-dominant DI

        Returns:
            Confidence score (0-1)
        """
        # ADX contribution: 25 = 0.3, 50 = 0.6
        adx_factor = min((adx - 25) / 50.0, 0.3) + 0.3

        # DI spread contribution
        di_spread = dominant_di - other_di
        # Spread of 10 = 0.2, spread of 30 = 0.4
        di_factor = min(di_spread / 75.0, 0.4)

        confidence = adx_factor + di_factor
        return min(max(confidence, 0.0), 1.0)

    def _calculate_ranging_confidence(self, adx: float) -> float:
        """
        Calculate confidence for ranging regime.

        Lower ADX = higher confidence in ranging.

        Args:
            adx: Current ADX value

        Returns:
            Confidence score (0-1)
        """
        if adx < self.ADX_RANGING_THRESHOLD:
            # Strong ranging: ADX < 20
            # ADX 20 = 0.6, ADX 10 = 0.8, ADX 0 = 1.0
            confidence = 0.6 + (self.ADX_RANGING_THRESHOLD - adx) / 50.0
        else:
            # Weak ranging: 20 <= ADX < 25
            # ADX 20 = 0.6, ADX 25 = 0.4
            confidence = 0.6 - (adx - self.ADX_RANGING_THRESHOLD) / 12.5

        return min(max(confidence, 0.0), 1.0)
