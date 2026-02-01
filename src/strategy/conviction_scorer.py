"""
AlphaStrike Trading Bot - Conviction Scoring System

Implements the 5-factor conviction scoring for trade quality assessment.
Only trades with conviction >= 70 are executed.

Scoring Factors:
- Timeframe Alignment: +30 pts (Daily/4H/1H agree)
- Ensemble Confidence: +25 pts (>80% model agreement)
- Regime Clarity: +20 pts (clear trend or range)
- Volume Confirmation: +15 pts (above average volume)
- Technical Setup: +10 pts (key level interaction)

Position Sizing Tiers:
- Score < 70: NO TRADE
- Score 70-84: SMALL (15% position, 0.25% risk)
- Score 85-94: MEDIUM (30% position, 0.40% risk)
- Score 95+: LARGE (50% position, 0.50% risk)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class PositionTier(Enum):
    """Position sizing tier based on conviction score."""

    NO_TRADE = "no_trade"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class ConvictionBreakdown:
    """Detailed breakdown of conviction score components."""

    timeframe_alignment: float  # 0-30
    ensemble_confidence: float  # 0-25
    regime_clarity: float  # 0-20
    volume_confirmation: float  # 0-15
    technical_setup: float  # 0-10

    @property
    def total(self) -> float:
        """Total conviction score (0-100)."""
        return (
            self.timeframe_alignment
            + self.ensemble_confidence
            + self.regime_clarity
            + self.volume_confirmation
            + self.technical_setup
        )


@dataclass
class ConvictionResult:
    """Result of conviction scoring."""

    score: float  # 0-100
    tier: PositionTier
    breakdown: ConvictionBreakdown
    position_size_pct: float  # 0.0 - 0.50
    risk_per_trade_pct: float  # 0.0 - 0.50
    stop_atr_multiplier: float  # 0.8 - 1.2
    signal: Literal["LONG", "SHORT", "HOLD"]
    reason: str


@dataclass
class TimeframeSignals:
    """Signals from multiple timeframes."""

    daily_trend: Literal["BULL", "BEAR", "NEUTRAL"]
    daily_adx: float
    four_hour_signal: Literal["LONG", "SHORT", "HOLD"]
    four_hour_confidence: float
    one_hour_momentum: Literal["BULLISH", "BEARISH", "NEUTRAL"]


@dataclass
class MarketContext:
    """Current market context for scoring."""

    regime: Literal["TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOLATILITY", "EXTREME_VOLATILITY"]
    regime_confidence: float  # 0-1
    volume_ratio: float  # Current volume / 20-period average
    atr_ratio: float  # Current ATR / 20-period average
    rsi: float  # 0-100
    price_vs_ema50: float  # % distance from EMA50
    price_vs_ema200: float  # % distance from EMA200
    bb_position: float  # -1 (below lower) to +1 (above upper)
    model_agreement_pct: float  # 0-1, percentage of models agreeing


class ConvictionScorer:
    """
    Calculates conviction scores for trade decisions.

    The conviction score determines whether to trade and at what size.
    It synthesizes multiple factors into a single 0-100 score.
    """

    # Score thresholds
    MIN_TRADE_SCORE = 70
    SMALL_TIER_MAX = 84
    MEDIUM_TIER_MAX = 94

    # Position sizing by tier
    TIER_SIZING = {
        PositionTier.NO_TRADE: {"position_pct": 0.0, "risk_pct": 0.0, "stop_atr": 0.0},
        PositionTier.SMALL: {"position_pct": 0.15, "risk_pct": 0.0025, "stop_atr": 1.2},
        PositionTier.MEDIUM: {"position_pct": 0.30, "risk_pct": 0.004, "stop_atr": 1.0},
        PositionTier.LARGE: {"position_pct": 0.50, "risk_pct": 0.005, "stop_atr": 0.8},
    }

    def __init__(self) -> None:
        """Initialize the conviction scorer."""
        logger.info("ConvictionScorer initialized")

    def calculate(
        self,
        timeframe_signals: TimeframeSignals,
        market_context: MarketContext,
    ) -> ConvictionResult:
        """
        Calculate conviction score and determine position tier.

        Args:
            timeframe_signals: Signals from Daily/4H/1H timeframes
            market_context: Current market conditions

        Returns:
            ConvictionResult with score, tier, and sizing parameters
        """
        # Calculate each component
        tf_score = self._score_timeframe_alignment(timeframe_signals)
        ensemble_score = self._score_ensemble_confidence(
            timeframe_signals.four_hour_confidence,
            market_context.model_agreement_pct,
        )
        regime_score = self._score_regime_clarity(market_context)
        volume_score = self._score_volume_confirmation(market_context.volume_ratio)
        technical_score = self._score_technical_setup(market_context)

        breakdown = ConvictionBreakdown(
            timeframe_alignment=tf_score,
            ensemble_confidence=ensemble_score,
            regime_clarity=regime_score,
            volume_confirmation=volume_score,
            technical_setup=technical_score,
        )

        total_score = breakdown.total
        tier = self._determine_tier(total_score)
        sizing = self.TIER_SIZING[tier]

        # Determine final signal direction
        signal = self._determine_signal(timeframe_signals, total_score)

        # Build reason string
        reason = self._build_reason(breakdown, tier, signal)

        result = ConvictionResult(
            score=total_score,
            tier=tier,
            breakdown=breakdown,
            position_size_pct=sizing["position_pct"],
            risk_per_trade_pct=sizing["risk_pct"],
            stop_atr_multiplier=sizing["stop_atr"],
            signal=signal,
            reason=reason,
        )

        logger.info(
            f"Conviction: {total_score:.1f} ({tier.value}) | "
            f"Signal: {signal} | TF:{tf_score:.0f} EN:{ensemble_score:.0f} "
            f"RG:{regime_score:.0f} VL:{volume_score:.0f} TC:{technical_score:.0f}"
        )

        return result

    def _score_timeframe_alignment(self, signals: TimeframeSignals) -> float:
        """
        Score timeframe alignment (0-30 points).

        Full points when all three timeframes agree on direction.
        """
        score = 0.0

        # Determine if 4H signal aligns with Daily trend
        four_hour_bullish = signals.four_hour_signal == "LONG"
        four_hour_bearish = signals.four_hour_signal == "SHORT"

        daily_bullish = signals.daily_trend == "BULL"
        daily_bearish = signals.daily_trend == "BEAR"

        one_hour_bullish = signals.one_hour_momentum == "BULLISH"
        one_hour_bearish = signals.one_hour_momentum == "BEARISH"

        # All three align bullish
        if four_hour_bullish and daily_bullish and one_hour_bullish:
            score = 30.0
        # All three align bearish
        elif four_hour_bearish and daily_bearish and one_hour_bearish:
            score = 30.0
        # Daily + 4H align, 1H neutral or slightly misaligned
        elif (four_hour_bullish and daily_bullish) or (four_hour_bearish and daily_bearish):
            score = 22.0
        # Only 4H + 1H align (counter-trend to daily)
        elif (four_hour_bullish and one_hour_bullish) or (four_hour_bearish and one_hour_bearish):
            score = 10.0
        # No alignment
        else:
            score = 0.0

        # Bonus for strong daily ADX
        if signals.daily_adx > 30:
            score = min(30.0, score + 3.0)

        return score

    def _score_ensemble_confidence(
        self,
        four_hour_confidence: float,
        model_agreement_pct: float,
    ) -> float:
        """
        Score ensemble confidence (0-25 points).

        Based on ML model confidence and inter-model agreement.
        """
        score = 0.0

        # Base score from 4H confidence (0-15 points)
        # Confidence ranges from 0.5 to 1.0 typically
        if four_hour_confidence >= 0.85:
            score += 15.0
        elif four_hour_confidence >= 0.75:
            score += 12.0
        elif four_hour_confidence >= 0.65:
            score += 8.0
        elif four_hour_confidence >= 0.55:
            score += 4.0

        # Model agreement bonus (0-10 points)
        # Agreement is 0.25 (1/4) to 1.0 (4/4)
        if model_agreement_pct >= 1.0:  # All 4 agree
            score += 10.0
        elif model_agreement_pct >= 0.75:  # 3/4 agree
            score += 6.0
        elif model_agreement_pct >= 0.5:  # 2/4 agree
            score += 2.0
        # Less than 50% agreement = 0 bonus

        return min(25.0, score)

    def _score_regime_clarity(self, context: MarketContext) -> float:
        """
        Score regime clarity (0-20 points).

        Clear trending or ranging regimes score higher.
        High volatility regimes score lower (unpredictable).
        """
        score = 0.0

        # Regime type scoring
        if context.regime in ("TRENDING_UP", "TRENDING_DOWN"):
            score += 15.0
        elif context.regime == "RANGING":
            score += 10.0
        elif context.regime == "HIGH_VOLATILITY":
            score += 5.0
        elif context.regime == "EXTREME_VOLATILITY":
            score += 0.0  # Don't trade extreme volatility

        # Regime confidence bonus
        score += context.regime_confidence * 5.0

        return min(20.0, score)

    def _score_volume_confirmation(self, volume_ratio: float) -> float:
        """
        Score volume confirmation (0-15 points).

        Above-average volume confirms the move.
        """
        if volume_ratio >= 2.0:
            return 15.0  # Strong volume surge
        elif volume_ratio >= 1.5:
            return 12.0  # Good volume
        elif volume_ratio >= 1.2:
            return 9.0  # Above average
        elif volume_ratio >= 1.0:
            return 6.0  # Average
        elif volume_ratio >= 0.8:
            return 3.0  # Below average
        else:
            return 0.0  # Low volume = weak move

    def _score_technical_setup(self, context: MarketContext) -> float:
        """
        Score technical setup quality (0-10 points).

        Key level interactions and indicator positioning.
        """
        score = 0.0

        # RSI in favorable zone (not extreme)
        if 30 <= context.rsi <= 70:
            score += 3.0
        elif 25 <= context.rsi <= 75:
            score += 1.5
        # Extreme RSI = 0 points

        # Price relationship to EMAs
        # Trend-following: price on same side as both EMAs
        if context.price_vs_ema50 > 0 and context.price_vs_ema200 > 0:
            score += 3.0  # Clear uptrend structure
        elif context.price_vs_ema50 < 0 and context.price_vs_ema200 < 0:
            score += 3.0  # Clear downtrend structure
        elif abs(context.price_vs_ema50) < 0.5:
            score += 1.0  # Near EMA50 (potential entry zone)

        # Bollinger Band position
        # Best entries: price near middle or bouncing from band
        if -0.3 <= context.bb_position <= 0.3:
            score += 2.0  # Near middle band
        elif -0.7 <= context.bb_position <= 0.7:
            score += 1.0  # Within bands

        # ATR ratio (prefer normal volatility)
        if 0.8 <= context.atr_ratio <= 1.3:
            score += 2.0  # Normal volatility
        elif 0.6 <= context.atr_ratio <= 1.5:
            score += 1.0  # Acceptable volatility

        return min(10.0, score)

    def _determine_tier(self, score: float) -> PositionTier:
        """Determine position tier from score."""
        if score < self.MIN_TRADE_SCORE:
            return PositionTier.NO_TRADE
        elif score <= self.SMALL_TIER_MAX:
            return PositionTier.SMALL
        elif score <= self.MEDIUM_TIER_MAX:
            return PositionTier.MEDIUM
        else:
            return PositionTier.LARGE

    def _determine_signal(
        self,
        signals: TimeframeSignals,
        score: float,
    ) -> Literal["LONG", "SHORT", "HOLD"]:
        """Determine final signal direction."""
        if score < self.MIN_TRADE_SCORE:
            return "HOLD"

        # Use 4H signal as primary direction
        if signals.four_hour_signal == "LONG":
            # Verify daily trend doesn't contradict
            if signals.daily_trend != "BEAR":
                return "LONG"
        elif signals.four_hour_signal == "SHORT":
            if signals.daily_trend != "BULL":
                return "SHORT"

        return "HOLD"

    def _build_reason(
        self,
        breakdown: ConvictionBreakdown,
        tier: PositionTier,
        signal: str,
    ) -> str:
        """Build human-readable reason string."""
        if tier == PositionTier.NO_TRADE:
            weak_factors = []
            if breakdown.timeframe_alignment < 20:
                weak_factors.append("TF misalignment")
            if breakdown.ensemble_confidence < 15:
                weak_factors.append("low ML confidence")
            if breakdown.regime_clarity < 10:
                weak_factors.append("unclear regime")
            if breakdown.volume_confirmation < 6:
                weak_factors.append("weak volume")

            return f"NO TRADE: {', '.join(weak_factors) or 'score below threshold'}"

        strengths = []
        if breakdown.timeframe_alignment >= 25:
            strengths.append("strong TF alignment")
        if breakdown.ensemble_confidence >= 20:
            strengths.append("high ML confidence")
        if breakdown.regime_clarity >= 15:
            strengths.append("clear regime")
        if breakdown.volume_confirmation >= 12:
            strengths.append("volume confirmed")

        return f"{signal} ({tier.value}): {', '.join(strengths) or 'threshold met'}"
