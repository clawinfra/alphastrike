"""
AlphaStrike Trading Bot - Signal Processor with Filters (US-023)

Filters and scales signals before execution based on technical indicators,
regime alignment, and model agreement metrics.

Implements multi-layer signal filtering with position scaling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from src.core.config import MarketRegime

logger = logging.getLogger(__name__)

# Type aliases
SignalType = Literal["LONG", "SHORT", "HOLD"]


@dataclass
class SignalResult:
    """
    Result of signal processing with filters applied.

    Attributes:
        signal: Final trading signal after filtering ("LONG", "SHORT", "HOLD")
        confidence: Confidence level in the signal (0-1)
        position_scale: Position size multiplier (0.05 to 1.0)
        filters_applied: List of all filters that were evaluated
        filters_triggered: List of filters that modified or blocked the signal
    """

    signal: SignalType
    confidence: float
    position_scale: float
    filters_applied: list[str] = field(default_factory=list)
    filters_triggered: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate position_scale bounds."""
        self.position_scale = max(0.05, min(1.0, self.position_scale))


class SignalProcessor:
    """
    Signal processor that filters and scales signals before execution.

    Signal Filters:
    1. RSI Extreme Filter: RSI < 30 blocks SHORT, RSI > 70 blocks LONG
    2. ADX Saturation Filter: ADX > 95 or ADX < 10 blocks signal
    3. Regime Alignment: Counter-trend scales to 40%
    4. MTF Misalignment: 4H trend opposes signal scales to 50%
    5. Model Agreement Filter: Agreement < 25% = HOLD
    6. Reversal Detection Override: Strong reversal signals bypass filters

    Position Scaling Matrix (cumulative):
    - Base: 1.0
    - + Counter-trend: 0.4 (result: 0.4)
    - + MTF misalignment: 0.5 (result: 0.2)
    - + Low agreement: 0.6 (result: 0.12)
    - Minimum floor: 0.05

    Example:
        processor = SignalProcessor()
        features = {
            "rsi": 45.0,
            "adx": 35.0,
            "trend_4h": "up",
        }
        result = processor.process_signal(
            raw_signal="LONG",
            confidence=0.75,
            features=features,
            model_agreement=0.80
        )
        # result.signal, result.position_scale
    """

    # Filter thresholds
    RSI_OVERSOLD_THRESHOLD: float = 30.0
    RSI_OVERBOUGHT_THRESHOLD: float = 70.0
    ADX_MIN_THRESHOLD: float = 10.0
    ADX_MAX_THRESHOLD: float = 95.0
    MODEL_AGREEMENT_MIN: float = 0.25
    REVERSAL_STRENGTH_THRESHOLD: float = 0.85

    # Position scaling factors
    COUNTER_TREND_SCALE: float = 0.4
    MTF_MISALIGNMENT_SCALE: float = 0.5
    LOW_AGREEMENT_SCALE: float = 0.6
    MIN_POSITION_SCALE: float = 0.05

    def __init__(self) -> None:
        """Initialize SignalProcessor."""
        logger.info("SignalProcessor initialized")

    def process_signal(
        self,
        raw_signal: SignalType,
        confidence: float,
        features: dict,
        model_agreement: float,
    ) -> SignalResult:
        """
        Process a raw signal through all filters and calculate position scale.

        This is the main entry point for signal processing. It applies filters
        in sequence and calculates the final position scale based on market
        conditions and model agreement.

        Args:
            raw_signal: Raw trading signal from ML ensemble ("LONG", "SHORT", "HOLD")
            confidence: Confidence level from model ensemble (0-1)
            features: Dictionary of technical features:
                - rsi: RSI indicator value (0-100)
                - adx: ADX indicator value (0-100)
                - trend_4h: 4-hour trend direction ("up", "down", "neutral")
                - reversal_strength: Optional reversal detection strength (0-1)
            model_agreement: Percentage of models agreeing on signal (0-1)

        Returns:
            SignalResult with final signal, confidence, position scale, and filter info
        """
        filters_applied: list[str] = []
        filters_triggered: list[str] = []

        # HOLD signals pass through unchanged
        if raw_signal == "HOLD":
            return SignalResult(
                signal="HOLD",
                confidence=confidence,
                position_scale=0.0,
                filters_applied=["hold_passthrough"],
                filters_triggered=[],
            )

        # Check for reversal override first
        reversal_strength = features.get("reversal_strength", 0.0)
        filters_applied.append("reversal_detection")
        if reversal_strength >= self.REVERSAL_STRENGTH_THRESHOLD:
            filters_triggered.append("reversal_override")
            logger.info(
                "Reversal override bypassing filters",
                extra={
                    "signal": raw_signal,
                    "reversal_strength": reversal_strength,
                },
            )
            # Strong reversal bypasses all filters
            return SignalResult(
                signal=raw_signal,
                confidence=confidence,
                position_scale=1.0,
                filters_applied=filters_applied,
                filters_triggered=filters_triggered,
            )

        # Apply signal filters
        filtered_signal, filter_triggers = self._apply_filters(
            signal=raw_signal,
            features=features,
        )
        filters_applied.extend([
            "rsi_extreme",
            "adx_saturation",
            "model_agreement",
        ])
        filters_triggered.extend(filter_triggers)

        # If signal was blocked by filters, return HOLD
        if filtered_signal == "HOLD":
            return SignalResult(
                signal="HOLD",
                confidence=0.0,
                position_scale=0.0,
                filters_applied=filters_applied,
                filters_triggered=filters_triggered,
            )

        # Apply model agreement filter
        if model_agreement < self.MODEL_AGREEMENT_MIN:
            filters_triggered.append(f"low_agreement({model_agreement:.2f})")
            logger.info(
                "Signal blocked by model agreement filter",
                extra={
                    "signal": raw_signal,
                    "model_agreement": model_agreement,
                    "threshold": self.MODEL_AGREEMENT_MIN,
                },
            )
            return SignalResult(
                signal="HOLD",
                confidence=0.0,
                position_scale=0.0,
                filters_applied=filters_applied,
                filters_triggered=filters_triggered,
            )

        # Get regime for position scaling
        regime = features.get("regime", "ranging")
        if isinstance(regime, MarketRegime):
            regime = regime.value

        # Calculate position scale
        filters_applied.extend(["regime_alignment", "mtf_alignment"])
        position_scale, scale_triggers = self._calculate_position_scale(
            signal=filtered_signal,
            features=features,
            regime=regime,
            model_agreement=model_agreement,
        )
        filters_triggered.extend(scale_triggers)

        logger.info(
            "Signal processed",
            extra={
                "raw_signal": raw_signal,
                "final_signal": filtered_signal,
                "confidence": confidence,
                "position_scale": position_scale,
                "filters_triggered": filters_triggered,
            },
        )

        return SignalResult(
            signal=filtered_signal,
            confidence=confidence,
            position_scale=position_scale,
            filters_applied=filters_applied,
            filters_triggered=filters_triggered,
        )

    def _apply_filters(
        self,
        signal: SignalType,
        features: dict,
    ) -> tuple[SignalType, list[str]]:
        """
        Apply blocking filters to the signal.

        Filters:
        1. RSI Extreme Filter: RSI < 30 blocks SHORT, RSI > 70 blocks LONG
        2. ADX Saturation Filter: ADX > 95 or ADX < 10 blocks signal

        Args:
            signal: Raw trading signal ("LONG" or "SHORT")
            features: Dictionary with rsi, adx values

        Returns:
            Tuple of (filtered_signal, list of triggered filter names)
        """
        triggered: list[str] = []

        # Get indicator values with defaults
        rsi = features.get("rsi", 50.0)
        adx = features.get("adx", 25.0)

        # RSI Extreme Filter
        if signal == "SHORT" and rsi < self.RSI_OVERSOLD_THRESHOLD:
            triggered.append(f"rsi_oversold({rsi:.1f})")
            logger.info(
                "Signal blocked by RSI oversold filter",
                extra={
                    "signal": signal,
                    "rsi": rsi,
                    "threshold": self.RSI_OVERSOLD_THRESHOLD,
                },
            )
            return "HOLD", triggered

        if signal == "LONG" and rsi > self.RSI_OVERBOUGHT_THRESHOLD:
            triggered.append(f"rsi_overbought({rsi:.1f})")
            logger.info(
                "Signal blocked by RSI overbought filter",
                extra={
                    "signal": signal,
                    "rsi": rsi,
                    "threshold": self.RSI_OVERBOUGHT_THRESHOLD,
                },
            )
            return "HOLD", triggered

        # ADX Saturation Filter
        if adx > self.ADX_MAX_THRESHOLD:
            triggered.append(f"adx_saturated_high({adx:.1f})")
            logger.info(
                "Signal blocked by ADX saturation filter (too high)",
                extra={
                    "signal": signal,
                    "adx": adx,
                    "threshold": self.ADX_MAX_THRESHOLD,
                },
            )
            return "HOLD", triggered

        if adx < self.ADX_MIN_THRESHOLD:
            triggered.append(f"adx_saturated_low({adx:.1f})")
            logger.info(
                "Signal blocked by ADX saturation filter (too low)",
                extra={
                    "signal": signal,
                    "adx": adx,
                    "threshold": self.ADX_MIN_THRESHOLD,
                },
            )
            return "HOLD", triggered

        return signal, triggered

    def _calculate_position_scale(
        self,
        signal: SignalType,
        features: dict,
        regime: str,
        model_agreement: float,
    ) -> tuple[float, list[str]]:
        """
        Calculate position scale based on regime alignment and market conditions.

        Position Scaling Matrix (cumulative multiplication):
        - Base: 1.0
        - + Counter-trend: 0.4 (result: 0.4)
        - + MTF misalignment: 0.5 (result: 0.2)
        - + Low agreement: 0.6 (result: 0.12)
        - Minimum floor: 0.05

        Args:
            signal: Trading signal ("LONG" or "SHORT")
            features: Dictionary with trend_4h, etc.
            regime: Current market regime string
            model_agreement: Model agreement percentage (0-1)

        Returns:
            Tuple of (position_scale, list of triggered scale factors)
        """
        scale = 1.0
        triggered: list[str] = []

        # 1. Check regime alignment (counter-trend detection)
        is_counter_trend = self._is_counter_trend(signal, regime)
        if is_counter_trend:
            scale *= self.COUNTER_TREND_SCALE
            triggered.append(f"counter_trend(scale={self.COUNTER_TREND_SCALE})")
            logger.debug(
                "Counter-trend scaling applied",
                extra={
                    "signal": signal,
                    "regime": regime,
                    "new_scale": scale,
                },
            )

        # 2. Check MTF (multi-timeframe) alignment
        trend_4h = features.get("trend_4h", "neutral")
        is_mtf_misaligned = self._is_mtf_misaligned(signal, trend_4h)
        if is_mtf_misaligned:
            scale *= self.MTF_MISALIGNMENT_SCALE
            triggered.append(f"mtf_misalignment(scale={self.MTF_MISALIGNMENT_SCALE})")
            logger.debug(
                "MTF misalignment scaling applied",
                extra={
                    "signal": signal,
                    "trend_4h": trend_4h,
                    "new_scale": scale,
                },
            )

        # 3. Check for low model agreement (between 25% and higher thresholds)
        # Low agreement threshold is between minimum and mid-level
        low_agreement_threshold = 0.50  # Scale down if agreement < 50%
        if model_agreement < low_agreement_threshold:
            scale *= self.LOW_AGREEMENT_SCALE
            triggered.append(f"low_agreement(scale={self.LOW_AGREEMENT_SCALE})")
            logger.debug(
                "Low agreement scaling applied",
                extra={
                    "model_agreement": model_agreement,
                    "new_scale": scale,
                },
            )

        # Apply minimum floor
        final_scale = max(scale, self.MIN_POSITION_SCALE)
        if final_scale != scale:
            triggered.append(f"min_floor_applied({self.MIN_POSITION_SCALE})")

        return final_scale, triggered

    def _is_counter_trend(self, signal: SignalType, regime: str) -> bool:
        """
        Determine if signal is counter to the current regime.

        Counter-trend conditions:
        - LONG signal in TRENDING_DOWN regime
        - SHORT signal in TRENDING_UP regime

        Args:
            signal: Trading signal ("LONG" or "SHORT")
            regime: Current market regime string

        Returns:
            True if signal is counter to trend
        """
        if signal == "LONG" and regime in ("trending_down", "trend_exhaustion"):
            return True
        if signal == "SHORT" and regime == "trending_up":
            return True
        return False

    def _is_mtf_misaligned(
        self,
        signal: SignalType,
        trend_4h: str,
    ) -> bool:
        """
        Determine if signal is misaligned with 4-hour trend.

        Misalignment conditions:
        - LONG signal when 4H trend is "down"
        - SHORT signal when 4H trend is "up"

        Args:
            signal: Trading signal ("LONG" or "SHORT")
            trend_4h: 4-hour trend direction ("up", "down", "neutral")

        Returns:
            True if signal opposes 4H trend
        """
        if signal == "LONG" and trend_4h == "down":
            return True
        if signal == "SHORT" and trend_4h == "up":
            return True
        return False
