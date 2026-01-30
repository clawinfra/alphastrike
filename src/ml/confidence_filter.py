"""
AlphaStrike Trading Bot - Confidence Filter for Prediction Bounds

US-015: Validates prediction confidence using multiple metrics to filter
out low-quality signals before they reach the trading system.

Implements composite scoring with regime-adjusted thresholds.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.config import ConfidenceFilterConfig, MarketRegime, get_settings

logger = logging.getLogger(__name__)

# Type aliases
SignalType = Literal["LONG", "SHORT", "HOLD"]
ModelOutput = dict[str, float]  # model_name -> prediction probability


@dataclass
class ScoreBreakdown:
    """Breakdown of confidence score components."""

    confidence_score: float
    agreement_score: float
    proximity_score: float
    stability_score: float
    composite_score: float
    regime_threshold: float
    raw_confidence: float
    model_agreement_pct: float
    threshold_distance: float
    stability_pct: float


@dataclass
class ConfidenceFilterState:
    """State for tracking recent predictions for stability calculation."""

    recent_predictions: deque[SignalType] = field(
        default_factory=lambda: deque(maxlen=5)
    )


class ConfidenceFilter:
    """
    Confidence filter for validating prediction quality before trading.

    Implements multi-factor confidence validation:
    1. Minimum Confidence (30%): raw_confidence >= 0.55
    2. Model Agreement (25%): >= 50% of models agree on direction
    3. Threshold Proximity (25%): >= 0.10 distance from 0.5
    4. Prediction Stability (20%): >= 60% same signal over last 5 predictions

    The composite score is compared against regime-adjusted thresholds
    to determine whether to accept or reject the signal.
    """

    # Component weights for composite score calculation
    WEIGHT_CONFIDENCE: float = 0.30
    WEIGHT_AGREEMENT: float = 0.25
    WEIGHT_PROXIMITY: float = 0.25
    WEIGHT_STABILITY: float = 0.20

    # Minimum thresholds for individual components
    MIN_RAW_CONFIDENCE: float = 0.55
    MIN_MODEL_AGREEMENT: float = 0.50
    MIN_THRESHOLD_PROXIMITY: float = 0.10
    MIN_STABILITY: float = 0.60

    # Regime-adjusted composite thresholds
    REGIME_THRESHOLDS: dict[str, float] = {
        "trending_up": 0.55,
        "trending_down": 0.55,
        "ranging": 0.70,
        "high_volatility": 0.65,
        "extreme_volatility": 0.75,
    }
    DEFAULT_THRESHOLD: float = 0.60

    def __init__(self, config: ConfidenceFilterConfig | None = None) -> None:
        """
        Initialize confidence filter.

        Args:
            config: Optional configuration. If not provided, uses default settings.
        """
        self.config = config or get_settings().confidence
        self._state: dict[str, ConfidenceFilterState] = {}

        logger.info(
            "ConfidenceFilter initialized",
            extra={
                "enabled": self.config.enabled,
                "min_raw_confidence": self.config.min_raw_confidence,
                "min_model_agreement": self.config.min_model_agreement,
                "stability_window": self.config.stability_window,
            },
        )

    def _get_state(self, symbol: str = "default") -> ConfidenceFilterState:
        """Get or create state for a symbol."""
        if symbol not in self._state:
            self._state[symbol] = ConfidenceFilterState(
                recent_predictions=deque(maxlen=self.config.stability_window)
            )
        return self._state[symbol]

    def _determine_signal_direction(self, weighted_avg: float) -> SignalType:
        """
        Determine signal direction from weighted average.

        Args:
            weighted_avg: Weighted average prediction (0-1)

        Returns:
            SignalType: "LONG" if > 0.5, "SHORT" if < 0.5, "HOLD" otherwise
        """
        if weighted_avg > 0.5:
            return "LONG"
        elif weighted_avg < 0.5:
            return "SHORT"
        return "HOLD"

    def _calculate_confidence_score(self, raw_confidence: float) -> float:
        """
        Calculate normalized confidence score.

        Score is 1.0 if confidence >= MIN_RAW_CONFIDENCE, scaled linearly below.

        Args:
            raw_confidence: Raw confidence value (0-1)

        Returns:
            Normalized score (0-1)
        """
        if raw_confidence >= self.MIN_RAW_CONFIDENCE:
            return 1.0
        return raw_confidence / self.MIN_RAW_CONFIDENCE

    def _calculate_agreement_score(
        self, model_outputs: ModelOutput, weighted_avg: float
    ) -> tuple[float, float]:
        """
        Calculate model agreement score.

        Args:
            model_outputs: Dict of model_name -> prediction probability
            weighted_avg: Weighted average to determine target direction

        Returns:
            Tuple of (normalized score, agreement percentage)
        """
        if not model_outputs:
            return 0.0, 0.0

        target_direction = self._determine_signal_direction(weighted_avg)
        if target_direction == "HOLD":
            return 0.5, 0.5

        agreeing_models = 0
        total_models = len(model_outputs)

        for _model_name, prediction in model_outputs.items():
            model_direction = self._determine_signal_direction(prediction)
            if model_direction == target_direction:
                agreeing_models += 1

        agreement_pct = agreeing_models / total_models if total_models > 0 else 0.0

        # Normalize score: 1.0 if >= MIN_MODEL_AGREEMENT, scaled below
        if agreement_pct >= self.MIN_MODEL_AGREEMENT:
            score = 1.0
        else:
            score = agreement_pct / self.MIN_MODEL_AGREEMENT

        return score, agreement_pct

    def _calculate_proximity_score(self, weighted_avg: float) -> tuple[float, float]:
        """
        Calculate threshold proximity score.

        Distance from 0.5 indicates confidence in direction.

        Args:
            weighted_avg: Weighted average prediction (0-1)

        Returns:
            Tuple of (normalized score, actual distance from 0.5)
        """
        distance = abs(weighted_avg - 0.5)

        # Normalize score: 1.0 if >= MIN_THRESHOLD_PROXIMITY, scaled below
        if distance >= self.MIN_THRESHOLD_PROXIMITY:
            score = 1.0
        else:
            score = distance / self.MIN_THRESHOLD_PROXIMITY

        return score, distance

    def _calculate_stability_score(
        self, current_signal: SignalType, recent_predictions: deque[SignalType]
    ) -> tuple[float, float]:
        """
        Calculate prediction stability score.

        Measures consistency of signal direction over recent predictions.

        Args:
            current_signal: Current predicted signal
            recent_predictions: Deque of recent signal predictions

        Returns:
            Tuple of (normalized score, stability percentage)
        """
        if not recent_predictions:
            # No history - assume neutral stability
            return 0.5, 0.5

        matching_count = sum(1 for pred in recent_predictions if pred == current_signal)
        stability_pct = matching_count / len(recent_predictions)

        # Normalize score: 1.0 if >= MIN_STABILITY, scaled below
        if stability_pct >= self.MIN_STABILITY:
            score = 1.0
        else:
            score = stability_pct / self.MIN_STABILITY

        return score, stability_pct

    def get_regime_threshold(self, regime: str) -> float:
        """
        Get confidence threshold for a market regime.

        Args:
            regime: Market regime string (e.g., "trending_up", "ranging")

        Returns:
            Regime-adjusted threshold value
        """
        # Handle MarketRegime enum values
        if isinstance(regime, MarketRegime):
            regime = regime.value

        return self.REGIME_THRESHOLDS.get(regime, self.DEFAULT_THRESHOLD)

    def calculate_composite_score(
        self,
        raw_confidence: float,
        model_outputs: ModelOutput,
        weighted_avg: float,
        recent_predictions: deque[SignalType] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Calculate composite confidence score.

        Combines multiple validation metrics into a single score:
        - Minimum Confidence (30%): raw_confidence >= 0.55
        - Model Agreement (25%): >= 50% of models agree on direction
        - Threshold Proximity (25%): >= 0.10 distance from 0.5
        - Prediction Stability (20%): >= 60% same signal over last 5 predictions

        Args:
            raw_confidence: Raw confidence from model ensemble
            model_outputs: Dict of model predictions
            weighted_avg: Weighted average prediction
            recent_predictions: Recent prediction history for stability

        Returns:
            Tuple of (composite_score, score_breakdown_dict)
        """
        if recent_predictions is None:
            recent_predictions = deque()

        # Calculate individual component scores
        conf_score = self._calculate_confidence_score(raw_confidence)
        agreement_score, agreement_pct = self._calculate_agreement_score(
            model_outputs, weighted_avg
        )
        proximity_score, threshold_distance = self._calculate_proximity_score(
            weighted_avg
        )

        current_signal = self._determine_signal_direction(weighted_avg)
        stability_score, stability_pct = self._calculate_stability_score(
            current_signal, recent_predictions
        )

        # Calculate weighted composite score
        composite = (
            conf_score * self.WEIGHT_CONFIDENCE
            + agreement_score * self.WEIGHT_AGREEMENT
            + proximity_score * self.WEIGHT_PROXIMITY
            + stability_score * self.WEIGHT_STABILITY
        )

        breakdown = {
            "confidence_score": conf_score,
            "agreement_score": agreement_score,
            "proximity_score": proximity_score,
            "stability_score": stability_score,
            "composite_score": composite,
            "raw_confidence": raw_confidence,
            "model_agreement_pct": agreement_pct,
            "threshold_distance": threshold_distance,
            "stability_pct": stability_pct,
        }

        return composite, breakdown

    def should_reject(
        self,
        signal: SignalType,
        raw_confidence: float,
        weighted_avg: float,
        model_outputs: ModelOutput,
        regime: str,
        symbol: str = "default",
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Determine if a signal should be rejected based on confidence metrics.

        If rejected, the signal should be converted to HOLD.

        Args:
            signal: Predicted trading signal (LONG/SHORT/HOLD)
            raw_confidence: Raw confidence from model ensemble (0-1)
            weighted_avg: Weighted average prediction from ensemble
            model_outputs: Dict of individual model predictions
            regime: Current market regime string
            symbol: Trading symbol for state tracking

        Returns:
            Tuple of:
            - should_reject: True if signal should be rejected (convert to HOLD)
            - rejection_reason: Human-readable reason if rejected, None otherwise
            - score_breakdown: Dict with all score components
        """
        # If confidence filter is disabled, never reject
        if not self.config.enabled:
            return False, None, {"filter_disabled": True}

        # HOLD signals are never rejected
        if signal == "HOLD":
            return False, None, {"signal_type": "HOLD", "bypassed": True}

        # Get state for this symbol
        state = self._get_state(symbol)

        # Calculate composite score
        composite_score, breakdown = self.calculate_composite_score(
            raw_confidence=raw_confidence,
            model_outputs=model_outputs,
            weighted_avg=weighted_avg,
            recent_predictions=state.recent_predictions,
        )

        # Get regime-adjusted threshold
        regime_threshold = self.get_regime_threshold(regime)
        breakdown["regime_threshold"] = regime_threshold
        breakdown["regime"] = regime

        # Update recent predictions history
        state.recent_predictions.append(signal)

        # Check if composite score meets threshold
        should_reject = composite_score < regime_threshold

        rejection_reason: str | None = None
        if should_reject:
            # Build detailed rejection reason
            reasons = []

            if breakdown["confidence_score"] < 1.0:
                reasons.append(
                    f"low_confidence({raw_confidence:.2f}<{self.MIN_RAW_CONFIDENCE})"
                )

            if breakdown["agreement_score"] < 1.0:
                reasons.append(
                    f"low_agreement({breakdown['model_agreement_pct']:.0%}<{self.MIN_MODEL_AGREEMENT:.0%})"
                )

            if breakdown["proximity_score"] < 1.0:
                reasons.append(
                    f"near_threshold({breakdown['threshold_distance']:.2f}<{self.MIN_THRESHOLD_PROXIMITY})"
                )

            if breakdown["stability_score"] < 1.0:
                reasons.append(
                    f"unstable({breakdown['stability_pct']:.0%}<{self.MIN_STABILITY:.0%})"
                )

            rejection_reason = (
                f"composite_score({composite_score:.3f})<threshold({regime_threshold:.2f}): "
                + ", ".join(reasons)
            )

            logger.info(
                "Signal rejected by confidence filter",
                extra={
                    "signal": signal,
                    "symbol": symbol,
                    "composite_score": f"{composite_score:.3f}",
                    "regime_threshold": regime_threshold,
                    "regime": regime,
                    "reason": rejection_reason,
                },
            )

        return should_reject, rejection_reason, breakdown

    def reset_state(self, symbol: str | None = None) -> None:
        """
        Reset filter state.

        Args:
            symbol: Specific symbol to reset, or None to reset all
        """
        if symbol is None:
            self._state.clear()
            logger.debug("Confidence filter state cleared for all symbols")
        elif symbol in self._state:
            del self._state[symbol]
            logger.debug(f"Confidence filter state cleared for {symbol}")
