"""
AlphaStrike Trading Bot - Confidence Calibration Layer

Adjusts raw ML confidence scores based on multiple factors:
- Model agreement (higher when all models agree)
- Regime clarity (higher in clear trends)
- Recent accuracy (hot/cold streak detection)

This prevents overconfidence in noisy or conflicting conditions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""

    raw_confidence: float
    calibrated_confidence: float
    agreement_multiplier: float
    regime_multiplier: float
    accuracy_multiplier: float
    factors_explanation: str


@dataclass
class ModelPrediction:
    """Individual model prediction for agreement calculation."""

    model_name: str
    direction: Literal["LONG", "SHORT", "HOLD"]
    confidence: float


class ConfidenceCalibrator:
    """
    Calibrates ML confidence scores for more reliable trading signals.

    The calibrator ensures that raw ML confidence is adjusted based on:
    1. Inter-model agreement - penalize when models disagree
    2. Regime clarity - penalize in unclear/volatile regimes
    3. Recent accuracy - boost during hot streaks, reduce during cold
    """

    # Multiplier ranges
    AGREEMENT_MULTIPLIERS = {
        4: 1.10,  # All 4 models agree - boost confidence
        3: 0.90,  # 3/4 agree - slight reduction
        2: 0.60,  # Only 2/4 agree - significant reduction
        1: 0.30,  # Only 1 model - major reduction
        0: 0.00,  # No consensus - no confidence
    }

    REGIME_MULTIPLIERS = {
        "TRENDING_UP": 1.0,
        "TRENDING_DOWN": 1.0,
        "RANGING": 0.85,
        "HIGH_VOLATILITY": 0.70,
        "EXTREME_VOLATILITY": 0.40,
        "UNKNOWN": 0.60,
    }

    # Accuracy tracking
    ACCURACY_WINDOW = 20  # Number of recent trades to consider
    HOT_STREAK_THRESHOLD = 0.65  # Above this = hot streak
    COLD_STREAK_THRESHOLD = 0.45  # Below this = cold streak

    def __init__(self) -> None:
        """Initialize the confidence calibrator."""
        self._recent_outcomes: deque[bool] = deque(maxlen=self.ACCURACY_WINDOW)
        self._calibration_count = 0

        logger.info("ConfidenceCalibrator initialized")

    def calibrate(
        self,
        raw_confidence: float,
        model_predictions: list[ModelPrediction],
        regime: str,
        regime_confidence: float,
    ) -> CalibrationResult:
        """
        Calibrate raw ML confidence based on multiple factors.

        Args:
            raw_confidence: Raw confidence from ML ensemble (0-1)
            model_predictions: Individual predictions from each model
            regime: Current market regime
            regime_confidence: Confidence in regime detection (0-1)

        Returns:
            CalibrationResult with adjusted confidence
        """
        self._calibration_count += 1

        # Calculate agreement multiplier
        agreement_mult, agreeing_count = self._calculate_agreement_multiplier(
            model_predictions
        )

        # Calculate regime multiplier
        regime_mult = self._calculate_regime_multiplier(regime, regime_confidence)

        # Calculate accuracy multiplier
        accuracy_mult = self._calculate_accuracy_multiplier()

        # Apply all multipliers
        calibrated = raw_confidence * agreement_mult * regime_mult * accuracy_mult

        # Clamp to valid range
        calibrated = max(0.0, min(1.0, calibrated))

        # Build explanation
        explanation = self._build_explanation(
            agreeing_count,
            len(model_predictions),
            regime,
            accuracy_mult,
        )

        result = CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            agreement_multiplier=agreement_mult,
            regime_multiplier=regime_mult,
            accuracy_multiplier=accuracy_mult,
            factors_explanation=explanation,
        )

        logger.debug(
            f"Calibration #{self._calibration_count}: "
            f"raw={raw_confidence:.2f} → calibrated={calibrated:.2f} "
            f"(agree={agreement_mult:.2f}, regime={regime_mult:.2f}, acc={accuracy_mult:.2f})"
        )

        return result

    def record_outcome(self, was_profitable: bool) -> None:
        """
        Record trade outcome for accuracy tracking.

        Args:
            was_profitable: Whether the trade was profitable
        """
        self._recent_outcomes.append(was_profitable)
        logger.debug(
            f"Outcome recorded: {'WIN' if was_profitable else 'LOSS'}, "
            f"Recent: {sum(self._recent_outcomes)}/{len(self._recent_outcomes)}"
        )

    def get_recent_accuracy(self) -> float:
        """Get accuracy over recent trades."""
        if not self._recent_outcomes:
            return 0.5  # Neutral if no history

        return sum(self._recent_outcomes) / len(self._recent_outcomes)

    def reset_accuracy_tracking(self) -> None:
        """Reset accuracy tracking (e.g., after model retrain)."""
        self._recent_outcomes.clear()
        logger.info("Accuracy tracking reset")

    def _calculate_agreement_multiplier(
        self,
        predictions: list[ModelPrediction],
    ) -> tuple[float, int]:
        """
        Calculate multiplier based on model agreement.

        Returns:
            Tuple of (multiplier, agreeing_model_count)
        """
        if not predictions:
            return 0.0, 0

        # Count models predicting each direction
        direction_counts: dict[str, int] = {"LONG": 0, "SHORT": 0, "HOLD": 0}
        for pred in predictions:
            direction_counts[pred.direction] += 1

        # Find the majority direction
        max_count = max(direction_counts.values())

        # Get multiplier based on agreement level
        multiplier = self.AGREEMENT_MULTIPLIERS.get(max_count, 0.5)

        return multiplier, max_count

    def _calculate_regime_multiplier(
        self,
        regime: str,
        regime_confidence: float,
    ) -> float:
        """Calculate multiplier based on market regime."""
        base_mult = self.REGIME_MULTIPLIERS.get(regime, 0.6)

        # Adjust based on regime detection confidence
        # Low confidence in regime = reduce overall confidence
        confidence_adjustment = 0.7 + (regime_confidence * 0.3)

        return base_mult * confidence_adjustment

    def _calculate_accuracy_multiplier(self) -> float:
        """
        Calculate multiplier based on recent trading accuracy.

        Hot streak (>65% accuracy): Boost confidence slightly
        Normal (45-65%): Neutral
        Cold streak (<45%): Reduce confidence
        """
        if len(self._recent_outcomes) < 5:
            return 1.0  # Need minimum history

        accuracy = self.get_recent_accuracy()

        if accuracy >= self.HOT_STREAK_THRESHOLD:
            # Hot streak - boost up to 1.15
            return 1.0 + (accuracy - self.HOT_STREAK_THRESHOLD) * 0.5

        elif accuracy <= self.COLD_STREAK_THRESHOLD:
            # Cold streak - reduce down to 0.7
            return 0.7 + (accuracy / self.COLD_STREAK_THRESHOLD) * 0.3

        else:
            # Normal range
            return 1.0

    def _build_explanation(
        self,
        agreeing_count: int,
        total_models: int,
        regime: str,
        accuracy_mult: float,
    ) -> str:
        """Build human-readable explanation of calibration."""
        parts = []

        # Agreement
        if agreeing_count == total_models:
            parts.append(f"all {total_models} models agree")
        elif agreeing_count >= 3:
            parts.append(f"{agreeing_count}/{total_models} models agree")
        else:
            parts.append(f"weak agreement ({agreeing_count}/{total_models})")

        # Regime
        regime_nice = regime.replace("_", " ").lower()
        parts.append(f"{regime_nice} regime")

        # Accuracy
        if accuracy_mult > 1.05:
            parts.append("hot streak")
        elif accuracy_mult < 0.9:
            parts.append("cold streak")

        return ", ".join(parts)


class AgreementGate:
    """
    Gate that blocks signals when model agreement is insufficient.

    Implements the Simons Protocol requirement for 3/4 model consensus.
    """

    MIN_AGREEMENT = 3  # Minimum models that must agree

    def __init__(self) -> None:
        """Initialize the agreement gate."""
        logger.info("AgreementGate initialized (min_agreement=3)")

    def check(
        self,
        predictions: list[ModelPrediction],
    ) -> tuple[bool, Literal["LONG", "SHORT", "HOLD"], float]:
        """
        Check if sufficient models agree on direction.

        Args:
            predictions: List of individual model predictions

        Returns:
            Tuple of (passed, consensus_direction, agreement_ratio)
        """
        if len(predictions) < self.MIN_AGREEMENT:
            return False, "HOLD", 0.0

        # Count directions
        direction_counts: dict[str, list[ModelPrediction]] = {
            "LONG": [],
            "SHORT": [],
            "HOLD": [],
        }

        for pred in predictions:
            direction_counts[pred.direction].append(pred)

        # Find majority
        majority_direction = max(direction_counts.keys(), key=lambda d: len(direction_counts[d]))
        majority_count = len(direction_counts[majority_direction])

        agreement_ratio = majority_count / len(predictions)

        # Check if sufficient agreement
        if majority_count >= self.MIN_AGREEMENT and majority_direction != "HOLD":
            logger.info(
                f"Agreement gate PASSED: {majority_count}/{len(predictions)} "
                f"agree on {majority_direction}"
            )
            return True, majority_direction, agreement_ratio  # type: ignore

        logger.info(
            f"Agreement gate BLOCKED: only {majority_count}/{len(predictions)} "
            f"agree (need {self.MIN_AGREEMENT})"
        )
        return False, "HOLD", agreement_ratio
