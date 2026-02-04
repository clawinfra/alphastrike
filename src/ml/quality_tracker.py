"""
AlphaStrike Trading Bot - Model Quality Tracker

Tracks rolling accuracy and regime-specific performance for ensemble models.
Provides graduated weight adjustments based on accuracy rather than binary exclusion.

Key Features:
- Rolling accuracy tracking (configurable window, default 100 predictions)
- Regime-specific accuracy (trending, ranging, volatile)
- Graduated weight scaling based on accuracy
- Persistence requirement (must be consistently low before exclusion)
- Diversity protection (always keep at least 2 models)
- Recovery mechanism (excluded models can return if accuracy improves)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


# Type aliases
RegimeType = Literal["trending", "ranging", "volatile", "unknown"]
SignalType = Literal["LONG", "SHORT", "HOLD"]


class ModelStatus(Enum):
    """Status of a model in the ensemble."""

    ACTIVE = "active"  # Full weight
    DEGRADED = "degraded"  # Reduced weight
    EXCLUDED = "excluded"  # Zero weight (but can recover)
    FAILED = "failed"  # Health check failure


@dataclass
class PredictionRecord:
    """Record of a single prediction for accuracy tracking."""

    model_name: str
    predicted_signal: SignalType
    actual_outcome: SignalType | None  # None until outcome is known
    regime: RegimeType
    timestamp: datetime
    confidence: float

    @property
    def is_correct(self) -> bool | None:
        """Check if prediction was correct. None if outcome unknown."""
        if self.actual_outcome is None:
            return None
        # HOLD predictions are always considered "correct" (no action taken)
        if self.predicted_signal == "HOLD":
            return True
        return self.predicted_signal == self.actual_outcome


@dataclass
class ModelAccuracyStats:
    """Accuracy statistics for a single model."""

    model_name: str
    overall_accuracy: float = 0.5
    trending_accuracy: float = 0.5
    ranging_accuracy: float = 0.5
    volatile_accuracy: float = 0.5
    total_predictions: int = 0
    correct_predictions: int = 0
    consecutive_wrong: int = 0
    status: ModelStatus = ModelStatus.ACTIVE
    weight_multiplier: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "model_name": self.model_name,
            "overall_accuracy": f"{self.overall_accuracy:.2%}",
            "trending_accuracy": f"{self.trending_accuracy:.2%}",
            "ranging_accuracy": f"{self.ranging_accuracy:.2%}",
            "volatile_accuracy": f"{self.volatile_accuracy:.2%}",
            "total_predictions": self.total_predictions,
            "status": self.status.value,
            "weight_multiplier": self.weight_multiplier,
        }


@dataclass
class QualityTrackerConfig:
    """Configuration for ModelQualityTracker."""

    # Rolling window size
    rolling_window: int = 100

    # Accuracy thresholds for weight adjustment
    full_weight_threshold: float = 0.55  # Above this = 100% weight
    degraded_threshold: float = 0.50  # 50-55% = 75% weight
    reduced_threshold: float = 0.45  # 45-50% = 50% weight
    exclusion_threshold: float = 0.45  # Below this = candidate for exclusion

    # Persistence requirements
    min_predictions_for_exclusion: int = 50  # Need 50+ predictions before excluding
    consecutive_wrong_for_alert: int = 10  # Alert after 10 consecutive wrong

    # Diversity protection
    min_active_models: int = 2  # Always keep at least 2 models

    # Recovery settings
    recovery_threshold: float = 0.52  # Accuracy needed to recover from exclusion
    recovery_window: int = 30  # Predictions to evaluate for recovery


class ModelQualityTracker:
    """
    Tracks model prediction accuracy and provides dynamic weight adjustments.

    Unlike simple health checks that only detect degenerate models,
    this tracker monitors actual prediction accuracy over time and
    adjusts model weights accordingly.

    Key principles:
    1. Graduated weight reduction (not binary exclusion)
    2. Regime-specific accuracy tracking
    3. Persistence requirement (avoid excluding on short bad streaks)
    4. Diversity protection (always keep min_active_models)
    5. Recovery mechanism (excluded models can return)

    Usage:
        tracker = ModelQualityTracker()

        # Record a prediction
        tracker.record_prediction(
            model_name="xgboost",
            predicted_signal="LONG",
            regime="trending",
            confidence=0.75
        )

        # Later, record the outcome
        tracker.record_outcome(
            model_name="xgboost",
            actual_outcome="LONG"  # or "SHORT" if prediction was wrong
        )

        # Get adjusted weights for ensemble
        weights = tracker.get_adjusted_weights(base_weights)
    """

    def __init__(self, config: QualityTrackerConfig | None = None) -> None:
        """
        Initialize the quality tracker.

        Args:
            config: Configuration for thresholds and windows
        """
        self.config = config or QualityTrackerConfig()

        # Prediction history per model (rolling window)
        self._predictions: dict[str, deque[PredictionRecord]] = {}

        # Pending predictions awaiting outcome
        self._pending: dict[str, list[PredictionRecord]] = {}

        # Current accuracy stats per model
        self._stats: dict[str, ModelAccuracyStats] = {}

        # Regime-specific prediction history
        self._regime_predictions: dict[str, dict[RegimeType, deque[bool]]] = {}

        logger.info(
            "ModelQualityTracker initialized",
            extra={
                "rolling_window": self.config.rolling_window,
                "exclusion_threshold": self.config.exclusion_threshold,
                "min_active_models": self.config.min_active_models,
            },
        )

    def _ensure_model_initialized(self, model_name: str) -> None:
        """Ensure tracking structures exist for a model."""
        if model_name not in self._predictions:
            self._predictions[model_name] = deque(maxlen=self.config.rolling_window)
            self._pending[model_name] = []
            self._stats[model_name] = ModelAccuracyStats(model_name=model_name)
            self._regime_predictions[model_name] = {
                "trending": deque(maxlen=self.config.rolling_window),
                "ranging": deque(maxlen=self.config.rolling_window),
                "volatile": deque(maxlen=self.config.rolling_window),
                "unknown": deque(maxlen=self.config.rolling_window),
            }

    def record_prediction(
        self,
        model_name: str,
        predicted_signal: SignalType,
        regime: RegimeType,
        confidence: float,
    ) -> None:
        """
        Record a prediction from a model.

        Call this when a model makes a prediction. The outcome will be
        recorded later via record_outcome().

        Args:
            model_name: Name of the model
            predicted_signal: The signal predicted ("LONG", "SHORT", "HOLD")
            regime: Current market regime
            confidence: Model's confidence in the prediction
        """
        self._ensure_model_initialized(model_name)

        record = PredictionRecord(
            model_name=model_name,
            predicted_signal=predicted_signal,
            actual_outcome=None,
            regime=regime,
            timestamp=datetime.utcnow(),
            confidence=confidence,
        )

        # Add to pending (awaiting outcome)
        self._pending[model_name].append(record)

        logger.debug(
            f"Recorded prediction for {model_name}",
            extra={
                "signal": predicted_signal,
                "regime": regime,
                "confidence": confidence,
            },
        )

    def record_outcome(
        self,
        model_name: str,
        actual_outcome: SignalType,
    ) -> None:
        """
        Record the actual outcome for pending predictions.

        Call this when we know what actually happened (price moved up/down).
        This resolves the oldest pending prediction for the model.

        Args:
            model_name: Name of the model
            actual_outcome: What actually happened ("LONG" if price went up, etc.)
        """
        self._ensure_model_initialized(model_name)

        if not self._pending[model_name]:
            logger.warning(f"No pending predictions for {model_name}")
            return

        # Get oldest pending prediction (FIFO)
        record = self._pending[model_name].pop(0)
        record.actual_outcome = actual_outcome

        # Add to history
        self._predictions[model_name].append(record)

        # Update regime-specific tracking
        is_correct = record.is_correct
        if is_correct is not None and record.predicted_signal != "HOLD":
            self._regime_predictions[model_name][record.regime].append(is_correct)

        # Update stats
        self._update_stats(model_name)

        logger.debug(
            f"Recorded outcome for {model_name}",
            extra={
                "predicted": record.predicted_signal,
                "actual": actual_outcome,
                "correct": is_correct,
                "regime": record.regime,
            },
        )

    def record_batch_outcomes(
        self,
        actual_outcome: SignalType,
    ) -> None:
        """
        Record the same outcome for all models' oldest pending predictions.

        Convenience method when all models predicted on the same candle.

        Args:
            actual_outcome: What actually happened
        """
        for model_name in list(self._pending.keys()):
            if self._pending[model_name]:
                self.record_outcome(model_name, actual_outcome)

    def _update_stats(self, model_name: str) -> None:
        """Update accuracy statistics for a model."""
        self._ensure_model_initialized(model_name)

        predictions = self._predictions[model_name]
        stats = self._stats[model_name]

        # Calculate overall accuracy (excluding HOLD predictions)
        actionable_predictions = [
            p for p in predictions
            if p.predicted_signal != "HOLD" and p.is_correct is not None
        ]

        if actionable_predictions:
            correct = sum(1 for p in actionable_predictions if p.is_correct)
            stats.total_predictions = len(actionable_predictions)
            stats.correct_predictions = correct
            stats.overall_accuracy = correct / len(actionable_predictions)
        else:
            stats.overall_accuracy = 0.5  # Neutral if no data

        # Calculate regime-specific accuracy
        for regime in ["trending", "ranging", "volatile"]:
            regime_history = self._regime_predictions[model_name][regime]  # type: ignore
            if len(regime_history) >= 10:  # Need minimum samples
                regime_accuracy = sum(regime_history) / len(regime_history)
                if regime == "trending":
                    stats.trending_accuracy = regime_accuracy
                elif regime == "ranging":
                    stats.ranging_accuracy = regime_accuracy
                elif regime == "volatile":
                    stats.volatile_accuracy = regime_accuracy

        # Track consecutive wrong predictions
        recent_correct = [
            p.is_correct for p in list(predictions)[-20:]
            if p.predicted_signal != "HOLD" and p.is_correct is not None
        ]
        consecutive_wrong = 0
        for is_correct in reversed(recent_correct):
            if not is_correct:
                consecutive_wrong += 1
            else:
                break
        stats.consecutive_wrong = consecutive_wrong

        # Update status and weight multiplier
        self._update_model_status(model_name)

        stats.last_updated = datetime.utcnow()

    def _update_model_status(self, model_name: str) -> None:
        """Update model status based on accuracy."""
        stats = self._stats[model_name]
        config = self.config

        # Check if we have enough data
        if stats.total_predictions < config.min_predictions_for_exclusion:
            # Not enough data - keep at active with slight caution
            stats.status = ModelStatus.ACTIVE
            stats.weight_multiplier = 0.9  # Slight reduction for new/untested models
            return

        accuracy = stats.overall_accuracy

        # Graduated weight adjustment
        if accuracy >= config.full_weight_threshold:
            stats.status = ModelStatus.ACTIVE
            stats.weight_multiplier = 1.0
        elif accuracy >= config.degraded_threshold:
            stats.status = ModelStatus.DEGRADED
            # Linear interpolation: 55% -> 1.0, 50% -> 0.75
            stats.weight_multiplier = 0.75 + (accuracy - config.degraded_threshold) * 5
        elif accuracy >= config.reduced_threshold:
            stats.status = ModelStatus.DEGRADED
            # Linear interpolation: 50% -> 0.75, 45% -> 0.5
            stats.weight_multiplier = 0.5 + (accuracy - config.reduced_threshold) * 5
        else:
            # Below exclusion threshold
            stats.status = ModelStatus.EXCLUDED
            stats.weight_multiplier = 0.0

        # Log significant status changes
        if stats.status == ModelStatus.EXCLUDED:
            logger.warning(
                f"Model {model_name} EXCLUDED due to low accuracy",
                extra={
                    "accuracy": f"{accuracy:.2%}",
                    "total_predictions": stats.total_predictions,
                    "threshold": f"{config.exclusion_threshold:.2%}",
                },
            )
        elif stats.consecutive_wrong >= config.consecutive_wrong_for_alert:
            logger.warning(
                f"Model {model_name} has {stats.consecutive_wrong} consecutive wrong predictions",
                extra={"accuracy": f"{accuracy:.2%}"},
            )

    def get_adjusted_weights(
        self,
        base_weights: dict[str, float],
        current_regime: RegimeType = "unknown",
        health_status: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """
        Get accuracy-adjusted weights for ensemble.

        Combines base weights with accuracy-based multipliers and
        enforces diversity protection.

        Args:
            base_weights: Original model weights (should sum to 1.0)
            current_regime: Current market regime for regime-specific adjustment
            health_status: Optional health check results (failed = 0 weight)

        Returns:
            Adjusted weights (normalized to sum to 1.0)
        """
        adjusted = {}

        for model_name, base_weight in base_weights.items():
            self._ensure_model_initialized(model_name)
            stats = self._stats[model_name]

            # Start with base weight
            weight = base_weight

            # Apply accuracy-based multiplier
            weight *= stats.weight_multiplier

            # Apply health check (if provided)
            if health_status is not None and not health_status.get(model_name, True):
                weight = 0.0
                stats.status = ModelStatus.FAILED

            # Apply regime-specific boost/penalty
            if current_regime != "unknown" and stats.total_predictions >= 20:
                regime_accuracy = self._get_regime_accuracy(model_name, current_regime)
                if regime_accuracy is not None:
                    # Boost models that perform well in current regime
                    if regime_accuracy >= 0.60:
                        weight *= 1.2  # 20% boost
                    elif regime_accuracy <= 0.40:
                        weight *= 0.7  # 30% penalty

            adjusted[model_name] = weight

        # Enforce diversity protection
        adjusted = self._enforce_diversity(adjusted, base_weights)

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        else:
            # All models excluded - fall back to base weights
            logger.error("All models have zero weight - falling back to base weights")
            return base_weights

        logger.debug(
            "Adjusted weights calculated",
            extra={
                "base_weights": base_weights,
                "adjusted_weights": {k: f"{v:.2%}" for k, v in adjusted.items()},
                "regime": current_regime,
            },
        )

        return adjusted

    def _get_regime_accuracy(
        self, model_name: str, regime: RegimeType
    ) -> float | None:
        """Get regime-specific accuracy for a model."""
        stats = self._stats.get(model_name)
        if stats is None:
            return None

        if regime == "trending":
            return stats.trending_accuracy
        elif regime == "ranging":
            return stats.ranging_accuracy
        elif regime == "volatile":
            return stats.volatile_accuracy
        return None

    def _enforce_diversity(
        self,
        adjusted: dict[str, float],
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Enforce minimum model diversity.

        If too many models would be excluded, restore the best-performing
        ones until we have min_active_models.
        """
        active_count = sum(1 for w in adjusted.values() if w > 0)

        if active_count >= self.config.min_active_models:
            return adjusted

        # Need to restore some models
        excluded = [
            (name, self._stats[name].overall_accuracy)
            for name, weight in adjusted.items()
            if weight == 0
        ]

        # Sort by accuracy (best first)
        excluded.sort(key=lambda x: x[1], reverse=True)

        # Restore models until we have enough
        needed = self.config.min_active_models - active_count
        for name, accuracy in excluded[:needed]:
            # Restore with reduced weight
            adjusted[name] = base_weights[name] * 0.5
            self._stats[name].status = ModelStatus.DEGRADED
            self._stats[name].weight_multiplier = 0.5
            logger.info(
                f"Restored {name} for diversity protection",
                extra={"accuracy": f"{accuracy:.2%}"},
            )

        return adjusted

    def check_recovery(self, model_name: str) -> bool:
        """
        Check if an excluded model should be recovered.

        Evaluates recent predictions to see if accuracy has improved
        above recovery threshold.

        Args:
            model_name: Name of the model

        Returns:
            True if model should be recovered
        """
        self._ensure_model_initialized(model_name)
        stats = self._stats[model_name]

        if stats.status != ModelStatus.EXCLUDED:
            return False

        # Check recent predictions
        predictions = list(self._predictions[model_name])[-self.config.recovery_window:]
        actionable = [
            p for p in predictions
            if p.predicted_signal != "HOLD" and p.is_correct is not None
        ]

        if len(actionable) < self.config.recovery_window // 2:
            return False  # Not enough recent data

        recent_accuracy = sum(1 for p in actionable if p.is_correct) / len(actionable)

        if recent_accuracy >= self.config.recovery_threshold:
            logger.info(
                f"Model {model_name} eligible for recovery",
                extra={
                    "recent_accuracy": f"{recent_accuracy:.2%}",
                    "threshold": f"{self.config.recovery_threshold:.2%}",
                },
            )
            stats.status = ModelStatus.DEGRADED
            stats.weight_multiplier = 0.5  # Start with reduced weight
            return True

        return False

    def get_stats(self, model_name: str) -> ModelAccuracyStats | None:
        """Get accuracy stats for a model."""
        return self._stats.get(model_name)

    def get_all_stats(self) -> dict[str, ModelAccuracyStats]:
        """Get accuracy stats for all tracked models."""
        return self._stats.copy()

    def get_summary(self) -> dict:
        """Get summary of all model quality metrics."""
        summary = {
            "models": {},
            "active_count": 0,
            "degraded_count": 0,
            "excluded_count": 0,
            "failed_count": 0,
        }

        for name, stats in self._stats.items():
            summary["models"][name] = stats.to_dict()
            if stats.status == ModelStatus.ACTIVE:
                summary["active_count"] += 1
            elif stats.status == ModelStatus.DEGRADED:
                summary["degraded_count"] += 1
            elif stats.status == ModelStatus.EXCLUDED:
                summary["excluded_count"] += 1
            elif stats.status == ModelStatus.FAILED:
                summary["failed_count"] += 1

        return summary

    def reset(self, model_name: str | None = None) -> None:
        """
        Reset tracking data.

        Args:
            model_name: Specific model to reset, or None for all models
        """
        if model_name is not None:
            if model_name in self._predictions:
                self._predictions[model_name].clear()
                self._pending[model_name].clear()
                self._stats[model_name] = ModelAccuracyStats(model_name=model_name)
                for regime in self._regime_predictions[model_name].values():
                    regime.clear()
                logger.info(f"Reset tracking for {model_name}")
        else:
            self._predictions.clear()
            self._pending.clear()
            self._stats.clear()
            self._regime_predictions.clear()
            logger.info("Reset all model tracking")
