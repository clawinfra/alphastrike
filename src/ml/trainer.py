"""
AlphaStrike Trading Bot - Model Trainer with Trigger-Based Retraining

Provides automated training pipeline for all ML models with trigger-based
retraining instead of time-interval based. Retraining is triggered by:

1. Regime change detection (market structure changed)
2. Model health degradation (degenerate predictions)
3. Feature drift above threshold (input distribution shifted)
4. Validation accuracy drop (model performance degraded)

Key insight: During high volatility, we REDUCE position sizes and RAISE
confidence thresholds rather than retraining on noisy data.

Features:
- Trains all 4 models (XGBoost, LightGBM, LSTM, RandomForest)
- Trigger-based retraining (not time-based)
- Cooldown period to prevent excessive retraining
- Volatility-aware confidence adjustment
- Comprehensive training reports
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.core.config import MarketRegime, get_settings
from src.data.database import Candle, Database
from src.features.pipeline import FeaturePipeline
from src.ml.lightgbm_model import LightGBMConfig, LightGBMModel
from src.ml.lstm_model import LSTMConfig, LSTMModel
from src.ml.random_forest_model import RandomForestConfig, RandomForestModel
from src.ml.xgboost_model import XGBoostConfig, XGBoostModel

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Label classes
LABEL_LONG = 2  # Price increased > threshold
LABEL_HOLD = 1  # Price change within threshold
LABEL_SHORT = 0  # Price decreased > threshold

# Label threshold (0.5% price change)
LABEL_THRESHOLD = 0.005

# Minimum samples for training
MIN_TRAINING_SAMPLES = 1000

# Cooldown between retraining (prevents excessive retraining)
MIN_RETRAIN_COOLDOWN_MINUTES = 30
MAX_RETRAIN_COOLDOWN_MINUTES = 120

# Trigger thresholds
REGIME_CHANGE_CONFIDENCE_MIN = 0.6  # Minimum confidence for regime change trigger
FEATURE_DRIFT_THRESHOLD = 0.25  # PSI threshold for drift trigger
MODEL_HEALTH_CHECK_FAILURES_THRESHOLD = 2  # Consecutive failures before trigger
VALIDATION_ACCURACY_MIN = 0.52  # Minimum accuracy before trigger (above random)

# Volatility thresholds
ATR_RATIO_HIGH = 2.0
ATR_RATIO_EXTREME = 3.0


# =============================================================================
# Trigger System
# =============================================================================


class RetrainingTrigger(Enum):
    """Types of events that can trigger model retraining."""

    REGIME_CHANGE = "regime_change"
    MODEL_HEALTH_DEGRADED = "model_health_degraded"
    FEATURE_DRIFT = "feature_drift"
    VALIDATION_ACCURACY_DROP = "validation_accuracy_drop"
    MANUAL = "manual"
    INITIAL = "initial"


@dataclass
class TriggerState:
    """
    State tracking for retraining triggers.

    Tracks the conditions that determine when retraining should occur.
    """

    # Regime tracking
    last_regime: MarketRegime | None = None
    regime_stable_since: datetime | None = None

    # Model health tracking
    consecutive_health_failures: int = 0
    last_health_check: datetime | None = None

    # Feature drift tracking
    last_drift_score: float = 0.0
    drift_above_threshold_since: datetime | None = None

    # Validation tracking
    last_validation_accuracy: float = 0.5
    accuracy_below_threshold_since: datetime | None = None

    # Volatility tracking
    current_volatility: float = 1.0
    in_high_volatility_mode: bool = False

    # Cooldown tracking
    last_retrain_time: datetime | None = None
    last_trigger: RetrainingTrigger | None = None


@dataclass
class VolatilityAdjustment:
    """
    Adjustments to apply during high volatility periods.

    Instead of retraining on noisy data, we adjust trading parameters.
    """

    confidence_multiplier: float = 1.0  # Multiply confidence thresholds
    position_scale: float = 1.0  # Scale down position sizes
    should_trade: bool = True  # Whether to trade at all

    @classmethod
    def for_volatility(cls, atr_ratio: float) -> VolatilityAdjustment:
        """
        Calculate adjustments based on current volatility.

        Args:
            atr_ratio: Current ATR ratio (1.0 = normal)

        Returns:
            VolatilityAdjustment with appropriate scaling
        """
        if atr_ratio >= ATR_RATIO_EXTREME:
            # Extreme volatility: minimal trading
            return cls(
                confidence_multiplier=1.5,  # 50% higher confidence required
                position_scale=0.25,  # 25% of normal size
                should_trade=True,  # Still trade but very conservatively
            )
        elif atr_ratio >= ATR_RATIO_HIGH:
            # High volatility: reduced trading
            return cls(
                confidence_multiplier=1.25,  # 25% higher confidence required
                position_scale=0.5,  # 50% of normal size
                should_trade=True,
            )
        elif atr_ratio >= 1.5:
            # Elevated volatility: slightly reduced
            return cls(
                confidence_multiplier=1.1,
                position_scale=0.75,
                should_trade=True,
            )
        else:
            # Normal or low volatility: no adjustment
            return cls(
                confidence_multiplier=1.0,
                position_scale=1.0,
                should_trade=True,
            )


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TrainingReport:
    """Result of model training process."""

    success: bool
    models_trained: list[str]
    validation_accuracy: dict[str, float]
    training_time_seconds: float
    timestamp: datetime
    trigger: RetrainingTrigger
    samples_used: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    feature_count: int = 0
    error_message: str | None = None
    model_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "models_trained": self.models_trained,
            "validation_accuracy": self.validation_accuracy,
            "training_time_seconds": self.training_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger.value,
            "samples_used": self.samples_used,
            "class_distribution": self.class_distribution,
            "feature_count": self.feature_count,
            "error_message": self.error_message,
            "model_paths": self.model_paths,
        }


@dataclass
class ModelInfo:
    """Information about a trained model."""

    name: str
    path: Path
    accuracy: float
    trained_at: datetime
    samples_used: int


# =============================================================================
# Model Trainer
# =============================================================================


class ModelTrainer:
    """
    Trainer for all ML models with trigger-based retraining.

    Key principle: During high volatility, we adjust trading parameters
    (reduce size, raise confidence) rather than retraining on noisy data.

    Retraining is triggered by:
    1. Regime change detection - market structure fundamentally changed
    2. Model health degradation - models producing degenerate predictions
    3. Feature drift - input distribution shifted significantly
    4. Validation accuracy drop - models underperforming

    Usage:
        trainer = ModelTrainer(database=db)

        # Check if retraining needed (call periodically)
        trigger = trainer.check_triggers(
            current_regime=regime,
            model_health_results=health,
            drift_score=drift,
            validation_accuracy=accuracy,
            volatility=atr_ratio,
        )

        if trigger is not None:
            report = await trainer.train_all_models(trigger=trigger)

        # Get volatility adjustments (use instead of retraining during vol)
        adjustment = trainer.get_volatility_adjustment(atr_ratio)
    """

    def __init__(
        self,
        database: Database,
        models_dir: Path | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        reload_callback: Callable[[], None] | None = None,
    ):
        """
        Initialize model trainer.

        Args:
            database: Database instance for fetching candles
            models_dir: Directory to save trained models
            feature_pipeline: Feature pipeline for feature calculation
            reload_callback: Callback to signal model hot reload to ensemble
        """
        settings = get_settings()
        self.database = database
        self.models_dir = models_dir or settings.models_dir
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.reload_callback = reload_callback

        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model instances (lazily created)
        self._xgboost: XGBoostModel | None = None
        self._lightgbm: LightGBMModel | None = None
        self._lstm: LSTMModel | None = None
        self._random_forest: RandomForestModel | None = None

        # Trigger state tracking
        self._trigger_state = TriggerState()

        logger.info(
            "ModelTrainer initialized with trigger-based retraining",
            extra={"models_dir": str(self.models_dir)},
        )

    def check_triggers(
        self,
        current_regime: MarketRegime,
        model_health_results: dict[str, bool],
        drift_score: float,
        validation_accuracy: float,
        volatility: float,
    ) -> RetrainingTrigger | None:
        """
        Check all retraining triggers and return the highest priority one.

        This method should be called periodically (e.g., every minute).
        It does NOT trigger retraining during high volatility - instead,
        use get_volatility_adjustment() to adjust trading parameters.

        Args:
            current_regime: Current detected market regime
            model_health_results: Dict of model name -> health status
            drift_score: Current feature drift score (PSI)
            validation_accuracy: Recent out-of-sample accuracy
            volatility: Current ATR ratio

        Returns:
            RetrainingTrigger if retraining should occur, None otherwise
        """
        now = datetime.utcnow()

        # Update volatility state
        self._trigger_state.current_volatility = volatility
        self._trigger_state.in_high_volatility_mode = volatility >= ATR_RATIO_HIGH

        # Check cooldown first
        if not self._cooldown_elapsed(volatility):
            return None

        # During high volatility, suppress retraining
        # (we use volatility adjustments instead)
        if volatility >= ATR_RATIO_HIGH:
            logger.debug(
                "Suppressing retraining triggers during high volatility",
                extra={"volatility": volatility, "threshold": ATR_RATIO_HIGH},
            )
            return None

        # Priority 1: Regime change (highest priority)
        regime_trigger = self._check_regime_change_trigger(current_regime, now)
        if regime_trigger:
            return regime_trigger

        # Priority 2: Model health degradation
        health_trigger = self._check_model_health_trigger(model_health_results, now)
        if health_trigger:
            return health_trigger

        # Priority 3: Feature drift
        drift_trigger = self._check_feature_drift_trigger(drift_score, now)
        if drift_trigger:
            return drift_trigger

        # Priority 4: Validation accuracy drop
        accuracy_trigger = self._check_accuracy_trigger(validation_accuracy, now)
        if accuracy_trigger:
            return accuracy_trigger

        return None

    def _cooldown_elapsed(self, volatility: float) -> bool:
        """
        Check if cooldown period has elapsed since last retraining.

        Cooldown is LONGER during low volatility (clean data, less urgent)
        and SHORTER after volatility subsides (need to capture new regime).

        Args:
            volatility: Current ATR ratio

        Returns:
            True if cooldown has elapsed
        """
        if self._trigger_state.last_retrain_time is None:
            return True

        # Adaptive cooldown: longer during calm markets, shorter after vol
        if volatility < 1.0:
            # Low volatility: longer cooldown (data is stable)
            cooldown_minutes = MAX_RETRAIN_COOLDOWN_MINUTES
        elif volatility < ATR_RATIO_HIGH:
            # Normal volatility: moderate cooldown
            cooldown_minutes = (MIN_RETRAIN_COOLDOWN_MINUTES + MAX_RETRAIN_COOLDOWN_MINUTES) // 2
        else:
            # Post high-volatility: shorter cooldown when vol subsides
            cooldown_minutes = MIN_RETRAIN_COOLDOWN_MINUTES

        elapsed = datetime.utcnow() - self._trigger_state.last_retrain_time
        return elapsed >= timedelta(minutes=cooldown_minutes)

    def _check_regime_change_trigger(
        self, current_regime: MarketRegime, now: datetime
    ) -> RetrainingTrigger | None:
        """Check if regime change should trigger retraining."""
        last_regime = self._trigger_state.last_regime

        if last_regime is None:
            # First regime detection, don't trigger
            self._trigger_state.last_regime = current_regime
            self._trigger_state.regime_stable_since = now
            return None

        if current_regime != last_regime:
            # Regime changed - require it to be stable for 5+ minutes
            if self._trigger_state.regime_stable_since is None:
                self._trigger_state.regime_stable_since = now

            # Wait for regime to stabilize (avoid false triggers)
            stability_duration = now - self._trigger_state.regime_stable_since
            if stability_duration >= timedelta(minutes=5):
                logger.info(
                    "Regime change trigger activated",
                    extra={
                        "old_regime": last_regime.value,
                        "new_regime": current_regime.value,
                        "stable_for": str(stability_duration),
                    },
                )
                self._trigger_state.last_regime = current_regime
                self._trigger_state.regime_stable_since = now
                return RetrainingTrigger.REGIME_CHANGE
        else:
            # Same regime, reset stability tracking
            self._trigger_state.regime_stable_since = now

        return None

    def _check_model_health_trigger(
        self, health_results: dict[str, bool], now: datetime
    ) -> RetrainingTrigger | None:
        """Check if model health degradation should trigger retraining."""
        self._trigger_state.last_health_check = now

        # Count unhealthy models
        unhealthy_count = sum(1 for healthy in health_results.values() if not healthy)
        total_models = len(health_results)

        if total_models == 0:
            return None

        # Trigger if majority of models are unhealthy
        if unhealthy_count >= (total_models // 2 + 1):
            self._trigger_state.consecutive_health_failures += 1

            if self._trigger_state.consecutive_health_failures >= MODEL_HEALTH_CHECK_FAILURES_THRESHOLD:
                logger.info(
                    "Model health trigger activated",
                    extra={
                        "unhealthy_count": unhealthy_count,
                        "total_models": total_models,
                        "consecutive_failures": self._trigger_state.consecutive_health_failures,
                    },
                )
                self._trigger_state.consecutive_health_failures = 0
                return RetrainingTrigger.MODEL_HEALTH_DEGRADED
        else:
            # Reset consecutive failures if models recovered
            self._trigger_state.consecutive_health_failures = 0

        return None

    def _check_feature_drift_trigger(
        self, drift_score: float, now: datetime
    ) -> RetrainingTrigger | None:
        """Check if feature drift should trigger retraining."""
        self._trigger_state.last_drift_score = drift_score

        if drift_score >= FEATURE_DRIFT_THRESHOLD:
            if self._trigger_state.drift_above_threshold_since is None:
                self._trigger_state.drift_above_threshold_since = now

            # Require drift to persist for 10+ minutes (avoid transient spikes)
            drift_duration = now - self._trigger_state.drift_above_threshold_since
            if drift_duration >= timedelta(minutes=10):
                logger.info(
                    "Feature drift trigger activated",
                    extra={
                        "drift_score": drift_score,
                        "threshold": FEATURE_DRIFT_THRESHOLD,
                        "duration": str(drift_duration),
                    },
                )
                self._trigger_state.drift_above_threshold_since = None
                return RetrainingTrigger.FEATURE_DRIFT
        else:
            self._trigger_state.drift_above_threshold_since = None

        return None

    def _check_accuracy_trigger(
        self, validation_accuracy: float, now: datetime
    ) -> RetrainingTrigger | None:
        """Check if accuracy drop should trigger retraining."""
        self._trigger_state.last_validation_accuracy = validation_accuracy

        if validation_accuracy < VALIDATION_ACCURACY_MIN:
            if self._trigger_state.accuracy_below_threshold_since is None:
                self._trigger_state.accuracy_below_threshold_since = now

            # Require low accuracy to persist for 15+ minutes
            low_accuracy_duration = now - self._trigger_state.accuracy_below_threshold_since
            if low_accuracy_duration >= timedelta(minutes=15):
                logger.info(
                    "Accuracy drop trigger activated",
                    extra={
                        "accuracy": validation_accuracy,
                        "threshold": VALIDATION_ACCURACY_MIN,
                        "duration": str(low_accuracy_duration),
                    },
                )
                self._trigger_state.accuracy_below_threshold_since = None
                return RetrainingTrigger.VALIDATION_ACCURACY_DROP
        else:
            self._trigger_state.accuracy_below_threshold_since = None

        return None

    def get_volatility_adjustment(self, volatility: float) -> VolatilityAdjustment:
        """
        Get trading adjustments for current volatility.

        Use this instead of retraining during high volatility periods.
        Reduces position sizes and raises confidence thresholds.

        Args:
            volatility: Current ATR ratio

        Returns:
            VolatilityAdjustment with scaling factors
        """
        return VolatilityAdjustment.for_volatility(volatility)

    async def train_all_models(
        self,
        trigger: RetrainingTrigger = RetrainingTrigger.MANUAL,
        symbol: str = "cmt_btcusdt",
        min_candles: int = MIN_TRAINING_SAMPLES,
    ) -> TrainingReport:
        """
        Train all models in the ensemble.

        Args:
            trigger: What triggered this retraining
            symbol: Trading symbol to fetch candles for
            min_candles: Minimum number of candles required

        Returns:
            TrainingReport with results and metrics
        """
        start_time = time.time()
        timestamp = datetime.utcnow()

        logger.info(
            f"Starting model training triggered by {trigger.value}",
            extra={"symbol": symbol, "min_candles": min_candles},
        )

        try:
            # Step 1: Fetch candles from database
            logger.info(f"Fetching {min_candles}+ candles for {symbol}")
            candles = await self._fetch_candles(symbol, min_candles)

            if len(candles) < min_candles:
                return TrainingReport(
                    success=False,
                    models_trained=[],
                    validation_accuracy={},
                    training_time_seconds=time.time() - start_time,
                    timestamp=timestamp,
                    trigger=trigger,
                    error_message=f"Insufficient candles: {len(candles)} < {min_candles}",
                )

            # Step 2: Calculate features
            logger.info("Calculating features for training data")
            X, feature_names = self._calculate_features(candles)

            if X.shape[0] < min_candles:
                return TrainingReport(
                    success=False,
                    models_trained=[],
                    validation_accuracy={},
                    training_time_seconds=time.time() - start_time,
                    timestamp=timestamp,
                    trigger=trigger,
                    error_message=f"Insufficient feature samples: {X.shape[0]}",
                )

            # Step 3: Generate labels
            logger.info("Generating training labels")
            y = self._generate_labels(candles)

            # Trim X to match y length (labels are shifted by 1)
            X = X[:-1]
            y = y[:-1]

            # Step 4: Balance samples
            logger.info("Balancing class distribution")
            X_balanced, y_balanced = self._balance_samples(X, y)

            # Get class distribution
            class_distribution = {
                "LONG": int(np.sum(y_balanced == LABEL_LONG)),
                "HOLD": int(np.sum(y_balanced == LABEL_HOLD)),
                "SHORT": int(np.sum(y_balanced == LABEL_SHORT)),
            }

            # Step 5: Time-ordered train/validation split (80/20)
            logger.info("Creating train/validation split")
            split_idx = int(len(X_balanced) * 0.8)
            X_train, X_val = X_balanced[:split_idx], X_balanced[split_idx:]
            y_train, y_val = y_balanced[:split_idx], y_balanced[split_idx:]

            # Convert to binary for models (LONG=1, else=0)
            y_train_binary = (y_train == LABEL_LONG).astype(np.float64)
            y_val_binary = (y_val == LABEL_LONG).astype(np.float64)

            # Step 6: Train all 4 models
            logger.info(f"Training models with {len(X_train)} samples")
            models_trained: list[str] = []
            validation_accuracy: dict[str, float] = {}
            model_paths: dict[str, str] = {}

            # Train XGBoost
            xgb_accuracy, xgb_path = self._train_xgboost(
                X_train, y_train_binary, X_val, y_val_binary, feature_names, timestamp
            )
            if xgb_accuracy is not None:
                models_trained.append("xgboost")
                validation_accuracy["xgboost"] = xgb_accuracy
                model_paths["xgboost"] = str(xgb_path)

            # Train LightGBM
            lgb_accuracy, lgb_path = self._train_lightgbm(
                X_train, y_train_binary, X_val, y_val_binary, timestamp
            )
            if lgb_accuracy is not None:
                models_trained.append("lightgbm")
                validation_accuracy["lightgbm"] = lgb_accuracy
                model_paths["lightgbm"] = str(lgb_path)

            # Train LSTM
            lstm_accuracy, lstm_path = self._train_lstm(
                X_train, y_train_binary, X_val, y_val_binary, timestamp
            )
            if lstm_accuracy is not None:
                models_trained.append("lstm")
                validation_accuracy["lstm"] = lstm_accuracy
                model_paths["lstm"] = str(lstm_path)

            # Train Random Forest
            rf_accuracy, rf_path = self._train_random_forest(
                X_train, y_train_binary, X_val, y_val_binary, timestamp
            )
            if rf_accuracy is not None:
                models_trained.append("random_forest")
                validation_accuracy["random_forest"] = rf_accuracy
                model_paths["random_forest"] = str(rf_path)

            # Step 7: Signal hot reload to ensemble
            if self.reload_callback is not None:
                logger.info("Signaling model hot reload to ensemble")
                self.reload_callback()

            # Update trigger state
            self._trigger_state.last_retrain_time = timestamp
            self._trigger_state.last_trigger = trigger

            training_time = time.time() - start_time

            logger.info(
                "Training completed successfully",
                extra={
                    "trigger": trigger.value,
                    "models_trained": models_trained,
                    "training_time": f"{training_time:.2f}s",
                    "validation_accuracy": validation_accuracy,
                },
            )

            return TrainingReport(
                success=True,
                models_trained=models_trained,
                validation_accuracy=validation_accuracy,
                training_time_seconds=training_time,
                timestamp=timestamp,
                trigger=trigger,
                samples_used=len(X_balanced),
                class_distribution=class_distribution,
                feature_count=X.shape[1],
                model_paths=model_paths,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return TrainingReport(
                success=False,
                models_trained=[],
                validation_accuracy={},
                training_time_seconds=time.time() - start_time,
                timestamp=timestamp,
                trigger=trigger,
                error_message=str(e),
            )

    def _generate_labels(self, candles: list[Candle]) -> NDArray[np.int64]:
        """
        Generate 3-class labels based on future price change.

        Labels:
        - LONG (2): Price increased > 0.5%
        - HOLD (1): Price change within +/- 0.5%
        - SHORT (0): Price decreased > 0.5%

        Args:
            candles: List of candles (oldest first)

        Returns:
            Array of labels for each candle
        """
        n = len(candles)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(n - 1):
            current_price = candles[i].close
            future_price = candles[i + 1].close

            if current_price <= 0:
                labels[i] = LABEL_HOLD
                continue

            price_change = (future_price - current_price) / current_price

            if price_change > LABEL_THRESHOLD:
                labels[i] = LABEL_LONG
            elif price_change < -LABEL_THRESHOLD:
                labels[i] = LABEL_SHORT
            else:
                labels[i] = LABEL_HOLD

        # Last candle has no future price, default to HOLD
        labels[-1] = LABEL_HOLD

        return labels

    def _balance_samples(
        self, X: NDArray[np.float64], y: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Balance samples for equal class distribution.

        Uses undersampling of majority classes to match minority class.

        Args:
            X: Feature matrix
            y: Label array

        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        # Find indices for each class
        long_indices = np.where(y == LABEL_LONG)[0]
        hold_indices = np.where(y == LABEL_HOLD)[0]
        short_indices = np.where(y == LABEL_SHORT)[0]

        # Find minimum class size
        min_count = min(len(long_indices), len(hold_indices), len(short_indices))

        if min_count == 0:
            logger.warning("One or more classes have zero samples, returning original data")
            return X, y

        # Randomly sample from each class to match minimum
        np.random.seed(42)  # Reproducibility
        long_sampled = np.random.choice(long_indices, min_count, replace=False)
        hold_sampled = np.random.choice(hold_indices, min_count, replace=False)
        short_sampled = np.random.choice(short_indices, min_count, replace=False)

        # Combine indices and sort (maintain time order within each subset)
        all_indices = np.concatenate([long_sampled, hold_sampled, short_sampled])
        all_indices = np.sort(all_indices)

        X_balanced = X[all_indices]
        y_balanced = y[all_indices]

        logger.debug(
            f"Balanced samples: {len(X_balanced)} (was {len(X)}), "
            f"each class has {min_count} samples"
        )

        return X_balanced, y_balanced

    async def _fetch_candles(
        self, symbol: str, min_candles: int
    ) -> list[Candle]:
        """Fetch candles from database."""
        # Get more candles than needed to ensure we have enough after processing
        limit = min_candles + 200
        candles = await self.database.get_candles(symbol, limit=limit)

        # Reverse to get oldest first (database returns newest first)
        candles = list(reversed(candles))

        return candles

    def _calculate_features(
        self, candles: list[Candle]
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Calculate features for all candles."""
        feature_names = self.feature_pipeline.feature_names
        n_features = len(feature_names)
        n_samples = len(candles)

        # We need a sliding window for feature calculation
        min_window = self.feature_pipeline.config.min_candles

        # Calculate features for each valid position
        valid_samples = n_samples - min_window
        if valid_samples <= 0:
            raise ValueError(
                f"Not enough candles for feature calculation: {n_samples} < {min_window}"
            )

        X = np.zeros((valid_samples, n_features), dtype=np.float64)

        for i in range(valid_samples):
            window = candles[i : i + min_window + 1]
            features = self.feature_pipeline.calculate_features(window, use_cache=False)

            for j, name in enumerate(feature_names):
                X[i, j] = features.get(name, 0.0)

        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, feature_names

    def _train_xgboost(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        feature_names: list[str],
        timestamp: datetime,
    ) -> tuple[float | None, Path | None]:
        """Train XGBoost model."""
        try:
            config = XGBoostConfig(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                early_stopping_rounds=10,
            )
            model = XGBoostModel(config)

            result = model.train(
                X=X_train,
                y=y_train,
                feature_names=feature_names,
                X_val=X_val,
                y_val=y_val,
            )

            if not result.success:
                logger.warning(f"XGBoost training failed: {result.error_message}")
                return None, None

            # Calculate validation accuracy
            predictions = model.predict(X_val)
            accuracy = float(np.mean((predictions > 0.5) == y_val))

            # Save model
            model_path = self.models_dir / f"xgboost_{timestamp.strftime('%Y%m%d_%H%M%S')}.joblib"
            model.save(model_path)

            self._xgboost = model

            logger.info(f"XGBoost trained: accuracy={accuracy:.4f}")
            return accuracy, model_path

        except Exception as e:
            logger.error(f"XGBoost training error: {e}", exc_info=True)
            return None, None

    def _train_lightgbm(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        timestamp: datetime,
    ) -> tuple[float | None, Path | None]:
        """Train LightGBM model."""
        try:
            config = LightGBMConfig(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.1,
            )
            model = LightGBMModel(config)

            result = model.train(X_train, y_train)

            if not result.success:
                logger.warning(f"LightGBM training failed: {result.error_message}")
                return None, None

            # Calculate validation accuracy
            predictions = model.predict(X_val)
            accuracy = float(np.mean((predictions > 0.5) == y_val))

            # Save model
            model_path = self.models_dir / f"lightgbm_{timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
            model.save(model_path)

            self._lightgbm = model

            logger.info(f"LightGBM trained: accuracy={accuracy:.4f}")
            return accuracy, model_path

        except Exception as e:
            logger.error(f"LightGBM training error: {e}", exc_info=True)
            return None, None

    def _train_lstm(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        timestamp: datetime,
    ) -> tuple[float | None, Path | None]:
        """Train LSTM model."""
        try:
            n_features = X_train.shape[1]
            config = LSTMConfig(
                input_size=n_features,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                sequence_length=20,
                batch_size=32,
                epochs=50,
                learning_rate=0.001,
            )
            model = LSTMModel(config)

            # Train the model
            model.train(X_train, y_train)

            # Calculate validation accuracy
            # Need enough samples for sequence creation
            if len(X_val) > config.sequence_length:
                predictions = model.predict(X_val)
                # Predictions are for samples after sequence window
                y_val_trimmed = y_val[config.sequence_length - 1 :]
                if len(predictions) == len(y_val_trimmed):
                    accuracy = float(np.mean((predictions > 0.5) == y_val_trimmed))
                else:
                    accuracy = 0.5  # Default if mismatch
            else:
                accuracy = 0.5  # Not enough validation data

            # Save model
            model_path = self.models_dir / f"lstm_{timestamp.strftime('%Y%m%d_%H%M%S')}.pt"
            model.save(model_path)

            self._lstm = model

            logger.info(f"LSTM trained: accuracy={accuracy:.4f}")
            return accuracy, model_path

        except Exception as e:
            logger.error(f"LSTM training error: {e}", exc_info=True)
            return None, None

    def _train_random_forest(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        X_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        timestamp: datetime,
    ) -> tuple[float | None, Path | None]:
        """Train Random Forest model."""
        try:
            config = RandomForestConfig(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
            )
            model = RandomForestModel(config)

            # Convert to integer labels for RandomForest
            y_train_int = y_train.astype(np.int64)

            model.train(X_train, y_train_int)

            # Calculate validation accuracy
            predictions = model.predict(X_val)
            accuracy = float(np.mean((predictions > 0.5) == y_val))

            # Save model
            model_path = self.models_dir / f"random_forest_{timestamp.strftime('%Y%m%d_%H%M%S')}.joblib"
            model.save(model_path)

            self._random_forest = model

            logger.info(f"RandomForest trained: accuracy={accuracy:.4f}")
            return accuracy, model_path

        except Exception as e:
            logger.error(f"RandomForest training error: {e}", exc_info=True)
            return None, None

    def should_retrain(
        self,
        current_regime: MarketRegime,
        model_health_results: dict[str, bool],
        drift_score: float,
        validation_accuracy: float,
        volatility: float,
    ) -> tuple[bool, RetrainingTrigger | None]:
        """
        Convenience method to check if retraining should occur.

        Args:
            current_regime: Current detected market regime
            model_health_results: Dict of model name -> health status
            drift_score: Current feature drift score (PSI)
            validation_accuracy: Recent out-of-sample accuracy
            volatility: Current ATR ratio

        Returns:
            Tuple of (should_retrain, trigger) where trigger is the reason
        """
        trigger = self.check_triggers(
            current_regime=current_regime,
            model_health_results=model_health_results,
            drift_score=drift_score,
            validation_accuracy=validation_accuracy,
            volatility=volatility,
        )
        return (trigger is not None, trigger)

    @property
    def last_training_time(self) -> datetime | None:
        """Get last training timestamp."""
        return self._trigger_state.last_retrain_time

    @property
    def last_trigger(self) -> RetrainingTrigger | None:
        """Get the trigger that caused the last retraining."""
        return self._trigger_state.last_trigger

    @property
    def trigger_state(self) -> TriggerState:
        """Get current trigger state for monitoring."""
        return self._trigger_state

    def get_model(self, name: str) -> XGBoostModel | LightGBMModel | LSTMModel | RandomForestModel | None:
        """
        Get a trained model by name.

        Args:
            name: Model name (xgboost, lightgbm, lstm, random_forest)

        Returns:
            Model instance or None if not trained
        """
        models: dict[str, XGBoostModel | LightGBMModel | LSTMModel | RandomForestModel | None] = {
            "xgboost": self._xgboost,
            "lightgbm": self._lightgbm,
            "lstm": self._lstm,
            "random_forest": self._random_forest,
        }
        return models.get(name)

    async def load_latest_models(self) -> dict[str, bool]:
        """
        Load the latest saved models from disk.

        Returns:
            Dictionary of model names and load status
        """
        load_status: dict[str, bool] = {}

        for model_type in ["xgboost", "lightgbm", "lstm", "random_forest"]:
            pattern = f"{model_type}_*.{'pt' if model_type == 'lstm' else 'txt' if model_type == 'lightgbm' else 'joblib'}"
            model_files = sorted(self.models_dir.glob(pattern), reverse=True)

            if not model_files:
                load_status[model_type] = False
                continue

            latest_file = model_files[0]

            try:
                if model_type == "xgboost":
                    self._xgboost = XGBoostModel()
                    self._xgboost.load(latest_file)
                    load_status["xgboost"] = True
                elif model_type == "lightgbm":
                    self._lightgbm = LightGBMModel()
                    self._lightgbm.load(latest_file)
                    load_status["lightgbm"] = True
                elif model_type == "lstm":
                    # LSTM requires config with correct input_size
                    # This is a limitation - we need to know feature count
                    n_features = self.feature_pipeline.feature_count
                    self._lstm = LSTMModel(LSTMConfig(input_size=n_features))
                    self._lstm.load(latest_file)
                    load_status["lstm"] = True
                elif model_type == "random_forest":
                    self._random_forest = RandomForestModel(RandomForestConfig())
                    self._random_forest.load(latest_file)
                    load_status["random_forest"] = True

                logger.info(f"Loaded {model_type} from {latest_file}")

            except Exception as e:
                logger.error(f"Failed to load {model_type}: {e}")
                load_status[model_type] = False

        return load_status
