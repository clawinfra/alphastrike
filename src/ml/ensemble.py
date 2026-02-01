"""
AlphaStrike Trading Bot - ML Ensemble Orchestrator (US-014)

Combines predictions from XGBoost, LightGBM, LSTM, and Random Forest models
using weighted averaging for robust trading signal generation.

Base Weights:
- XGBoost: 30%
- LightGBM: 25%
- LSTM: 25%
- Random Forest: 20%

Dynamic Weight Adjustment:
- Accuracy > 55%: Full weight
- Accuracy 50-55%: 75% weight
- Accuracy 45-50%: 50% weight
- Accuracy < 45% (persistent): Excluded (0% weight)
- Regime-specific boosts for models performing well in current regime

Signal Generation:
- LONG: weighted_avg > 0.75
- SHORT: weighted_avg < 0.25
- HOLD: otherwise

Requires minimum 2 healthy models for signal generation.
Diversity protection ensures we never exclude all but 1 model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.ml.lightgbm_model import LightGBMConfig, LightGBMModel
from src.ml.lstm_model import LSTMConfig, LSTMModel
from src.ml.quality_tracker import (
    ModelQualityTracker,
    QualityTrackerConfig,
    SignalType,
)
from src.ml.random_forest_model import RandomForestConfig, RandomForestModel
from src.ml.xgboost_model import XGBoostConfig, XGBoostModel

logger = logging.getLogger(__name__)

# Type alias for regime
RegimeType = Literal["trending", "ranging", "volatile", "unknown"]


# Default model weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "xgboost": 0.30,
    "lightgbm": 0.25,
    "lstm": 0.25,
    "random_forest": 0.20,
}

# Signal thresholds
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.25

# Minimum healthy models required
MIN_HEALTHY_MODELS = 2


class MLEnsemble:
    """
    ML Ensemble Orchestrator for trading signal generation.

    Combines predictions from multiple ML models using weighted averaging:
    - XGBoost (30%): Gradient boosting for tabular features
    - LightGBM (25%): Fast gradient boosting with leaf-wise growth
    - LSTM (25%): Sequential pattern recognition
    - Random Forest (20%): Ensemble of decision trees

    Includes health monitoring and automatic weight redistribution
    when models become unhealthy.

    Example:
        ensemble = MLEnsemble(models_dir=Path("models"))
        signal, confidence, outputs, weighted_avg = ensemble.predict(features)
        print(f"Signal: {signal}, Confidence: {confidence:.2%}")
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        weights: dict[str, float] | None = None,
        lstm_input_size: int = 20,
        quality_config: QualityTrackerConfig | None = None,
    ) -> None:
        """
        Initialize the ML Ensemble.

        Args:
            models_dir: Directory containing saved model files for hot reload.
                        Expected files: xgboost.joblib, lightgbm.txt, lstm.pt, random_forest.joblib
            weights: Custom model weights (must sum to 1.0). Uses defaults if None.
            lstm_input_size: Number of input features for LSTM model.
            quality_config: Configuration for model quality tracking. Uses defaults if None.
        """
        self.models_dir = models_dir or Path("models")
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.lstm_input_size = lstm_input_size

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.001):
            raise ValueError(f"Model weights must sum to 1.0, got {weight_sum}")

        # Initialize models
        self._xgboost: XGBoostModel | None = None
        self._lightgbm: LightGBMModel | None = None
        self._lstm: LSTMModel | None = None
        self._random_forest: RandomForestModel | None = None

        # Track model health status
        self._model_health: dict[str, bool] = {
            "xgboost": False,
            "lightgbm": False,
            "lstm": False,
            "random_forest": False,
        }

        # Quality tracker for dynamic weight adjustment based on accuracy
        self._quality_tracker = ModelQualityTracker(
            config=quality_config or QualityTrackerConfig()
        )

        logger.info(
            "MLEnsemble initialized",
            extra={
                "models_dir": str(self.models_dir),
                "weights": self.weights,
            },
        )

    @property
    def xgboost(self) -> XGBoostModel | None:
        """Get XGBoost model instance."""
        return self._xgboost

    @property
    def lightgbm(self) -> LightGBMModel | None:
        """Get LightGBM model instance."""
        return self._lightgbm

    @property
    def lstm(self) -> LSTMModel | None:
        """Get LSTM model instance."""
        return self._lstm

    @property
    def random_forest(self) -> RandomForestModel | None:
        """Get Random Forest model instance."""
        return self._random_forest

    def set_models(
        self,
        xgboost: XGBoostModel | None = None,
        lightgbm: LightGBMModel | None = None,
        lstm: LSTMModel | None = None,
        random_forest: RandomForestModel | None = None,
    ) -> None:
        """
        Set model instances directly (alternative to loading from disk).

        Args:
            xgboost: Trained XGBoost model
            lightgbm: Trained LightGBM model
            lstm: Trained LSTM model
            random_forest: Trained Random Forest model
        """
        if xgboost is not None:
            self._xgboost = xgboost
        if lightgbm is not None:
            self._lightgbm = lightgbm
        if lstm is not None:
            self._lstm = lstm
        if random_forest is not None:
            self._random_forest = random_forest

        # Update health status
        self._update_all_health_status()

    def _update_all_health_status(self) -> None:
        """Update health status for all models."""
        for model_name in self._model_health:
            self._model_health[model_name] = self.check_model_health(model_name)

    def check_model_health(self, model_name: str) -> bool:
        """
        Check if a specific model is healthy.

        A model is considered healthy if:
        1. It exists and is trained
        2. Its health_check() method returns True

        Args:
            model_name: Name of the model ("xgboost", "lightgbm", "lstm", "random_forest")

        Returns:
            True if model is healthy, False otherwise
        """
        model = self._get_model_by_name(model_name)

        if model is None:
            logger.debug(f"Model {model_name} not loaded")
            return False

        # Check if model is trained
        if not getattr(model, "is_trained", False) and not getattr(
            model, "_is_trained", False
        ):
            logger.debug(f"Model {model_name} not trained")
            return False

        # Run model's health check
        try:
            is_healthy = model.health_check()
            if not is_healthy:
                logger.warning(f"Model {model_name} failed health check")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {model_name}: {e}")
            return False

    def _get_model_by_name(
        self, model_name: str
    ) -> XGBoostModel | LightGBMModel | LSTMModel | RandomForestModel | None:
        """Get model instance by name."""
        model_map: dict[
            str, XGBoostModel | LightGBMModel | LSTMModel | RandomForestModel | None
        ] = {
            "xgboost": self._xgboost,
            "lightgbm": self._lightgbm,
            "lstm": self._lstm,
            "random_forest": self._random_forest,
        }
        return model_map.get(model_name)

    def check_and_reload_models(self) -> None:
        """
        Hot reload models from the models directory.

        Loads models from:
        - models/xgboost.joblib
        - models/lightgbm.txt
        - models/lstm.pt
        - models/random_forest.joblib

        Only reloads models that exist on disk. Updates health status after reload.
        """
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        # XGBoost
        xgboost_path = self.models_dir / "xgboost.joblib"
        if xgboost_path.exists():
            try:
                self._xgboost = XGBoostModel(XGBoostConfig())
                self._xgboost.load(xgboost_path)
                logger.info(f"Loaded XGBoost model from {xgboost_path}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
                self._xgboost = None

        # LightGBM
        lightgbm_path = self.models_dir / "lightgbm.txt"
        if lightgbm_path.exists():
            try:
                self._lightgbm = LightGBMModel(LightGBMConfig())
                self._lightgbm.load(lightgbm_path)
                logger.info(f"Loaded LightGBM model from {lightgbm_path}")
            except Exception as e:
                logger.error(f"Failed to load LightGBM model: {e}")
                self._lightgbm = None

        # LSTM
        lstm_path = self.models_dir / "lstm.pt"
        if lstm_path.exists():
            try:
                self._lstm = LSTMModel(LSTMConfig(input_size=self.lstm_input_size))
                self._lstm.load(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
                self._lstm = None

        # Random Forest
        rf_path = self.models_dir / "random_forest.joblib"
        if rf_path.exists():
            try:
                self._random_forest = RandomForestModel(RandomForestConfig())
                self._random_forest.load(rf_path)
                logger.info(f"Loaded Random Forest model from {rf_path}")
            except Exception as e:
                logger.error(f"Failed to load Random Forest model: {e}")
                self._random_forest = None

        # Update health status for all models
        self._update_all_health_status()

        healthy_count = sum(self._model_health.values())
        logger.info(
            f"Model reload complete: {healthy_count}/4 healthy models",
            extra={"health_status": self._model_health},
        )

    def _redistribute_weights(self, unhealthy: list[str]) -> dict[str, float]:
        """
        Redistribute weights when some models are unhealthy.

        Proportionally redistributes the weights of unhealthy models
        to healthy models based on their original weight ratios.

        Args:
            unhealthy: List of unhealthy model names

        Returns:
            New weight distribution with unhealthy models set to 0
        """
        if not unhealthy:
            return self.weights.copy()

        new_weights = self.weights.copy()

        # Set unhealthy model weights to 0
        unhealthy_weight_sum = 0.0
        for model_name in unhealthy:
            if model_name in new_weights:
                unhealthy_weight_sum += new_weights[model_name]
                new_weights[model_name] = 0.0

        # Get healthy models and their total weight
        healthy_models = [m for m in new_weights if m not in unhealthy]
        healthy_weight_sum = sum(new_weights[m] for m in healthy_models)

        if healthy_weight_sum <= 0:
            # No healthy models - return zeros
            return {m: 0.0 for m in new_weights}

        # Redistribute proportionally
        for model_name in healthy_models:
            original_weight = new_weights[model_name]
            proportion = original_weight / healthy_weight_sum
            redistributed = proportion * unhealthy_weight_sum
            new_weights[model_name] = original_weight + redistributed

        logger.debug(
            f"Weights redistributed: unhealthy={unhealthy}, new_weights={new_weights}"
        )

        return new_weights

    def predict(
        self,
        features: dict[str, Any],
        current_regime: RegimeType = "unknown",
    ) -> tuple[str, float, dict[str, float], float]:
        """
        Generate ensemble prediction from all healthy models.

        Args:
            features: Dictionary of feature values. Expected to contain:
                - "array": numpy array of shape (n_features,) or (n_samples, n_features)
                - Or individual feature keys that will be converted to array
            current_regime: Current market regime for quality-adjusted weighting.
                           One of "trending", "ranging", "volatile", or "unknown".

        Returns:
            Tuple of:
                - signal: "LONG", "SHORT", or "HOLD"
                - confidence: Confidence score (0.0 to 1.0)
                - model_outputs: Dict of each model's prediction
                - weighted_avg: The combined weighted average prediction

        Note:
            Returns ("HOLD", 0.0, {}, 0.5) if fewer than MIN_HEALTHY_MODELS are available.
        """
        # Update health status
        self._update_all_health_status()

        # Count healthy models
        healthy_models = [m for m, healthy in self._model_health.items() if healthy]
        unhealthy_models = [m for m, healthy in self._model_health.items() if not healthy]

        if len(healthy_models) < MIN_HEALTHY_MODELS:
            logger.warning(
                f"Insufficient healthy models: {len(healthy_models)} < {MIN_HEALTHY_MODELS}"
            )
            return ("HOLD", 0.0, {}, 0.5)

        # Get base redistributed weights (for unhealthy models)
        base_weights = self._redistribute_weights(unhealthy_models)

        # Apply quality-adjusted weights based on accuracy tracking
        effective_weights = self._quality_tracker.get_adjusted_weights(
            base_weights=base_weights,
            current_regime=current_regime,
            health_status=self._model_health,
        )

        # Extract feature array
        feature_array = self._extract_feature_array(features)

        # Get predictions from each healthy model
        model_outputs: dict[str, float] = {}
        model_confidences: dict[str, float] = {}

        # XGBoost prediction
        if self._model_health["xgboost"] and self._xgboost is not None:
            try:
                pred = self._xgboost.predict(feature_array)
                pred_value = float(np.mean(pred))
                model_outputs["xgboost"] = pred_value
                # Estimate confidence from prediction extremity
                model_confidences["xgboost"] = abs(pred_value - 0.5) * 2
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
                self._model_health["xgboost"] = False

        # LightGBM prediction
        if self._model_health["lightgbm"] and self._lightgbm is not None:
            try:
                pred = self._lightgbm.predict(feature_array)
                pred_value = float(np.mean(pred))
                model_outputs["lightgbm"] = pred_value
                model_confidences["lightgbm"] = abs(pred_value - 0.5) * 2
            except Exception as e:
                logger.error(f"LightGBM prediction failed: {e}")
                self._model_health["lightgbm"] = False

        # LSTM prediction
        if self._model_health["lstm"] and self._lstm is not None:
            try:
                pred = self._lstm.predict(feature_array)
                pred_value = float(np.mean(pred))
                model_outputs["lstm"] = pred_value
                model_confidences["lstm"] = abs(pred_value - 0.5) * 2
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
                self._model_health["lstm"] = False

        # Random Forest prediction
        if self._model_health["random_forest"] and self._random_forest is not None:
            try:
                pred = self._random_forest.predict(feature_array)
                pred_value = float(np.mean(pred))
                model_outputs["random_forest"] = pred_value
                model_confidences["random_forest"] = abs(pred_value - 0.5) * 2
            except Exception as e:
                logger.error(f"Random Forest prediction failed: {e}")
                self._model_health["random_forest"] = False

        # Check if we still have enough models after potential failures
        if len(model_outputs) < MIN_HEALTHY_MODELS:
            logger.warning(
                f"Too many prediction failures: only {len(model_outputs)} models succeeded"
            )
            return ("HOLD", 0.0, model_outputs, 0.5)

        # Record predictions with quality tracker for accuracy tracking
        for model_name, pred_value in model_outputs.items():
            # Convert prediction to signal for tracking
            pred_signal = self._determine_signal(pred_value)
            confidence = model_confidences.get(model_name, 0.5)
            self._quality_tracker.record_prediction(
                model_name=model_name,
                predicted_signal=pred_signal,
                regime=current_regime,
                confidence=confidence,
            )

        # Calculate weighted average using quality-adjusted weights
        weighted_sum = 0.0
        total_weight = 0.0
        for model_name, pred_value in model_outputs.items():
            weight = effective_weights.get(model_name, 0.0)
            weighted_sum += pred_value * weight
            total_weight += weight

        if total_weight > 0:
            weighted_avg = weighted_sum / total_weight
        else:
            weighted_avg = 0.5

        # Determine signal
        signal = self._determine_signal(weighted_avg)

        # Calculate confidence
        confidence = self._calculate_confidence(signal, weighted_avg)

        logger.info(
            f"Ensemble prediction: signal={signal}, confidence={confidence:.2%}, "
            f"weighted_avg={weighted_avg:.4f}, regime={current_regime}",
            extra={
                "model_outputs": model_outputs,
                "effective_weights": effective_weights,
                "quality_stats": self._quality_tracker.get_all_stats(),
            },
        )

        return (signal, confidence, model_outputs, weighted_avg)

    def _extract_feature_array(self, features: dict[str, Any]) -> np.ndarray:
        """
        Extract numpy array from features dictionary.

        Args:
            features: Dictionary containing either:
                - "array" key with numpy array
                - Individual feature keys

        Returns:
            Numpy array suitable for model prediction
        """
        if "array" in features:
            arr = np.asarray(features["array"], dtype=np.float64)
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        # Convert dict values to array
        values = [float(v) for k, v in features.items() if isinstance(v, (int, float))]
        arr = np.array(values, dtype=np.float64).reshape(1, -1)
        return arr

    def _determine_signal(self, weighted_avg: float) -> SignalType:
        """
        Determine trading signal from weighted average.

        Args:
            weighted_avg: Combined prediction value (0-1)

        Returns:
            "LONG", "SHORT", or "HOLD"
        """
        if weighted_avg > LONG_THRESHOLD:
            return "LONG"
        elif weighted_avg < SHORT_THRESHOLD:
            return "SHORT"
        else:
            return "HOLD"

    def _calculate_confidence(self, signal: str, weighted_avg: float) -> float:
        """
        Calculate confidence score using threshold-relative formula.

        Confidence indicates how far beyond the threshold the prediction is:
        - LONG: (weighted_avg - long_threshold) / (1.0 - long_threshold)
        - SHORT: (short_threshold - weighted_avg) / short_threshold
        - HOLD: 0.0

        Args:
            signal: The determined signal ("LONG", "SHORT", or "HOLD")
            weighted_avg: The combined prediction value

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if signal == "LONG":
            confidence = (weighted_avg - LONG_THRESHOLD) / (1.0 - LONG_THRESHOLD)
        elif signal == "SHORT":
            confidence = (SHORT_THRESHOLD - weighted_avg) / SHORT_THRESHOLD
        else:
            confidence = 0.0

        # Clip to valid range
        return float(np.clip(confidence, 0.0, 1.0))

    def get_health_status(self) -> dict[str, bool]:
        """
        Get current health status of all models.

        Returns:
            Dictionary mapping model names to health status
        """
        self._update_all_health_status()
        return self._model_health.copy()

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the ensemble configuration.

        Returns:
            Dictionary with ensemble metadata
        """
        return {
            "models_dir": str(self.models_dir),
            "weights": self.weights.copy(),
            "health_status": self._model_health.copy(),
            "healthy_count": sum(self._model_health.values()),
            "min_required": MIN_HEALTHY_MODELS,
            "thresholds": {
                "long": LONG_THRESHOLD,
                "short": SHORT_THRESHOLD,
            },
        }

    def record_outcome(self, actual_signal: str) -> None:
        """
        Record actual market outcome for all models that made predictions.

        Should be called after the market confirms the signal direction.
        This updates accuracy statistics used for dynamic weight adjustment.

        Args:
            actual_signal: The actual market outcome - "LONG" (price went up),
                          "SHORT" (price went down), or "HOLD" (sideways)

        Example:
            # After prediction
            signal, conf, outputs, avg = ensemble.predict(features, regime="trending")
            # ... time passes, market moves ...
            # After outcome is known
            ensemble.record_outcome("LONG")  # Price actually went up
        """
        # Use batch method to record outcome for all models at once
        self._quality_tracker.record_batch_outcomes(actual_signal)  # type: ignore[arg-type]
        logger.debug(f"Recorded outcome for all models: {actual_signal}")

    def get_quality_stats(self) -> dict[str, Any]:
        """
        Get quality statistics for all models.

        Returns:
            Dictionary with per-model accuracy stats including:
            - overall_accuracy: Rolling accuracy across all regimes
            - regime_accuracy: Per-regime accuracy breakdown
            - status: Current model status (ACTIVE, DEGRADED, EXCLUDED)
            - weight_multiplier: Current weight adjustment factor
        """
        return self._quality_tracker.get_all_stats()

    def check_model_recovery(self, model_name: str) -> bool:
        """
        Check if an excluded/degraded model has recovered.

        Args:
            model_name: Name of the model to check

        Returns:
            True if the model has recovered and been restored to active status
        """
        return self._quality_tracker.check_recovery(model_name)

    def reset_quality_stats(self) -> None:
        """
        Reset all quality tracking statistics.

        Use with caution - typically after significant model retraining
        or when starting a new trading session with fresh models.
        """
        self._quality_tracker = ModelQualityTracker(
            config=self._quality_tracker.config
        )
        logger.info("Quality tracking statistics reset")
