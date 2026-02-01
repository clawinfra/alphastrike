"""
AlphaStrike Trading Bot - XGBoost Model Wrapper (US-010)

XGBoost classifier for directional prediction with health checks
and feature alignment support between training and inference.

Features:
- Binary classification with probability outputs (0-1)
- Feature alignment between train/inference
- Health check to detect degenerate models
- Model persistence (save/load)
- Comprehensive training results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class XGBoostConfig(BaseModel):
    """Configuration for XGBoost model."""

    n_estimators: int = Field(default=100, ge=1, description="Number of boosting rounds")
    max_depth: int = Field(default=6, ge=1, le=20, description="Maximum tree depth")
    learning_rate: float = Field(
        default=0.1, gt=0.0, le=1.0, description="Learning rate (eta)"
    )
    subsample: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Row subsampling ratio"
    )
    colsample_bytree: float = Field(
        default=0.8, gt=0.0, le=1.0, description="Column subsampling ratio per tree"
    )
    reg_alpha: float = Field(default=0.1, ge=0.0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0.0, description="L2 regularization")

    # Class imbalance handling
    scale_pos_weight: float = Field(default=1.0, ge=0.0, description="Weight for positive class")
    min_child_weight: float = Field(default=1.0, ge=0.0, description="Min sum of instance weight in child")

    # Additional parameters
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    n_jobs: int = Field(default=-1, description="Number of parallel threads (-1 = all)")
    early_stopping_rounds: int | None = Field(
        default=10, ge=1, description="Early stopping rounds (None to disable)"
    )
    eval_metric: str = Field(default="logloss", description="Evaluation metric")


# =============================================================================
# Training Result
# =============================================================================


@dataclass
class TrainingResult:
    """Result of model training."""

    success: bool
    train_samples: int
    train_time_seconds: float
    best_iteration: int | None = None
    best_score: float | None = None
    train_logloss: float | None = None
    val_logloss: float | None = None
    feature_names: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "train_samples": self.train_samples,
            "train_time_seconds": self.train_time_seconds,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "train_logloss": self.train_logloss,
            "val_logloss": self.val_logloss,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "error_message": self.error_message,
        }


# =============================================================================
# XGBoost Model Wrapper
# =============================================================================


class XGBoostModel:
    """
    XGBoost model wrapper for binary classification.

    Provides probability predictions in [0, 1] range where:
    - Values > 0.5 indicate positive class (e.g., LONG signal)
    - Values < 0.5 indicate negative class (e.g., SHORT signal)
    - Values near 0.5 indicate uncertainty

    Includes health checks to detect degenerate models that:
    - Predict same class > 95% of the time
    - Have prediction variance < 0.01

    Usage:
        config = XGBoostConfig(n_estimators=200, max_depth=5)
        model = XGBoostModel(config)

        # Train
        result = model.train(X_train, y_train)
        print(f"Training success: {result.success}")

        # Predict probabilities
        probs = model.predict(X_test)  # Shape: (n_samples,)

        # Health check
        if not model.health_check():
            print("Model may be degenerate!")

        # Save/load
        model.save(Path("model.joblib"))
        model.load(Path("model.joblib"))
    """

    def __init__(self, config: XGBoostConfig | None = None):
        """
        Initialize XGBoost model.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or XGBoostConfig()
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self._is_trained: bool = False
        self._last_predictions: NDArray[np.float64] | None = None

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained and self._model is not None

    @property
    def feature_names(self) -> list[str] | None:
        """Get feature names used during training."""
        return self._feature_names

    @property
    def n_features(self) -> int | None:
        """Get number of features expected by the model."""
        if self._feature_names is not None:
            return len(self._feature_names)
        return None

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str] | None = None,
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.float64] | None = None,
        validation_split: float = 0.2,
    ) -> TrainingResult:
        """
        Train the XGBoost model.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,) with values 0 or 1
            feature_names: Optional list of feature names
            X_val: Optional validation features (if not provided, split from X)
            y_val: Optional validation labels
            validation_split: Fraction of data for validation if X_val not provided

        Returns:
            TrainingResult with training metrics and status
        """
        start_time = time.time()

        try:
            # Validate inputs
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            if len(X.shape) != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            if len(y.shape) != 1:
                raise ValueError(f"y must be 1D, got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
                )
            if X.shape[0] < 10:
                raise ValueError(f"Need at least 10 samples, got {X.shape[0]}")

            n_samples, n_features = X.shape

            # Handle NaN/Inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Store feature names
            if feature_names is not None:
                if len(feature_names) != n_features:
                    raise ValueError(
                        f"feature_names length {len(feature_names)} != n_features {n_features}"
                    )
                self._feature_names = list(feature_names)
            else:
                self._feature_names = [f"f{i}" for i in range(n_features)]

            # Create validation set if not provided
            if X_val is None or y_val is None:
                split_idx = int(n_samples * (1 - validation_split))
                if split_idx < 5:
                    split_idx = n_samples  # No validation for tiny datasets
                    X_train, y_train = X, y
                    X_val_final, y_val_final = None, None
                else:
                    X_train, y_train = X[:split_idx], y[:split_idx]
                    X_val_final = X[split_idx:]
                    y_val_final = y[split_idx:]
            else:
                X_train, y_train = X, y
                X_val_final = np.asarray(X_val, dtype=np.float64)
                y_val_final = np.asarray(y_val, dtype=np.float64)
                X_val_final = np.nan_to_num(X_val_final, nan=0.0, posinf=0.0, neginf=0.0)

            # Build model parameters
            model_params: dict[str, Any] = {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bytree,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "scale_pos_weight": self.config.scale_pos_weight,
                "min_child_weight": self.config.min_child_weight,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                "objective": "binary:logistic",
                "eval_metric": self.config.eval_metric,
            }

            # Add early stopping if validation set will be available
            if X_val_final is not None and y_val_final is not None:
                if self.config.early_stopping_rounds is not None:
                    model_params["early_stopping_rounds"] = self.config.early_stopping_rounds

            # Create model
            self._model = xgb.XGBClassifier(**model_params)

            # Prepare eval_set for training
            if X_val_final is not None and y_val_final is not None:
                eval_set = [(X_train, y_train), (X_val_final, y_val_final)]
            else:
                eval_set = [(X_train, y_train)]

            # Train
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            self._is_trained = True

            # Extract training results
            train_time = time.time() - start_time

            best_iteration = getattr(self._model, "best_iteration", None)
            best_score = getattr(self._model, "best_score", None)

            # Get training metrics from evals_result
            evals_result = getattr(self._model, "evals_result_", {})
            train_logloss = None
            val_logloss = None

            if "validation_0" in evals_result:
                metric_key = self.config.eval_metric
                if metric_key in evals_result["validation_0"]:
                    train_logloss = evals_result["validation_0"][metric_key][-1]

            if "validation_1" in evals_result:
                metric_key = self.config.eval_metric
                if metric_key in evals_result["validation_1"]:
                    val_logloss = evals_result["validation_1"][metric_key][-1]

            # Get feature importances
            feature_importances = self._get_feature_importances()

            logger.info(
                f"XGBoost training completed: {n_samples} samples, "
                f"{n_features} features, {train_time:.2f}s"
            )

            return TrainingResult(
                success=True,
                train_samples=n_samples,
                train_time_seconds=train_time,
                best_iteration=best_iteration,
                best_score=best_score,
                train_logloss=train_logloss,
                val_logloss=val_logloss,
                feature_names=self._feature_names,
                feature_importances=feature_importances,
            )

        except Exception as e:
            train_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"XGBoost training failed: {error_msg}")

            return TrainingResult(
                success=False,
                train_samples=X.shape[0] if hasattr(X, "shape") else 0,
                train_time_seconds=train_time,
                error_message=error_msg,
            )

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict class probabilities for positive class.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            Probabilities in [0, 1] range, shape (n_samples,)

        Raises:
            RuntimeError: If model has not been trained
            ValueError: If feature dimensions don't match
        """
        if not self.is_trained or self._model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = np.asarray(X, dtype=np.float64)

        # Handle 1D input (single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if len(X.shape) != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")

        # Feature alignment
        X = self._align_features(X)

        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get probability of positive class
        proba = self._model.predict_proba(X)

        # proba has shape (n_samples, 2) for binary classification
        # Column 1 is probability of positive class
        positive_proba: NDArray[np.float64] = proba[:, 1].astype(np.float64)

        # Clip to ensure [0, 1] range
        positive_proba = np.clip(positive_proba, 0.0, 1.0)

        # Store for health check
        self._last_predictions = positive_proba

        return positive_proba

    def _align_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Align inference features to match training features.

        If inference has more features, extra columns are dropped.
        If inference has fewer features, missing columns are filled with zeros.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Aligned features matching training feature count
        """
        if self._feature_names is None:
            return X

        expected_features = len(self._feature_names)
        actual_features = X.shape[1]

        if actual_features == expected_features:
            return X

        if actual_features > expected_features:
            # Truncate extra features
            logger.warning(
                f"Feature count mismatch: got {actual_features}, expected {expected_features}. "
                f"Truncating extra features."
            )
            return X[:, :expected_features]

        # Pad missing features with zeros
        logger.warning(
            f"Feature count mismatch: got {actual_features}, expected {expected_features}. "
            f"Padding with zeros."
        )
        n_samples = X.shape[0]
        padded = np.zeros((n_samples, expected_features), dtype=np.float64)
        padded[:, :actual_features] = X
        return padded

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model file

        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained or self._model is None:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "config": self.config.model_dump(),
        }

        joblib.dump(model_data, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            model_data = joblib.load(path)

            if "model" not in model_data:
                raise ValueError("Invalid model file: missing 'model' key")

            self._model = model_data["model"]
            self._feature_names = model_data.get("feature_names")

            # Optionally restore config
            if "config" in model_data:
                self.config = XGBoostConfig(**model_data["config"])

            self._is_trained = True
            logger.info(f"XGBoost model loaded from {path}")

        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e

    def health_check(self) -> bool:
        """
        Check if model is healthy (not degenerate).

        A model is considered unhealthy/degenerate if:
        - It predicts the same class > 95% of the time
        - Its probability variance < 0.01

        Uses the last batch of predictions from predict().
        If no predictions available, generates synthetic test data.

        Returns:
            True if model is healthy, False if degenerate
        """
        if not self.is_trained or self._model is None:
            logger.warning("Health check called on untrained model")
            return False

        # Use last predictions or generate test data
        if self._last_predictions is not None and len(self._last_predictions) >= 10:
            predictions = self._last_predictions
        else:
            # Generate synthetic test data for health check
            predictions = self._generate_health_check_predictions()

        if predictions is None or len(predictions) < 10:
            logger.warning("Insufficient predictions for health check")
            return True  # Assume healthy if can't verify

        # Check 1: Same class prediction dominance (> 95%)
        positive_ratio = np.mean(predictions > 0.5)
        same_class_dominant = positive_ratio > 0.95 or positive_ratio < 0.05

        if same_class_dominant:
            logger.warning(
                f"Degenerate model detected: same class ratio = {max(positive_ratio, 1 - positive_ratio):.2%}"
            )
            return False

        # Check 2: Low variance (< 0.01)
        variance = float(np.var(predictions))

        if variance < 0.01:
            logger.warning(
                f"Degenerate model detected: probability variance = {variance:.4f}"
            )
            return False

        logger.debug(
            f"Model health check passed: positive_ratio={positive_ratio:.2%}, variance={variance:.4f}"
        )
        return True

    def _generate_health_check_predictions(self) -> NDArray[np.float64] | None:
        """
        Generate predictions on synthetic data for health check.

        Creates random features within reasonable ranges and predicts.

        Returns:
            Array of predictions or None if generation fails
        """
        if self._feature_names is None:
            return None

        n_features = len(self._feature_names)
        n_samples = 100

        # Generate random features (normalized around 0 with unit variance)
        np.random.seed(42)  # Reproducible for consistent health checks
        X_synthetic = np.random.randn(n_samples, n_features)

        try:
            return self.predict(X_synthetic)
        except Exception as e:
            logger.warning(f"Failed to generate health check predictions: {e}")
            return None

    def _get_feature_importances(self) -> dict[str, float]:
        """
        Get feature importances from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._model is None or self._feature_names is None:
            return {}

        try:
            importances = self._model.feature_importances_
            return {
                name: float(imp)
                for name, imp in zip(self._feature_names, importances)
            }
        except Exception as e:
            logger.warning(f"Failed to get feature importances: {e}")
            return {}

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "is_trained": self.is_trained,
            "n_features": self.n_features,
            "feature_names": self._feature_names,
            "config": self.config.model_dump() if self.config else None,
            "best_iteration": getattr(self._model, "best_iteration", None)
            if self._model
            else None,
        }
