"""
AlphaStrike Trading Bot - LightGBM Model Wrapper

LightGBM model for fast gradient boosting prediction (25% ensemble weight).
Implements leaf-wise tree growth with degenerate tree detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model."""

    n_estimators: int = 100
    num_leaves: int = 31  # CRITICAL: must be > 1
    max_depth: int = -1
    learning_rate: float = 0.1
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 0.0  # L2 regularization
    is_unbalance: bool = False  # Handle class imbalance
    verbose: int = -1  # Suppress LightGBM output
    random_state: int = 42
    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"


@dataclass
class TrainingResult:
    """Result of model training."""

    success: bool
    n_samples: int
    n_features: int
    training_iterations: int
    best_iteration: int
    validation_loss: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None


class LightGBMModel:
    """
    LightGBM model wrapper for AlphaStrike ensemble.

    Implements the same interface as XGBoost wrapper for consistency.
    Includes critical validation for num_leaves and degenerate tree detection.
    """

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        """
        Initialize LightGBM model.

        Args:
            config: LightGBM configuration. Uses defaults if not provided.
        """
        self.config = config or LightGBMConfig()
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] | None = None
        self._is_trained: bool = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TrainingResult:
        """
        Train the LightGBM model.

        CRITICAL: Validates num_leaves > 1 before training to prevent
        degenerate trees that predict a single class.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels (0 or 1 for binary classification)
            feature_names: Optional list of feature names. If provided, these
                are stored in the model file for reproducibility and debugging.
                Without this, features are named feature_0, feature_1, etc.

        Returns:
            TrainingResult with training metrics and status
        """
        # CRITICAL: Validate num_leaves > 1
        if self.config.num_leaves <= 1:
            error_msg = (
                f"CRITICAL: num_leaves must be > 1, got {self.config.num_leaves}. "
                "num_leaves=1 creates degenerate trees that predict single class."
            )
            logger.error(error_msg)
            return TrainingResult(
                success=False,
                n_samples=len(y),
                n_features=X.shape[1] if X.ndim > 1 else 1,
                training_iterations=0,
                best_iteration=0,
                error_message=error_msg,
            )

        # Validate input data
        if len(X) == 0 or len(y) == 0:
            return TrainingResult(
                success=False,
                n_samples=0,
                n_features=0,
                training_iterations=0,
                best_iteration=0,
                error_message="Empty training data provided",
            )

        if len(X) != len(y):
            return TrainingResult(
                success=False,
                n_samples=len(y),
                n_features=X.shape[1] if X.ndim > 1 else 1,
                training_iterations=0,
                best_iteration=0,
                error_message=f"X and y length mismatch: {len(X)} vs {len(y)}",
            )

        # Check for class imbalance
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return TrainingResult(
                success=False,
                n_samples=len(y),
                n_features=X.shape[1] if X.ndim > 1 else 1,
                training_iterations=0,
                best_iteration=0,
                error_message=f"Only one class in training data: {unique_classes}",
            )

        n_samples, n_features = X.shape

        # Use provided feature names or generate generic ones
        if feature_names is not None:
            if len(feature_names) != n_features:
                logger.warning(
                    f"Feature names length {len(feature_names)} != n_features {n_features}, "
                    "falling back to generic names"
                )
                self._feature_names = [f"feature_{i}" for i in range(n_features)]
            else:
                self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"feature_{i}" for i in range(n_features)]

        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(
                X,
                label=y,
                feature_name=self._feature_names,
            )

            # Build parameters
            params = self._get_params()

            # Train the model
            self._model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=self.config.n_estimators,
            )

            self._is_trained = True

            # Get feature importances
            importance_dict = {}
            if self._model is not None:
                importances = self._model.feature_importance(importance_type="gain")
                for i, imp in enumerate(importances):
                    importance_dict[self._feature_names[i]] = float(imp)

            logger.info(
                "LightGBM training completed",
                extra={
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_estimators": self.config.n_estimators,
                    "num_leaves": self.config.num_leaves,
                },
            )

            return TrainingResult(
                success=True,
                n_samples=n_samples,
                n_features=n_features,
                training_iterations=self.config.n_estimators,
                best_iteration=self._model.best_iteration if self._model else 0,
                feature_importances=importance_dict,
            )

        except Exception as e:
            error_msg = f"LightGBM training failed: {e!s}"
            logger.error(error_msg, exc_info=True)
            return TrainingResult(
                success=False,
                n_samples=n_samples,
                n_features=n_features,
                training_iterations=0,
                best_iteration=0,
                error_message=error_msg,
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions (probabilities 0-1).

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of probabilities between 0 and 1

        Raises:
            RuntimeError: If model is not trained
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # LightGBM predict returns probabilities directly for binary classification
        probabilities = self._model.predict(X, num_iteration=self._model.best_iteration)

        # Ensure output is numpy array with correct shape
        probabilities = np.asarray(probabilities)

        # Clip to valid probability range
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return probabilities

    def save(self, path: Path) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            RuntimeError: If model is not trained
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError("Model must be trained before saving")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        self._model.save_model(str(path))

        logger.info(f"LightGBM model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model = lgb.Booster(model_file=str(path))
        self._is_trained = True

        # Recover feature names from model
        self._feature_names = self._model.feature_name()

        logger.info(f"LightGBM model loaded from {path}")

    def health_check(self) -> bool:
        """
        Check if the model is healthy (not degenerate).

        Detects single-class prediction pattern which indicates
        degenerate trees that always predict the same class.

        Returns:
            True if model is healthy, False if degenerate
        """
        if self._model is None or not self._is_trained:
            logger.warning("Health check failed: model not trained")
            return False

        try:
            # Generate synthetic test data to check prediction distribution
            n_samples = 100
            n_features = self._model.num_feature()

            # Create varied synthetic data
            np.random.seed(42)
            test_data = np.random.randn(n_samples, n_features)

            # Get predictions
            predictions = self.predict(test_data)

            # Check for degenerate predictions (all same value)
            pred_std = np.std(predictions)
            pred_min = np.min(predictions)
            pred_max = np.max(predictions)

            # Degenerate model check: if all predictions are nearly identical
            if pred_std < 1e-6:
                logger.warning(
                    "Health check FAILED: degenerate model (constant predictions)",
                    extra={
                        "prediction_std": pred_std,
                        "prediction_min": pred_min,
                        "prediction_max": pred_max,
                    },
                )
                return False

            # Check for single-class prediction pattern
            # If all predictions are < 0.01 or > 0.99, model is likely degenerate
            all_low = np.all(predictions < 0.01)
            all_high = np.all(predictions > 0.99)

            if all_low or all_high:
                logger.warning(
                    "Health check FAILED: single-class prediction pattern",
                    extra={
                        "all_predictions_low": all_low,
                        "all_predictions_high": all_high,
                        "prediction_mean": float(np.mean(predictions)),
                    },
                )
                return False

            # Check for reasonable variance in predictions
            # A healthy model should have some spread in predictions
            prediction_range = pred_max - pred_min
            if prediction_range < 0.01:
                logger.warning(
                    "Health check FAILED: insufficient prediction variance",
                    extra={
                        "prediction_range": prediction_range,
                        "prediction_std": pred_std,
                    },
                )
                return False

            logger.debug(
                "Health check PASSED",
                extra={
                    "prediction_std": pred_std,
                    "prediction_range": prediction_range,
                    "prediction_mean": float(np.mean(predictions)),
                },
            )
            return True

        except Exception as e:
            logger.error(f"Health check failed with exception: {e!s}")
            return False

    def _get_params(self) -> dict[str, Any]:
        """Get LightGBM training parameters from config."""
        return {
            "objective": self.config.objective,
            "metric": self.config.metric,
            "boosting_type": self.config.boosting_type,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "lambda_l1": self.config.reg_alpha,
            "lambda_l2": self.config.reg_lambda,
            "is_unbalance": self.config.is_unbalance,
            "verbose": self.config.verbose,
            "seed": self.config.random_state,
        }

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def feature_names(self) -> list[str] | None:
        """Get feature names used in training."""
        return self._feature_names
