"""
AlphaStrike Trading Bot - Random Forest Model Implementation

Provides a Random Forest classifier wrapper for ensemble predictions
with health monitoring and prediction diversity detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""

    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"  # Important for imbalanced data
    n_jobs: int = -1
    random_state: int | None = 42


@dataclass
class TrainingResult:
    """Result from model training."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importances: np.ndarray
    n_samples: int
    n_features: int
    training_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RandomForestModel:
    """
    Random Forest model wrapper for trading signal prediction.

    Uses sklearn's RandomForestClassifier with health monitoring
    for prediction diversity and forest agreement detection.

    Example:
        config = RandomForestConfig(n_estimators=200, max_depth=15)
        model = RandomForestModel(config)
        result = model.train(X_train, y_train)
        probabilities = model.predict(X_test)  # Returns probabilities [0, 1]
    """

    def __init__(self, config: RandomForestConfig) -> None:
        """
        Initialize Random Forest model.

        Args:
            config: Model configuration parameters
        """
        self.config = config
        self._model: RandomForestClassifier | None = None
        self._is_trained: bool = False
        self._last_predictions: np.ndarray | None = None
        self._n_features: int | None = None

        logger.info(
            "RandomForestModel initialized",
            extra={
                "n_estimators": config.n_estimators,
                "max_depth": config.max_depth,
                "class_weight": config.class_weight,
            },
        )

    def _create_model(self) -> RandomForestClassifier:
        """Create a new RandomForestClassifier with configured parameters."""
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> TrainingResult:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,) with binary values (0 or 1)

        Returns:
            TrainingResult with metrics and feature importances

        Raises:
            ValueError: If X and y have incompatible shapes
        """
        import time

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )

        start_time = time.time()

        # Create and train model
        self._model = self._create_model()
        self._model.fit(X, y)
        self._is_trained = True
        self._n_features = X.shape[1]

        training_time = time.time() - start_time

        # Calculate training metrics
        y_pred = self._model.predict(X)

        # Handle potential zero division in metrics
        # Note: sklearn accepts numeric zero_division but stubs are typed as str only
        accuracy = float(accuracy_score(y, y_pred))
        precision = float(precision_score(y, y_pred, zero_division=0.0))  # type: ignore[arg-type]
        recall = float(recall_score(y, y_pred, zero_division=0.0))  # type: ignore[arg-type]
        f1 = float(f1_score(y, y_pred, zero_division=0.0))  # type: ignore[arg-type]

        result = TrainingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            feature_importances=self._model.feature_importances_.copy(),
            n_samples=X.shape[0],
            n_features=X.shape[1],
            training_time_seconds=training_time,
            metadata={
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
                "class_distribution": {
                    "class_0": int(np.sum(y == 0)),
                    "class_1": int(np.sum(y == 1)),
                },
            },
        )

        logger.info(
            "RandomForest training completed",
            extra={
                "accuracy": f"{accuracy:.4f}",
                "f1": f"{f1:.4f}",
                "n_samples": X.shape[0],
                "training_time": f"{training_time:.2f}s",
            },
        )

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Array of probabilities in range [0, 1] for positive class

        Raises:
            RuntimeError: If model is not trained
            ValueError: If feature count doesn't match training data
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X.shape[1]}"
            )

        # Get probability of positive class (class 1)
        proba_result = self._model.predict_proba(X)
        probabilities: np.ndarray = proba_result[:, 1]  # type: ignore[index]

        # Store for health check
        self._last_predictions = probabilities.copy()

        return probabilities

    def save(self, path: Path) -> None:
        """
        Save model to disk using joblib.

        Args:
            path: File path to save model

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Cannot save untrained model")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        save_data = {
            "model": self._model,
            "config": self.config,
            "n_features": self._n_features,
        }

        joblib.dump(save_data, path)

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If file format is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        save_data = joblib.load(path)

        if not isinstance(save_data, dict) or "model" not in save_data:
            raise ValueError(f"Invalid model file format: {path}")

        self._model = save_data["model"]
        self._n_features = save_data.get("n_features")

        # Update config if present
        if "config" in save_data:
            self.config = save_data["config"]

        self._is_trained = True

        logger.info(f"Model loaded from {path}")

    def health_check(self) -> bool:
        """
        Check model health based on prediction diversity.

        Health checks:
        1. Model is trained
        2. Prediction diversity: outputs should have variation (not all same value)
        3. Forest diversity: individual trees should have different predictions

        Returns:
            True if model is healthy, False otherwise
        """
        if not self._is_trained or self._model is None:
            logger.warning("Health check failed: model not trained")
            return False

        # Check 1: Verify we have predictions to analyze
        if self._last_predictions is None or len(self._last_predictions) < 2:
            logger.warning("Health check: insufficient predictions for analysis")
            return True  # Not enough data to determine unhealthy

        # Check 2: Prediction diversity - outputs should vary
        prediction_std = float(np.std(self._last_predictions))
        if prediction_std < 0.001:
            logger.warning(
                f"Health check failed: low prediction diversity (std={prediction_std:.6f})"
            )
            return False

        # Check 3: Forest diversity - trees should disagree sometimes
        # Sample a few predictions and check if all trees agree
        if hasattr(self._model, "estimators_") and len(self._model.estimators_) > 1:
            # Use a subset of predictions for efficiency
            sample_size = min(10, len(self._last_predictions))

            # Get predictions from individual trees
            tree_predictions: list[np.ndarray] = []
            for tree in self._model.estimators_[:10]:  # Check first 10 trees
                tree_preds = tree.predict_proba(
                    np.zeros((sample_size, self._n_features or 1))
                )[:, 1] if self._n_features else np.zeros(sample_size)
                tree_predictions.append(tree_preds)

            if tree_predictions:
                tree_pred_array = np.array(tree_predictions)
                # Check variance across trees
                tree_variance = float(np.mean(np.var(tree_pred_array, axis=0)))
                if tree_variance < 0.0001:
                    logger.warning(
                        f"Health check failed: low forest diversity (var={tree_variance:.6f})"
                    )
                    return False

        logger.debug("Health check passed")
        return True

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def feature_importances(self) -> np.ndarray | None:
        """Get feature importances if model is trained."""
        if self._model is None:
            return None
        return self._model.feature_importances_.copy()

    def get_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all individual trees.

        Useful for analyzing forest diversity and prediction uncertainty.

        Args:
            X: Feature matrix

        Returns:
            Array of shape (n_estimators, n_samples) with individual tree predictions

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        tree_predictions = []
        for tree in self._model.estimators_:
            tree_probs = tree.predict_proba(X)[:, 1]
            tree_predictions.append(tree_probs)

        return np.array(tree_predictions)

    def get_prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty based on tree disagreement.

        Args:
            X: Feature matrix

        Returns:
            Array of uncertainty values (standard deviation across trees)

        Raises:
            RuntimeError: If model is not trained
        """
        tree_predictions = self.get_tree_predictions(X)
        uncertainty = np.std(tree_predictions, axis=0)
        return uncertainty
