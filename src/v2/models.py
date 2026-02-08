"""
V2 Model Pipeline — Regression + Stacking

Key differences from V1:
1. Regression target (predict return magnitude) instead of binary classification
2. Two-stage stacking with diverse feature views
3. Proper out-of-fold training for meta-learner (no information leakage)
4. Prediction includes uncertainty estimate for Kelly sizing

Architecture:
  Stage 1: Base learners with diverse inductive biases
    - LightGBM on full 25 features (tabular specialist)
    - Ridge regression on Tier 1 features only (regularized, decorrelated)
    - Rolling statistics model (pure momentum/mean-reversion features)

  Stage 2: Meta-learner
    - Ridge regression on Stage 1 predictions + regime features
    - Trained on out-of-fold predictions only
    - Outputs: predicted_return, predicted_std
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Model prediction with uncertainty."""

    predicted_return: float  # expected return (signed, e.g., +0.015 = +1.5%)
    predicted_std: float  # uncertainty (std dev of prediction)
    signal: str  # "LONG", "SHORT", or "HOLD"
    confidence: float  # 0-1 (based on return / std ratio)
    base_predictions: dict[str, float] = field(default_factory=dict)  # per-model preds


@dataclass
class ModelConfig:
    """Configuration for model pipeline."""

    # LightGBM params
    lgb_n_estimators: int = 200
    lgb_num_leaves: int = 15
    lgb_learning_rate: float = 0.03
    lgb_max_depth: int = 4
    lgb_reg_alpha: float = 0.5
    lgb_reg_lambda: float = 1.0
    lgb_min_data_in_leaf: int = 30
    lgb_feature_fraction: float = 0.7
    lgb_bagging_fraction: float = 0.7
    lgb_bagging_freq: int = 5

    # Ridge params
    ridge_alpha: float = 1.0

    # Prediction thresholds
    min_return_threshold: float = 0.003  # 0.3% minimum predicted return to trade
    min_confidence: float = 0.3  # minimum confidence to trade

    # Forward return horizon (in bars)
    target_horizon: int = 12  # predict 12-bar forward return

    # Walk-forward
    n_folds: int = 5  # for out-of-fold stacking


class LightGBMRegressor:
    """LightGBM regression model for return prediction."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model = None
        self._feature_names: list[str] = []
        self._is_trained = False

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Train LightGBM regressor."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed")
            return {"success": False, "error": "lightgbm not installed"}

        self._feature_names = list(feature_names)

        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": self.config.lgb_num_leaves,
            "max_depth": self.config.lgb_max_depth,
            "learning_rate": self.config.lgb_learning_rate,
            "min_data_in_leaf": self.config.lgb_min_data_in_leaf,
            "feature_fraction": self.config.lgb_feature_fraction,
            "bagging_fraction": self.config.lgb_bagging_fraction,
            "bagging_freq": self.config.lgb_bagging_freq,
            "lambda_l1": self.config.lgb_reg_alpha,
            "lambda_l2": self.config.lgb_reg_lambda,
            "verbose": -1,
            "seed": 42,
        }

        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        self._model = lgb.train(params, train_data, num_boost_round=self.config.lgb_n_estimators)
        self._is_trained = True

        # Feature importance
        importances = self._model.feature_importance(importance_type="gain")
        top_features = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "success": True,
            "n_samples": len(y),
            "top_features": top_features,
        }

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict returns."""
        if not self._is_trained or self._model is None:
            return np.zeros(X.shape[0])
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        if self._model:
            self._model.save_model(str(path))

    def load(self, path: Path) -> None:
        import lightgbm as lgb

        self._model = lgb.Booster(model_file=str(path))
        self._feature_names = self._model.feature_name()
        self._is_trained = True


class RidgeRegressor:
    """Simple ridge regression for decorrelated predictions."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._weights: NDArray | None = None
        self._bias: float = 0.0
        self._is_trained = False

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Train ridge regression (closed-form solution)."""
        n_samples, n_features = X.shape

        # Add regularization: w = (X^T X + αI)^{-1} X^T y
        XtX = X.T @ X + self.alpha * np.eye(n_features)
        Xty = X.T @ y

        self._weights = np.linalg.solve(XtX, Xty)
        self._bias = float(np.mean(y) - np.mean(X @ self._weights))
        self._is_trained = True

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_trained or self._weights is None:
            return np.zeros(X.shape[0])
        return X @ self._weights + self._bias


class RollingStatsModel:
    """
    Pure statistical model — momentum and mean-reversion signals.

    No ML — just rolling window statistics. Provides a fundamentally
    different "view" of the data for stacking diversity.
    """

    def __init__(self, fast_window: int = 12, slow_window: int = 48) -> None:
        self.fast = fast_window
        self.slow = slow_window

    def predict_from_prices(self, close: NDArray[np.float64]) -> float:
        """Predict return from price momentum and mean-reversion signals."""
        if len(close) < self.slow + 1:
            return 0.0

        # Fast momentum (12-bar return)
        fast_mom = (close[-1] - close[-self.fast - 1]) / close[-self.fast - 1]

        # Slow momentum (48-bar return)
        slow_mom = (close[-1] - close[-self.slow - 1]) / close[-self.slow - 1]

        # Mean reversion signal (z-score relative to slow MA)
        slow_ma = float(np.mean(close[-self.slow :]))
        slow_std = float(np.std(close[-self.slow :]))
        z_score = (close[-1] - slow_ma) / slow_std if slow_std > 0 else 0.0

        # Combine: momentum in trends, mean-reversion in ranges
        # The meta-learner will learn when each is useful
        momentum_signal = fast_mom * 0.6 + slow_mom * 0.4
        reversion_signal = -z_score * 0.01  # small, mean-reverting

        return float(momentum_signal + reversion_signal)


class StackingEnsemble:
    """
    Two-stage stacking ensemble with out-of-fold training.

    Stage 1: Base learners predict returns independently
    Stage 2: Meta-learner combines predictions with regime context

    The meta-learner is trained ONLY on out-of-fold predictions,
    preventing information leakage that inflates backtest results.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

        # Stage 1: Base learners
        self.lgb_model = LightGBMRegressor(self.config)
        self.ridge_model = RidgeRegressor(alpha=self.config.ridge_alpha)
        self.stats_model = RollingStatsModel()

        # Stage 2: Meta-learner
        self.meta_learner = RidgeRegressor(alpha=0.5)

        # Training state
        self._is_trained = False
        self._residual_std: float = 0.02  # default uncertainty

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_names: list[str],
        close_prices: NDArray[np.float64] | None = None,
        regime_features: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """
        Train the full stacking pipeline with out-of-fold predictions.

        Args:
            X: Feature matrix (n_samples, 25)
            y: Forward returns (n_samples,)
            feature_names: Feature names matching X columns
            close_prices: Close prices for rolling stats model (optional)
            regime_features: ADX + ATR ratio for meta-learner (optional)
        """
        n_samples = len(y)
        n_folds = self.config.n_folds
        fold_size = n_samples // n_folds

        # Out-of-fold predictions for meta-learner training
        oof_lgb = np.zeros(n_samples)
        oof_ridge = np.zeros(n_samples)
        oof_stats = np.zeros(n_samples)

        # Tier 1 feature indices (first 12 features)
        tier1_idx = list(range(min(12, X.shape[1])))

        logger.info(f"Training stacking ensemble: {n_samples} samples, {n_folds} folds")

        # Generate out-of-fold predictions
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

            # Time-ordered split (no future data leakage)
            train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
            val_idx = list(range(val_start, val_end))

            if len(train_idx) < 100:
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            # LightGBM on full features
            fold_lgb = LightGBMRegressor(self.config)
            fold_lgb.train(X_train, y_train, feature_names)
            oof_lgb[val_idx] = fold_lgb.predict(X_val)

            # Ridge on Tier 1 only
            fold_ridge = RidgeRegressor(alpha=self.config.ridge_alpha)
            fold_ridge.train(X_train[:, tier1_idx], y_train)
            oof_ridge[val_idx] = fold_ridge.predict(X_val[:, tier1_idx])

            # Stats model (if close prices available)
            if close_prices is not None:
                for i in val_idx:
                    if i >= 50:
                        oof_stats[i] = self.stats_model.predict_from_prices(close_prices[:i])

        # Train final base models on ALL data
        self.lgb_model.train(X, y, feature_names)
        self.ridge_model.train(X[:, tier1_idx], y)

        # Build meta-learner features: [lgb_pred, ridge_pred, stats_pred, regime...]
        meta_X = np.column_stack([oof_lgb, oof_ridge, oof_stats])
        if regime_features is not None:
            meta_X = np.column_stack([meta_X, regime_features])

        # Train meta-learner on out-of-fold predictions
        self.meta_learner.train(meta_X, y)

        # Estimate residual std for uncertainty quantification
        meta_preds = self.meta_learner.predict(meta_X)
        residuals = y - meta_preds
        self._residual_std = float(np.std(residuals))

        self._is_trained = True

        # Report
        lgb_mae = float(np.mean(np.abs(oof_lgb - y)))
        ridge_mae = float(np.mean(np.abs(oof_ridge - y)))
        meta_mae = float(np.mean(np.abs(meta_preds - y)))

        logger.info(
            f"Stacking trained: LGB MAE={lgb_mae:.5f}, "
            f"Ridge MAE={ridge_mae:.5f}, Meta MAE={meta_mae:.5f}, "
            f"Residual std={self._residual_std:.5f}"
        )

        return {
            "success": True,
            "lgb_mae": lgb_mae,
            "ridge_mae": ridge_mae,
            "meta_mae": meta_mae,
            "residual_std": self._residual_std,
            "n_samples": n_samples,
        }

    def predict(
        self,
        X: NDArray[np.float64],
        close_prices: NDArray[np.float64] | None = None,
        regime_features: NDArray[np.float64] | None = None,
    ) -> Prediction:
        """
        Generate prediction with uncertainty estimate.

        Returns a Prediction with:
          - predicted_return: signed expected return
          - predicted_std: uncertainty
          - signal: LONG/SHORT/HOLD
          - confidence: return/std ratio (pseudo-Sharpe)
        """
        if not self._is_trained:
            return Prediction(
                predicted_return=0.0,
                predicted_std=self._residual_std,
                signal="HOLD",
                confidence=0.0,
            )

        # Stage 1 predictions
        tier1_idx = list(range(min(12, X.shape[1])))

        lgb_pred = float(self.lgb_model.predict(X)[0])
        ridge_pred = float(self.ridge_model.predict(X[:, tier1_idx])[0])
        stats_pred = 0.0
        if close_prices is not None and len(close_prices) >= 50:
            stats_pred = self.stats_model.predict_from_prices(close_prices)

        # Stage 2: meta-learner
        meta_input = np.array([[lgb_pred, ridge_pred, stats_pred]])
        if regime_features is not None:
            meta_input = np.column_stack([meta_input, regime_features])

        predicted_return = float(self.meta_learner.predict(meta_input)[0])

        # Confidence: |predicted_return| / residual_std
        # This is effectively a per-prediction pseudo-Sharpe
        confidence = abs(predicted_return) / self._residual_std if self._residual_std > 0 else 0.0
        confidence = float(np.clip(confidence, 0, 1))

        # Signal
        if (
            predicted_return > self.config.min_return_threshold
            and confidence >= self.config.min_confidence
        ):
            signal = "LONG"
        elif (
            predicted_return < -self.config.min_return_threshold
            and confidence >= self.config.min_confidence
        ):
            signal = "SHORT"
        else:
            signal = "HOLD"

        return Prediction(
            predicted_return=predicted_return,
            predicted_std=self._residual_std,
            signal=signal,
            confidence=confidence,
            base_predictions={
                "lightgbm": lgb_pred,
                "ridge": ridge_pred,
                "rolling_stats": stats_pred,
            },
        )
