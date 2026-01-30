"""
AlphaStrike Trading Bot - Model Trainer with Auto-Retraining (US-016)

Provides automated training pipeline for all ML models with:
- Data collection from database (1000+ candles)
- Feature calculation using FeaturePipeline
- 3-class label generation (LONG/SHORT/HOLD)
- Balanced sampling for equal class distribution
- Time-ordered train/validation split (80/20)
- Out-of-sample validation
- Model persistence with timestamps
- Dynamic retraining intervals based on volatility
- Signal hot reload to ensemble

Features:
- Trains all 4 models (XGBoost, LightGBM, LSTM, RandomForest)
- Adaptive retraining interval (13-90 minutes based on volatility)
- Comprehensive training reports
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.core.config import get_settings
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

# Retraining interval bounds (minutes)
RETRAINING_INTERVAL_MIN = 13
RETRAINING_INTERVAL_MAX = 90

# ATR ratio thresholds for volatility classification
ATR_RATIO_HIGH = 2.0
ATR_RATIO_LOW = 1.0


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
    Trainer for all ML models in the AlphaStrike ensemble.

    Handles the complete training pipeline:
    1. Data collection from database (1000+ candles)
    2. Feature calculation using FeaturePipeline
    3. Label generation (3-class: LONG/SHORT/HOLD)
    4. Balanced sampling for equal class distribution
    5. Time-ordered train/validation split (80/20)
    6. Train all 4 models
    7. Out-of-sample validation
    8. Save models with timestamps
    9. Signal hot reload to ensemble

    Usage:
        trainer = ModelTrainer(database=db)
        report = await trainer.train_all_models()
        if report.success:
            print(f"Trained {len(report.models_trained)} models")
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

        # Training state
        self._last_training_time: datetime | None = None
        self._current_retraining_interval: int = 60  # Default 60 minutes

        logger.info(
            "ModelTrainer initialized",
            extra={"models_dir": str(self.models_dir)},
        )

    async def train_all_models(
        self,
        symbol: str = "cmt_btcusdt",
        min_candles: int = MIN_TRAINING_SAMPLES,
    ) -> TrainingReport:
        """
        Train all models in the ensemble.

        Args:
            symbol: Trading symbol to fetch candles for
            min_candles: Minimum number of candles required

        Returns:
            TrainingReport with results and metrics
        """
        start_time = time.time()
        timestamp = datetime.utcnow()

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

            # Update training state
            self._last_training_time = timestamp

            training_time = time.time() - start_time

            logger.info(
                "Training completed successfully",
                extra={
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

    def _calculate_retraining_interval(self, volatility: float) -> int:
        """
        Calculate retraining interval based on market volatility.

        Uses ATR ratio to determine volatility level:
        - High volatility (ATR ratio > 2.0): 13-30 minutes
        - Normal volatility: 45-60 minutes
        - Low volatility (ATR ratio < 1.0): 60-90 minutes

        Args:
            volatility: ATR ratio (current ATR / historical average ATR)

        Returns:
            Retraining interval in minutes (13-90)
        """
        if volatility > ATR_RATIO_HIGH:
            # High volatility: shorter intervals (13-30 minutes)
            # Scale linearly from 30 at ATR=2.0 to 13 at ATR=4.0+
            interval = max(
                RETRAINING_INTERVAL_MIN,
                int(30 - (volatility - ATR_RATIO_HIGH) * (30 - RETRAINING_INTERVAL_MIN) / 2.0),
            )
        elif volatility < ATR_RATIO_LOW:
            # Low volatility: longer intervals (60-90 minutes)
            # Scale linearly from 60 at ATR=1.0 to 90 at ATR=0.5 or lower
            interval = min(
                RETRAINING_INTERVAL_MAX,
                int(60 + (ATR_RATIO_LOW - volatility) * (RETRAINING_INTERVAL_MAX - 60) / 0.5),
            )
        else:
            # Normal volatility: 45-60 minutes
            # Linear interpolation between low and high thresholds
            normalized = (volatility - ATR_RATIO_LOW) / (ATR_RATIO_HIGH - ATR_RATIO_LOW)
            interval = int(60 - normalized * 15)  # 60 at low end, 45 at high end

        self._current_retraining_interval = interval
        return interval

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

    def should_retrain(self, volatility: float) -> bool:
        """
        Check if models should be retrained based on interval.

        Args:
            volatility: Current ATR ratio

        Returns:
            True if retraining is due
        """
        if self._last_training_time is None:
            return True

        interval_minutes = self._calculate_retraining_interval(volatility)
        elapsed_minutes = (datetime.utcnow() - self._last_training_time).total_seconds() / 60

        return elapsed_minutes >= interval_minutes

    @property
    def last_training_time(self) -> datetime | None:
        """Get last training timestamp."""
        return self._last_training_time

    @property
    def current_retraining_interval(self) -> int:
        """Get current retraining interval in minutes."""
        return self._current_retraining_interval

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
