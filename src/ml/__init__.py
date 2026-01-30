"""
AlphaStrike Trading Bot - Machine Learning Module

Contains ML models for trading signal prediction.
"""

from src.ml.confidence_filter import ConfidenceFilter, ScoreBreakdown
from src.ml.ensemble import MLEnsemble
from src.ml.lightgbm_model import LightGBMConfig, LightGBMModel
from src.ml.lightgbm_model import TrainingResult as LightGBMTrainingResult
from src.ml.lstm_model import LSTMConfig, LSTMModel, LSTMNetwork
from src.ml.lstm_model import TrainingResult as LSTMTrainingResult
from src.ml.random_forest_model import RandomForestConfig, RandomForestModel
from src.ml.random_forest_model import TrainingResult as RFTrainingResult
from src.ml.trainer import ModelTrainer, TrainingReport
from src.ml.xgboost_model import XGBoostConfig, XGBoostModel
from src.ml.xgboost_model import TrainingResult as XGBoostTrainingResult

__all__ = [
    # Confidence Filter
    "ConfidenceFilter",
    "ScoreBreakdown",
    # Ensemble
    "MLEnsemble",
    # LightGBM
    "LightGBMConfig",
    "LightGBMModel",
    "LightGBMTrainingResult",
    # LSTM
    "LSTMConfig",
    "LSTMModel",
    "LSTMNetwork",
    "LSTMTrainingResult",
    # Random Forest
    "RandomForestConfig",
    "RandomForestModel",
    "RFTrainingResult",
    # XGBoost
    "XGBoostConfig",
    "XGBoostModel",
    "XGBoostTrainingResult",
    # Trainer
    "ModelTrainer",
    "TrainingReport",
]
