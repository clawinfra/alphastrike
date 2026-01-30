"""
AlphaStrike Trading Bot - Machine Learning Module

Contains ML models for trading signal prediction.
"""

from src.ml.lstm_model import LSTMConfig, LSTMModel, LSTMNetwork
from src.ml.lstm_model import TrainingResult as LSTMTrainingResult
from src.ml.random_forest_model import RandomForestConfig, RandomForestModel
from src.ml.random_forest_model import TrainingResult as RFTrainingResult
from src.ml.xgboost_model import XGBoostConfig, XGBoostModel
from src.ml.xgboost_model import TrainingResult as XGBoostTrainingResult

__all__ = [
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
]
