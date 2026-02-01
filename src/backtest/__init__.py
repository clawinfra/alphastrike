"""
AlphaStrike Backtesting System

Historical backtesting infrastructure for strategy evaluation.
"""

from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade
from src.backtest.metrics import BacktestMetrics, MetricsCalculator
from src.backtest.report import ReportGenerator
from src.backtest.simulator import ExecutionSimulator, SimulatedFill

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "BacktestTrade",
    "DataLoader",
    "ExecutionSimulator",
    "MetricsCalculator",
    "ReportGenerator",
    "SimulatedFill",
]
