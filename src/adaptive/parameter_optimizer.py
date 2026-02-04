"""
Bayesian Parameter Optimizer using Optuna

Automatically finds optimal trading parameters for each symbol using
sample-efficient Bayesian optimization. Runs on retune triggers from
PerformanceTracker.

Key Design Decisions:
1. Bayesian (not Grid/GA) - 30-50 iterations vs 1000+ for grid search
2. Trigger-based (not calendar) - Markets don't follow schedules
3. Walk-forward validation - Prevents overfitting to historical data
4. Symbol-specific - Each asset optimized independently
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import numpy as np

from src.adaptive.asset_config import AdaptiveAssetConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    symbol: str
    success: bool
    message: str

    # Best parameters found
    best_params: dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0  # Objective value (e.g., Sharpe ratio)

    # Validation metrics
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    in_sample_return: float = 0.0
    out_of_sample_return: float = 0.0

    # Optimization stats
    n_trials: int = 0
    optimization_time_seconds: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trigger_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "success": self.success,
            "message": self.message,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "in_sample_sharpe": self.in_sample_sharpe,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "in_sample_return": self.in_sample_return,
            "out_of_sample_return": self.out_of_sample_return,
            "n_trials": self.n_trials,
            "optimization_time_seconds": self.optimization_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "trigger_reason": self.trigger_reason,
        }


@dataclass
class ParameterSpace:
    """Defines the search space for optimization."""

    # Conviction threshold: Higher = fewer but higher quality signals
    conviction_threshold_min: float = 55.0
    conviction_threshold_max: float = 85.0

    # Stop loss: ATR multiplier for stop distance
    stop_atr_multiplier_min: float = 1.5
    stop_atr_multiplier_max: float = 3.5

    # Take profit: ATR multiplier for TP distance
    take_profit_atr_multiplier_min: float = 2.0
    take_profit_atr_multiplier_max: float = 5.0

    # Position sizing: Multiplier on base position size
    position_size_multiplier_min: float = 0.3
    position_size_multiplier_max: float = 1.5

    # Short conviction penalty: Extra points needed for shorts
    short_conviction_penalty_min: int = 0
    short_conviction_penalty_max: int = 15

    # Binary options
    allow_short_toggle: bool = True  # Whether to optimize short_enabled
    allow_daily_trend_toggle: bool = True  # Whether to optimize require_daily_trend


class ParameterOptimizer:
    """
    Bayesian optimizer for trading parameters using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) for sample-efficient search.
    Integrates with walk-forward validation to prevent overfitting.
    """

    def __init__(
        self,
        backtest_func: Callable[[str, dict], dict],
        n_trials: int = 50,
        timeout_seconds: int | None = 300,
        min_sharpe_threshold: float = 0.5,
        min_out_of_sample_ratio: float = 0.7,
        results_dir: Path | None = None,
    ):
        """
        Initialize optimizer.

        Args:
            backtest_func: Function(symbol, params) -> {"sharpe": float, "return": float, ...}
            n_trials: Number of optimization trials (30-50 typically sufficient)
            timeout_seconds: Max optimization time per symbol
            min_sharpe_threshold: Minimum acceptable Sharpe ratio
            min_out_of_sample_ratio: OOS Sharpe must be >= this ratio of IS Sharpe
            results_dir: Directory to save optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna not installed. Run: pip install optuna")

        self.backtest_func = backtest_func
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.min_sharpe_threshold = min_sharpe_threshold
        self.min_out_of_sample_ratio = min_out_of_sample_ratio
        self.results_dir = results_dir or Path("data/optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track optimization history per symbol
        self._history: dict[str, list[OptimizationResult]] = {}

        # Default parameter space
        self.param_space = ParameterSpace()

    def optimize(
        self,
        symbol: str,
        trigger_reason: str = "manual",
        current_config: AdaptiveAssetConfig | None = None,
        warm_start: bool = True,
    ) -> OptimizationResult:
        """
        Run Bayesian optimization to find best parameters for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            trigger_reason: Why optimization was triggered
            current_config: Current config to warm-start from
            warm_start: Whether to seed with current config

        Returns:
            OptimizationResult with best parameters and validation metrics
        """
        import time
        start_time = time.time()

        logger.info(f"Starting optimization for {symbol} (reason: {trigger_reason})")

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",  # Maximize Sharpe ratio
            sampler=TPESampler(seed=42, n_startup_trials=10),
            study_name=f"{symbol}_optimization",
        )

        # Warm start with current config if available
        if warm_start and current_config:
            self._enqueue_current_config(study, current_config)

        # Run optimization
        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)
            try:
                result = self.backtest_func(symbol, params)
                sharpe = result.get("sharpe", 0.0)

                # Penalize if too few trades
                n_trades = result.get("n_trades", 0)
                if n_trades < 10:
                    sharpe *= 0.5  # Heavy penalty for too few trades

                return sharpe
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -10.0  # Return bad value for failed trials

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout_seconds,
                show_progress_bar=False,
            )
        except Exception as e:
            return OptimizationResult(
                symbol=symbol,
                success=False,
                message=f"Optimization failed: {e}",
                trigger_reason=trigger_reason,
            )

        # Get best parameters
        best_params = study.best_params
        best_sharpe = study.best_value

        # Run walk-forward validation on best params
        validation = self._validate_params(symbol, best_params)

        elapsed = time.time() - start_time

        # Check if results pass validation
        success = self._check_validation(validation)

        result = OptimizationResult(
            symbol=symbol,
            success=success,
            message="Optimization successful" if success else "Failed validation checks",
            best_params=best_params,
            best_value=best_sharpe,
            in_sample_sharpe=validation.get("in_sample_sharpe", 0.0),
            out_of_sample_sharpe=validation.get("out_of_sample_sharpe", 0.0),
            in_sample_return=validation.get("in_sample_return", 0.0),
            out_of_sample_return=validation.get("out_of_sample_return", 0.0),
            n_trials=len(study.trials),
            optimization_time_seconds=elapsed,
            trigger_reason=trigger_reason,
        )

        # Save result
        self._save_result(result)

        # Track history
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append(result)

        logger.info(
            f"Optimization complete for {symbol}: "
            f"Sharpe={best_sharpe:.2f}, OOS={validation.get('out_of_sample_sharpe', 0):.2f}, "
            f"success={success}"
        )

        return result

    def _sample_params(self, trial: optuna.Trial) -> dict:
        """Sample parameters from search space."""
        ps = self.param_space

        params = {
            "conviction_threshold": trial.suggest_float(
                "conviction_threshold",
                ps.conviction_threshold_min,
                ps.conviction_threshold_max,
            ),
            "stop_atr_multiplier": trial.suggest_float(
                "stop_atr_multiplier",
                ps.stop_atr_multiplier_min,
                ps.stop_atr_multiplier_max,
            ),
            "take_profit_atr_multiplier": trial.suggest_float(
                "take_profit_atr_multiplier",
                ps.take_profit_atr_multiplier_min,
                ps.take_profit_atr_multiplier_max,
            ),
            "position_size_multiplier": trial.suggest_float(
                "position_size_multiplier",
                ps.position_size_multiplier_min,
                ps.position_size_multiplier_max,
            ),
            "short_conviction_penalty": trial.suggest_int(
                "short_conviction_penalty",
                ps.short_conviction_penalty_min,
                ps.short_conviction_penalty_max,
            ),
        }

        # Binary options
        if ps.allow_short_toggle:
            params["short_enabled"] = trial.suggest_categorical(
                "short_enabled", [True, False]
            )

        if ps.allow_daily_trend_toggle:
            params["require_daily_trend_for_short"] = trial.suggest_categorical(
                "require_daily_trend_for_short", [True, False]
            )

        return params

    def _enqueue_current_config(
        self, study: optuna.Study, config: AdaptiveAssetConfig
    ) -> None:
        """Add current config as first trial (warm start)."""
        study.enqueue_trial({
            "conviction_threshold": config.conviction_threshold,
            "stop_atr_multiplier": config.stop_atr_multiplier,
            "take_profit_atr_multiplier": config.take_profit_atr_multiplier,
            "position_size_multiplier": config.position_size_multiplier,
            "short_conviction_penalty": config.short_conviction_penalty,
            "short_enabled": config.short_enabled,
            "require_daily_trend_for_short": config.require_daily_trend_for_short,
        })

    def _validate_params(self, symbol: str, params: dict) -> dict:
        """
        Run walk-forward validation on optimized parameters.

        Splits data into in-sample (optimization) and out-of-sample (validation).
        """
        # Run backtest with walk-forward validation flag
        params_with_validation = {**params, "_walk_forward": True}

        try:
            result = self.backtest_func(symbol, params_with_validation)
            return {
                "in_sample_sharpe": result.get("in_sample_sharpe", result.get("sharpe", 0)),
                "out_of_sample_sharpe": result.get("out_of_sample_sharpe", 0),
                "in_sample_return": result.get("in_sample_return", result.get("return", 0)),
                "out_of_sample_return": result.get("out_of_sample_return", 0),
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}

    def _check_validation(self, validation: dict) -> bool:
        """Check if optimization result passes validation criteria."""
        if not validation:
            return False

        oos_sharpe = validation.get("out_of_sample_sharpe", 0)
        is_sharpe = validation.get("in_sample_sharpe", 0)

        # Must have positive OOS Sharpe
        if oos_sharpe < self.min_sharpe_threshold:
            logger.warning(f"OOS Sharpe {oos_sharpe:.2f} below threshold {self.min_sharpe_threshold}")
            return False

        # OOS must not be much worse than IS (overfitting check)
        if is_sharpe > 0 and oos_sharpe / is_sharpe < self.min_out_of_sample_ratio:
            logger.warning(
                f"OOS/IS ratio {oos_sharpe/is_sharpe:.2f} below threshold "
                f"{self.min_out_of_sample_ratio}"
            )
            return False

        return True

    def _save_result(self, result: OptimizationResult) -> None:
        """Save optimization result to disk."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.symbol}_{timestamp}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved optimization result to {filepath}")

    def apply_result(
        self,
        result: OptimizationResult,
        config: AdaptiveAssetConfig,
        require_validation: bool = True,
    ) -> AdaptiveAssetConfig:
        """
        Apply optimization result to asset config.

        Args:
            result: Optimization result with best params
            config: Config to update
            require_validation: Only apply if validation passed

        Returns:
            Updated config (same object, modified in place)
        """
        if require_validation and not result.success:
            logger.warning(f"Not applying failed optimization to {config.symbol}")
            return config

        # Update config with optimized parameters
        params = result.best_params

        if "conviction_threshold" in params:
            config.conviction_threshold = params["conviction_threshold"]
        if "stop_atr_multiplier" in params:
            config.stop_atr_multiplier = params["stop_atr_multiplier"]
        if "take_profit_atr_multiplier" in params:
            config.take_profit_atr_multiplier = params["take_profit_atr_multiplier"]
        if "position_size_multiplier" in params:
            config.position_size_multiplier = params["position_size_multiplier"]
        if "short_conviction_penalty" in params:
            config.short_conviction_penalty = params["short_conviction_penalty"]
        if "short_enabled" in params:
            config.short_enabled = params["short_enabled"]
        if "require_daily_trend_for_short" in params:
            config.require_daily_trend_for_short = params["require_daily_trend_for_short"]

        # Update metadata
        config.mark_tuned()
        config.notes = f"Auto-tuned: {result.trigger_reason} (Sharpe={result.best_value:.2f})"

        logger.info(f"Applied optimization to {config.symbol}: {params}")

        return config

    def get_history(self, symbol: str) -> list[OptimizationResult]:
        """Get optimization history for a symbol."""
        return self._history.get(symbol, [])


class QuickOptimizer:
    """
    Lightweight optimizer for specific parameter adjustments.

    Use when you only need to tune one or two parameters (e.g., stop distance
    after consecutive losses), not a full optimization sweep.
    """

    def __init__(
        self,
        backtest_func: Callable[[str, dict], dict],
    ):
        self.backtest_func = backtest_func

    def optimize_stops(
        self,
        symbol: str,
        current_stop_atr: float,
        search_range: tuple[float, float] = (0.8, 1.5),
        n_steps: int = 7,
    ) -> dict:
        """
        Quick optimization of stop distance only.

        Useful after consecutive stop-outs to find better stop placement.

        Args:
            symbol: Trading pair
            current_stop_atr: Current stop ATR multiplier
            search_range: Multiplier range around current (e.g., 0.8-1.5x)
            n_steps: Number of steps to test

        Returns:
            {"best_stop_atr": float, "best_sharpe": float, "tested": list}
        """
        results = []

        min_mult = current_stop_atr * search_range[0]
        max_mult = current_stop_atr * search_range[1]

        for stop_atr in np.linspace(min_mult, max_mult, n_steps):
            try:
                result = self.backtest_func(symbol, {"stop_atr_multiplier": stop_atr})
                results.append({
                    "stop_atr": stop_atr,
                    "sharpe": result.get("sharpe", 0),
                    "win_rate": result.get("win_rate", 0),
                })
            except Exception:
                continue

        if not results:
            return {"best_stop_atr": current_stop_atr, "best_sharpe": 0, "tested": []}

        best = max(results, key=lambda x: x["sharpe"])

        return {
            "best_stop_atr": best["stop_atr"],
            "best_sharpe": best["sharpe"],
            "tested": results,
        }

    def optimize_position_size(
        self,
        symbol: str,
        current_size_mult: float,
        search_range: tuple[float, float] = (0.5, 1.5),
        n_steps: int = 5,
    ) -> dict:
        """
        Quick optimization of position size multiplier.

        Useful after drawdown breach to find better sizing.
        """
        results = []

        min_mult = current_size_mult * search_range[0]
        max_mult = current_size_mult * search_range[1]

        for size_mult in np.linspace(min_mult, max_mult, n_steps):
            try:
                result = self.backtest_func(symbol, {"position_size_multiplier": size_mult})
                results.append({
                    "size_mult": size_mult,
                    "sharpe": result.get("sharpe", 0),
                    "max_dd": result.get("max_drawdown", 0),
                })
            except Exception:
                continue

        if not results:
            return {"best_size_mult": current_size_mult, "best_sharpe": 0, "tested": []}

        # Optimize for Sharpe but penalize high drawdown
        def score(r):
            dd_penalty = max(0, r["max_dd"] - 0.05) * 10  # Penalize DD > 5%
            return r["sharpe"] - dd_penalty

        best = max(results, key=score)

        return {
            "best_size_mult": best["size_mult"],
            "best_sharpe": best["sharpe"],
            "tested": results,
        }
