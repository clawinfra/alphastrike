"""
Walk-Forward Validation Framework

The ONLY honest way to evaluate a trading strategy.

Instead of one big train/test split (which can be cherry-picked),
walk-forward rolls through time:

  Window 1: [===TRAIN 6mo===][==TEST 1mo==]
  Window 2:    [===TRAIN 6mo===][==TEST 1mo==]
  Window 3:       [===TRAIN 6mo===][==TEST 1mo==]
  ...

The model is retrained on each window with the latest data,
then tested on the next month it has NEVER seen.

Final metrics are the AVERAGE across all test windows.
This is what your live performance will look like.

Also includes permutation testing: shuffle labels N times,
run the same strategy, confirm real results are statistically
significant vs random.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: int  # index
    train_end: int
    test_start: int
    test_end: int
    n_train_samples: int
    n_test_samples: int

    # Performance
    total_return: float  # cumulative return in test window
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    profit_factor: float

    # Model quality
    prediction_mae: float  # MAE of return predictions
    prediction_corr: float  # correlation of predicted vs actual returns
    direction_accuracy: float  # % of times predicted direction was correct


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward windows."""

    windows: list[WindowResult]
    n_windows: int

    # Aggregated metrics (THESE are the real numbers)
    avg_return: float
    avg_sharpe: float
    avg_max_dd: float
    avg_win_rate: float
    avg_n_trades: float
    avg_profit_factor: float
    avg_direction_accuracy: float

    # Stability metrics
    sharpe_std: float  # std dev of Sharpe across windows
    return_std: float  # std dev of returns across windows
    worst_window_return: float
    best_window_return: float

    # Statistical significance
    pct_profitable_windows: float  # % of windows with positive return
    t_stat: float  # t-statistic for returns > 0
    p_value: float  # p-value for returns > 0

    # Permutation test results (if run)
    permutation_p_value: float | None = None  # % of shuffled runs that beat real
    permutation_n_runs: int = 0


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""

    # Window sizes (in bars)
    train_window: int = 4320  # 6 months of hourly bars (180 * 24)
    test_window: int = 720  # 1 month of hourly bars (30 * 24)
    step_size: int = 720  # advance by 1 month between windows

    # Minimum requirements
    min_train_samples: int = 2000
    min_test_samples: int = 200

    # Permutation testing
    n_permutations: int = 1000  # number of shuffled runs
    permutation_seed: int = 42

    # Strategy params (matching V2 defaults)
    min_return_threshold: float = 0.003
    leverage: float = 5.0
    taker_fee: float = 0.0005
    funding_rate_per_8h: float = 0.0001
    max_position_pct: float = 0.05


class WalkForwardValidator:
    """
    Walk-forward backtesting framework.

    Usage:
        validator = WalkForwardValidator()
        result = validator.run(
            X=feature_matrix,
            y=forward_returns,
            close_prices=close,
            feature_names=names,
        )
        print(f"Average Sharpe: {result.avg_sharpe:.2f}")
        print(f"p-value: {result.p_value:.4f}")
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def run(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        close_prices: NDArray[np.float64],
        feature_names: list[str],
        run_permutation_test: bool = False,
    ) -> WalkForwardResult:
        """
        Run full walk-forward validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Forward returns (n_samples,)
            close_prices: Close prices aligned with X
            feature_names: Feature names
            run_permutation_test: Whether to run permutation testing

        Returns:
            WalkForwardResult with aggregated metrics
        """
        n_samples = len(y)
        windows: list[WindowResult] = []
        window_id = 0

        # Generate windows
        train_start = 0
        while True:
            train_end = train_start + self.config.train_window
            test_start = train_end
            test_end = test_start + self.config.test_window

            if test_end > n_samples:
                break

            if train_end - train_start < self.config.min_train_samples:
                train_start += self.config.step_size
                continue

            if test_end - test_start < self.config.min_test_samples:
                train_start += self.config.step_size
                continue

            logger.info(
                f"Window {window_id}: train[{train_start}:{train_end}] "
                f"test[{test_start}:{test_end}]"
            )

            # Run single window
            result = self._run_window(
                window_id=window_id,
                X=X,
                y=y,
                close_prices=close_prices,
                feature_names=feature_names,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            windows.append(result)
            window_id += 1

            # Advance
            train_start += self.config.step_size

        if not windows:
            logger.error("No valid windows generated!")
            return self._empty_result()

        # Aggregate
        aggregated = self._aggregate(windows)

        # Permutation test
        if run_permutation_test:
            perm_p = self._permutation_test(X, y, close_prices, feature_names)
            aggregated.permutation_p_value = perm_p
            aggregated.permutation_n_runs = self.config.n_permutations
            logger.info(f"Permutation test p-value: {perm_p:.4f}")

        return aggregated

    def _run_window(
        self,
        window_id: int,
        X: NDArray,
        y: NDArray,
        close_prices: NDArray,
        feature_names: list[str],
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
    ) -> WindowResult:
        """Run a single walk-forward window."""
        from src.v2.models import ModelConfig, StackingEnsemble

        # Split data
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        close_train = close_prices[train_start:train_end]
        close_test = close_prices[test_start:test_end]

        # Train model on this window
        model = StackingEnsemble(ModelConfig())
        model.train(X_train, y_train, feature_names, close_train)

        # Generate predictions on test set
        predictions = []
        for i in range(len(X_test)):
            pred = model.predict(
                X_test[i : i + 1],
                close_prices=close_test[: i + 1] if i > 0 else None,
            )
            predictions.append(pred)

        # Simulate trading
        trades, equity_curve = self._simulate_trades(
            predictions, y_test, close_test
        )

        # Calculate metrics
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else []

        pred_returns = np.array([p.predicted_return for p in predictions])
        pred_mae = float(np.mean(np.abs(pred_returns - y_test)))
        pred_corr = float(np.corrcoef(pred_returns, y_test)[0, 1]) if len(pred_returns) > 1 else 0.0
        if np.isnan(pred_corr):
            pred_corr = 0.0

        # Direction accuracy
        correct_dir = np.sum(np.sign(pred_returns) == np.sign(y_test))
        dir_accuracy = float(correct_dir / len(y_test)) if len(y_test) > 0 else 0.0

        # Sharpe
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(24 * 365))
        else:
            sharpe = 0.0

        # Max drawdown
        max_dd = self._max_drawdown(equity_curve)

        # Win rate & profit factor
        winning = [t for t in trades if t > 0]
        losing = [t for t in trades if t <= 0]
        win_rate = len(winning) / len(trades) if trades else 0.0
        total_wins = sum(winning) if winning else 0
        total_losses = abs(sum(losing)) if losing else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0.0

        return WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            n_train_samples=len(y_train),
            n_test_samples=len(y_test),
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=len(trades),
            profit_factor=profit_factor,
            prediction_mae=pred_mae,
            prediction_corr=pred_corr,
            direction_accuracy=dir_accuracy,
        )

    def _simulate_trades(
        self,
        predictions: list,
        actual_returns: NDArray,
        close_prices: NDArray,
    ) -> tuple[list[float], NDArray]:
        """
        Simulate trading based on predictions.

        Returns (trade_pnls, equity_curve)
        """
        balance = 10000.0  # standardized starting balance
        equity = [balance]
        trade_pnls: list[float] = []
        position = None  # (direction, entry_idx, size)

        for i, pred in enumerate(predictions):
            current_price = close_prices[i] if i < len(close_prices) else close_prices[-1]

            # Check exit for open position
            if position is not None:
                direction, entry_idx, size = position
                holding_bars = i - entry_idx

                # Simple exit: hold for target horizon or signal reversal
                should_exit = (
                    holding_bars >= 12  # target horizon
                    or (direction == "LONG" and pred.signal == "SHORT")
                    or (direction == "SHORT" and pred.signal == "LONG")
                )

                if should_exit:
                    actual_ret = actual_returns[i] if i < len(actual_returns) else 0.0
                    # Approximate P&L over holding period
                    cumulative_ret = sum(
                        actual_returns[j]
                        for j in range(entry_idx, min(i + 1, len(actual_returns)))
                    )

                    if direction == "SHORT":
                        cumulative_ret = -cumulative_ret

                    notional = size * self.config.leverage
                    pnl = notional * cumulative_ret
                    costs = notional * self.config.taker_fee * 2  # entry + exit
                    funding = notional * self.config.funding_rate_per_8h * max(1, holding_bars // 8)
                    net_pnl = pnl - costs - funding

                    balance += net_pnl
                    trade_pnls.append(net_pnl)
                    position = None

            # Check entry
            if position is None and pred.signal != "HOLD":
                size = balance * min(
                    self.config.max_position_pct,
                    abs(pred.predicted_return) / max(pred.predicted_std, 0.001) * 0.02,
                )
                if size >= 50:
                    position = (pred.signal, i, size)

            equity.append(balance)

        # Close any remaining position
        if position is not None:
            direction, entry_idx, size = position
            cumulative_ret = sum(
                actual_returns[j]
                for j in range(entry_idx, len(actual_returns))
            )
            if direction == "SHORT":
                cumulative_ret = -cumulative_ret
            notional = size * self.config.leverage
            holding_bars = len(actual_returns) - entry_idx
            pnl = notional * cumulative_ret
            costs = notional * self.config.taker_fee * 2
            funding = notional * self.config.funding_rate_per_8h * max(1, holding_bars // 8)
            balance += pnl - costs - funding
            trade_pnls.append(pnl - costs - funding)
            equity.append(balance)

        return trade_pnls, np.array(equity)

    def _permutation_test(
        self,
        X: NDArray,
        y: NDArray,
        close_prices: NDArray,
        feature_names: list[str],
    ) -> float:
        """
        Permutation test: shuffle labels, run strategy, compare.

        Returns p-value (fraction of shuffled runs that beat real).
        """
        logger.info(f"Running {self.config.n_permutations} permutation tests...")

        # Get real performance (simplified — use first window only for speed)
        real_result = self.run(X, y, close_prices, feature_names, run_permutation_test=False)
        real_sharpe = real_result.avg_sharpe

        rng = np.random.RandomState(self.config.permutation_seed)
        beats_real = 0

        for perm_i in range(self.config.n_permutations):
            # Shuffle returns (break any real signal)
            y_shuffled = y.copy()
            rng.shuffle(y_shuffled)

            # Run same strategy
            perm_result = self.run(
                X, y_shuffled, close_prices, feature_names,
                run_permutation_test=False,
            )

            if perm_result.avg_sharpe >= real_sharpe:
                beats_real += 1

            if (perm_i + 1) % 100 == 0:
                logger.info(
                    f"Permutation {perm_i + 1}/{self.config.n_permutations}: "
                    f"{beats_real} beat real so far"
                )

        p_value = beats_real / self.config.n_permutations
        return p_value

    def _max_drawdown(self, equity: NDArray) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity) < 2:
            return 0.0
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    def _aggregate(self, windows: list[WindowResult]) -> WalkForwardResult:
        """Aggregate results across windows."""
        returns = [w.total_return for w in windows]
        sharpes = [w.sharpe_ratio for w in windows]

        avg_return = float(np.mean(returns))
        avg_sharpe = float(np.mean(sharpes))
        return_std = float(np.std(returns))

        # t-test: are returns significantly > 0?
        if return_std > 0 and len(returns) > 1:
            t_stat = avg_return / (return_std / np.sqrt(len(returns)))
            # Approximate p-value (one-sided) using normal approx
            from scipy.stats import t as t_dist

            p_value = float(1 - t_dist.cdf(t_stat, df=len(returns) - 1))
        else:
            t_stat = 0.0
            p_value = 1.0

        return WalkForwardResult(
            windows=windows,
            n_windows=len(windows),
            avg_return=avg_return,
            avg_sharpe=avg_sharpe,
            avg_max_dd=float(np.mean([w.max_drawdown for w in windows])),
            avg_win_rate=float(np.mean([w.win_rate for w in windows])),
            avg_n_trades=float(np.mean([w.n_trades for w in windows])),
            avg_profit_factor=float(np.mean([w.profit_factor for w in windows])),
            avg_direction_accuracy=float(np.mean([w.direction_accuracy for w in windows])),
            sharpe_std=float(np.std(sharpes)),
            return_std=return_std,
            worst_window_return=float(np.min(returns)),
            best_window_return=float(np.max(returns)),
            pct_profitable_windows=float(np.mean([1 if r > 0 else 0 for r in returns])),
            t_stat=float(t_stat),
            p_value=p_value,
        )

    def _empty_result(self) -> WalkForwardResult:
        """Return empty result when no windows generated."""
        return WalkForwardResult(
            windows=[], n_windows=0,
            avg_return=0, avg_sharpe=0, avg_max_dd=0, avg_win_rate=0,
            avg_n_trades=0, avg_profit_factor=0, avg_direction_accuracy=0,
            sharpe_std=0, return_std=0, worst_window_return=0,
            best_window_return=0, pct_profitable_windows=0,
            t_stat=0, p_value=1.0,
        )
