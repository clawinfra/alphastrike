"""
Adaptive Trading Manager

The central orchestrator that coordinates all adaptive components:
- PerformanceTracker: Detects when retuning is needed
- ParameterOptimizer: Finds optimal parameters
- RegimeAwareParams: Real-time regime adjustments
- HotReloadManager: Zero-downtime updates

This is the "brain" that makes the system self-tuning.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.adaptive.asset_config import (
    AdaptiveAssetConfig,
    load_asset_config,
)
from src.adaptive.hot_reload import HotReloadManager
from src.adaptive.performance_tracker import (
    PerformanceTracker,
    RetuneTrigger,
    Trade,
)
from src.adaptive.regime_params import (
    AdjustedParams,
    RegimeAwareParams,
    create_symbol_specific_adjustments,
)
from src.core.config import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveState:
    """Current state of the adaptive system."""

    symbol: str
    config: AdaptiveAssetConfig
    current_regime: MarketRegime = MarketRegime.RANGING
    regime_confidence: float = 0.5
    adjusted_params: AdjustedParams | None = None
    pending_retune: bool = False
    retune_reason: str = ""
    last_optimization: datetime | None = None
    optimization_count: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "config": self.config.to_dict(),
            "current_regime": self.current_regime.value,
            "regime_confidence": self.regime_confidence,
            "adjusted_params": self.adjusted_params.to_dict() if self.adjusted_params else None,
            "pending_retune": self.pending_retune,
            "retune_reason": self.retune_reason,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "optimization_count": self.optimization_count,
        }


class AdaptiveManager:
    """
    Central manager for the adaptive trading system.

    Responsibilities:
    1. Load and manage per-symbol configurations
    2. Track performance and detect retune triggers
    3. Run parameter optimization when triggered
    4. Apply real-time regime adjustments
    5. Coordinate hot reloads
    6. Save learned parameters for persistence

    Usage:
        manager = AdaptiveManager(backtest_func)
        manager.initialize(["BTCUSDT", "ETHUSDT", "SOLUSDT"])

        # On each trade completion
        trigger = manager.record_trade(trade)
        if trigger:
            await manager.handle_trigger(trigger)

        # Before each trade decision
        params = manager.get_adjusted_params("BTCUSDT", regime, confidence)
    """

    def __init__(
        self,
        backtest_func: Callable[[str, dict], dict] | None = None,
        config_dir: Path | None = None,
        learned_params_dir: Path | None = None,
        auto_optimize: bool = True,
        auto_save: bool = True,
    ):
        """
        Initialize adaptive manager.

        Args:
            backtest_func: Function(symbol, params) -> {sharpe, return, n_trades, ...}
            config_dir: Directory for asset config YAML files
            learned_params_dir: Directory for learned parameter JSON files
            auto_optimize: Automatically run optimization on triggers
            auto_save: Automatically save learned parameters
        """
        self.backtest_func = backtest_func
        self.config_dir = config_dir or Path("configs/assets")
        self.learned_params_dir = learned_params_dir or Path("data/learned_params")
        self.auto_optimize = auto_optimize
        self.auto_save = auto_save

        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.learned_params_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.performance_tracker = PerformanceTracker(
            window_days=30,
            win_rate_threshold=0.40,  # Lowered from 0.5 - more realistic
            max_drawdown_threshold=0.10,  # 10% max DD before retune
            max_consecutive_losses=3,
            min_trades_for_evaluation=10,
        )
        self.regime_params = RegimeAwareParams(confidence_scaling=True)
        self.hot_reload = HotReloadManager()

        # State per symbol
        self._states: dict[str, AdaptiveState] = {}
        self._optimizer = None  # Lazy load

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def initialize(self, symbols: list[str]) -> None:
        """
        Initialize adaptive state for all symbols.

        Loads existing configs and learned parameters.
        """
        logger.info(f"Initializing adaptive manager for {len(symbols)} symbols")

        for symbol in symbols:
            # Load or create config
            config = load_asset_config(symbol, self.config_dir)

            # Load learned parameters if they exist
            learned_path = self.learned_params_dir / f"{symbol.lower()}_params.json"
            if learned_path.exists():
                try:
                    with open(learned_path) as f:
                        learned = json.load(f)
                    # Apply learned params to config
                    self._apply_learned_params(config, learned.get("config", {}))
                    logger.info(f"Loaded learned params for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to load learned params for {symbol}: {e}")

            # Create symbol-specific regime adjustments
            symbol_adjustments = create_symbol_specific_adjustments(symbol)
            _symbol_regime_params = RegimeAwareParams(
                custom_adjustments=symbol_adjustments,
                confidence_scaling=True,
            )

            # Initialize state
            self._states[symbol] = AdaptiveState(
                symbol=symbol,
                config=config,
            )

            logger.info(
                f"Initialized {symbol}: conviction={config.conviction_threshold:.0f}, "
                f"stop={config.stop_atr_multiplier:.2f}x, "
                f"size={config.position_size_multiplier:.2f}x"
            )

    def _apply_learned_params(self, config: AdaptiveAssetConfig, params: dict) -> None:
        """Apply learned parameters to config."""
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
        if "require_daily_trend_for_short" in params:
            config.require_daily_trend_for_short = params["require_daily_trend_for_short"]

    def record_trade(self, trade: Trade) -> RetuneTrigger | None:
        """
        Record a completed trade and check for retune triggers.

        Args:
            trade: Completed trade record

        Returns:
            RetuneTrigger if optimization needed, None otherwise
        """
        trigger = self.performance_tracker.record_trade(trade)

        if trigger:
            symbol = trade.symbol
            if symbol in self._states:
                self._states[symbol].pending_retune = True
                self._states[symbol].retune_reason = trigger.reason

            logger.warning(
                f"Retune triggered for {symbol}: {trigger.trigger_type.value} - "
                f"{trigger.reason}"
            )

        return trigger

    async def handle_trigger(self, trigger: RetuneTrigger) -> dict:
        """
        Handle a retune trigger by running optimization.

        Args:
            trigger: The retune trigger to handle

        Returns:
            Result dict with optimization status
        """
        symbol = trigger.symbol

        if not self.auto_optimize:
            logger.info(f"Auto-optimize disabled, skipping optimization for {symbol}")
            return {"status": "skipped", "reason": "auto_optimize disabled"}

        if not self.backtest_func:
            logger.warning(f"No backtest function provided, cannot optimize {symbol}")
            return {"status": "error", "reason": "no backtest function"}

        async with self._lock:
            return await self._run_optimization(symbol, trigger.reason)

    async def _run_optimization(self, symbol: str, reason: str) -> dict:
        """Run parameter optimization for a symbol."""
        from src.adaptive.parameter_optimizer import OPTUNA_AVAILABLE, ParameterOptimizer

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed, using quick optimization")
            return await self._run_quick_optimization(symbol, reason)

        # Lazy load optimizer
        if self._optimizer is None:
            self._optimizer = ParameterOptimizer(
                backtest_func=self.backtest_func,
                n_trials=40,  # Balance between quality and speed
                timeout_seconds=300,
                min_sharpe_threshold=0.3,
            )

        state = self._states.get(symbol)
        if not state:
            return {"status": "error", "reason": f"Unknown symbol {symbol}"}

        logger.info(f"Starting optimization for {symbol} (reason: {reason})")

        try:
            result = self._optimizer.optimize(
                symbol=symbol,
                trigger_reason=reason,
                current_config=state.config,
                warm_start=True,
            )

            if result.success:
                # Apply optimized parameters
                self._optimizer.apply_result(result, state.config)

                # Save to disk
                if self.auto_save:
                    self._save_learned_params(symbol, state.config)

                # Update state
                state.pending_retune = False
                state.retune_reason = ""
                state.last_optimization = datetime.now(UTC)
                state.optimization_count += 1

                # Clear trigger
                self.performance_tracker.clear_trigger(symbol)

                logger.info(
                    f"Optimization successful for {symbol}: "
                    f"Sharpe={result.best_value:.2f}, OOS={result.out_of_sample_sharpe:.2f}"
                )

                return {
                    "status": "success",
                    "sharpe": result.best_value,
                    "oos_sharpe": result.out_of_sample_sharpe,
                    "params": result.best_params,
                }
            else:
                logger.warning(f"Optimization failed validation for {symbol}: {result.message}")
                return {"status": "failed_validation", "reason": result.message}

        except Exception as e:
            logger.error(f"Optimization error for {symbol}: {e}")
            return {"status": "error", "reason": str(e)}

    async def _run_quick_optimization(self, symbol: str, reason: str) -> dict:
        """Fallback optimization when Optuna isn't available."""
        from src.adaptive.parameter_optimizer import QuickOptimizer

        state = self._states.get(symbol)
        if not state:
            return {"status": "error", "reason": f"Unknown symbol {symbol}"}

        quick_opt = QuickOptimizer(self.backtest_func)

        # Optimize based on trigger type
        if "consecutive" in reason.lower() or "stop" in reason.lower():
            # Optimize stops after consecutive losses
            result = quick_opt.optimize_stops(
                symbol,
                state.config.stop_atr_multiplier,
            )
            if result["best_sharpe"] > 0:
                state.config.stop_atr_multiplier = result["best_stop_atr"]
                if self.auto_save:
                    self._save_learned_params(symbol, state.config)
                return {"status": "success", "type": "stops", **result}

        elif "drawdown" in reason.lower():
            # Optimize position size after drawdown
            result = quick_opt.optimize_position_size(
                symbol,
                state.config.position_size_multiplier,
            )
            if result["best_sharpe"] > 0:
                state.config.position_size_multiplier = result["best_size_mult"]
                if self.auto_save:
                    self._save_learned_params(symbol, state.config)
                return {"status": "success", "type": "position_size", **result}

        return {"status": "no_improvement"}

    def _save_learned_params(self, symbol: str, config: AdaptiveAssetConfig) -> None:
        """Save learned parameters to JSON file."""
        filepath = self.learned_params_dir / f"{symbol.lower()}_params.json"

        data = {
            "config": {
                "conviction_threshold": config.conviction_threshold,
                "stop_atr_multiplier": config.stop_atr_multiplier,
                "take_profit_atr_multiplier": config.take_profit_atr_multiplier,
                "position_size_multiplier": config.position_size_multiplier,
                "short_conviction_penalty": config.short_conviction_penalty,
                "require_daily_trend_for_short": config.require_daily_trend_for_short,
            },
            "thresholds": {
                "win_rate_low": self.performance_tracker.win_rate_threshold,
                "win_rate_high": 0.6,
                "consecutive_losses": self.performance_tracker.max_consecutive_losses,
                "catastrophic_r": -2.0,
                "large_loss_r": -1.5,
                "stop_loss_rate": 0.5,
                "small_win_r": 1.0,
            },
            "symbol": symbol,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved learned params for {symbol} to {filepath}")

    def update_regime(
        self,
        symbol: str,
        regime: MarketRegime,
        confidence: float,
    ) -> None:
        """
        Update detected regime for a symbol.

        Call this when regime detector produces new classification.
        """
        if symbol in self._states:
            self._states[symbol].current_regime = regime
            self._states[symbol].regime_confidence = confidence

    def get_adjusted_params(
        self,
        symbol: str,
        regime: MarketRegime | None = None,
        confidence: float | None = None,
    ) -> AdjustedParams:
        """
        Get regime-adjusted parameters for trading.

        Args:
            symbol: Trading pair
            regime: Override regime (uses stored if None)
            confidence: Override confidence (uses stored if None)

        Returns:
            AdjustedParams with regime-adjusted values
        """
        state = self._states.get(symbol)
        if not state:
            # Return defaults for unknown symbol
            default_config = AdaptiveAssetConfig(symbol=symbol)
            return self.regime_params.adjust(
                default_config,
                regime or MarketRegime.RANGING,
                confidence or 0.5,
            )

        # Use provided or stored regime
        use_regime = regime or state.current_regime
        use_confidence = confidence if confidence is not None else state.regime_confidence

        # Get regime-adjusted params
        adjusted = self.regime_params.adjust(
            state.config,
            use_regime,
            use_confidence,
        )

        # Store for reference
        state.adjusted_params = adjusted

        return adjusted

    def get_config(self, symbol: str) -> AdaptiveAssetConfig | None:
        """Get current config for a symbol."""
        state = self._states.get(symbol)
        return state.config if state else None

    def get_state(self, symbol: str) -> AdaptiveState | None:
        """Get full adaptive state for a symbol."""
        return self._states.get(symbol)

    def get_all_states(self) -> dict[str, AdaptiveState]:
        """Get all symbol states."""
        return self._states.copy()

    def get_performance_report(self) -> str:
        """Get performance report from tracker."""
        return self.performance_tracker.generate_report()

    async def force_retune(self, symbol: str, reason: str = "manual") -> dict:
        """
        Force optimization for a symbol regardless of triggers.

        Useful for initial tuning or manual intervention.
        """
        async with self._lock:
            return await self._run_optimization(symbol, reason)

    async def retune_all(self, reason: str = "scheduled") -> dict[str, dict]:
        """
        Run optimization for all symbols.

        Use for periodic full re-optimization.
        """
        results = {}
        for symbol in self._states:
            results[symbol] = await self.force_retune(symbol, reason)
        return results

    def get_summary(self) -> str:
        """Get human-readable summary of adaptive state."""
        lines = ["=" * 60, "ADAPTIVE TRADING SYSTEM STATUS", "=" * 60, ""]

        for symbol, state in sorted(self._states.items()):
            status = "⚠️ NEEDS RETUNE" if state.pending_retune else "✓ OK"
            lines.extend([
                f"{symbol}: {status}",
                "-" * 40,
                f"  Regime: {state.current_regime.value} ({state.regime_confidence:.0%})",
                f"  Conviction: {state.config.conviction_threshold:.0f}",
                f"  Stop: {state.config.stop_atr_multiplier:.2f}x ATR",
                f"  TP: {state.config.take_profit_atr_multiplier:.2f}x ATR",
                f"  Size: {state.config.position_size_multiplier:.2f}x",
                f"  Optimizations: {state.optimization_count}",
            ])
            if state.last_optimization:
                lines.append(f"  Last optimized: {state.last_optimization.date()}")
            if state.pending_retune:
                lines.append(f"  Retune reason: {state.retune_reason}")
            lines.append("")

        return "\n".join(lines)
