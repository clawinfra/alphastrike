"""
AlphaStrike Adaptive Trading System

Self-tuning, per-asset optimization with regime-aware parameter adjustment.

Components:
- AdaptiveAssetConfig: Per-asset tunable parameters
- PerformanceTracker: Rolling performance metrics and retune triggers
- ParameterOptimizer: Bayesian optimization for parameter tuning
- WalkForwardValidator: Prevents overfitting via rolling validation
- RegimeAwareParams: Real-time regime-based parameter adjustment
- AdaptiveManager: Central orchestrator for all adaptive components
- LLMAdvisor: Ollama + DeepSeek integration for complex decisions
- HotReloadManager: Zero-downtime model and config updates
"""

from src.adaptive.asset_config import (
    AdaptiveAssetConfig,
    load_asset_config,
    save_asset_config,
    load_all_asset_configs,
    initialize_default_configs,
)
from src.adaptive.performance_tracker import (
    PerformanceTracker,
    AssetPerformance,
    Trade,
    RetuneTrigger,
)
from src.adaptive.llm_advisor import (
    LLMAdvisor,
    AdvisorDecision,
    OllamaConfig,
    TOOLS,
)
from src.adaptive.hot_reload import (
    HotReloadManager,
    TradingState,
    ReloadResult,
)
from src.adaptive.dynamic_leverage import (
    DynamicLeverageManager,
    LeverageState,
)
from src.adaptive.regime_params import (
    RegimeAwareParams,
    RegimeAdjustment,
    AdjustedParams,
    create_symbol_specific_adjustments,
    DEFAULT_REGIME_ADJUSTMENTS,
)
from src.adaptive.adaptive_manager import (
    AdaptiveManager,
    AdaptiveState,
)

# Optional imports (require additional dependencies)
try:
    from src.adaptive.parameter_optimizer import (
        ParameterOptimizer,
        QuickOptimizer,
        OptimizationResult,
        ParameterSpace,
    )
    from src.adaptive.walk_forward import (
        WalkForwardValidator,
        WalkForwardResult,
        WalkForwardWindow,
        quick_walk_forward_split,
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

__all__ = [
    # Asset Configuration
    "AdaptiveAssetConfig",
    "load_asset_config",
    "save_asset_config",
    "load_all_asset_configs",
    "initialize_default_configs",
    # Performance Tracking
    "PerformanceTracker",
    "AssetPerformance",
    "Trade",
    "RetuneTrigger",
    # Regime-Aware Parameters
    "RegimeAwareParams",
    "RegimeAdjustment",
    "AdjustedParams",
    "create_symbol_specific_adjustments",
    "DEFAULT_REGIME_ADJUSTMENTS",
    # Adaptive Manager
    "AdaptiveManager",
    "AdaptiveState",
    # Parameter Optimization (optional)
    "ParameterOptimizer",
    "QuickOptimizer",
    "OptimizationResult",
    "ParameterSpace",
    "OPTIMIZER_AVAILABLE",
    # Walk-Forward Validation (optional)
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardWindow",
    "quick_walk_forward_split",
    # LLM Advisor
    "LLMAdvisor",
    "AdvisorDecision",
    "OllamaConfig",
    "TOOLS",
    # Hot Reload
    "HotReloadManager",
    "TradingState",
    "ReloadResult",
    # Dynamic Leverage
    "DynamicLeverageManager",
    "LeverageState",
]
