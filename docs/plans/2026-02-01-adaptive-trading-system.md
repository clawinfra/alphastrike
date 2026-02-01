# Adaptive Self-Tuning Trading System Architecture

## The Problem

Current backtest results show:
- **BTC**: 75% win rate ✓
- **ETH**: 100% win rate ✓
- **SOL**: 33% win rate ✗ (same parameters don't work)

Each asset has different volatility, liquidity, and behavior patterns. A one-size-fits-all approach is suboptimal.

## Proposed Solution: Adaptive Asset Manager with LLM Oversight

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE TRADING SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ BTC Config   │   │ ETH Config   │   │ SOL Config   │                │
│  │ thresh: 70   │   │ thresh: 70   │   │ thresh: 80   │ ← Auto-tuned  │
│  │ stop: 1.2ATR │   │ stop: 1.0ATR │   │ stop: 1.8ATR │                │
│  │ short: BEAR  │   │ short: BEAR  │   │ short: false │                │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┼──────────────────┘                         │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   ADAPTIVE MANAGER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │Performance  │  │Data Quality │  │ Retune      │              │   │
│  │  │Tracker      │→ │Validator    │→ │Scheduler    │              │   │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘              │   │
│  └───────────────────────────────────────────┼──────────────────────┘   │
│                                              ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   LLM DECISION LAYER                             │   │
│  │                   (Ollama + DeepSeek)                            │   │
│  │                                                                   │   │
│  │  Tools Available:                                                 │   │
│  │  • retune_asset(symbol, params)                                  │   │
│  │  • disable_asset(symbol, reason)                                 │   │
│  │  • adjust_threshold(symbol, value)                               │   │
│  │  • trigger_hot_reload()                                          │   │
│  │  • send_alert(message)                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                              │                          │
│                                              ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   HOT RELOAD MANAGER                             │   │
│  │  • Save state → Retrain → Validate → Reload → Resume            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Per-Asset Adaptive Configuration

Each asset maintains its own optimized parameters based on rolling performance:

```python
@dataclass
class AdaptiveAssetConfig:
    symbol: str

    # Tunable parameters
    conviction_threshold: float = 70  # 60-85 range
    stop_atr_multiplier: float = 1.2  # 0.8-2.0 range
    position_size_mult: float = 1.0   # 0.5-1.5 range
    short_enabled: bool = True
    require_daily_trend_for_short: bool = True

    # Quality thresholds
    min_trades_for_tuning: int = 20
    min_data_quality_score: float = 0.8

    # Performance tracking
    rolling_window_days: int = 30
    win_rate_threshold: float = 0.5   # Retune if below
    max_drawdown_threshold: float = 0.05
```

### 2. Data Quality Validator

Before tuning, validate data quality:

```python
class DataQualityValidator:
    def calculate_quality_score(self, candles: list[Candle]) -> float:
        """
        Score from 0-1 based on:
        - Gap detection (missing candles)
        - Outlier detection (suspicious wicks)
        - Volume consistency
        - Price continuity
        """

    def has_sufficient_data(self, symbol: str) -> bool:
        """Check if enough quality data for tuning."""
```

### 3. Retune Triggers

Automatic retuning is triggered when:

| Trigger | Condition | Action |
|---------|-----------|--------|
| Performance Drop | Win rate < 50% over 20 trades | Optimize parameters |
| Drawdown Breach | DD > 5% | Reduce position size |
| Regime Change | Volatility shift detected | Recalibrate stops |
| Scheduled | Every 7 days | Full validation |
| Data Quality | New data available | Incremental update |

### 4. LLM Decision Layer (Ollama + DeepSeek)

For complex decisions that require reasoning:

```python
TOOLS = [
    {
        "name": "retune_asset",
        "description": "Trigger parameter optimization for an asset",
        "parameters": {
            "symbol": "string",
            "reason": "string"
        }
    },
    {
        "name": "disable_asset",
        "description": "Temporarily disable trading for an asset",
        "parameters": {
            "symbol": "string",
            "duration_hours": "int",
            "reason": "string"
        }
    },
    {
        "name": "adjust_conviction",
        "description": "Adjust conviction threshold for an asset",
        "parameters": {
            "symbol": "string",
            "new_threshold": "float",
            "reason": "string"
        }
    },
    {
        "name": "trigger_hot_reload",
        "description": "Reload all models and configs without restart",
        "parameters": {}
    },
    {
        "name": "send_alert",
        "description": "Send alert to operator",
        "parameters": {
            "message": "string",
            "severity": "string"  # info, warning, critical
        }
    }
]
```

### 5. Hot Reload Manager

Zero-downtime model updates:

```python
class HotReloadManager:
    async def reload_models(self) -> bool:
        """
        1. Save current state (open positions, pending orders)
        2. Train new models in background
        3. Validate new models on holdout data
        4. If validation passes, swap models atomically
        5. If validation fails, keep old models
        6. Resume trading
        """

    async def reload_configs(self) -> bool:
        """Reload asset configs from disk."""

    async def graceful_restart(self) -> None:
        """Full restart with state preservation."""
```

## Implementation Plan

### Phase 1: Per-Asset Configuration
- Create AdaptiveAssetConfig dataclass
- Store configs in JSON/YAML
- Load per-asset parameters in backtest/live

### Phase 2: Performance Tracker
- Track per-asset rolling metrics
- Detect performance degradation
- Generate retune signals

### Phase 3: Parameter Optimizer
- Bayesian optimization for threshold tuning
- Walk-forward validation
- Prevent overfitting

### Phase 4: LLM Integration
- Ollama client for DeepSeek
- Tool definitions for trading actions
- Context building from performance data

### Phase 5: Hot Reload
- State serialization
- Model swapping
- Validation gates

## Example LLM Interaction

```
SYSTEM: You are the trading strategy optimizer for AlphaStrike.
Analyze performance and decide if adjustments are needed.

USER:
Asset Performance Report (Last 30 days):
- BTC: 8 trades, 75% win rate, +2.1% return, max DD 1.2%
- ETH: 6 trades, 83% win rate, +1.8% return, max DD 0.9%
- SOL: 5 trades, 20% win rate, -3.2% return, max DD 4.1%

Current configs:
- All assets: conviction_threshold=70, stop_atr_mult=1.2

What actions should be taken?

ASSISTANT: Based on the performance analysis:

1. **SOL is significantly underperforming** (20% win rate vs 75-83% for others)
2. **SOL max drawdown (4.1%) is approaching the 5% limit**
3. **BTC and ETH are performing well** - no changes needed

I recommend the following actions:

<tool_call>
{"name": "adjust_conviction", "parameters": {"symbol": "SOLUSDT", "new_threshold": 80, "reason": "Low win rate requires higher conviction filter"}}
</tool_call>

<tool_call>
{"name": "adjust_stop", "parameters": {"symbol": "SOLUSDT", "new_atr_mult": 1.8, "reason": "SOL higher volatility needs wider stops"}}
</tool_call>

<tool_call>
{"name": "send_alert", "parameters": {"message": "SOL parameters adjusted due to underperformance. Monitoring for 10 trades before further action.", "severity": "info"}}
</tool_call>
```

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| AdaptiveAssetConfig | ✅ Complete | `src/adaptive/asset_config.py` |
| PerformanceTracker | ✅ Complete | `src/adaptive/performance_tracker.py` |
| LLMAdvisor (Ollama) | ✅ Complete | `src/adaptive/llm_advisor.py` |
| HotReloadManager | ✅ Complete | `src/adaptive/hot_reload.py` |
| Default Configs | ✅ Complete | `configs/assets/*.yaml` |
| Demo Script | ✅ Complete | `scripts/adaptive_demo.py` |
| Data Quality Validator | ⏳ Pending | - |
| Parameter Optimizer | ⏳ Pending | - |

## File Structure

```
src/
├── adaptive/
│   ├── __init__.py           # Module exports
│   ├── asset_config.py       # AdaptiveAssetConfig dataclass + YAML I/O
│   ├── performance_tracker.py # Rolling metrics + retune triggers
│   ├── hot_reload.py         # Zero-downtime model updates
│   └── llm_advisor.py        # Ollama + DeepSeek tool calling
configs/
└── assets/
    ├── btcusdt.yaml          # BTC config (default params)
    ├── ethusdt.yaml          # ETH config (slightly tighter stops)
    └── solusdt.yaml          # SOL config (conservative, shorts disabled)
scripts/
└── adaptive_demo.py          # Demo script for testing
```
