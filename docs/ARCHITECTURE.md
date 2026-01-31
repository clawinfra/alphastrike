# AlphaStrike Trading Bot - Architecture Document

**Version:** 2.2
**Last Updated:** January 2026
**Status:** Production

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Architecture](#3-component-architecture)
4. [Data Flow](#4-data-flow)
5. [ML Pipeline](#5-ml-pipeline)
6. [Risk Management](#6-risk-management)
7. [Execution Engine](#7-execution-engine)
8. [Configuration](#8-configuration)
9. [Deployment](#9-deployment)
10. [Monitoring](#10-monitoring)

---

## 1. System Overview

### 1.1 Purpose

AlphaStrike is an autonomous algorithmic trading system for cryptocurrency perpetual futures. It combines ML-based signal generation with adaptive risk management to execute trades on WEEX exchange.

### 1.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Modularity** | Independent components with clear interfaces |
| **Adaptability** | Regime-aware strategy adjustment |
| **Resilience** | Self-healing models, graceful degradation |
| **Observability** | Comprehensive logging, AI explanations |
| **Safety** | Multi-layer risk controls, circuit breakers |

### 1.3 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| ML Framework | XGBoost, LightGBM, PyTorch (LSTM), Scikit-learn |
| Database | SQLite (local), PostgreSQL (optional) |
| Async | asyncio, aiohttp |
| Configuration | Pydantic Settings |
| Logging | Structured JSON logging |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ALPHASTRIKE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ DATA LAYER  │──▶│DATA GATEWAY │──▶│FEATURE LAYER│──▶│  FEATURE    │         │
│  │             │   │             │   │             │   │  VALIDATOR  │         │
│  │ • WebSocket │   │ • Staleness │   │ • Technical │   │             │         │
│  │ • REST API  │   │ • Sequence  │   │ • Micro-    │   │ • PSI       │         │
│  │ • Database  │   │ • Range     │   │   structure │   │ • KS Test   │         │
│  │ • Cache     │   │ • Circuit   │   │ • Cross-    │   │ • CUSUM     │         │
│  │             │   │   Breaker   │   │   Asset     │   │ • Z-Score   │         │
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────┬──────┘         │
│                                                               │                 │
│                                                               ▼                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ EXECUTION   │◀──│ RISK LAYER  │◀──│  STRATEGY   │◀──│  ML LAYER   │         │
│  │ LAYER       │   │             │   │  LAYER      │   │             │         │
│  │             │   │ • Position  │   │             │   │ • XGBoost   │         │
│  │ • Order Mgr │   │   Sizer     │   │ • Signal    │   │ • LightGBM  │         │
│  │ • Slippage  │   │ • Exposure  │   │   Filter    │   │ • LSTM      │         │
│  │ • AI Logger │   │ • Risk Mgr  │   │ • Regime    │   │ • RF        │         │
│  │ • Pos Sync  │   │ • Portfolio │   │ • Exit Mgr  │   │ • Ensemble  │         │
│  └──────┬──────┘   └─────────────┘   └─────────────┘   └─────────────┘         │
│         │                                                                       │
│         ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                           WEEX EXCHANGE                                  │   │
│  │  • Perpetual Futures  • 20x Max Leverage  • 8 Trading Pairs             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Data Layer

#### 3.1.1 WebSocket Client (`src/data/websocket_client.py`)

**Purpose:** Real-time market data streaming

**Responsibilities:**
- Connect to WEEX WebSocket endpoints
- Subscribe to candle, ticker, orderbook channels
- Handle reconnection with exponential backoff
- Parse and validate incoming messages

**Key Classes:**
```python
class WebSocketClient:
    async def connect() -> None
    async def subscribe(channels: list[str]) -> None
    async def on_message(message: dict) -> None
    async def reconnect() -> None
```

#### 3.1.2 REST Client (`src/data/rest_client.py`)

**Purpose:** Exchange API interactions

**Responsibilities:**
- Account balance and position queries
- Order placement and management
- Leverage setting
- Stop-loss and take-profit orders

**Key Methods:**
```python
class RESTClient:
    async def get_account_balance() -> dict
    async def get_positions() -> list[Position]
    async def place_order(request: OrderRequest) -> OrderResult
    async def set_leverage(symbol: str, leverage: int) -> bool
    async def place_stop_loss_order(...) -> dict
    async def place_take_profit_order(...) -> dict
```

#### 3.1.3 Database (`src/data/database.py`)

**Purpose:** Persistent storage

**Tables:**
| Table | Purpose |
|-------|---------|
| `candles` | Historical OHLCV data |
| `trades` | Executed trade records |
| `ai_log_uploads` | AI explanations |
| `training_cache` | Model training data |
| `performance_metrics` | Rolling metrics |

#### 3.1.4 Data Gateway (`src/data/data_gateway.py`)

**Purpose:** Filter bad/stale data before it reaches the feature layer

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA GATEWAY                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Data                                                       │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   QUALITY GATES                          │   │
│   │                                                          │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │   │
│   │   │Staleness│→│Sequence │→│ Price   │→│  OHLC   │      │   │
│   │   │ Check   │ │  Check  │ │ Range   │ │  Logic  │      │   │
│   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘      │   │
│   │        │           │           │           │            │   │
│   │        ▼           ▼           ▼           ▼            │   │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐                  │   │
│   │   │ Volume  │→│Complete-│→│ Spread  │                  │   │
│   │   │ Check   │ │  ness   │ │ Check   │                  │   │
│   │   └─────────┘ └─────────┘ └─────────┘                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│             ┌────────────┼────────────┐                         │
│             │            │            │                         │
│           Pass         Fail      Degraded                       │
│             │            │            │                         │
│             ▼            ▼            ▼                         │
│        Feature       Fallback    Confidence                     │
│         Layer        Provider    Adjustment                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Quality Gates:**

| Gate | Check | Threshold | Action |
|------|-------|-----------|--------|
| Staleness | Timestamp age | < 5 seconds | Use fallback |
| Sequence | Gap detection | No gaps > 1 | Backfill/alert |
| Price Range | vs 24h range | Within 50% | Reject + alert |
| OHLC Logic | High >= Low | Boolean | Reject candle |
| Volume | Volume bounds | < 100x average | Cap or reject |
| Completeness | Required fields | 100% | Reject record |
| Spread | Bid-ask spread | < 5% majors | Flag illiquid |

**Key Classes:**
```python
class DataGateway:
    async def process(raw_data: dict) -> GatewayResult
    def validate_candle(candle: Candle) -> ValidationResult
    def check_all_gates(data: dict) -> list[GateResult]

class StalenessChecker:
    def check(timestamp: datetime, max_age: float) -> bool
    def get_age_seconds(timestamp: datetime) -> float

class RangeValidator:
    def validate_price(price: float, symbol: str) -> bool
    def update_24h_range(symbol: str, high: float, low: float) -> None

class AnomalyDetector:
    def detect_spike(value: float, history: list[float]) -> bool
    def detect_gap(current: float, previous: float, threshold: float) -> bool

class DataCircuitBreaker:
    def record_failure(gate: str) -> None
    def is_open() -> bool
    def reset() -> None

class FallbackDataProvider:
    async def get_fallback(symbol: str) -> dict
    def cache_valid_data(symbol: str, data: dict) -> None
```

**Configuration:**
```python
staleness_threshold_seconds: float = 5.0
price_range_threshold: float = 0.5  # 50% of 24h range
volume_spike_multiplier: float = 100.0
circuit_breaker_threshold: int = 5  # failures before open
circuit_breaker_reset_seconds: float = 60.0
```

**Integration:**
- Receives data from WebSocket client before buffering
- Emits validated data to candle buffer
- Triggers fallback provider on circuit breaker open
- Logs all rejections for monitoring

**Metrics Emitted:**
- `data_gateway.pass_rate` - Percentage of data passing all gates
- `data_gateway.rejection_by_gate` - Count by gate type
- `data_gateway.staleness_seconds` - Average data age
- `data_gateway.circuit_breaker_state` - Open/closed state
- `data_gateway.fallback_usage_rate` - Fallback activation frequency

### 3.2 Feature Layer

#### 3.2.1 Feature Pipeline (`src/features/pipeline.py`)

**Purpose:** Coordinate feature calculation

**Architecture:**
```
Raw Data → Technical → Microstructure → Cross-Asset → Schema Alignment → Output
```

**Canonical Features (86 total):**

| Category | Count | Key Features |
|----------|-------|--------------|
| Technical | 35 | RSI, ADX, ATR, MACD, BB, EMA |
| Microstructure | 7 | OI, funding rate, orderbook |
| Fee Features | 5 | Maker/taker, breakeven |
| Cross-Asset | 3 | BTC correlation |
| Time-Based | 5 | Session, cyclical |
| Volatility | 4 | ATR ratio, regime |

#### 3.2.2 Technical Features (`src/features/technical.py`)

**Indicators Calculated:**
```python
# Momentum
RSI(7), RSI(14), RSI(21), RSI_slope, RSI_divergence
Stochastic(K, D), MACD(line, signal, histogram)

# Trend
ADX(14), +DI, -DI, ADX_ROC(5), ADX_slope
EMA(9, 21, 50), EMA_crossovers

# Volatility
ATR(14), ATR_ratio, Bollinger_Bands, ATR_percentile

# Volume
Volume_ratio, OBV, Volume_profile
```

#### 3.2.3 Microstructure Features (`src/features/microstructure.py`)

**Features:**
- `orderbook_imbalance` - Bid/ask depth ratio
- `funding_rate` - Current funding rate
- `open_interest` - Total OI
- `volume_profile` - Recent trade distribution
- `large_trade_ratio` - Whale activity detection

#### 3.2.4 Feature Validation (`src/features/feature_validator.py`)

**Purpose:** Detect feature drift from training distribution and degrade confidence accordingly

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE VALIDATION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Feature Vector (86 features)                                   │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────────────────────────────────────┐     │
│   │               DRIFT DETECTION                          │     │
│   │                                                        │     │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │     │
│   │   │   PSI   │ │ KS Test │ │  CUSUM  │ │ Z-Score │    │     │
│   │   │ Distrib │ │ Statist │ │  Mean   │ │ Outlier │    │     │
│   │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │     │
│   │        │           │           │           │          │     │
│   │        └───────────┴───────────┴───────────┘          │     │
│   │                        │                               │     │
│   │                        ▼                               │     │
│   │              ┌─────────────────┐                      │     │
│   │              │  Health Score   │                      │     │
│   │              │   Calculation   │                      │     │
│   │              └────────┬────────┘                      │     │
│   └───────────────────────┼────────────────────────────────┘     │
│                           │                                       │
│              ┌────────────┴────────────┐                         │
│              │                         │                         │
│         Health ≥ 70%              Health < 70%                   │
│              │                         │                         │
│              ▼                         ▼                         │
│         Full Signal              Degraded Signal                 │
│         confidence               confidence × multiplier         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Drift Detection Methods:**

| Method | Use Case | Threshold | Description |
|--------|----------|-----------|-------------|
| PSI | Distribution shift | > 0.25 | Population Stability Index comparing current vs training distribution |
| KS Test | Statistical difference | p < 0.01 | Kolmogorov-Smirnov test for distribution divergence |
| CUSUM | Mean shift | > 5 * std | Cumulative sum control chart for persistent drift |
| Z-Score | Point outliers | \|z\| > 3.5 | Individual feature value outlier detection |

**Confidence Degradation Formula:**

```python
# Health score calculation (0-100)
health_score = 100 - (
    psi_penalty +          # 0-30 points
    ks_penalty +           # 0-20 points
    cusum_penalty +        # 0-25 points
    outlier_penalty        # 0-25 points
)

# Confidence multiplier based on health
if health_score >= 70:
    multiplier = 1.0
elif health_score >= 50:
    multiplier = 0.7 + (health_score - 50) * 0.015  # 0.7-1.0
elif health_score >= 30:
    multiplier = 0.4 + (health_score - 30) * 0.015  # 0.4-0.7
else:
    multiplier = 0.1  # Minimum multiplier, signals near-halt

# Final confidence
adjusted_confidence = raw_confidence * multiplier
```

**Key Classes:**
```python
class FeatureValidator:
    def validate(features: dict) -> ValidationResult
    def get_health_score() -> float
    def get_drifted_features() -> list[str]

class DriftDetector:
    def calculate_psi(current: np.array, reference: np.array) -> float
    def ks_test(current: np.array, reference: np.array) -> tuple[float, float]
    def cusum_test(values: list[float], target_mean: float) -> float
    def z_score_check(value: float, mean: float, std: float) -> float

class FeatureRangeChecker:
    def check_bounds(feature: str, value: float) -> bool
    def get_valid_range(feature: str) -> tuple[float, float]

class FeatureCompletenessChecker:
    def check_required_features(features: dict) -> list[str]
    def get_missing_features(features: dict) -> list[str]
```

**Reference Distribution Storage:**
```json
// models/reference_distributions.json
{
  "version": "2026-01-30",
  "features": {
    "rsi_14": {
      "mean": 50.2,
      "std": 15.3,
      "percentiles": [10, 25, 50, 75, 90],
      "values": [30.1, 40.5, 50.0, 60.2, 70.8]
    },
    "adx_14": {
      "mean": 28.5,
      "std": 12.1,
      "percentiles": [10, 25, 50, 75, 90],
      "values": [15.0, 20.5, 26.0, 35.0, 48.0]
    }
    // ... all 86 features
  }
}
```

**Configuration:**
```python
psi_threshold: float = 0.25
ks_p_value_threshold: float = 0.01
cusum_multiplier: float = 5.0
z_score_threshold: float = 3.5
min_health_for_trading: float = 30.0
reference_update_interval_hours: int = 168  # Weekly
```

**Integration with ML Ensemble:**
- Called before each prediction
- Adjusts confidence based on feature health
- Logs drifted features for monitoring
- Triggers alerts when health drops below thresholds

**Metrics Emitted:**
- `feature_validation.health_score` - Current health score (0-100)
- `feature_validation.drifted_count` - Number of features currently drifted
- `feature_validation.confidence_multiplier` - Applied multiplier
- `feature_validation.psi_by_feature` - PSI values per feature
- `feature_validation.alerts_triggered` - Count of validation alerts

### 3.3 ML Layer

#### 3.3.1 Ensemble (`src/ml/ensemble.py`)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      ML ENSEMBLE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│   │ XGBoost │   │LightGBM │   │  LSTM   │   │   RF    │        │
│   │  (30%)  │   │  (25%)  │   │  (25%)  │   │  (20%)  │        │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘        │
│        │             │             │             │              │
│        └─────────────┴─────────────┴─────────────┘              │
│                          │                                       │
│                    Weighted Average                              │
│                          │                                       │
│                    ┌─────▼─────┐                                │
│                    │ Confidence│                                │
│                    │ Calibrator│                                │
│                    └─────┬─────┘                                │
│                          │                                       │
│              ┌───────────┴───────────┐                          │
│              │                       │                          │
│        Signal (LONG/SHORT/HOLD)  Confidence (0-1)               │
└─────────────────────────────────────────────────────────────────┘
```

**Key Methods:**
```python
class MLEnsemble:
    async def predict(features: dict) -> tuple[str, float, dict, float]
    def check_model_health(model_name: str) -> bool
    async def check_and_reload_models() -> None
    def _redistribute_weights(unhealthy: list[str]) -> dict
```

**Confidence Calculation (Fixed):**
```python
# Threshold-relative formula
if signal == 'LONG':
    confidence = (weighted_avg - long_threshold) / (1.0 - long_threshold)
elif signal == 'SHORT':
    confidence = (short_threshold - weighted_avg) / short_threshold
```

#### 3.3.2 Individual Models

**XGBoost Model (`src/ml/xgboost_model.py`):**
- Gradient boosting with regularization
- Best for tabular data with mixed features
- 30% ensemble weight

**LightGBM Model (`src/ml/lightgbm_model.py`):**
- Fast gradient boosting
- Leaf-wise growth strategy
- 25% ensemble weight
- **Critical:** Requires num_leaves > 1 validation

**LSTM Model (`src/ml/lstm_model.py`):**
- Sequence pattern recognition
- PyTorch implementation
- 25% ensemble weight
- GPU acceleration when available

**Random Forest (`src/ml/random_forest_model.py`):**
- Ensemble of decision trees
- Robust to outliers
- 20% ensemble weight

#### 3.3.3 Trainer (`src/ml/trainer.py`)

**Training Pipeline:**
```
Trigger Detection → Data Collection → Feature Engineering → Label Generation →
Training → Validation → Model Export → Hot Reload
```

**Trigger-Based Retraining (NOT time-interval based):**

The trainer uses intelligent triggers instead of fixed time intervals. Key insight: During high volatility, data is noisy - training on noise = overfitting to noise.

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRIGGER-BASED RETRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   check_triggers() called periodically                          │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                           │
│   │ High Volatility │──► SUPPRESS retraining                    │
│   │   (ATR > 2.0)   │    Use VolatilityAdjustment instead       │
│   └─────────────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────┐                                           │
│   │ Check Cooldown  │──► If not elapsed, return None            │
│   └─────────────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Priority 1: REGIME_CHANGE (5 min stability required)   │   │
│   │ Priority 2: MODEL_HEALTH_DEGRADED (2 consecutive)      │   │
│   │ Priority 3: FEATURE_DRIFT (PSI > 0.25 for 10+ min)     │   │
│   │ Priority 4: VALIDATION_ACCURACY_DROP (< 52% for 15 min)│   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Retraining Triggers:**

| Trigger | Condition | Persistence | Priority |
|---------|-----------|-------------|----------|
| REGIME_CHANGE | Market regime changed | 5 minutes stable | 1 (highest) |
| MODEL_HEALTH_DEGRADED | Majority models unhealthy | 2 consecutive checks | 2 |
| FEATURE_DRIFT | PSI > 0.25 | 10+ minutes | 3 |
| VALIDATION_ACCURACY_DROP | Accuracy < 52% | 15+ minutes | 4 |
| MANUAL | Operator request | Immediate | - |
| INITIAL | First run | Immediate | - |

**Volatility Adjustment (Use Instead of Retraining):**

During high volatility, adjust trading parameters rather than retraining on noisy data:

| ATR Ratio | Confidence Multiplier | Position Scale | Action |
|-----------|----------------------|----------------|--------|
| ≥ 3.0 (Extreme) | 1.5x | 0.25x | Minimal trading |
| ≥ 2.0 (High) | 1.25x | 0.50x | Reduced trading |
| ≥ 1.5 (Elevated) | 1.1x | 0.75x | Slightly reduced |
| < 1.5 (Normal) | 1.0x | 1.0x | Normal trading |

**Adaptive Cooldown (Inverted Logic):**

| Volatility | Cooldown | Rationale |
|------------|----------|-----------|
| Low (< 1.0) | 120 min | Data stable, no rush to retrain |
| Normal (1.0-2.0) | 75 min | Moderate cooldown |
| Post high-vol | 30 min | Capture new regime quickly |

**Key Classes:**
```python
class RetrainingTrigger(Enum):
    REGIME_CHANGE = "regime_change"
    MODEL_HEALTH_DEGRADED = "model_health_degraded"
    FEATURE_DRIFT = "feature_drift"
    VALIDATION_ACCURACY_DROP = "validation_accuracy_drop"
    MANUAL = "manual"
    INITIAL = "initial"

class VolatilityAdjustment:
    confidence_multiplier: float  # Raise thresholds
    position_scale: float         # Reduce sizes
    should_trade: bool            # Trade at all?

class ModelTrainer:
    def check_triggers(...) -> RetrainingTrigger | None
    def get_volatility_adjustment(volatility: float) -> VolatilityAdjustment
    async def train_all_models(trigger: RetrainingTrigger) -> TrainingReport
```

**Key Parameters:**
- Cooldown: 30-120 minutes (inverted volatility-adaptive)
- Training samples: 1000+ candles
- Validation split: 20% out-of-sample
- Label generation: 3-class (LONG/SHORT/HOLD)

#### 3.3.4 Prediction Confidence Bounds (`src/ml/confidence_filter.py`)

**Purpose:** Reject low-confidence ML signals before they reach the strategy layer, preventing noisy predictions from triggering trades.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                 PREDICTION CONFIDENCE BOUNDS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ML Ensemble Output                                             │
│   (signal, raw_confidence, weighted_avg, model_outputs)         │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────────────────────────────────────┐     │
│   │              CONFIDENCE VALIDATION                     │     │
│   │                                                        │     │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │     │
│   │   │ Minimum │ │ Model   │ │Threshold│ │Prediction│    │     │
│   │   │Confidence│ │Agreement│ │Proximity│ │Stability │    │     │
│   │   │  Check  │ │  Check  │ │  Check  │ │  Check   │    │     │
│   │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │     │
│   │        │           │           │           │          │     │
│   │        └───────────┴───────────┴───────────┘          │     │
│   │                        │                               │     │
│   │                        ▼                               │     │
│   │              ┌─────────────────┐                      │     │
│   │              │ Composite Score │                      │     │
│   │              │   Calculation   │                      │     │
│   │              └────────┬────────┘                      │     │
│   └───────────────────────┼────────────────────────────────┘     │
│                           │                                       │
│              ┌────────────┴────────────┐                         │
│              │                         │                         │
│        Score ≥ Threshold         Score < Threshold               │
│              │                         │                         │
│              ▼                         ▼                         │
│         Pass Signal              REJECT → HOLD                   │
│         to Strategy              (Log rejection reason)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Confidence Validation Checks:**

| Check | Description | Threshold | Weight |
|-------|-------------|-----------|--------|
| Minimum Confidence | Raw confidence from ensemble | ≥ 0.55 | 30% |
| Model Agreement | % of models agreeing on direction | ≥ 50% (2/4) | 25% |
| Threshold Proximity | Distance from decision boundary | ≥ 0.10 from 0.5 | 25% |
| Prediction Stability | Consistency over last N predictions | ≥ 60% same signal | 20% |

**Composite Score Calculation:**

```python
class ConfidenceFilter:
    def calculate_composite_score(
        self,
        raw_confidence: float,
        model_outputs: dict[str, float],
        weighted_avg: float,
        recent_predictions: list[str]
    ) -> tuple[float, dict]:

        # 1. Minimum confidence check (0-1)
        conf_score = min(raw_confidence / 0.55, 1.0) if raw_confidence >= 0.55 else 0.0

        # 2. Model agreement (0-1)
        signal_direction = 'LONG' if weighted_avg > 0.5 else 'SHORT'
        agreeing = sum(1 for v in model_outputs.values()
                      if (v > 0.5) == (weighted_avg > 0.5))
        agreement_score = agreeing / len(model_outputs)

        # 3. Threshold proximity (0-1)
        distance_from_boundary = abs(weighted_avg - 0.5)
        proximity_score = min(distance_from_boundary / 0.25, 1.0)

        # 4. Prediction stability (0-1)
        if recent_predictions:
            same_signal = sum(1 for p in recent_predictions[-5:]
                            if p == signal_direction)
            stability_score = same_signal / len(recent_predictions[-5:])
        else:
            stability_score = 0.5  # Neutral for first prediction

        # Weighted composite
        composite = (
            conf_score * 0.30 +
            agreement_score * 0.25 +
            proximity_score * 0.25 +
            stability_score * 0.20
        )

        breakdown = {
            'confidence_score': conf_score,
            'agreement_score': agreement_score,
            'proximity_score': proximity_score,
            'stability_score': stability_score,
            'composite_score': composite
        }

        return composite, breakdown
```

**Rejection Thresholds:**

| Market Regime | Minimum Composite Score | Rationale |
|---------------|------------------------|-----------|
| Trending (ADX > 25) | 0.55 | Lower bar in clear trends |
| Ranging (ADX < 20) | 0.70 | Higher bar in noisy markets |
| High Volatility | 0.65 | Moderate bar, expect noise |
| Extreme Volatility | 0.75 | Highest bar, very selective |
| Default | 0.60 | Standard threshold |

**Key Classes:**

```python
class ConfidenceFilter:
    def __init__(self, config: ConfidenceFilterConfig):
        self.min_confidence = config.min_confidence
        self.min_agreement = config.min_agreement
        self.min_proximity = config.min_proximity
        self.stability_window = config.stability_window
        self.recent_predictions: deque = deque(maxlen=10)

    def should_reject(
        self,
        signal: str,
        raw_confidence: float,
        weighted_avg: float,
        model_outputs: dict[str, float],
        regime: str
    ) -> tuple[bool, str, dict]:
        """
        Returns: (should_reject, rejection_reason, score_breakdown)
        """

        # HOLD signals always pass (no action needed)
        if signal == 'HOLD':
            return False, None, {}

        # Calculate composite score
        composite, breakdown = self.calculate_composite_score(
            raw_confidence, model_outputs, weighted_avg,
            list(self.recent_predictions)
        )

        # Get regime-adjusted threshold
        threshold = self.get_regime_threshold(regime)

        # Record prediction for stability tracking
        self.recent_predictions.append(signal)

        # Rejection decision
        if composite < threshold:
            reason = self._determine_rejection_reason(breakdown, threshold)
            return True, reason, breakdown

        return False, None, breakdown

    def get_regime_threshold(self, regime: str) -> float:
        thresholds = {
            'trending_up': 0.55,
            'trending_down': 0.55,
            'ranging': 0.70,
            'high_volatility': 0.65,
            'extreme_volatility': 0.75,
        }
        return thresholds.get(regime, 0.60)

    def _determine_rejection_reason(
        self,
        breakdown: dict,
        threshold: float
    ) -> str:
        reasons = []
        if breakdown['confidence_score'] < 0.5:
            reasons.append('low_raw_confidence')
        if breakdown['agreement_score'] < 0.5:
            reasons.append('model_disagreement')
        if breakdown['proximity_score'] < 0.4:
            reasons.append('near_decision_boundary')
        if breakdown['stability_score'] < 0.4:
            reasons.append('unstable_predictions')
        return '|'.join(reasons) if reasons else 'below_composite_threshold'

class ConfidenceFilterConfig:
    min_confidence: float = 0.55
    min_agreement: float = 0.50
    min_proximity: float = 0.10
    stability_window: int = 5
    base_threshold: float = 0.60
```

**Integration with ML Ensemble:**

```python
# In ensemble.py predict() method
async def predict(self, features: dict) -> tuple[str, float, dict, float]:
    # ... existing ensemble prediction logic ...

    # Apply confidence filter BEFORE returning
    should_reject, reason, breakdown = self.confidence_filter.should_reject(
        signal=signal,
        raw_confidence=confidence,
        weighted_avg=weighted_avg,
        model_outputs=model_outputs,
        regime=self.regime_detector.current_regime
    )

    if should_reject:
        logger.info(
            f"Signal REJECTED by confidence filter",
            extra={
                'original_signal': signal,
                'rejection_reason': reason,
                'score_breakdown': breakdown,
                'regime': self.regime_detector.current_regime
            }
        )
        # Convert to HOLD with zero confidence
        return 'HOLD', 0.0, model_outputs, weighted_avg

    return signal, confidence, model_outputs, weighted_avg
```

**Configuration:**

```python
# Confidence Filter Settings
confidence_filter_enabled: bool = True
min_raw_confidence: float = 0.55
min_model_agreement_ratio: float = 0.50
min_threshold_proximity: float = 0.10
prediction_stability_window: int = 5
base_composite_threshold: float = 0.60
regime_threshold_adjustments: dict = {
    'trending': -0.05,
    'ranging': +0.10,
    'high_volatility': +0.05,
    'extreme_volatility': +0.15
}
```

**Rejection Logging:**

```json
{
  "timestamp": "2026-01-30T12:00:00Z",
  "event": "signal_rejected",
  "original_signal": "LONG",
  "symbol": "cmt_btcusdt",
  "rejection_reason": "model_disagreement|near_decision_boundary",
  "score_breakdown": {
    "confidence_score": 0.72,
    "agreement_score": 0.25,
    "proximity_score": 0.36,
    "stability_score": 0.60,
    "composite_score": 0.48
  },
  "regime": "ranging",
  "threshold_used": 0.70,
  "raw_model_outputs": {
    "xgboost": 0.68,
    "lightgbm": 0.42,
    "lstm": 0.71,
    "random_forest": 0.38
  }
}
```

**Metrics Emitted:**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `confidence_filter.rejection_rate` | % of signals rejected | > 80% (too strict) |
| `confidence_filter.pass_rate` | % of signals passing | < 10% (too strict) |
| `confidence_filter.avg_composite_score` | Rolling average score | < 0.40 (model issues) |
| `confidence_filter.rejection_by_reason` | Count by rejection type | - |
| `confidence_filter.regime_rejection_rate` | Rejection rate per regime | - |

**Benefits:**

1. **Noise Reduction**: Filters out signals where the ML layer is uncertain
2. **Model Health Signal**: High rejection rates indicate model or data quality issues
3. **Regime Awareness**: Adapts strictness based on market conditions
4. **Stability Enforcement**: Prevents whipsawing on rapidly changing predictions
5. **Transparency**: Full logging of why signals were rejected for analysis

### 3.4 Strategy Layer

#### 3.4.1 Signal Processor (`src/execution/signal_processor.py`)

**Signal Generation Flow:**
```
ML Prediction → Threshold Check → Filter Application →
Position Scaling → Risk Validation → Order Generation
```

**Filters Applied:**
1. RSI extreme filter
2. ADX saturation filter
3. Regime alignment filter
4. MTF (multi-timeframe) alignment
5. Model agreement filter
6. Reversal detection override

#### 3.4.2 Regime Detector (`src/strategy/regime_detector.py`)

**Regime Classification:**
```python
class RegimeDetector:
    def detect_regime(features: dict) -> RegimeState:
        # Trend detection
        if adx > 25:
            if plus_di > minus_di:
                regime = 'trending_up'
            else:
                regime = 'trending_down'
        elif adx < 20:
            regime = 'ranging'

        # Volatility overlay
        if atr_ratio > 2.0:
            volatility = 'extreme'
        elif atr_ratio > 1.5:
            volatility = 'high'

        # Exhaustion detection
        if adx > 60 and adx_roc < -2:
            regime = 'trend_exhaustion'
```

#### 3.4.3 Exit Manager (`src/strategy/exit_manager.py`)

**Exit Strategy Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      EXIT MANAGER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────┐    ┌────────────────┐    ┌──────────────┐ │
│   │ Multi-Level TP │    │ Trailing Stop  │    │ Time-Based   │ │
│   │                │    │                │    │              │ │
│   │ TP1: 40% @1.5x │    │ Initial: 2.0x  │    │ Max: 24-96h  │ │
│   │ TP2: 35% @2.5x │    │ Trail: 1.5x    │    │ Quality adj  │ │
│   │ TP3: 25% trail │    │ BE+0.3x buffer │    │ Profit ext   │ │
│   └────────────────┘    └────────────────┘    └──────────────┘ │
│                                                                  │
│   Regime Multipliers:                                           │
│   • Trending: 1.25x wider stops                                 │
│   • Ranging: 0.8x tighter stops                                 │
│   • High Vol: 2.0x wider stops                                  │
│   • Exhaustion: 0.7x tighter stops                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Risk Layer

#### 3.5.1 Position Sizer (`src/risk/position_sizer.py`)

**Half-Kelly Formula:**
```python
kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio
adaptive_kelly = kelly_fraction * kelly_multiplier  # 0.3-0.7 range

position_size = (
    capital *
    adaptive_kelly *
    confidence_factor *
    volatility_factor *
    drawdown_factor
)
```

#### 3.5.2 Adaptive Exposure (`src/risk/adaptive_exposure.py`)

**Exposure Limit Calculation:**
```python
adaptive_per_trade = base_limit * vol_factor * regime_factor * dd_factor
adaptive_total = total_limit * vol_factor * regime_factor * dd_factor
adaptive_per_pair = per_pair_limit * vol_factor * concentration_factor
```

**Scaling Factors:**
| Factor | Low | High | Effect |
|--------|-----|------|--------|
| Volatility | 0.6 | 1.3 | High vol = tighter |
| Regime | 0.7 | 1.2 | Trending = more |
| Drawdown | 0.5 | 1.0 | In DD = tighter |
| Concentration | 0.7 | 1.2 | Fewer pos = more per pair |

#### 3.5.3 Risk Manager (`src/risk/risk_manager.py`)

**Validation Checks:**
```python
class RiskManager:
    async def validate_order(order: OrderRequest) -> RiskCheck:
        # 1. Close orders bypass all checks
        if order.is_close:
            return RiskCheck(allowed=True)

        # 2. Exposure limits
        exposure_check = await self.check_exposure(order)

        # 3. Drawdown limits
        drawdown_check = self.check_drawdown()

        # 4. Leverage validation
        leverage_check = self.validate_leverage(order.leverage)

        # 5. Never add to losers
        loser_check = self.check_losing_position(order)

        return aggregate_checks(...)
```

### 3.6 Execution Layer

#### 3.6.1 Order Manager (`src/execution/order_manager.py`)

**Order Flow:**
```
Signal → Slippage Estimation → Order Optimization →
Leverage Setting → Risk Validation → Order Placement →
Fill Tracking → AI Logging
```

#### 3.6.2 Position Sync (`src/execution/position_sync.py`)

**Responsibilities:**
- Periodic exchange position sync
- Orphan position detection
- Stop-loss order management
- Take-profit order management
- Exit condition processing

#### 3.6.3 AI Logger (`src/execution/ai_logger.py`)

**Log Structure:**
```json
{
  "orderId": "12345",
  "symbol": "cmt_btcusdt",
  "side": "LONG",
  "timestamp": "2026-01-30T12:00:00Z",
  "ai_explanation": {
    "signal": "LONG",
    "confidence": 0.78,
    "weighted_average": 0.82,
    "model_outputs": {...},
    "regime": "trending_up",
    "risk_checks": [...],
    "reasoning": "Strong bullish momentum..."
  }
}
```

---

## 4. Data Flow

### 4.1 Real-Time Data Flow

```
WEEX WebSocket
      │
      ▼
┌─────────────────┐
│ WebSocket Client│
│ • Parse candles │
│ • Raw data recv │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DATA GATEWAY   │◀─── Quality Gates
│ • Staleness     │     • Reject bad data
│ • Range check   │     • Fallback provider
│ • Circuit break │     • Alert on issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Candle Buffer   │
│ • 100 candles   │
│ • Per symbol    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Feature Pipeline│────▶│ Feature Cache   │
│ • 86 features   │     │ • 250ms TTL     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│FEATURE VALIDATOR│◀─── Drift Detection
│ • PSI check     │     • Health score
│ • Outlier check │     • Confidence adj
│ • Completeness  │     • Alert on drift
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ML Ensemble     │
│ • 4 models      │
│ • Weighted avg  │
│ • Adj confidence│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│CONFIDENCE FILTER│◀─── Prediction Bounds
│ • Min confidence│     • Model agreement
│ • Composite scr │     • Stability check
│ • Regime thresh │     • Reject → HOLD
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Signal Processor│
│ • Filters       │
│ • Scaling       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Risk Validation │
│ • Exposure      │
│ • Drawdown      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Order Execution │
│ • WEEX API      │
│ • AI Logging    │
└─────────────────┘
```

### 4.2 Training Data Flow

```
Historical Candles → Feature Calculation → Label Generation →
Train/Val Split → Model Training → Validation → Model Export
```

---

## 5. ML Pipeline

### 5.1 Model Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  0. Trigger Detection (NEW)                                      │
│     ├── Check regime change (5 min stability)                   │
│     ├── Check model health (2 consecutive failures)             │
│     ├── Check feature drift (PSI > 0.25 for 10 min)             │
│     ├── Check accuracy drop (< 52% for 15 min)                  │
│     └── SUPPRESS during high volatility (ATR > 2.0)             │
│                                                                  │
│  1. Data Collection                                              │
│     └── Fetch 1000+ candles from database                       │
│                                                                  │
│  2. Feature Engineering                                          │
│     └── Calculate 86 canonical features per candle              │
│                                                                  │
│  3. Label Generation                                             │
│     └── 3-class: LONG (>0.5% up), SHORT (>0.5% down), HOLD     │
│     └── Balanced sampling for equal class distribution          │
│                                                                  │
│  4. Train/Validation Split                                       │
│     └── 80% train, 20% validation (time-ordered)                │
│                                                                  │
│  5. Model Training                                               │
│     └── XGBoost, LightGBM, LSTM, RF in parallel                │
│                                                                  │
│  6. Validation                                                   │
│     └── Out-of-sample accuracy, AUC, calibration               │
│                                                                  │
│  7. Export                                                       │
│     └── Save to models/ directory with timestamp                │
│                                                                  │
│  8. Hot Reload                                                   │
│     └── Ensemble detects new models and loads automatically     │
└─────────────────────────────────────────────────────────────────┘
```

**Why Trigger-Based Instead of Time-Based:**

| Approach | Problem | Result |
|----------|---------|--------|
| Time-based (old) | Retrains during high volatility | Training on noisy data = overfitting |
| Trigger-based (new) | Only retrains on meaningful events | Cleaner models, less churn |

During high volatility, the 0.5% label threshold becomes statistical noise when markets move 5%+ daily. Instead of retraining, we use `VolatilityAdjustment` to reduce position sizes and raise confidence thresholds.

### 5.2 Self-Healing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-HEALING FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Periodic Check (every minute)                                   │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────┐                                           │
│  │ Volatility Check │                                           │
│  │   ATR > 2.0?     │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│      ┌────┴────┐                                                │
│      │         │                                                │
│    HIGH       NORMAL                                            │
│      │         │                                                │
│      ▼         ▼                                                │
│  ┌────────┐ ┌──────────────────┐                               │
│  │ Apply  │ │ Check Triggers   │                               │
│  │ Vol    │ │                  │                               │
│  │ Adjust │ │ • Regime change? │                               │
│  │ ments  │ │ • Health failed? │                               │
│  │        │ │ • Drift > 0.25?  │                               │
│  │ • ↑ conf│ │ • Accuracy < 52%│                               │
│  │ • ↓ size│ └────────┬────────┘                               │
│  └────────┘          │                                          │
│                 ┌────┴────┐                                     │
│                 │         │                                     │
│            No Trigger   Trigger                                 │
│                 │         │                                     │
│                 ▼         ▼                                     │
│             Continue   ┌─────────────────┐                     │
│             Trading    │ Retrain Models  │                     │
│                        │ (with trigger   │                     │
│                        │  reason logged) │                     │
│                        └────────┬────────┘                     │
│                                 │                               │
│                                 ▼                               │
│                        ┌─────────────────┐                     │
│                        │ Hot Reload to   │                     │
│                        │ Ensemble        │                     │
│                        └─────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Model Health Check (Per Prediction):**
```
Prediction Request
       │
       ▼
┌──────────────────┐
│ Health Check     │
│ • Degenerate?    │
│ • Max prob > 95% │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
 Healthy  Unhealthy
    │         │
    ▼         ▼
 Normal   Increment Counter
    │         │
    │    Counter >= 2? (consecutive)
    │         │
    │    ┌────┴────┐
    │    │         │
    │   No       Yes
    │    │         │
    │    ▼         ▼
    │ Redistribute  Trigger:
    │   Weights    MODEL_HEALTH_DEGRADED
    │              (retrain at next check)
    │              │
    └──────────────┘
```

---

## 6. Risk Management

### 6.1 Risk Control Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-LAYER RISK CONTROL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 0: Data Quality Gates                                     │
│  ├── Staleness check (< 5 seconds)                              │
│  ├── Sequence gap detection                                     │
│  ├── Price range validation (50% of 24h)                        │
│  ├── OHLC logic validation                                      │
│  ├── Volume spike detection (< 100x avg)                        │
│  └── Data circuit breaker (5 failures = open)                   │
│                                                                  │
│  Layer 0.5: Feature Validation                                   │
│  ├── PSI drift detection (> 0.25 = drifted)                     │
│  ├── KS test (p < 0.01 = diverged)                              │
│  ├── CUSUM mean shift (> 5*std)                                 │
│  ├── Z-score outliers (|z| > 3.5)                               │
│  ├── Health score calculation                                   │
│  └── Confidence degradation (health < 70% = reduced)            │
│                                                                  │
│  Layer 0.75: Prediction Confidence Bounds                        │
│  ├── Minimum confidence check (≥ 0.55)                          │
│  ├── Model agreement check (≥ 50% same direction)               │
│  ├── Threshold proximity (≥ 0.10 from decision boundary)        │
│  ├── Prediction stability (≥ 60% consistent over 5 signals)     │
│  ├── Composite score calculation (weighted average)             │
│  ├── Regime-adjusted thresholds (0.55-0.75)                     │
│  └── Reject low-confidence → convert to HOLD                    │
│                                                                  │
│  Layer 1: Signal Filters                                         │
│  ├── RSI extreme filter                                         │
│  ├── ADX saturation filter                                      │
│  ├── Regime alignment                                           │
│  └── Model agreement threshold                                  │
│                                                                  │
│  Layer 2: Position Sizing                                        │
│  ├── Half-Kelly sizing                                          │
│  ├── Confidence scaling                                         │
│  ├── Volatility adjustment                                      │
│  └── Leverage-adjusted sizing                                   │
│                                                                  │
│  Layer 3: Exposure Limits                                        │
│  ├── Per-trade: 5-20%                                           │
│  ├── Per-pair: 15-45%                                           │
│  ├── Total: 30-95%                                              │
│  └── Correlation-adjusted                                       │
│                                                                  │
│  Layer 4: Drawdown Controls                                      │
│  ├── Daily: 5-7% adaptive                                       │
│  ├── Total: 15% hard limit                                      │
│  └── Position-count adjusted                                    │
│                                                                  │
│  Layer 5: Circuit Breakers                                       │
│  ├── Max leverage: 20x                                          │
│  ├── Min balance: $9,000                                        │
│  ├── Close orders: Never blocked                                │
│  └── API failure: Graceful degradation                          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Position Scaling Matrix

| Risk Factor | Scale | Cumulative |
|-------------|-------|------------|
| Base | 1.0 | 1.0 |
| + Counter-trend | 0.4 | 0.4 |
| + MTF misalignment | 0.5 | 0.2 |
| + Low agreement | 0.6 | 0.12 |
| Minimum floor | - | 0.05 |

---

## 7. Execution Engine

### 7.1 Order Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                     ORDER LIFECYCLE                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Signal Generated                                              │
│     └── ML ensemble outputs LONG/SHORT with confidence           │
│                                                                   │
│  2. Pre-Trade Checks                                              │
│     ├── Slippage estimation                                      │
│     ├── Order type selection                                     │
│     └── Size optimization                                        │
│                                                                   │
│  3. Risk Validation                                               │
│     ├── Exposure check                                           │
│     ├── Drawdown check                                           │
│     ├── Leverage validation                                      │
│     └── Losing position check                                    │
│                                                                   │
│  4. Leverage Setting                                              │
│     └── Skip if position exists                                  │
│                                                                   │
│  5. Order Placement                                               │
│     ├── Preset stop-loss                                         │
│     ├── Preset take-profit                                       │
│     └── Main order                                               │
│                                                                   │
│  6. Post-Trade                                                    │
│     ├── AI log generation                                        │
│     ├── Position sync                                            │
│     ├── Exchange stop order                                      │
│     └── Exchange TP order                                        │
│                                                                   │
│  7. Exit Management                                               │
│     ├── Trailing stop monitoring                                 │
│     ├── Multi-level TP execution                                 │
│     └── Time-based exit check                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Configuration

### 8.1 Configuration Hierarchy

```
Environment Variables (.env)
         │
         ▼
    Settings Class (config.py)
         │
    ┌────┴────┐
    │         │
 Defaults  Overrides
    │         │
    └────┬────┘
         │
         ▼
   Runtime Config
```

### 8.2 Key Configuration Parameters

**Trading Thresholds:**
```python
long_threshold: float = 0.75
short_threshold: float = 0.25
high_confidence_threshold: float = 0.75
```

**Position Sizing:**
```python
min_notional_value: float = 150.0
max_position_per_pair: float = 0.45
pyramiding_enabled: bool = True
pyramiding_max_per_pair: float = 0.50
```

**Risk Controls:**
```python
max_leverage: int = 20
daily_drawdown_limit: float = 0.05
total_drawdown_limit: float = 0.15
min_balance_threshold: float = 9000.0
```

**Trade Frequency:**
```python
trade_cooldown_seconds: int = 3600
max_trades_per_day: int = 12
max_trades_per_symbol_per_day: int = 2
```

---

## 9. Deployment

### 9.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Ubuntu Server                          │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ Main Process │  │   Trainer    │  │   Monitor    │  │    │
│  │  │              │  │              │  │              │  │    │
│  │  │ • Trading    │  │ • Retraining │  │ • Metrics    │  │    │
│  │  │ • Signals    │  │ • Validation │  │ • Alerts     │  │    │
│  │  │ • Execution  │  │ • Hot reload │  │ • Dashboard  │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │   SQLite     │  │    Models    │  │     Logs     │  │    │
│  │  │   Database   │  │   Directory  │  │   Directory  │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  WEEX Exchange  │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Process Management

**Systemd Service:**
```ini
[Unit]
Description=AlphaStrike Trading Bot
After=network.target

[Service]
Type=simple
User=bowen
WorkingDirectory=/home/bowen/projects/alphastrike
ExecStart=/usr/bin/uv run python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 10. Monitoring

### 10.1 Metrics Dashboard

| Metric | Update Frequency | Alert Threshold |
|--------|------------------|-----------------|
| Net PnL | Real-time | -5% daily |
| Win Rate | Per trade | <40% rolling |
| Model Health | Per prediction | <2 healthy |
| Memory Usage | 1 minute | >1.8GB |
| API Success | Per request | <95% |
| Position Sync | 1 minute | Any mismatch |

### 10.2 Alert System

**Critical Alerts:**
- Total drawdown > 10%
- All models unhealthy
- Position sync failure
- API connectivity loss
- Memory > 90%
- Data gateway circuit breaker open
- Feature validation health < 30%

**Warning Alerts:**
- Daily drawdown > 3%
- Win rate < 45%
- Single model unhealthy
- Feature distribution shift
- High latency (>1s)
- Data staleness > 10 seconds
- PSI drift > 0.25 on any feature
- More than 5 features drifted simultaneously
- Feature confidence multiplier < 50%
- Confidence filter rejection rate > 80% (filter too strict)
- Confidence filter pass rate < 10% (model quality issue)
- Average composite score < 0.40 (systematic model degradation)

---

## Appendix A: File Structure

```
alphastrike/
├── src/
│   ├── core/
│   │   └── config.py           # Configuration management
│   ├── data/
│   │   ├── websocket_client.py # Real-time data
│   │   ├── rest_client.py      # Exchange API
│   │   ├── database.py         # Persistence
│   │   ├── data_gateway.py     # Data quality gates
│   │   └── fallback_provider.py # Fallback data source
│   ├── features/
│   │   ├── pipeline.py         # Feature orchestration
│   │   ├── technical.py        # Technical indicators
│   │   ├── microstructure.py   # Orderbook features
│   │   └── feature_validator.py # Feature drift detection
│   ├── ml/
│   │   ├── ensemble.py         # ML ensemble
│   │   ├── confidence_filter.py # Prediction confidence bounds
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_model.py
│   │   ├── random_forest_model.py
│   │   └── trainer.py          # Model training
│   ├── strategy/
│   │   ├── regime_detector.py  # Market regime
│   │   ├── exit_manager.py     # Exit strategy
│   │   ├── adaptive_thresholds.py
│   │   └── adaptive_leverage.py
│   ├── risk/
│   │   ├── portfolio.py        # Portfolio state
│   │   ├── position_sizer.py   # Position sizing
│   │   ├── risk_manager.py     # Risk validation
│   │   └── adaptive_exposure.py
│   ├── execution/
│   │   ├── signal_processor.py # Signal generation
│   │   ├── order_manager.py    # Order execution
│   │   ├── position_sync.py    # Exchange sync
│   │   └── ai_logger.py        # AI explanations
│   └── utils/
│       ├── fee_calculator.py   # Fee calculations
│       └── drift_detector.py   # Statistical drift detection
├── models/                      # Trained models
│   └── reference_distributions.json  # Feature baseline stats
├── logs/                        # Application logs
├── ai_logs/                     # AI explanations
├── data/                        # Database files
├── scripts/
│   ├── train_dual_models.py
│   ├── pnl_report.py
│   ├── monitor_post_fix.py
│   └── generate_reference_distributions.py  # Create baseline stats
├── main.py                      # Entry point
└── .env                         # Configuration
```

---

*Document History:*
- v1.0 (December 2025): Initial architecture
- v2.0 (January 2026): Updated based on production learnings
- v2.1 (January 2026): Added Prediction Confidence Bounds (Section 3.3.4)
- v2.2 (January 2026): Refactored to Trigger-Based Retraining (Section 3.3.3, 5.1, 5.2)
  - Replaced time-interval retraining with trigger-based approach
  - Added VolatilityAdjustment for high-vol periods (adjust trading, not training)
  - Inverted cooldown logic (longer during calm, shorter after vol subsides)
  - Four retraining triggers: regime_change, model_health_degraded, feature_drift, validation_accuracy_drop
