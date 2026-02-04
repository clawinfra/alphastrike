# Hyperliquid Multi-Asset ML Training Plan

## Overview

Train ML models on Hyperliquid historical data to enable production-compliant backtesting that accurately predicts live performance.

## Architecture Compliance

Per ARCHITECTURE.md Section 5.1 (ML Pipeline):

```
Historical Candles → Feature Calculation → Label Generation →
Train/Val Split → Model Training → Validation → Model Export
```

## Implementation Steps

### Phase 1: Data Collection

**1.1 Create `scripts/hyperliquid_data_collector.py`**

Fetches historical candles from Hyperliquid and saves to SQLite database.

```python
# For each asset in universe:
# 1. Fetch 2000+ hourly candles (83+ days)
# 2. Convert to UnifiedCandle format
# 3. Save to database via Database.save_candles()
```

**Assets to collect:**
- BTC (crypto_major)
- ETH (crypto_major)
- PAXG (gold_proxy) - key diversifier
- SOL (crypto_l1)
- AAVE (crypto_defi)

**Data requirements:**
- Minimum: 1000 candles per asset (per architecture)
- Recommended: 2000+ candles for robust training
- Interval: 1h (matches production)

### Phase 2: Feature Engineering

**2.1 Use existing `src/features/pipeline.py`**

The FeaturePipeline calculates 86 canonical features:
- Technical indicators (35 features)
- Microstructure features (7 features)
- Fee features (5 features)
- Cross-asset features (3 features)
- Time features (5 features)
- Volatility features (4 features)

**Cross-asset features require:**
- BTC prices for correlation
- ETH prices for correlation
- Market index (optional)

### Phase 3: Label Generation

**3.1 Per architecture Section 5.1:**

```
3-class: LONG (>0.5% up), SHORT (>0.5% down), HOLD
Balanced sampling for equal class distribution
```

**Implementation:**
```python
LABEL_THRESHOLD = 0.005  # 0.5%

for i in range(len(candles) - 1):
    future_return = (candles[i+1].close - candles[i].close) / candles[i].close
    if future_return > LABEL_THRESHOLD:
        labels[i] = "LONG"  # 1
    elif future_return < -LABEL_THRESHOLD:
        labels[i] = "SHORT"  # 0
    else:
        labels[i] = "HOLD"  # 0.5 or excluded
```

### Phase 4: Model Training

**4.1 Train all 4 models per architecture:**

| Model | Config | Purpose |
|-------|--------|---------|
| XGBoost | max_depth=3, n_est=150 | Gradient boosting |
| LightGBM | num_leaves=15, n_est=150 | Fast gradient boosting |
| LSTM | hidden=32, layers=1 | Sequential patterns |
| RandomForest | n_est=100, max_depth=10 | Ensemble baseline |

**4.2 Train/Validation Split:**
- 80% training (time-ordered)
- 20% validation (time-ordered)
- NO shuffle (preserve temporal order)

**4.3 Save models to:**
```
models/
  xgboost_hyperliquid_{symbol}.joblib
  lightgbm_hyperliquid_{symbol}.txt
  lstm_hyperliquid_{symbol}.pt
  random_forest_hyperliquid_{symbol}.joblib
```

### Phase 5: Multi-Asset Ensemble

**5.1 Option A: Single model for all assets**
- Train on combined data from all assets
- Simpler, may miss asset-specific patterns

**5.2 Option B: Per-asset models (Recommended)**
- Train separate models for each asset
- Captures asset-specific patterns
- More complex but more accurate

### Phase 6: Production Backtest

**6.1 Use `src/backtest/multi_asset_engine.py`**

The engine uses:
- `FeaturePipeline.calculate_features()` - same as training
- `MLEnsemble.predict()` - loads trained models
- `SignalProcessor.process_signal()` - production filters

**6.2 Verify parity:**
- Backtest uses EXACT same code as production
- Only data source differs (historical vs live)

## File Changes

| File | Action | Purpose |
|------|--------|---------|
| `scripts/hyperliquid_data_collector.py` | CREATE | Data collection |
| `scripts/train_hyperliquid_models.py` | CREATE | Model training |
| `scripts/hyperliquid_production_backtest.py` | MODIFY | Use trained models |
| `src/backtest/multi_asset_engine.py` | EXISTS | Already created |

## Verification

1. **Data Quality Check**
   - At least 1000 candles per asset
   - No gaps > 2 hours
   - Valid OHLCV (O <= H, L <= C, etc.)

2. **Model Quality Check**
   - Validation accuracy > 52%
   - No degenerate predictions (variance > 0.01)
   - AUC > 0.52

3. **Backtest Sanity Check**
   - Non-zero trades executed
   - Reasonable win rate (40-60%)
   - No obvious bugs (negative balance, etc.)

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Data Collection | 1 hour | Hyperliquid API |
| Phase 2: Feature Engineering | Existing | FeaturePipeline |
| Phase 3: Label Generation | 30 min | Data collected |
| Phase 4: Model Training | 2 hours | Labels generated |
| Phase 5: Multi-Asset Ensemble | 1 hour | Models trained |
| Phase 6: Production Backtest | 30 min | Ensemble ready |

**Total: ~5 hours**

## Success Criteria

1. All 4 models trained with >52% validation accuracy
2. Production backtest executes non-zero trades
3. Backtest results match expected range (not wildly different from prototype)
4. Code path identical between backtest and production
