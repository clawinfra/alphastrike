"""
AlphaStrike V2 — Redesigned Strategy Core

Key differences from V1:
1. Regression targets instead of binary classification
2. Two-stage stacking instead of naive ensemble
3. Unified regime detection (single source of truth)
4. Walk-forward validation (the only honest backtest)
5. Kelly-based position sizing
6. Correlation-aware portfolio risk
7. Mean reversion re-enabled for RANGE regime
8. Funding rate and realistic slippage in all calculations
"""
