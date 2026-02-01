# AlphaStrike Trading Bot - Product Requirements Document (PRD)

**Version:** 2.1
**Last Updated:** January 2026
**Status:** Production
**Owner:** Bowen Li

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals and Objectives](#3-goals-and-objectives)
4. [Target Users](#4-target-users)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Success Metrics](#7-success-metrics)
8. [Constraints and Assumptions](#8-constraints-and-assumptions)
9. [Risk Analysis](#9-risk-analysis)
10. [Appendices](#10-appendices)

---

## 1. Executive Summary

### 1.1 Product Vision

AlphaStrike is an autonomous algorithmic trading bot designed for cryptocurrency perpetual futures trading. The system features a multi-exchange abstraction layer supporting any CEX or DEX via adapter pattern, combined with machine learning ensemble predictions and adaptive risk management to generate profitable trading signals while maintaining strict risk controls.

### 1.2 Core Philosophy

> **"Trade Big, Trade Less, Trade with Confidence"**

The system prioritizes high-quality, high-conviction trades over frequent low-confidence trades, ensuring that each trade has meaningful profit potential relative to transaction costs.

### 1.3 Key Differentiators

| Feature | Description |
|---------|-------------|
| **ML Ensemble** | 4-model ensemble (XGBoost, LightGBM, LSTM, Random Forest) with dynamic weighting |
| **Regime Adaptation** | Automatic strategy adjustment based on market conditions |
| **Adaptive Risk** | Multi-layer risk management with position scaling |
| **Self-Healing** | Automatic model health detection and retraining |
| **Competition Compliance** | Full WEEX competition rule compliance |
| **Exchange Agnostic** | Adapter pattern supports any CEX/DEX via OpenAPI specs |

---

## 2. Problem Statement

### 2.1 Market Context

Cryptocurrency futures markets operate 24/7 with high volatility, requiring:
- Continuous monitoring impossible for human traders
- Rapid decision-making in volatile conditions
- Disciplined risk management to prevent catastrophic losses
- Adaptation to changing market regimes

### 2.2 Core Problems Addressed

| Problem | Impact | AlphaStrike Solution |
|---------|--------|---------------------|
| **Manual Trading Limitations** | Missed opportunities, emotional decisions | Autonomous 24/7 operation |
| **Market Regime Changes** | Strategy invalidation | Adaptive regime detection |
| **Risk Management** | Large drawdowns | Multi-layer risk controls |
| **Fee Erosion** | Profits consumed by fees | Fee-aware trading with minimum thresholds |
| **Model Degradation** | Declining performance | Self-healing with auto-retraining |

### 2.3 Production Learnings

From production deployment, key issues were identified and addressed:

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| 9.6% win rate | LightGBM degenerate trees | Balanced retraining |
| 985% fee/PnL ratio | Overtrading + small positions | Quality-based filtering |
| 185 trades/day | Per-symbol cooldown bypass | Global cooldown |
| Model SHORT bias | Training data imbalance | Balanced labels |

---

## 3. Goals and Objectives

### 3.1 Primary Goals

| Goal | Metric | Target |
|------|--------|--------|
| **Profitability** | Net PnL | Positive monthly returns |
| **Risk Control** | Maximum Drawdown | <15% total |
| **Win Rate** | Winning trades / Total trades | >55% |
| **Fee Efficiency** | Fees / Gross PnL | <30% |
| **Uptime** | System availability | >99.5% |

### 3.2 Secondary Goals

| Goal | Metric | Target |
|------|--------|--------|
| **Sharpe Ratio** | Risk-adjusted returns | >2.0 |
| **Profit Factor** | Gross profit / Gross loss | >1.5 |
| **Trade Quality** | Average profit per trade | >$2.00 |
| **Competition Compliance** | Rule violations | 0 |

### 3.3 Non-Goals (Explicitly Out of Scope)

- High-frequency trading (HFT) strategies
- Arbitrage between exchanges
- Manual intervention in trading decisions
- Social/copy trading features

---

## 4. Target Users

### 4.1 Primary User

**Algorithmic Trading Operator**
- Technical background in software/ML
- Understanding of financial markets
- Capable of monitoring and maintaining the system
- Responsible for capital allocation decisions

### 4.2 User Stories

#### US-001: Autonomous Trading
> As a trading operator, I want the bot to trade autonomously 24/7 so that I don't miss market opportunities while away.

**Acceptance Criteria:**
- System operates continuously without manual intervention
- Trades execute based on ML signals and risk rules
- All trades are logged with AI explanations
- Alerts sent for critical events

#### US-002: Risk Protection
> As a trading operator, I want the system to protect my capital through strict risk limits so that I don't experience catastrophic losses.

**Acceptance Criteria:**
- Daily drawdown limit of 5-7% triggers trading pause
- Total drawdown limit of 15% stops all trading
- Position limits prevent over-concentration
- Circuit breakers activate on extreme conditions

#### US-003: Performance Monitoring
> As a trading operator, I want real-time visibility into trading performance so that I can assess strategy effectiveness.

**Acceptance Criteria:**
- Dashboard shows live PnL, positions, and metrics
- AI logs explain every trading decision
- Performance reports generated daily/weekly
- Alerts for performance degradation

#### US-004: Regime Adaptation
> As a trading operator, I want the bot to adapt to different market conditions so that it remains profitable across market cycles.

**Acceptance Criteria:**
- System detects trending, ranging, and volatile markets
- Strategy parameters adjust based on regime
- Position sizing scales with confidence
- Leverage adapts to volatility

#### US-005: Competition Compliance
> As a trading operator, I want to ensure full compliance with WEEX competition rules so that I'm not disqualified.

**Acceptance Criteria:**
- Minimum 10 trades executed
- AI log submitted for every trade
- Maximum 20x leverage enforced
- No wash trading detected

#### US-006: Multi-Exchange Support
> As a trading operator, I want to deploy the bot on different exchanges (CEX or DEX) without modifying core trading logic.

**Acceptance Criteria:**
- Exchange selection via configuration (environment variable)
- Unified data models work across all exchanges
- Exchange-specific details encapsulated in adapters
- New exchanges added by providing OpenAPI specification
- Symbol normalization automatic (e.g., cmt_btcusdt ↔ BTCUSDT)

---

## 5. Functional Requirements

### 5.1 Signal Generation

#### FR-001: ML Ensemble Predictions
**Priority:** P0 (Critical)

The system SHALL generate trading signals using a 4-model ML ensemble:

| Model | Weight | Role |
|-------|--------|------|
| XGBoost | 30% | Primary trend prediction |
| LightGBM | 25% | Fast gradient boosting |
| LSTM | 25% | Sequence pattern recognition |
| Random Forest | 20% | Robust ensemble member |

**Requirements:**
- FR-001.1: Ensemble outputs weighted average prediction (0-1 scale)
- FR-001.2: Confidence calculated using threshold-relative formula
- FR-001.3: Minimum 2 healthy models required for signal generation
- FR-001.4: Dynamic weight adjustment based on recent performance

#### FR-002: Signal Thresholds
**Priority:** P0 (Critical)

The system SHALL use adaptive thresholds for signal generation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| LONG Threshold | 0.75 | weighted_avg > 0.75 for LONG |
| SHORT Threshold | 0.25 | weighted_avg < 0.25 for SHORT |
| High Confidence | 0.75 | Minimum for full position |
| Adaptive Range | 0.65-0.85 | Regime-adjusted thresholds |

#### FR-003: Signal Filters
**Priority:** P0 (Critical)

The system SHALL apply the following signal filters:

| Filter | Condition | Action |
|--------|-----------|--------|
| RSI Extreme | RSI < 30 for SHORT, RSI > 70 for LONG | Block signal |
| ADX Saturation | ADX > 95 or ADX < 10 | Block signal |
| Regime Alignment | Counter-trend in trending market | Scale position 40% |
| MTF Misalignment | 4H trend opposes signal | Scale position 50% |
| Model Agreement | Agreement < 25% | Treat as HOLD |

### 5.2 Feature Pipeline

#### FR-004: Canonical Features
**Priority:** P0 (Critical)

The system SHALL calculate and maintain 86 canonical features:

| Category | Count | Examples |
|----------|-------|----------|
| Technical | 35 | RSI(7,14,21), ADX, ATR, MACD, BB |
| Microstructure | 7 | Orderbook imbalance, funding rate |
| Fee Features | 5 | Maker/taker fees, breakeven |
| Cross-Asset | 3 | BTC correlation, ETH/BTC ratio |
| Time-Based | 5 | Session indicators, cyclical encoding |
| Volatility | 4 | ATR ratio, realized vol, regime |

#### FR-005: Feature Stability Monitoring
**Priority:** P1 (High)

The system SHALL monitor feature distributions:
- FR-005.1: Track feature distributions using KS-test
- FR-005.2: Alert when distribution shift exceeds threshold
- FR-005.3: Automatic feature retirement if importance drops

### 5.3 Risk Management

#### FR-006: Position Sizing
**Priority:** P0 (Critical)

The system SHALL use Half-Kelly position sizing with adaptive parameters:

| Confidence | Leverage | Position % |
|------------|----------|------------|
| >= 0.85 | 5x | 40% |
| >= 0.80 | 4x | 35% |
| >= 0.75 | 3x | 30% |
| < 0.75 | - | No trade |

#### FR-007: Exposure Limits
**Priority:** P0 (Critical)

The system SHALL enforce exposure limits:

| Limit Type | Base Value | Adaptive Range |
|------------|------------|----------------|
| Per-Trade | 10% | 5-20% |
| Per-Pair | 25% | 15-45% |
| Total | 80% | 30-95% |
| Pyramiding | 50% | When profitable |

#### FR-008: Drawdown Controls
**Priority:** P0 (Critical)

The system SHALL implement drawdown controls:

| Control | Threshold | Action |
|---------|-----------|--------|
| Daily Drawdown | 5% (adaptive) | Pause trading 10 minutes |
| Total Drawdown | 15% | Stop all trading |
| Minimum Balance | $9,000 | Close all positions |

#### FR-009: Circuit Breakers
**Priority:** P0 (Critical)

The system SHALL implement circuit breakers:
- FR-009.1: Maximum 20x leverage hard cap
- FR-009.2: Close orders never blocked by risk checks
- FR-009.3: Orphan position detection and reconciliation
- FR-009.4: API failure graceful degradation

### 5.4 Exit Management

#### FR-010: Multi-Level Take Profit
**Priority:** P0 (Critical)

The system SHALL implement 3-level take profit:

| Level | ATR Multiplier | Position % | Trigger |
|-------|---------------|------------|---------|
| TP1 | 1.5-2.0x | 40% | First target |
| TP2 | 2.5-3.0x | 35% | Extended move |
| TP3 | Trailing | 25% | Let winner run |

#### FR-011: Trailing Stop
**Priority:** P0 (Critical)

The system SHALL implement ATR-based trailing stops:
- FR-011.1: Initial stop at 2.0x ATR from entry
- FR-011.2: Trailing distance of 1.5x ATR
- FR-011.3: Break-even + 0.3x ATR buffer after TP1
- FR-011.4: Minimum 15-minute hold before trailing activates

#### FR-012: Time-Based Exits
**Priority:** P1 (High)

The system SHALL implement time-based exit rules:
- FR-012.1: Maximum hold time 24-96 hours (adaptive)
- FR-012.2: Extension for profitable trades
- FR-012.3: Extension for trending markets

### 5.5 Order Execution

#### FR-013: Order Types
**Priority:** P0 (Critical)

The system SHALL select optimal order types:

| Scenario | Order Type |
|----------|------------|
| High urgency signal | Market order |
| Wide spread (>0.05%) | Limit order at mid |
| Large order (>2% of book) | Split + TWAP |
| Stop loss | Market order |
| Take profit | Limit order |

#### FR-014: Slippage Optimization
**Priority:** P1 (High)

The system SHALL implement slippage optimization:
- FR-014.1: Pre-trade slippage estimation from orderbook
- FR-014.2: Post-trade slippage tracking and analysis
- FR-014.3: Slippage model retraining based on actual fills

#### FR-015: Exchange-Side Orders
**Priority:** P0 (Critical)

The system SHALL place exchange-side protective orders:
- FR-015.1: Preset stop-loss at order entry
- FR-015.2: Exchange-side TP orders (profit_plan)
- FR-015.3: Order update rate limiting (30-120 seconds)

### 5.6 Exchange Abstraction Layer

#### FR-016: Exchange Adapter Pattern
**Priority:** P0 (Critical)

The system SHALL provide a unified exchange interface via adapter pattern:

| Component | Purpose |
|-----------|---------|
| `ExchangeRESTProtocol` | Abstract interface for REST operations |
| `ExchangeWebSocketProtocol` | Abstract interface for real-time data |
| `ExchangeAdapter` | Combines REST and WebSocket with symbol normalization |
| `ExchangeFactory` | Runtime adapter creation from configuration |

**Requirements:**
- FR-016.1: All trading logic uses unified data models (UnifiedOrder, UnifiedPosition, etc.)
- FR-016.2: Symbol format normalized to `BTCUSDT` across all exchanges
- FR-016.3: Exchange selection via `EXCHANGE_NAME` environment variable
- FR-016.4: Adapter registration allows adding new exchanges without core changes

#### FR-017: Unified Data Models
**Priority:** P0 (Critical)

The system SHALL use exchange-agnostic data models:

| Model | Purpose |
|-------|---------|
| `UnifiedOrder` | Order request with standardized fields |
| `UnifiedOrderResult` | Order response/fill information |
| `UnifiedPosition` | Position state across exchanges |
| `UnifiedAccountBalance` | Account balance with margin info |
| `UnifiedTicker` | Real-time price data |
| `UnifiedOrderbook` | Bid/ask depth |
| `UnifiedCandle` | OHLCV data |

#### FR-018: OpenAPI Integration
**Priority:** P1 (High)

The system SHALL support OpenAPI-driven adapter generation:
- FR-018.1: Parse OpenAPI 3.x specifications
- FR-018.2: Auto-map endpoints to protocol methods
- FR-018.3: Generate adapter skeleton code
- FR-018.4: Confidence scoring for endpoint mappings

### 5.7 Market Regime Detection

#### FR-019: Regime Classification
**Priority:** P0 (Critical)

The system SHALL classify market regimes:

| Regime | Detection Criteria | Strategy Adjustment |
|--------|-------------------|---------------------|
| Trending Up | ADX > 25, Price > EMA20, +DI > -DI | 1.2x position, wider stops |
| Trending Down | ADX > 25, Price < EMA20, -DI > +DI | 1.2x position, wider stops |
| Ranging | ADX < 20 | 0.8x position, tighter stops |
| High Volatility | ATR Ratio > 1.5 | 0.6x position, 2x wider stops |
| Extreme Volatility | ATR Ratio > 2.0 | 0.3x position, minimal trading |
| Trend Exhaustion | ADX > 60 + ROC < -2 | Block new entries |

### 5.8 Model Management

#### FR-020: Self-Healing Models
**Priority:** P1 (High)

The system SHALL implement self-healing model management:
- FR-017.1: Health check on every prediction (degenerate detection)
- FR-017.2: Unhealthy counter threshold (50 consecutive checks)
- FR-017.3: Automatic corrupted model deletion
- FR-017.4: Force retrain signal for trainer subprocess

#### FR-021: Model Retraining
**Priority:** P1 (High)

The system SHALL implement automatic retraining:
- FR-018.1: Dynamic retraining interval (13-90 minutes based on volatility)
- FR-018.2: Out-of-sample validation before deployment
- FR-018.3: Automatic rollback if new model underperforms
- FR-018.4: Balanced training data (equal LONG/SHORT labels)

### 5.9 Competition Compliance

#### FR-022: WEEX Competition Rules
**Priority:** P0 (Critical)

The system SHALL comply with WEEX competition rules:
- FR-019.1: Minimum 10 trades executed
- FR-019.2: AI log submitted for every trade decision
- FR-019.3: Maximum leverage 20x enforced
- FR-019.4: No wash trading (WashTradingPrevention class)
- FR-019.5: Log completeness validation

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Requirement | Metric | Target |
|-------------|--------|--------|
| NFR-001 | Prediction latency | <500ms |
| NFR-002 | Feature calculation | <700ms |
| NFR-003 | Order placement | <2s |
| NFR-004 | WebSocket latency | <100ms |
| NFR-005 | Memory usage | <2GB |
| NFR-006 | CPU usage | <80% sustained |

### 6.2 Reliability

| Requirement | Metric | Target |
|-------------|--------|--------|
| NFR-007 | System uptime | >99.5% |
| NFR-008 | Data feed availability | >99.9% |
| NFR-009 | Order execution success | >99% |
| NFR-010 | Position sync accuracy | 100% |

### 6.3 Scalability

| Requirement | Description |
|-------------|-------------|
| NFR-011 | Support 8 trading pairs concurrently |
| NFR-012 | Handle 1000+ signals per hour |
| NFR-013 | Process 100+ candles per symbol |
| NFR-014 | Store 30 days of historical data |

### 6.4 Security

| Requirement | Description |
|-------------|-------------|
| NFR-015 | API keys encrypted at rest |
| NFR-016 | No credentials in logs |
| NFR-017 | Rate limiting on API calls |
| NFR-018 | Audit trail for all trades |

### 6.5 Observability

| Requirement | Description |
|-------------|-------------|
| NFR-019 | Structured logging (JSON format) |
| NFR-020 | Real-time metrics dashboard |
| NFR-021 | Alert system for critical events |
| NFR-022 | AI log explanations for every trade |

---

## 7. Success Metrics

### 7.1 Key Performance Indicators (KPIs)

| KPI | Definition | Target | Measurement |
|-----|------------|--------|-------------|
| **Net PnL** | Total profit after fees | Positive monthly | Daily tracking |
| **Win Rate** | Winning trades / Total | >55% | Rolling 100 trades |
| **Profit Factor** | Gross profit / Gross loss | >1.5 | Rolling 30 days |
| **Max Drawdown** | Peak-to-trough decline | <15% | Continuous |
| **Fee Efficiency** | Fees / Gross profit | <30% | Rolling 7 days |
| **Trade Quality** | Average profit per trade | >$2.00 | Rolling 50 trades |
| **Sharpe Ratio** | Risk-adjusted return | >2.0 | Rolling 30 days |

### 7.2 Operational Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| System uptime | >99.5% | <95% |
| Model health | 4/4 healthy | <2 healthy |
| Position sync | 100% accurate | Any mismatch |
| API success rate | >99% | <95% |
| Memory usage | <2GB | >1.8GB |

### 7.3 Success Criteria for Launch

| Criteria | Requirement |
|----------|-------------|
| Paper trading | 7 days profitable |
| Win rate | >50% over 100 trades |
| Max drawdown | <10% during testing |
| System stability | No crashes in 72 hours |
| Competition compliance | All rules validated |

---

## 8. Constraints and Assumptions

### 8.1 Technical Constraints

| Constraint | Description |
|------------|-------------|
| Exchange | Configurable via adapter pattern (WEEX, Hyperliquid, Binance, etc.) |
| Leverage | Maximum 20x (competition limit, configurable per exchange) |
| Pairs | 8 approved pairs for WEEX competition |
| Order limit | 200 active orders on exchange |
| API rate | Subject to exchange-specific rate limits |

### 8.2 Business Constraints

| Constraint | Description |
|------------|-------------|
| Capital | $1,000-$10,000 initial |
| Competition period | Fixed duration |
| Minimum trades | 10 trades required |
| AI logging | Mandatory for all trades |

### 8.3 Assumptions

| Assumption | Risk if Invalid |
|------------|-----------------|
| WEEX API remains stable | System unusable |
| Market has tradeable volatility | No profit opportunities |
| ML models generalize to future data | Strategy failure |
| Historical patterns repeat | Backtests invalid |
| Funding rates remain reasonable | Margin erosion |

---

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model overfitting | Medium | High | Out-of-sample validation, rolling retraining |
| API failure | Low | High | Retry logic, graceful degradation |
| Data feed lag | Medium | Medium | Timestamp validation, stale data rejection |
| Position sync failure | Low | Critical | Periodic reconciliation, alerts |

### 9.2 Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Flash crash | Low | Critical | Circuit breakers, exchange stops |
| Regime change | High | Medium | Regime detection, adaptive parameters |
| Correlation breakdown | Medium | High | Correlation limits, stress testing |
| Liquidity crisis | Low | High | Order size limits, slippage checks |

### 9.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Server downtime | Low | High | Cloud hosting, auto-restart |
| Memory leak | Medium | Medium | Periodic cleanup, monitoring |
| Log storage full | Low | Medium | Auto-cleanup, retention policy |
| Configuration error | Medium | High | Validation, safe defaults |

---

## 10. Appendices

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **ATR** | Average True Range - volatility measure |
| **ADX** | Average Directional Index - trend strength |
| **Kelly Criterion** | Position sizing formula based on edge |
| **Pyramiding** | Adding to winning positions |
| **Regime** | Market condition classification |
| **Slippage** | Difference between expected and actual fill price |
| **TWAP** | Time-Weighted Average Price execution |

### 10.2 Approved Trading Pairs

1. BTC/USDT
2. ETH/USDT
3. BNB/USDT
4. SOL/USDT
5. XRP/USDT
6. DOGE/USDT
7. LTC/USDT
8. ADA/USDT

### 10.3 Configuration Reference

See `docs/ARCHITECTURE.md` for detailed configuration reference.

### 10.4 Related Documents

- [Architecture Document](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*Document History:*
- v1.0 (December 2025): Initial release
- v2.0 (January 2026): Updated based on production learnings
- v2.1 (January 2026): Added multi-exchange abstraction layer (FR-016 through FR-018)
  - Exchange adapter pattern with unified protocols
  - Unified data models for exchange-agnostic trading
  - OpenAPI integration for automatic adapter generation
  - Removed "WEEX only" constraint - now supports any CEX/DEX
