# AlphaStrike Trading Bot - Architecture Document

**Version:** 2.4
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

AlphaStrike is an autonomous algorithmic trading system for cryptocurrency perpetual futures. It features a multi-exchange abstraction layer supporting any CEX or DEX via adapter pattern, combined with ML-based signal generation and adaptive risk management.

### 1.2 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Modularity** | Independent components with clear interfaces |
| **Adaptability** | Regime-aware strategy adjustment |
| **Resilience** | Self-healing models, graceful degradation |
| **Observability** | Comprehensive logging, AI explanations |
| **Safety** | Multi-layer risk controls, circuit breakers |
| **Exchange Agnostic** | Adapter pattern decouples trading logic from exchange specifics |

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
│  │ EXCHANGE    │──▶│DATA GATEWAY │──▶│FEATURE LAYER│──▶│  FEATURE    │         │
│  │ ABSTRACTION │   │             │   │             │   │  VALIDATOR  │         │
│  │             │   │ • Staleness │   │ • Technical │   │             │         │
│  │ • Adapters  │   │ • Sequence  │   │ • Micro-    │   │ • PSI       │         │
│  │ • Protocols │   │ • Range     │   │   structure │   │ • KS Test   │         │
│  │ • Factory   │   │ • Circuit   │   │ • Cross-    │   │ • CUSUM     │         │
│  │ • Models    │   │   Breaker   │   │   Asset     │   │ • Z-Score   │         │
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
│  │                      EXCHANGE ABSTRACTION LAYER                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │ WEEX Adapter │  │ Hyperliquid  │  │   Binance    │  │   Generic   │  │   │
│  │  │              │  │   Adapter    │  │   Adapter    │  │   OpenAPI   │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Exchange Abstraction Layer

The exchange abstraction layer provides a unified interface for interacting with any cryptocurrency exchange (CEX or DEX). It decouples trading logic from exchange-specific details through the adapter pattern.

#### 3.1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXCHANGE ABSTRACTION LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Trading Logic (order_manager.py, etc.)                        │
│           │                                                      │
│           ▼                                                      │
│   ┌───────────────────────────────────────────────────────┐     │
│   │              UNIFIED PROTOCOLS                         │     │
│   │                                                        │     │
│   │   ┌──────────────────┐  ┌──────────────────────┐      │     │
│   │   │ExchangeRESTProto │  │ExchangeWebSocketProto│      │     │
│   │   │                  │  │                      │      │     │
│   │   │ • get_ticker()   │  │ • connect()          │      │     │
│   │   │ • place_order()  │  │ • subscribe_candles()│      │     │
│   │   │ • get_positions()│  │ • on_candle()        │      │     │
│   │   │ • get_balance()  │  │ • on_ticker()        │      │     │
│   │   └──────────────────┘  └──────────────────────┘      │     │
│   └───────────────────────────────────────────────────────┘     │
│                           │                                      │
│                           ▼                                      │
│   ┌───────────────────────────────────────────────────────┐     │
│   │                EXCHANGE FACTORY                        │     │
│   │                                                        │     │
│   │  get_exchange_adapter(name: str) -> ExchangeAdapter   │     │
│   │  • Reads EXCHANGE_NAME from config                    │     │
│   │  • Returns appropriate adapter instance               │     │
│   └───────────────────────────────────────────────────────┘     │
│                           │                                      │
│              ┌────────────┼────────────┬────────────┐           │
│              │            │            │            │           │
│              ▼            ▼            ▼            ▼           │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐│
│   │WEEX Adapter  │ │Hyperliquid  │ │Binance       │ │Generic ││
│   │              │ │Adapter       │ │Adapter       │ │OpenAPI ││
│   │ • mappers.py │ │              │ │              │ │Adapter ││
│   │ • adapter.py │ │ • signer.py  │ │              │ │        ││
│   │ • websocket  │ │ • wallet     │ │              │ │        ││
│   └──────────────┘ └──────────────┘ └──────────────┘ └────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 Unified Protocols (`src/exchange/protocols.py`)

**Purpose:** Define abstract interfaces that all exchange adapters must implement.

**Key Protocols:**

```python
@runtime_checkable
class ExchangeRESTProtocol(Protocol):
    """REST API interface for any exchange."""

    @property
    def exchange_name(self) -> str: ...
    @property
    def capabilities(self) -> ExchangeCapabilities: ...

    # Market Data
    async def get_ticker(self, symbol: str) -> UnifiedTicker: ...
    async def get_orderbook(self, symbol: str, limit: int = 20) -> UnifiedOrderbook: ...
    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[UnifiedCandle]: ...
    async def get_funding_rate(self, symbol: str) -> float: ...
    async def get_symbol_info(self, symbol: str) -> SymbolInfo: ...

    # Account
    async def get_account_balance(self) -> UnifiedAccountBalance: ...
    async def get_positions(self, symbol: str | None = None) -> list[UnifiedPosition]: ...

    # Orders
    async def place_order(self, order: UnifiedOrder) -> UnifiedOrderResult: ...
    async def cancel_order(self, symbol: str, order_id: str) -> bool: ...
    async def get_open_orders(self, symbol: str | None = None) -> list[UnifiedOrderResult]: ...

    # Leverage & Stops
    async def set_leverage(self, symbol: str, leverage: int) -> bool: ...
    async def place_stop_loss(...) -> UnifiedOrderResult: ...
    async def place_take_profit(...) -> UnifiedOrderResult: ...

@runtime_checkable
class ExchangeWebSocketProtocol(Protocol):
    """WebSocket interface for real-time data."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def subscribe_candles(self, symbols: list[str], interval: str) -> None: ...
    async def subscribe_tickers(self, symbols: list[str]) -> None: ...
    def on_candle(self, callback: Callable[[UnifiedCandle], None]) -> None: ...
    def on_ticker(self, callback: Callable[[UnifiedTicker], None]) -> None: ...

class ExchangeAdapter(ABC):
    """Combined adapter providing REST and WebSocket access."""

    @property
    def rest(self) -> ExchangeRESTProtocol: ...
    @property
    def websocket(self) -> ExchangeWebSocketProtocol: ...
    def normalize_symbol(self, symbol: str) -> str: ...  # BTCUSDT -> exchange format
    def denormalize_symbol(self, symbol: str) -> str: ...  # exchange format -> BTCUSDT
```

#### 3.1.3 Unified Data Models (`src/exchange/models.py`)

**Purpose:** Exchange-agnostic data structures used throughout the system.

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `UnifiedOrder` | Order request | symbol, side, type, size, price, leverage |
| `UnifiedOrderResult` | Order response/fill | order_id, status, filled_size, avg_price |
| `UnifiedPosition` | Position state | symbol, side, size, entry_price, unrealized_pnl |
| `UnifiedAccountBalance` | Account info | total_balance, available_balance, margin_used |
| `UnifiedTicker` | Price data | symbol, bid, ask, last, volume_24h |
| `UnifiedOrderbook` | Market depth | bids, asks, timestamp |
| `UnifiedCandle` | OHLCV data | open, high, low, close, volume, timestamp |
| `UnifiedTrade` | Trade execution | price, size, side, timestamp |
| `SymbolInfo` | Contract details | tick_size, lot_size, min_notional |
| `ExchangeCapabilities` | Exchange features | supports_stop_loss, max_leverage |

**Enums:**

| Enum | Values |
|------|--------|
| `OrderSide` | BUY, SELL |
| `OrderType` | MARKET, LIMIT |
| `PositionSide` | LONG, SHORT |
| `TimeInForce` | GTC, IOC, FOK |
| `OrderStatus` | PENDING, OPEN, FILLED, CANCELLED, FAILED |
| `MarginMode` | CROSS, ISOLATED |

#### 3.1.4 Exchange Factory (`src/exchange/factory.py`)

**Purpose:** Runtime adapter creation and registration.

```python
class ExchangeFactory:
    """Factory for creating exchange adapters."""

    _adapters: dict[str, type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeAdapter]) -> None:
        """Register an adapter class for an exchange."""
        cls._adapters[name.lower()] = adapter_class

    async def create(self, exchange_name: str | None = None) -> ExchangeAdapter:
        """Create and initialize an exchange adapter."""
        name = exchange_name or get_settings().exchange.name.value
        adapter_class = self._adapters.get(name.lower())
        if not adapter_class:
            raise ValueError(f"Unknown exchange: {name}")
        adapter = adapter_class()
        await adapter.initialize()
        return adapter

async def get_exchange_adapter(exchange_name: str | None = None) -> ExchangeAdapter:
    """Convenience function to get an exchange adapter."""
    factory = ExchangeFactory()
    return await factory.create(exchange_name)
```

**Adapter Registration:**

Each adapter registers itself on import:

```python
# src/exchange/adapters/weex/__init__.py
from src.exchange.adapters.weex.adapter import WEEXAdapter
from src.exchange.factory import register_adapter

register_adapter("weex", WEEXAdapter)
```

#### 3.1.5 WEEX Adapter (`src/exchange/adapters/weex/`)

**Purpose:** WEEX-specific implementation of the exchange protocols.

**Components:**

| File | Purpose |
|------|---------|
| `adapter.py` | `WEEXAdapter` combining REST and WebSocket |
| `mappers.py` | Translation between unified models and WEEX formats |
| `websocket.py` | WEEX WebSocket implementation |

**Symbol Normalization:**

| Direction | Example |
|-----------|---------|
| Normalize (unified → WEEX) | `BTCUSDT` → `cmt_btcusdt` |
| Denormalize (WEEX → unified) | `cmt_btcusdt` → `BTCUSDT` |

**Key Mapper Methods:**

```python
class WEEXMapper:
    @staticmethod
    def to_weex_symbol(unified_symbol: str) -> str:
        """BTCUSDT -> cmt_btcusdt"""
        return f"cmt_{unified_symbol.lower()}"

    @staticmethod
    def from_weex_symbol(weex_symbol: str) -> str:
        """cmt_btcusdt -> BTCUSDT"""
        return weex_symbol.replace("cmt_", "").upper()

    @staticmethod
    def to_unified_order_result(weex_response: dict) -> UnifiedOrderResult: ...

    @staticmethod
    def to_unified_position(weex_position: dict) -> UnifiedPosition: ...
```

#### 3.1.6 OpenAPI Integration (`src/exchange/openapi/`)

**Purpose:** Parse OpenAPI specifications and generate adapter mappings.

**Components:**

| File | Purpose |
|------|---------|
| `parser.py` | Parse OpenAPI 3.x specifications |
| `mapper.py` | Map endpoints to protocol methods |

**OpenAPI Parser:**

```python
class OpenAPIParser:
    def parse_file(self, path: str | Path) -> ParsedOpenAPISpec:
        """Parse an OpenAPI specification file."""

    def parse_url(self, url: str) -> ParsedOpenAPISpec:
        """Fetch and parse an OpenAPI spec from URL."""

@dataclass
class ParsedOpenAPISpec:
    title: str
    version: str
    base_url: str
    endpoints: list[APIEndpoint]
    schemas: dict[str, dict]
    authentication: list[AuthenticationInfo]
```

**Protocol Mapper:**

```python
class ProtocolMapper:
    """Maps OpenAPI endpoints to ExchangeRESTProtocol methods."""

    def generate_mapping(self, spec: ParsedOpenAPISpec) -> ProtocolMappingResult:
        """Analyze spec and suggest endpoint-to-protocol mappings."""

    def print_mapping_report(self, result: ProtocolMappingResult) -> None:
        """Print human-readable mapping report."""

@dataclass
class ProtocolMappingResult:
    spec_title: str
    base_url: str
    mappings: dict[str, EndpointMapping]  # protocol method -> endpoint
    unmapped_endpoints: list[APIEndpoint]
    missing_protocol_methods: list[str]

    def get_coverage(self) -> float:
        """Get protocol coverage percentage."""

    def generate_adapter_skeleton(self) -> str:
        """Generate starter code for a new adapter."""
```

**Endpoint Matching:**

The mapper uses keyword matching, HTTP method validation, and path analysis to suggest mappings:

| Protocol Method | Keywords | Expected HTTP Method |
|-----------------|----------|---------------------|
| `get_ticker` | ticker, price, quote | GET |
| `place_order` | order, place, create | POST |
| `get_positions` | position, positions | GET |
| `set_leverage` | leverage, set | POST |

#### 3.1.7 Exception Hierarchy (`src/exchange/exceptions.py`)

**Purpose:** Unified exception types for exchange operations.

```python
class ExchangeError(Exception):
    """Base exception for all exchange errors."""

class AuthenticationError(ExchangeError):
    """API key/secret invalid or expired."""

class RateLimitError(ExchangeError):
    """Rate limit exceeded."""

class InsufficientBalanceError(ExchangeError):
    """Not enough balance for operation."""

class OrderError(ExchangeError):
    """Base class for order-related errors."""

class InvalidOrderError(OrderError):
    """Order parameters invalid."""

class OrderNotFoundError(OrderError):
    """Order ID not found."""

class PositionError(ExchangeError):
    """Position-related error."""

class SymbolNotFoundError(ExchangeError):
    """Trading pair not found on exchange."""
```

#### 3.1.8 Configuration (`src/core/config.py`)

**Exchange Configuration:**

```python
class ExchangeType(str, Enum):
    WEEX = "weex"
    HYPERLIQUID = "hyperliquid"
    BINANCE = "binance"

class ExchangeConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXCHANGE_")

    name: ExchangeType = ExchangeType.WEEX
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""  # WEEX, some CEXs
    wallet_private_key: str = ""  # For DEX
    rest_url: str = ""  # Override default
    ws_url: str = ""
    openapi_spec_path: str = ""  # For generic adapter
    rate_limit_requests: int = 100
```

**Environment Variables:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `EXCHANGE_NAME` | Select exchange | `weex`, `hyperliquid` |
| `EXCHANGE_API_KEY` | API key | `xxx-xxx` |
| `EXCHANGE_API_SECRET` | API secret | `xxx` |
| `EXCHANGE_API_PASSPHRASE` | Passphrase (if required) | `xxx` |
| `EXCHANGE_WALLET_PRIVATE_KEY` | DEX wallet key | `0x...` |

#### 3.1.9 Usage Example

```python
from src.exchange import get_exchange_adapter, UnifiedOrder, OrderSide, OrderType

async def main():
    # Get adapter (uses EXCHANGE_NAME from config)
    adapter = await get_exchange_adapter()

    # Or specify explicitly
    adapter = await get_exchange_adapter("weex")

    # Use unified interface
    balance = await adapter.rest.get_account_balance()
    print(f"Balance: ${balance.total_balance:.2f}")

    # Place order using unified model
    order = UnifiedOrder(
        symbol="BTCUSDT",  # Unified format
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        size=0.01,
        leverage=5,
    )
    result = await adapter.rest.place_order(order)
    print(f"Order {result.order_id}: {result.status}")

    # Symbol normalization is automatic
    positions = await adapter.rest.get_positions("BTCUSDT")
    for pos in positions:
        print(f"{pos.symbol}: {pos.size} @ {pos.entry_price}")
```

### 3.2 Data Layer

#### 3.2.1 WebSocket Client (`src/data/websocket_client.py`)

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

#### 3.2.2 REST Client (`src/data/rest_client.py`) [Deprecated]

> **Note:** The REST client in `src/data/rest_client.py` is deprecated. New code should use the exchange abstraction layer via `get_exchange_adapter()`. The legacy client remains for backwards compatibility.

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

#### 3.2.3 Database (`src/data/database.py`)

**Purpose:** Persistent storage

**Tables:**
| Table | Purpose |
|-------|---------|
| `candles` | Historical OHLCV data |
| `trades` | Executed trade records |
| `ai_log_uploads` | AI explanations |
| `training_cache` | Model training data |
| `performance_metrics` | Rolling metrics |

#### 3.2.4 Data Gateway (`src/data/data_gateway.py`)

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

### 3.3 Feature Layer

#### 3.3.1 Feature Pipeline (`src/features/pipeline.py`)

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

#### 3.3.2 Technical Features (`src/features/technical.py`)

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

#### 3.3.3 Microstructure Features (`src/features/microstructure.py`)

**Features:**
- `orderbook_imbalance` - Bid/ask depth ratio
- `funding_rate` - Current funding rate
- `open_interest` - Total OI
- `volume_profile` - Recent trade distribution
- `large_trade_ratio` - Whale activity detection

#### 3.3.4 Feature Validation (`src/features/feature_validator.py`)

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

### 3.4 ML Layer

#### 3.4.1 Ensemble (`src/ml/ensemble.py`)

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

#### 3.4.2 Individual Models

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

#### 3.4.3 Trainer (`src/ml/trainer.py`)

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

#### 3.4.4 Prediction Confidence Bounds (`src/ml/confidence_filter.py`)

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

#### 3.4.5 Model Quality Tracker (`src/ml/quality_tracker.py`)

**Purpose:** Track rolling prediction accuracy per model and dynamically adjust ensemble weights based on performance, using graduated reduction instead of binary exclusion.

**Why Graduated Weight Reduction (NOT Binary Exclusion):**

| Approach | Problem | Consequence |
|----------|---------|-------------|
| Binary exclusion | Remove model at accuracy < 50% | Lose diversification, regime-blind |
| Graduated weights | Scale weight 0-100% by accuracy | Preserve ensemble benefits |

Key insight: A model with 48% accuracy in trending markets might have 62% accuracy in ranging markets. Binary exclusion would lose this regime-specific value.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                  MODEL QUALITY TRACKER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   For each prediction:                                          │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              RECORD PREDICTION                           │   │
│   │  • model_name, predicted_signal, regime, confidence     │   │
│   │  • Added to pending queue (awaiting outcome)            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   When outcome known (price moved):                             │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              RECORD OUTCOME                              │   │
│   │  • Match to oldest pending prediction (FIFO)            │   │
│   │  • Calculate correctness                                 │   │
│   │  • Update rolling window (100 predictions)              │   │
│   │  • Update regime-specific accuracy                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   When ensemble predicts:                                        │
│       │                                                          │
│       ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │           GET ADJUSTED WEIGHTS                           │   │
│   │                                                          │   │
│   │   Base weight (30%)                                      │   │
│   │        │                                                 │   │
│   │        ▼                                                 │   │
│   │   ┌───────────────┐                                     │   │
│   │   │ Accuracy      │  >55% → 100% weight                 │   │
│   │   │ Multiplier    │  50-55% → 75% weight                │   │
│   │   │               │  45-50% → 50% weight                │   │
│   │   │               │  <45% → 0% weight (excluded)        │   │
│   │   └───────┬───────┘                                     │   │
│   │           │                                              │   │
│   │           ▼                                              │   │
│   │   ┌───────────────┐                                     │   │
│   │   │ Regime Boost/ │  >60% in regime → 1.2x boost       │   │
│   │   │ Penalty       │  <40% in regime → 0.7x penalty     │   │
│   │   └───────┬───────┘                                     │   │
│   │           │                                              │   │
│   │           ▼                                              │   │
│   │   ┌───────────────┐                                     │   │
│   │   │ Diversity     │  Always keep ≥2 models active      │   │
│   │   │ Protection    │  Restore best excluded if needed   │   │
│   │   └───────────────┘                                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Graduated Weight Scaling:**

| Overall Accuracy | Status | Weight Multiplier | Rationale |
|-----------------|--------|-------------------|-----------|
| ≥ 55% | ACTIVE | 100% | Full confidence |
| 50-55% | DEGRADED | 75% | Slightly reduced |
| 45-50% | DEGRADED | 50% | Significantly reduced |
| < 45% (persistent) | EXCLUDED | 0% | Candidate for exclusion |

**Regime-Specific Adjustments:**

| Regime Accuracy | Action | Example |
|-----------------|--------|---------|
| ≥ 60% | +20% boost | Model excels in current regime |
| ≤ 40% | -30% penalty | Model struggles in current regime |

**Diversity Protection:**

The tracker ensures we never exclude too many models:

```python
min_active_models = 2  # Always keep at least 2 active

# If exclusion would leave < 2 models:
# 1. Sort excluded models by accuracy
# 2. Restore best performers with 50% weight
# 3. Mark as DEGRADED (not ACTIVE)
```

**Recovery Mechanism:**

Excluded models can recover if recent accuracy improves:

```python
recovery_threshold = 0.52  # 52% accuracy needed
recovery_window = 30       # Evaluated over last 30 predictions

# If excluded model achieves 52%+ over last 30 predictions:
# 1. Restore to DEGRADED status
# 2. Start with 50% weight
# 3. Can return to ACTIVE if accuracy continues improving
```

**Key Classes:**

```python
class ModelStatus(Enum):
    ACTIVE = "active"      # Full weight
    DEGRADED = "degraded"  # Reduced weight
    EXCLUDED = "excluded"  # Zero weight (can recover)
    FAILED = "failed"      # Health check failure

class ModelQualityTracker:
    def record_prediction(
        model_name: str,
        predicted_signal: SignalType,
        regime: RegimeType,
        confidence: float
    ) -> None

    def record_outcome(
        model_name: str,
        actual_outcome: SignalType
    ) -> None

    def get_adjusted_weights(
        base_weights: dict[str, float],
        current_regime: RegimeType,
        health_status: dict[str, bool]
    ) -> dict[str, float]

    def check_recovery(model_name: str) -> bool

@dataclass
class QualityTrackerConfig:
    rolling_window: int = 100
    full_weight_threshold: float = 0.55
    degraded_threshold: float = 0.50
    reduced_threshold: float = 0.45
    exclusion_threshold: float = 0.45
    min_predictions_for_exclusion: int = 50
    min_active_models: int = 2
    recovery_threshold: float = 0.52
    recovery_window: int = 30
```

**Integration with MLEnsemble:**

```python
class MLEnsemble:
    def __init__(self, ...):
        self._quality_tracker = ModelQualityTracker()

    def predict(self, features: dict, current_regime: RegimeType):
        # 1. Get base redistributed weights
        base_weights = self._redistribute_weights(unhealthy)

        # 2. Apply quality-adjusted weights
        effective_weights = self._quality_tracker.get_adjusted_weights(
            base_weights, current_regime, self._model_health
        )

        # 3. Get predictions from each model
        for model in models:
            pred = model.predict(features)
            self._quality_tracker.record_prediction(
                model_name, signal, regime, confidence
            )

        # 4. Weighted average using quality-adjusted weights
        ...

    def record_outcome(self, actual_signal: str) -> None:
        """Call this when market outcome is known."""
        self._quality_tracker.record_batch_outcomes(actual_signal)
```

**Configuration:**

```python
# Quality Tracker Settings
quality_rolling_window: int = 100
quality_full_weight_threshold: float = 0.55
quality_degraded_threshold: float = 0.50
quality_exclusion_threshold: float = 0.45
quality_min_predictions: int = 50
quality_min_active_models: int = 2
quality_recovery_threshold: float = 0.52
quality_recovery_window: int = 30
```

**Metrics Emitted:**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `quality_tracker.model_accuracy` | Per-model rolling accuracy | < 45% |
| `quality_tracker.model_status` | Current status per model | EXCLUDED count > 2 |
| `quality_tracker.regime_accuracy` | Accuracy per model per regime | < 40% in any |
| `quality_tracker.active_count` | Number of active models | < 2 |
| `quality_tracker.weight_adjustments` | Applied weight multipliers | - |

**Benefits:**

1. **Preserves Diversification**: Doesn't throw away models that struggle temporarily
2. **Regime-Aware**: Weights models based on current market conditions
3. **Self-Healing**: Excluded models can recover
4. **Transparent**: Full logging of accuracy and weight adjustments
5. **Conservative**: Requires persistence before exclusion (50+ predictions)

### 3.5 Strategy Layer

#### 3.5.1 Signal Processor (`src/execution/signal_processor.py`)

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

#### 3.5.2 Regime Detector (`src/strategy/regime_detector.py`)

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

#### 3.5.3 Exit Manager (`src/strategy/exit_manager.py`)

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

### 3.6 Risk Layer

#### 3.6.1 Position Sizer (`src/risk/position_sizer.py`)

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

#### 3.6.2 Adaptive Exposure (`src/risk/adaptive_exposure.py`)

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

#### 3.6.3 Risk Manager (`src/risk/risk_manager.py`)

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

### 3.7 Execution Layer

#### 3.7.1 Order Manager (`src/execution/order_manager.py`)

**Order Flow:**
```
Signal → Slippage Estimation → Order Optimization →
Leverage Setting → Risk Validation → Order Placement →
Fill Tracking → AI Logging
```

#### 3.7.2 Position Sync (`src/execution/position_sync.py`)

**Responsibilities:**
- Periodic exchange position sync
- Orphan position detection
- Stop-loss order management
- Take-profit order management
- Exit condition processing

#### 3.7.3 AI Logger (`src/execution/ai_logger.py`)

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
Exchange (WEEX, Hyperliquid, etc.)
      │
      ▼
┌─────────────────┐
│Exchange Adapter │
│ • WebSocket     │
│ • Symbol norm   │
│ • Unified models│
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
│ • Exchange API  │
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
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EXCHANGE ABSTRACTION LAYER                  │    │
│  │  ┌─────────┐  ┌────────────┐  ┌─────────┐  ┌─────────┐  │    │
│  │  │  WEEX   │  │ Hyperliquid│  │ Binance │  │ Generic │  │    │
│  │  │ Adapter │  │   Adapter  │  │ Adapter │  │ OpenAPI │  │    │
│  │  └─────────┘  └────────────┘  └─────────┘  └─────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
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
│   │   └── config.py           # Configuration management (includes ExchangeConfig)
│   ├── exchange/               # NEW: Exchange Abstraction Layer
│   │   ├── __init__.py         # Package exports
│   │   ├── protocols.py        # ExchangeRESTProtocol, ExchangeWebSocketProtocol
│   │   ├── models.py           # UnifiedOrder, UnifiedPosition, etc.
│   │   ├── factory.py          # ExchangeFactory, get_exchange_adapter()
│   │   ├── exceptions.py       # ExchangeError hierarchy
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # Base adapter class
│   │   │   └── weex/           # WEEX-specific adapter
│   │   │       ├── __init__.py
│   │   │       ├── adapter.py  # WEEXAdapter, WEEXRESTClient
│   │   │       ├── websocket.py # WEEXWebSocket
│   │   │       └── mappers.py  # WEEXMapper (unified <-> WEEX translation)
│   │   └── openapi/            # OpenAPI integration
│   │       ├── __init__.py
│   │       ├── parser.py       # OpenAPIParser
│   │       └── mapper.py       # ProtocolMapper
│   ├── data/
│   │   ├── websocket_client.py # Real-time data [deprecated - use exchange layer]
│   │   ├── rest_client.py      # Exchange API [deprecated - use exchange layer]
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
│   │   ├── quality_tracker.py  # Model accuracy tracking
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
│   │   ├── order_manager.py    # Order execution (uses ExchangeRESTProtocol)
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
└── .env                         # Configuration (EXCHANGE_NAME, EXCHANGE_API_KEY, etc.)
```

---

*Document History:*
- v1.0 (December 2025): Initial architecture
- v2.0 (January 2026): Updated based on production learnings
- v2.1 (January 2026): Added Prediction Confidence Bounds (Section 3.4.4)
- v2.2 (January 2026): Refactored to Trigger-Based Retraining (Section 3.4.3, 5.1, 5.2)
  - Replaced time-interval retraining with trigger-based approach
  - Added VolatilityAdjustment for high-vol periods (adjust trading, not training)
  - Inverted cooldown logic (longer during calm, shorter after vol subsides)
  - Four retraining triggers: regime_change, model_health_degraded, feature_drift, validation_accuracy_drop
- v2.3 (January 2026): Added Model Quality Tracker (Section 3.4.5)
  - Graduated weight reduction instead of binary model exclusion
  - Rolling accuracy tracking per model (100-prediction window)
  - Regime-specific accuracy tracking and weight boosts/penalties
  - Diversity protection (always keep ≥2 active models)
  - Recovery mechanism for excluded models (52% threshold over 30 predictions)
- v2.4 (January 2026): Added Exchange Abstraction Layer (Section 3.1)
  - Multi-exchange support via adapter pattern (WEEX, Hyperliquid, Binance, etc.)
  - Unified protocols: ExchangeRESTProtocol, ExchangeWebSocketProtocol
  - Unified data models: UnifiedOrder, UnifiedPosition, UnifiedTicker, etc.
  - Exchange factory for runtime adapter selection
  - Symbol normalization (BTCUSDT ↔ cmt_btcusdt)
  - OpenAPI integration for automatic adapter generation
  - WEEX adapter refactored from src/data/ clients
  - Deprecated src/data/rest_client.py and websocket_client.py
