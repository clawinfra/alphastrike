"""
Simons-Style Statistical Arbitrage Engine

Implements the core strategies that made Medallion Fund successful:
1. Pairs Trading - Trade spreads between correlated assets
2. Mean Reversion - Z-score based entry/exit
3. Momentum Capture - Short-term momentum with quick exits
4. Volatility Regime Detection - Adjust strategy per regime

Key insight: Medallion doesn't predict direction - it exploits statistical patterns
that revert to mean. Small edge × many trades × leverage = extraordinary returns.

ARCHITECTURE NOTE:
All TA thresholds are now loaded from adaptive config rather than hardcoded.
The system learns optimal parameters through rolling performance optimization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Late import to avoid circular dependency
def _load_adaptive_config():
    """Lazy load adaptive config to avoid circular imports."""
    from src.adaptive.simons_config import load_or_create_config
    return load_or_create_config()


@dataclass
class SimonsSignal:
    """Signal from Simons strategy engine."""
    symbol: str
    direction: Literal["LONG", "SHORT", "HOLD"]
    conviction: float  # 0-100
    strategy: str  # Which strategy generated this signal
    edge: float  # Expected edge per trade
    holding_period: int  # Expected holding period in candles
    z_score: float = 0.0  # For mean reversion signals
    pair_symbol: str = ""  # For pairs trading
    spread_z: float = 0.0  # Spread z-score for pairs


@dataclass
class PairRelationship:
    """Tracks a tradeable pair relationship."""
    symbol_a: str
    symbol_b: str
    correlation: float
    spread_mean: float
    spread_std: float
    half_life: float  # Mean reversion half-life in candles
    last_spread: float = 0.0
    spread_z: float = 0.0


class SimonsStrategyEngine:
    """
    Statistical arbitrage engine implementing Medallion-style strategies.

    Core Philosophy:
    - Don't predict, exploit patterns
    - Mean reversion is the most reliable edge in RANGING markets
    - Trend-following in TRENDING markets
    - Regime detection is critical for strategy selection
    - Many small bets > few large bets
    - Speed of mean reversion determines position size

    ADAPTIVE PARAMETERS:
    All thresholds are loaded from adaptive config and can be auto-tuned
    based on rolling performance metrics. No hardcoded magic numbers.
    """

    # High correlation pairs for spread trading
    # These are structural relationships, not tunable parameters
    TRADEABLE_PAIRS = [
        # Major crypto correlations (highest reliability)
        ("BTCUSDT", "ETHUSDT", 0.90),
        ("BTCUSDT", "BNBUSDT", 0.85),
        # L1 blockchain correlations
        ("ETHUSDT", "SOLUSDT", 0.82),
        ("ETHUSDT", "AVAXUSDT", 0.80),
        ("SOLUSDT", "AVAXUSDT", 0.78),
        ("SOLUSDT", "NEARUSDT", 0.75),
        ("AVAXUSDT", "NEARUSDT", 0.74),
        ("NEARUSDT", "APTUSDT", 0.76),
        # DeFi correlations
        ("AAVEUSDT", "UNIUSDT", 0.82),
        ("AAVEUSDT", "LINKUSDT", 0.78),
        ("UNIUSDT", "LINKUSDT", 0.75),
        # Altcoin correlations
        ("XRPUSDT", "DOGEUSDT", 0.70),
        ("LINKUSDT", "FETUSDT", 0.68),
        # Cross-sector correlations
        ("BTCUSDT", "SOLUSDT", 0.80),
        ("ETHUSDT", "BNBUSDT", 0.78),
    ]

    # Lookback periods (structural, not tunable)
    FAST_LOOKBACK = 20
    SLOW_LOOKBACK = 100

    def __init__(self, config_path: Path | None = None):
        """
        Initialize with adaptive config.

        Args:
            config_path: Optional path to config file. If None, uses default.
        """
        # Load adaptive configuration
        self._config, self._optimizer = _load_adaptive_config()

        # State
        self.pair_relationships: dict[str, PairRelationship] = {}
        self._initialize_pairs()

        logger.info(
            f"SimonsEngine initialized with adaptive config: "
            f"entry_z={self.ENTRY_Z:.2f}, pairs_z={self.PAIRS_ENTRY_Z:.2f}, "
            f"min_conv={self.MIN_CONVICTION:.0f}"
        )

    # === Dynamic Properties (from adaptive config) ===

    @property
    def ENTRY_Z(self) -> float:
        """Z-score threshold for mean reversion entry."""
        return self._config.entry_z_score

    @property
    def EXIT_Z(self) -> float:
        """Z-score threshold for exit."""
        return self._config.exit_z_score

    @property
    def EXTREME_Z(self) -> float:
        """Z-score for very high conviction."""
        return self._config.extreme_z_score

    @property
    def PAIRS_ENTRY_Z(self) -> float:
        """Z-score threshold for pairs entry."""
        return self._config.pairs_entry_z

    @property
    def PAIRS_EXIT_Z(self) -> float:
        """Z-score threshold for pairs exit."""
        return self._config.pairs_exit_z

    @property
    def SPREAD_LOOKBACK(self) -> int:
        """Lookback for spread calculations."""
        return self._config.spread_lookback

    @property
    def TRENDING_ADX(self) -> float:
        """ADX threshold for trending market."""
        return self._config.trending_adx

    @property
    def HIGH_VOL_THRESHOLD(self) -> float:
        """ATR ratio threshold for high volatility."""
        return self._config.high_vol_atr_ratio

    @property
    def RSI_OVERSOLD(self) -> float:
        """RSI threshold for oversold condition."""
        return self._config.rsi_oversold

    @property
    def RSI_OVERBOUGHT(self) -> float:
        """RSI threshold for overbought condition."""
        return self._config.rsi_overbought

    @property
    def MIN_VOLUME_RATIO(self) -> float:
        """Minimum volume ratio to trade."""
        return self._config.min_volume_ratio

    @property
    def MIN_CONVICTION(self) -> float:
        """Minimum conviction to enter a trade."""
        return self._config.min_conviction

    # === Config Management ===

    def record_trade(self, trade: dict) -> None:
        """Record a trade for performance tracking and optimization."""
        self._optimizer.record_trade(trade)

    def optimize_parameters(self) -> None:
        """
        Optimize parameters based on recent performance.
        Call this periodically (e.g., daily or after N trades).
        """
        if self._optimizer.should_tune():
            self._config = self._optimizer.optimize()
            logger.info(
                f"Parameters optimized: entry_z={self.ENTRY_Z:.2f}, "
                f"pairs_z={self.PAIRS_ENTRY_Z:.2f}, rsi_os={self.RSI_OVERSOLD:.0f}"
            )

    def reload_config(self) -> None:
        """Hot-reload config from file."""
        self._config = self._optimizer.load()
        logger.info("Config reloaded from file")

    def get_config(self):
        """Get current adaptive config."""
        return self._config

    def _initialize_pairs(self):
        """Initialize pair relationships for spread trading."""
        for sym_a, sym_b, corr in self.TRADEABLE_PAIRS:
            key = f"{sym_a}_{sym_b}"
            self.pair_relationships[key] = PairRelationship(
                symbol_a=sym_a,
                symbol_b=sym_b,
                correlation=corr,
                spread_mean=0.0,
                spread_std=1.0,
                half_life=10.0,  # Default, will be calculated
            )

    def detect_market_regime(self, candles: list) -> tuple[str, float]:
        """
        Detect market regime from BTC or provided candles.

        Returns:
            (regime, confidence) where regime is "TRENDING", "RANGING", or "VOLATILE"
        """
        if len(candles) < 50:
            return "UNKNOWN", 0.0

        closes = np.array([c.close for c in candles[-50:]])
        highs = np.array([c.high for c in candles[-50:]])
        lows = np.array([c.low for c in candles[-50:]])

        # Calculate ADX to detect trending
        # Simplified ADX calculation
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(abs(highs[1:] - closes[:-1]),
                                  abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:])

        # Directional movement
        plus_dm = np.maximum(highs[1:] - highs[:-1], 0)
        minus_dm = np.maximum(lows[:-1] - lows[1:], 0)

        # Where plus_dm > minus_dm, set minus_dm to 0 and vice versa
        mask = plus_dm <= minus_dm
        plus_dm[mask] = 0
        mask = minus_dm <= plus_dm
        minus_dm[mask] = 0

        # Calculate DI
        plus_di = 100 * np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-14:]) / atr if atr > 0 else 0

        # Calculate ADX
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        adx = dx  # Simplified - normally would smooth

        # Volatility ratio
        current_atr = np.mean(tr[-5:])
        avg_atr = np.mean(tr[-20:])
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Determine regime
        if vol_ratio > self.HIGH_VOL_THRESHOLD:
            return "VOLATILE", float(min(1.0, vol_ratio / 2))
        elif adx > self.TRENDING_ADX:
            return "TRENDING", float(min(1.0, adx / 50))
        else:
            return "RANGING", float(min(1.0, (self.TRENDING_ADX - adx) / self.TRENDING_ADX))

    def update_pairs(self, all_candles: dict) -> None:
        """
        Update pair statistics from recent price data.

        Call this periodically to recalibrate spread parameters.
        """
        for key, pair in self.pair_relationships.items():
            candles_a = all_candles.get(pair.symbol_a, [])
            candles_b = all_candles.get(pair.symbol_b, [])

            if len(candles_a) < self.SPREAD_LOOKBACK or len(candles_b) < self.SPREAD_LOOKBACK:
                continue

            # Calculate log price ratio (spread)
            prices_a = np.array([c.close for c in candles_a[-self.SPREAD_LOOKBACK:]])
            prices_b = np.array([c.close for c in candles_b[-self.SPREAD_LOOKBACK:]])

            # Normalize prices and calculate spread
            norm_a = prices_a / prices_a[0]
            norm_b = prices_b / prices_b[0]
            spread = np.log(norm_a / norm_b)

            pair.spread_mean = float(np.mean(spread))
            pair.spread_std = float(np.std(spread))
            if pair.spread_std > 0:
                pair.last_spread = spread[-1]
                pair.spread_z = (spread[-1] - pair.spread_mean) / pair.spread_std

            # Estimate half-life of mean reversion
            pair.half_life = self._estimate_half_life(spread)

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """Estimate mean reversion half-life using OLS regression."""
        if len(spread) < 10:
            return 10.0

        # Regress spread change on lagged spread
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        # OLS: spread_diff = alpha + beta * spread_lag
        try:
            beta = np.cov(spread_diff, spread_lag)[0, 1] / np.var(spread_lag)
            if beta < 0:
                half_life = -np.log(2) / beta
                return float(np.clip(half_life, 1, 100))
        except Exception:
            pass

        return 10.0  # Default

    def generate_signals(
        self,
        all_candles: dict,
        current_positions: dict,
    ) -> list[SimonsSignal]:
        """
        Generate trading signals using multiple Simons strategies.

        CRITICAL: Uses market regime detection to select appropriate strategies:
        - RANGING: Mean reversion works best
        - TRENDING: Avoid mean reversion against trend, focus on pairs
        - VOLATILE: Reduce position sizes, widen stops

        Returns list of signals sorted by conviction.
        """
        signals = []

        # Detect market regime from BTC (market proxy)
        btc_candles = all_candles.get("BTCUSDT", [])
        regime, regime_confidence = self.detect_market_regime(btc_candles)

        # Update pair statistics
        self.update_pairs(all_candles)

        # 1. Pairs Trading - Re-enabled with spread-based signals
        # Note: Backtest must track pairs together for proper P&L
        pairs_signals = self._generate_pairs_signals(all_candles, current_positions)
        signals.extend(pairs_signals)

        # 2. Mean Reversion Signals - Generate always, let caller filter by regime
        # Previously only in ranging/low volatility, but this filtered too much
        mr_signals = self._generate_mean_reversion_signals(all_candles, current_positions)

        # Adjust conviction based on regime (but don't filter out)
        if regime == "TRENDING":
            # Reduce conviction in trending markets (mean reversion is riskier)
            for sig in mr_signals:
                if regime_confidence >= 0.7:
                    sig.conviction *= 0.6  # Strong trend - very cautious
                else:
                    sig.conviction *= 0.8  # Weak trend - moderately cautious
        elif regime == "VOLATILE":
            # Reduce conviction in volatile markets
            for sig in mr_signals:
                sig.conviction *= 0.75

        signals.extend(mr_signals)

        # 3. Momentum Signals (disabled - see _generate_momentum_signals)
        mom_signals = self._generate_momentum_signals(all_candles, current_positions)
        signals.extend(mom_signals)

        # Adjust conviction based on regime
        if regime == "VOLATILE":
            # Reduce all convictions in high volatility (higher risk)
            for sig in signals:
                sig.conviction *= 0.8

        # Sort by conviction (highest first)
        signals.sort(key=lambda s: s.conviction, reverse=True)

        return signals

    def _generate_pairs_signals(
        self,
        all_candles: dict,
        current_positions: dict,
    ) -> list[SimonsSignal]:
        """
        Generate pairs trading signals.

        Pairs trading is our most reliable strategy because:
        - Hedged positions reduce directional risk
        - Correlation-based spreads revert more reliably than single assets
        - Lower z-score threshold acceptable due to hedging

        When spread z-score > 2: Short the outperformer, long the underperformer
        When spread z-score < -2: Long the outperformer, short the underperformer
        """
        signals = []

        for key, pair in self.pair_relationships.items():
            # Use lower threshold for pairs (hedged, more reliable)
            if abs(pair.spread_z) < self.PAIRS_ENTRY_Z:
                continue

            # Skip if already in position
            if pair.symbol_a in current_positions or pair.symbol_b in current_positions:
                continue

            # Calculate conviction based on z-score magnitude
            # Pairs trading gets HIGHER base conviction (hedged = reliable)
            z_mag = abs(pair.spread_z)
            if z_mag >= self.EXTREME_Z:
                conviction = 95.0  # Very high for extreme spread
            elif z_mag >= 2.5:
                conviction = 80.0 + (z_mag - 2.5) * 10  # 80-95 for z 2.5-4
            elif z_mag >= self.PAIRS_ENTRY_Z:
                conviction = 65.0 + (z_mag - self.PAIRS_ENTRY_Z) * 20  # 65-80 for z 1.8-2.5
            else:
                continue

            # Adjust for half-life (faster mean reversion = higher conviction)
            if pair.half_life < 5:
                conviction *= 1.15
            elif pair.half_life > 20:
                conviction *= 0.85

            # Boost conviction based on correlation strength
            if pair.correlation >= 0.85:
                conviction *= 1.1  # High correlation = more reliable
            elif pair.correlation < 0.70:
                conviction *= 0.9  # Low correlation = less reliable

            conviction = min(100, conviction)

            # Expected edge based on z-score
            edge = (z_mag - 0.5) * pair.spread_std * 100  # In percentage points

            if pair.spread_z > self.ENTRY_Z:
                # Spread too high: A outperformed B, expect reversion
                # Short A, Long B
                signals.append(SimonsSignal(
                    symbol=pair.symbol_a,
                    direction="SHORT",
                    conviction=conviction,
                    strategy="pairs",
                    edge=edge,
                    holding_period=int(pair.half_life * 2),
                    z_score=pair.spread_z,
                    pair_symbol=pair.symbol_b,
                    spread_z=pair.spread_z,
                ))
                signals.append(SimonsSignal(
                    symbol=pair.symbol_b,
                    direction="LONG",
                    conviction=conviction * 0.9,  # Slightly lower for B
                    strategy="pairs",
                    edge=edge,
                    holding_period=int(pair.half_life * 2),
                    z_score=-pair.spread_z,
                    pair_symbol=pair.symbol_a,
                    spread_z=pair.spread_z,
                ))

            elif pair.spread_z < -self.ENTRY_Z:
                # Spread too low: B outperformed A, expect reversion
                # Long A, Short B
                signals.append(SimonsSignal(
                    symbol=pair.symbol_a,
                    direction="LONG",
                    conviction=conviction,
                    strategy="pairs",
                    edge=edge,
                    holding_period=int(pair.half_life * 2),
                    z_score=pair.spread_z,
                    pair_symbol=pair.symbol_b,
                    spread_z=pair.spread_z,
                ))
                signals.append(SimonsSignal(
                    symbol=pair.symbol_b,
                    direction="SHORT",
                    conviction=conviction * 0.9,
                    strategy="pairs",
                    edge=edge,
                    holding_period=int(pair.half_life * 2),
                    z_score=-pair.spread_z,
                    pair_symbol=pair.symbol_a,
                    spread_z=pair.spread_z,
                ))

        return signals

    def _generate_mean_reversion_signals(
        self,
        all_candles: dict,
        current_positions: dict,
    ) -> list[SimonsSignal]:
        """
        Generate single-asset mean reversion signals.

        Single-asset mean reversion is riskier than pairs because:
        - No hedge against directional moves
        - Trends can persist beyond z-score extremes
        - Requires stricter filters: higher z-score + RSI + volume confirmation
        """
        signals = []

        for symbol, candles in all_candles.items():
            if len(candles) < self.SLOW_LOOKBACK:
                continue

            if symbol in current_positions:
                continue

            closes = np.array([c.close for c in candles[-self.SLOW_LOOKBACK:]])
            volumes = np.array([c.volume for c in candles[-self.SLOW_LOOKBACK:]])

            # Calculate z-score relative to moving average
            ma = np.mean(closes)
            std = np.std(closes)
            if std == 0:
                continue

            current_price = closes[-1]
            z_score = (current_price - ma) / std

            # Use higher threshold for single-asset (2.5 instead of 2.0)
            if abs(z_score) < self.ENTRY_Z:
                continue

            # RSI confirmation for mean reversion (stricter thresholds)
            rsi = self._calculate_rsi(closes, 14)

            # Volume spike confirmation - mean reversion more reliable on high volume
            avg_volume = np.mean(volumes[:-1])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Skip if volume is too low (lack of conviction)
            if volume_ratio < self.MIN_VOLUME_RATIO:
                continue

            # Mean reversion: Asymmetric LONG-biased strategy
            # LONGs: Lower threshold (crypto has upward bias)
            # SHORTs: Higher threshold (only on extreme conditions)

            # SHORT mean reversion - only on extreme overbought (stricter threshold)
            # Uses adaptive RSI_OVERBOUGHT threshold
            if z_score > self.EXTREME_Z and rsi > self.RSI_OVERBOUGHT:
                # Very overbought - expect reversion down
                conviction = 40 + min(25, (z_score - self.EXTREME_Z) * 15)
                if rsi > 90:
                    conviction += 10
                if volume_ratio > self._config.volume_surge_threshold:
                    conviction += 5

                signals.append(SimonsSignal(
                    symbol=symbol,
                    direction="SHORT",
                    conviction=min(75, conviction),  # Cap lower for shorts
                    strategy="mean_reversion",
                    edge=(z_score - 0.5) * std / current_price * 100,
                    holding_period=self.FAST_LOOKBACK,
                    z_score=z_score,
                ))

            # LONG mean reversion - more aggressive (lower threshold)
            # Uses adaptive RSI_OVERSOLD threshold
            elif z_score < -self.ENTRY_Z and rsi < self.RSI_OVERSOLD:
                # Oversold - expect reversion up
                conviction = 45 + min(25, (abs(z_score) - self.ENTRY_Z) * 12)
                if rsi < 15:
                    conviction += 15
                if volume_ratio > self._config.volume_surge_threshold:
                    conviction += 10

                signals.append(SimonsSignal(
                    symbol=symbol,
                    direction="LONG",
                    conviction=min(85, conviction),  # Cap at 85
                    strategy="mean_reversion",
                    edge=(abs(z_score) - 0.5) * std / current_price * 100,
                    holding_period=self.FAST_LOOKBACK,
                    z_score=z_score,
                ))

        return signals

    def _generate_momentum_signals(
        self,
        all_candles: dict,
        current_positions: dict,
    ) -> list[SimonsSignal]:
        """
        Generate short-term momentum signals.

        DISABLED: Momentum in crypto has consistently poor performance due to:
        - High volatility causes whipsaws
        - Stop losses get hit frequently (27% win rate in backtests)
        - Trend reversals are sudden

        Focus on pairs trading and mean reversion instead.
        """
        # Momentum trading disabled - consistently unprofitable
        return []

        # Original code below (disabled)
        signals = []

        for symbol, candles in all_candles.items():  # noqa: E501
            if len(candles) < self.SLOW_LOOKBACK:
                continue

            if symbol in current_positions:
                continue

            closes = np.array([c.close for c in candles[-self.SLOW_LOOKBACK:]])
            volumes = np.array([c.volume for c in candles[-self.SLOW_LOOKBACK:]])

            # Short-term momentum (5-period rate of change)
            mom_5 = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0

            # Medium-term momentum (20-period rate of change)
            mom_20 = (closes[-1] - closes[-21]) / closes[-21] if closes[-21] > 0 else 0

            # Long-term trend (50-period)
            mom_50 = (closes[-1] - closes[-51]) / closes[-51] if len(closes) > 50 and closes[-51] > 0 else 0

            # Volume confirmation - require volume surge
            avg_volume = np.mean(volumes[-20:-1])
            recent_volume = np.mean(volumes[-3:])
            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Skip if no volume surge (momentum without volume is fake)
            if volume_surge < 1.3:
                continue

            # Don't trade momentum if overextended
            z_score = (closes[-1] - np.mean(closes)) / np.std(closes)
            if abs(z_score) > 1.2:  # More conservative
                continue

            # Require all timeframes aligned (5, 20, 50 period)
            # This is much stricter but produces higher quality signals
            if mom_5 > 0.03 and mom_20 > 0.01 and mom_50 > 0:
                # Strong bullish momentum, all timeframes aligned
                conviction = 35 + min(25, mom_5 * 400)
                if volume_surge > 2.0:
                    conviction += 10

                signals.append(SimonsSignal(
                    symbol=symbol,
                    direction="LONG",
                    conviction=min(70, conviction),  # Cap very low for momentum
                    strategy="momentum",
                    edge=mom_5 * 30,  # Conservative edge estimate
                    holding_period=8,  # Slightly longer hold
                    z_score=z_score,
                ))

            elif mom_5 < -0.03 and mom_20 < -0.01 and mom_50 < 0:
                # Strong bearish momentum, all timeframes aligned
                conviction = 35 + min(25, abs(mom_5) * 400)
                if volume_surge > 2.0:
                    conviction += 10

                signals.append(SimonsSignal(
                    symbol=symbol,
                    direction="SHORT",
                    conviction=min(70, conviction),  # Cap very low
                    strategy="momentum",
                    edge=abs(mom_5) * 30,
                    holding_period=8,
                    z_score=z_score,
                ))

        return signals

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def should_exit_position(
        self,
        symbol: str,
        entry_z_score: float,
        current_candles: dict,
        holding_candles: int,
        max_holding: int,
        strategy: str = "mean_reversion",
    ) -> tuple[bool, str]:
        """
        Determine if a position should be closed.

        Returns (should_exit, reason).
        """
        candles = current_candles.get(symbol, [])
        if len(candles) < self.SLOW_LOOKBACK:
            return False, ""

        closes = np.array([c.close for c in candles[-self.SLOW_LOOKBACK:]])
        std = np.std(closes)
        if std == 0:
            return False, ""

        current_z = (closes[-1] - np.mean(closes)) / std

        # Use different exit threshold for pairs (more lenient)
        exit_threshold = self.PAIRS_EXIT_Z if strategy == "pairs" else self.EXIT_Z

        # Exit if z-score has reverted to near mean
        if abs(current_z) < exit_threshold:
            return True, "mean_reversion_complete"

        # Exit if z-score has crossed zero (full reversion)
        if entry_z_score > 0 and current_z < 0:
            return True, "crossed_mean"
        if entry_z_score < 0 and current_z > 0:
            return True, "crossed_mean"

        # Time-based exit
        if holding_candles >= max_holding:
            return True, "max_holding_reached"

        return False, ""


class SimonsPositionSizer:
    """
    Position sizing optimized for statistical arbitrage.

    Key principles:
    - Higher conviction = larger position (but still capped)
    - Faster mean reversion = larger position
    - More diversified portfolio = more total exposure allowed

    NOTE: Position sizing parameters are now loaded from adaptive config.
    """

    # Kelly criterion fraction (we use fractional Kelly for safety)
    KELLY_FRACTION = 0.25

    def __init__(self, config=None):
        """
        Initialize with adaptive config.

        Args:
            config: SimonsAdaptiveConfig instance. If None, loads default.
        """
        if config is None:
            config, _ = _load_adaptive_config()
        self._config = config

    @property
    def MAX_SINGLE_POSITION(self) -> float:
        """Max position size for single-asset trades."""
        return self._config.base_position_pct * 1.5

    @property
    def MAX_PAIRS_POSITION(self) -> float:
        """Max position size for pairs (total, split between legs)."""
        return self._config.base_position_pct * 2

    @property
    def MAX_PORTFOLIO_EXPOSURE(self) -> float:
        """Max total portfolio exposure."""
        return self._config.max_portfolio_exposure

    def calculate_size(
        self,
        signal: SimonsSignal,
        balance: float,
        current_exposure: float,
    ) -> float:
        """
        Calculate position size based on signal characteristics.

        Sizing philosophy (from adaptive config):
        - Pairs trading: LARGEST (hedged, most reliable)
        - Mean reversion: MEDIUM (single-asset, less reliable)
        - Momentum: SMALLEST (high failure rate)
        """
        remaining_capacity = self.MAX_PORTFOLIO_EXPOSURE - current_exposure
        if remaining_capacity <= 0:
            return 0.0

        # Base size from conviction (uses adaptive base_position_pct)
        base_size = (signal.conviction / 100) * self._config.base_position_pct * balance

        # Adjust for strategy type - multipliers from adaptive config
        if signal.strategy == "pairs":
            base_size *= self._config.pairs_size_multiplier
            max_size = self.MAX_PAIRS_POSITION * balance / 2
        elif signal.strategy == "mean_reversion":
            base_size *= self._config.mr_size_multiplier
            max_size = self.MAX_SINGLE_POSITION * 0.8 * balance
        else:  # momentum
            base_size *= 0.5
            max_size = self.MAX_SINGLE_POSITION * 0.5 * balance

        # Adjust for expected edge (Kelly-inspired)
        if signal.edge > 0:
            kelly_size = (signal.edge / 100) * balance * self.KELLY_FRACTION
            base_size = min(base_size, kelly_size * 2)

        # Apply caps
        final_size = min(base_size, max_size, remaining_capacity * balance)

        return max(0, final_size)
