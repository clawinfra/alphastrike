#!/usr/bin/env python3
"""
Medallion V2 Backtest

LightGBM-based trading strategy with:
- BULLISH regime filter (60%+ confidence)
- Multi-tier ML signal filtering (65-70+ conviction)
- Dynamic leverage (adjusts for volatility and drawdown)

Target metrics:
- CAGR: 66%+
- Max Drawdown: <5%
- Sharpe Ratio: 2.5+

Usage:
    python scripts/medallion_v2_backtest.py --days 180 --base-leverage 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class Position:
    """Track an open position."""
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    size: float
    entry_time: datetime
    entry_candle_idx: int
    strategy: str
    conviction: float


@dataclass
class Trade:
    """Completed trade record."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy: str
    conviction: float


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""
    assets: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "BNB", "XRP", "SOL", "AVAX", "NEAR", "APT",
        "AAVE", "UNI", "LINK", "FET", "DOGE", "PAXG", "SPX",
    ])
    days: int = 90
    initial_balance: float = 10000.0

    # Dynamic leverage (reduces during emergencies, otherwise uses base)
    base_leverage: float = 5.0
    min_leverage: float = 1.0
    max_leverage: float = 10.0

    # Position sizing
    max_portfolio_exposure: float = 0.40  # 40% max total
    max_single_position: float = 0.05     # 5% max per position

    # Trading costs
    slippage_bps: float = 5.0
    taker_fee: float = 0.0005

    # Signal thresholds
    min_conviction: float = 50.0
    ml_long_threshold: float = 0.55   # Relaxed from 0.60
    ml_short_threshold: float = 0.45  # Relaxed from 0.40


@dataclass
class Metrics:
    """Performance metrics."""
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    final_balance: float = 0.0

    # Strategy breakdown (ML only - pairs and mean reversion disabled)
    ml_trades: int = 0
    ml_pnl: float = 0.0


class MedallionV2Engine:
    """
    Medallion V2 backtest engine.

    Strategy:
    - LightGBM-only ML predictions (proven superior to ensemble)
    - Strict BULLISH regime filter (60%+ confidence)
    - ML tier filtering (65-70+ conviction)
    - Dynamic leverage (adjusts for volatility and drawdown)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []

        # Dynamic leverage state
        self.current_leverage = config.base_leverage
        self.leverage_history: list[tuple[datetime, float, str]] = []

        # Position tracking
        self.positions: dict[str, Position] = {}

        # Components (lazy init)
        self.client = None
        self.feature_pipeline = None
        self.ml_models = {}

    async def _init_components(self):
        """Initialize trading components."""
        from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
        from src.features.pipeline import FeaturePipeline
        from src.ml.lightgbm_model import LightGBMModel

        self.client = HyperliquidRESTClient()
        await self.client.initialize()

        self.feature_pipeline = FeaturePipeline()

        # Load LightGBM models (proven superior to ensemble for crypto)
        models_dir = Path("models")
        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            lgb_path = models_dir / f"lightgbm_hyperliquid_{asset.lower()}.txt"
            if lgb_path.exists():
                try:
                    model = LightGBMModel()
                    model.load(lgb_path)
                    self.ml_models[symbol] = model
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")

        logger.info(f"Loaded {len(self.ml_models)} LightGBM models")

    def _get_ml_signal(
        self,
        symbol: str,
        features: dict,
    ) -> tuple[str, float]:
        """Get LightGBM prediction (best performer for crypto)."""
        if symbol not in self.ml_models or not features:
            return "HOLD", 0.0

        try:
            feature_names = self.feature_pipeline.feature_names
            X = np.array([[features.get(name, 0.0) for name in feature_names]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            pred = self.ml_models[symbol].predict(X)[0]

            # Relaxed thresholds for more signals
            if pred > self.config.ml_long_threshold:
                confidence = min(100, (pred - 0.5) * 250)
                return "LONG", confidence
            elif pred < self.config.ml_short_threshold:
                confidence = min(100, (0.5 - pred) * 250)
                return "SHORT", confidence

            return "HOLD", 0.0
        except Exception:
            return "HOLD", 0.0

    def _detect_market_regime(
        self,
        btc_candles: list,
        lookback: int = 50,
    ) -> tuple[str, float]:
        """
        Detect market regime using BTC as the market leader.

        Returns:
            (regime, strength): regime is BULLISH/BEARISH/RANGING, strength 0-100
        """
        if len(btc_candles) < lookback:
            return "RANGING", 50.0

        closes = np.array([c.close for c in btc_candles[-lookback:]])

        # Calculate multiple trend indicators
        # 1. Price vs moving averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes)
        current_price = closes[-1]

        # 2. Trend direction (linear regression slope)
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        slope_pct = slope / closes[0] * 100  # Normalize as percentage

        # 3. Recent momentum (last 10 vs previous 10)
        recent_avg = np.mean(closes[-10:])
        prev_avg = np.mean(closes[-20:-10])
        momentum = (recent_avg - prev_avg) / prev_avg * 100

        # Determine regime
        bullish_signals = 0
        bearish_signals = 0

        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if sma_20 > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if slope_pct > 0.1:  # Upward trend
            bullish_signals += 2
        elif slope_pct < -0.1:  # Downward trend
            bearish_signals += 2

        if momentum > 1:  # Strong upward momentum
            bullish_signals += 1
        elif momentum < -1:  # Strong downward momentum
            bearish_signals += 1

        # Calculate regime and strength
        total_signals = bullish_signals + bearish_signals
        if bullish_signals >= 4:
            regime = "BULLISH"
            strength = min(100, bullish_signals / total_signals * 100 + 20)
        elif bearish_signals >= 4:
            regime = "BEARISH"
            strength = min(100, bearish_signals / total_signals * 100 + 20)
        else:
            regime = "RANGING"
            strength = 50.0

        return regime, strength

    def _calculate_exposure(self, balance: float) -> float:
        """Calculate current portfolio exposure."""
        return sum(p.size / balance for p in self.positions.values())

    def _update_dynamic_leverage(
        self,
        btc_candles: list,
        balance: float,
        peak_balance: float,
        timestamp: datetime,
    ) -> None:
        """
        Update dynamic leverage based on volatility and drawdown.

        Conservative approach: only reduce during emergencies.
        """
        if len(btc_candles) < 20:
            return

        # Calculate current volatility
        closes = [c.close for c in btc_candles[-20:]]
        highs = [c.high for c in btc_candles[-20:]]
        lows = [c.low for c in btc_candles[-20:]]

        tr_values = []
        for j in range(1, len(closes)):
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]),
                abs(lows[j] - closes[j-1])
            )
            tr_values.append(tr)
        atr = sum(tr_values) / len(tr_values)
        current_vol = atr / closes[-1]

        # Calculate current drawdown
        drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0

        # CONSERVATIVE: Only reduce leverage during emergencies
        # Otherwise use base leverage to maximize returns
        new_leverage = self.config.base_leverage

        emergency_reason = None
        if drawdown > 0.10:  # 10% DD = emergency, reduce to protect capital
            new_leverage = self.config.base_leverage * 0.5
            emergency_reason = f"EMERGENCY DD ({drawdown:.1%})"
        elif current_vol > 0.05:  # 5% volatility = extreme, reduce exposure
            new_leverage = self.config.base_leverage * 0.6
            emergency_reason = f"extreme volatility ({current_vol:.1%})"

        # Clamp to valid range
        new_leverage = max(self.config.min_leverage, min(self.config.max_leverage, new_leverage))

        # Only log changes
        if abs(new_leverage - self.current_leverage) / self.current_leverage > 0.10:
            old_lev = self.current_leverage
            self.current_leverage = new_leverage
            if emergency_reason:
                self.leverage_history.append((timestamp, new_leverage, emergency_reason))
                logger.debug(f"Leverage: {old_lev:.1f}x → {new_leverage:.1f}x ({emergency_reason})")

    async def run(self) -> Metrics:
        """Run the full backtest."""
        await self._init_components()

        # Fetch candles
        all_candles = await self._fetch_candles()
        if not all_candles:
            return Metrics()

        logger.info(f"Running Medallion V2 backtest with {len(all_candles)} assets...")

        # Pre-calculate features
        all_features = self._precalculate_features(all_candles)

        # Initialize state
        balance = self.config.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trade_id = 0

        min_len = min(len(c) for c in all_candles.values())
        warmup = 150

        if min_len < warmup + 50:
            logger.error(f"Insufficient data: {min_len} candles")
            return Metrics()

        logger.info(f"Simulation: {min_len} candles, warmup={warmup}")

        returns = []

        for i in range(warmup, min_len):
            current_candles = {
                symbol: candles[:i+1]
                for symbol, candles in all_candles.items()
            }
            timestamp = list(all_candles.values())[0][i].timestamp
            current_exposure = self._calculate_exposure(balance)

            # Update dynamic leverage every 24 hours after sufficient trade history
            if i % 24 == 0 and len(self.trades) >= 20:
                btc_candles = current_candles.get("BTCUSDT", [])
                self._update_dynamic_leverage(btc_candles, balance, peak_balance, timestamp)

            # === CHECK EXITS ===
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                candles = current_candles.get(symbol, [])
                if not candles:
                    continue

                price = candles[-1].close
                holding = i - pos.entry_candle_idx

                if pos.direction == "LONG":
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price

                # Risk management exits
                should_exit = (
                    pnl_pct <= -0.01 or  # 1% stop loss
                    pnl_pct >= 0.04 or   # 4% take profit
                    holding >= 36        # 36 hour time exit
                )

                if should_exit:
                    pnl = pos.size * pnl_pct * self.current_leverage
                    fees = pos.size * self.config.taker_fee * 2
                    net_pnl = pnl - fees
                    balance += net_pnl

                    trade_id += 1
                    trade_record = Trade(
                        id=str(trade_id),
                        symbol=symbol,
                        side=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=price,
                        size_usd=pos.size,
                        entry_time=pos.entry_time,
                        exit_time=timestamp,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        strategy=pos.strategy,
                        conviction=pos.conviction,
                    )
                    self.trades.append(trade_record)
                    del self.positions[symbol]

            # === GENERATE NEW SIGNALS ===

            btc_candles = current_candles.get("BTCUSDT", [])

            # Only trade in BULLISH regime with 60%+ confidence
            regime, regime_conf = self._detect_market_regime(btc_candles)
            if regime != "BULLISH" or regime_conf < 60:
                continue

            # Process ML signals
            for symbol, candles in current_candles.items():
                if symbol in self.positions:
                    continue

                if current_exposure >= self.config.max_portfolio_exposure:
                    break

                if len(candles) < 50:
                    continue

                # Get features
                feature_idx = i - warmup
                symbol_features = all_features.get(symbol, [])
                features = symbol_features[feature_idx] if 0 <= feature_idx < len(symbol_features) else None

                # Get ML signal
                ml_direction, ml_conv = self._get_ml_signal(symbol, features)

                # High-selectivity ML strategy - quality over quantity
                direction = "HOLD"
                conviction = 0.0
                strategy = ""

                # Calculate momentum indicators
                mom_12h = 0.0
                mom_24h = 0.0
                volatility = 0.0
                if len(candles) >= 48:
                    recent_returns = []
                    for j in range(1, min(48, len(candles))):
                        ret = (candles[-j].close - candles[-j-1].close) / candles[-j-1].close
                        recent_returns.append(ret)
                    mom_12h = sum(recent_returns[:12]) if len(recent_returns) >= 12 else 0
                    mom_24h = sum(recent_returns[:24]) if len(recent_returns) >= 24 else 0
                    volatility = np.std(recent_returns[:24]) if len(recent_returns) >= 24 else 0

                # Multi-tier ML signal filtering
                if ml_direction == "LONG" and ml_conv >= 65:
                    # Tier 1: Very high confidence (70+) with momentum
                    if ml_conv >= 70 and mom_12h > 0.01:
                        direction = "LONG"
                        conviction = min(100, ml_conv + 10)
                        strategy = "ml_tier1"
                    # Tier 2: High confidence with strong momentum and low volatility
                    elif ml_conv >= 65 and mom_12h > 0.015 and volatility < 0.03:
                        direction = "LONG"
                        conviction = ml_conv
                        strategy = "ml_tier2"
                    # Tier 3: High confidence with consistent uptrend
                    elif ml_conv >= 65 and mom_12h > 0.005 and mom_24h > 0.01:
                        direction = "LONG"
                        conviction = ml_conv - 5
                        strategy = "ml_tier3"

                if direction == "HOLD" or conviction < self.config.min_conviction:
                    continue

                # Position sizing based on conviction
                size = min(
                    balance * self.config.max_single_position,
                    balance * (1 - current_exposure) * 0.5,
                ) * (conviction / 100)

                if size < 50:
                    continue

                price = candles[-1].close
                if direction == "LONG":
                    price *= (1 + self.config.slippage_bps / 10000)
                else:
                    price *= (1 - self.config.slippage_bps / 10000)

                self.positions[symbol] = Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=price,
                    size=size,
                    entry_time=timestamp,
                    entry_candle_idx=i,
                    strategy=strategy,
                    conviction=conviction,
                )
                current_exposure += size / balance

            # Track equity
            self.equity_curve.append((timestamp, balance))

            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance
            if dd > max_drawdown:
                max_drawdown = dd

            if len(self.equity_curve) > 1:
                prev = self.equity_curve[-2][1]
                returns.append((balance - prev) / prev)

        # Close remaining positions
        for pos in list(self.positions.values()):
            candles = all_candles.get(pos.symbol, [])
            if candles:
                price = candles[-1].close
                if pos.direction == "LONG":
                    pnl_pct = (price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - price) / pos.entry_price
                pnl = pos.size * pnl_pct * self.current_leverage
                fees = pos.size * self.config.taker_fee * 2
                balance += pnl - fees

        if self.client:
            await self.client.close()

        return self._calculate_metrics(balance, max_drawdown, returns)

    async def _fetch_candles(self) -> dict:
        """Fetch candles for all assets."""
        logger.info(f"Fetching candles for {len(self.config.assets)} assets...")
        all_candles = {}

        for asset in self.config.assets:
            symbol = f"{asset}USDT"
            try:
                limit = self.config.days * 24 + 200
                candles = await self.client.get_candles(
                    symbol=symbol,
                    interval="1h",
                    limit=limit,
                )
                if candles:
                    all_candles[symbol] = candles
                    logger.info(f"  {symbol}: {len(candles)} candles")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed - {e}")

        return all_candles

    def _precalculate_features(self, all_candles: dict) -> dict:
        """Pre-calculate features for all assets."""
        logger.info("Pre-calculating features...")
        all_features = {}
        min_window = self.feature_pipeline.config.min_candles

        for symbol, candles in all_candles.items():
            if len(candles) < min_window + 10:
                continue

            features_list = []
            for i in range(min_window, len(candles)):
                window = candles[max(0, i - min_window):i + 1]
                try:
                    features = self.feature_pipeline.calculate_features(window, use_cache=False)
                    features_list.append(features)
                except Exception:
                    features_list.append(None)

            all_features[symbol] = features_list

        return all_features

    def _calculate_metrics(
        self,
        final_balance: float,
        max_drawdown: float,
        returns: list,
    ) -> Metrics:
        """Calculate performance metrics."""
        m = Metrics()
        m.total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance
        m.final_balance = final_balance
        m.max_drawdown = max_drawdown
        m.total_trades = len(self.trades)

        hours = len(returns)
        years = hours / (24 * 365)
        if years > 0 and final_balance > 0:
            m.cagr = (final_balance / self.config.initial_balance) ** (1 / years) - 1

        if returns and np.std(returns) > 0:
            m.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(24 * 365))

        neg_rets = [r for r in returns if r < 0]
        if neg_rets and np.std(neg_rets) > 0:
            m.sortino_ratio = float(np.mean(returns) / np.std(neg_rets) * np.sqrt(24 * 365))

        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]
            m.win_rate = len(wins) / len(self.trades)

            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 1
            m.profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # All trades are ML trades
            m.ml_trades = len(self.trades)
            m.ml_pnl = sum(t.pnl for t in self.trades)

        return m


def print_results(config: BacktestConfig, metrics: Metrics, trades: list):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("MEDALLION V2 BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Assets: {len(config.assets)}")
    print(f"  Period: {config.days} days")
    print(f"  Initial: ${config.initial_balance:,.0f}")
    print(f"  Leverage: {config.base_leverage:.0f}x base (dynamic: {config.min_leverage:.0f}x-{config.max_leverage:.0f}x)")

    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print("=" * 70)

    cagr_ok = "OK" if metrics.cagr >= 0.66 else "X"
    dd_ok = "OK" if metrics.max_drawdown <= 0.05 else "X"
    sharpe_ok = "OK" if metrics.sharpe_ratio >= 2.5 else "X"

    print(f"\n{'Metric':<25} {'Value':>15} {'Target':>15} {'Status':>8}")
    print("-" * 65)
    print(f"{'CAGR':<25} {metrics.cagr:>14.1%} {'>66%':>15} {cagr_ok:>8}")
    print(f"{'Max Drawdown':<25} {metrics.max_drawdown:>14.1%} {'<5%':>15} {dd_ok:>8}")
    print(f"{'Sharpe Ratio':<25} {metrics.sharpe_ratio:>14.2f} {'>2.5':>15} {sharpe_ok:>8}")
    print(f"{'Sortino Ratio':<25} {metrics.sortino_ratio:>14.2f}")
    print(f"{'Win Rate':<25} {metrics.win_rate:>14.1%}")
    print(f"{'Profit Factor':<25} {metrics.profit_factor:>14.2f}")
    print(f"{'Total Trades':<25} {metrics.total_trades:>14}")

    print(f"\n{'=' * 70}")
    print("STRATEGY BREAKDOWN")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Trades':>10} {'P&L':>15}")
    print("-" * 50)
    print(f"{'ML Signals':<20} {metrics.ml_trades:>10} ${metrics.ml_pnl:>+13,.2f}")

    print(f"\n{'=' * 70}")
    print("PORTFOLIO")
    print("=" * 70)
    print(f"  Initial: ${config.initial_balance:,.2f}")
    print(f"  Final:   ${metrics.final_balance:,.2f}")
    print(f"  Return:  {metrics.total_return:+.1%}")

    if trades:
        print(f"\n{'=' * 70}")
        print("RECENT TRADES (last 15)")
        print("=" * 70)
        for t in trades[-15:]:
            pnl_str = f"+${t.pnl:,.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):,.2f}"
            print(f"  {t.entry_time.strftime('%m-%d %H:%M')} {t.symbol:<15} {t.side:<6} {t.strategy:<12} {pnl_str:>12}")

    print("\n" + "=" * 70)
    targets = sum([metrics.cagr >= 0.66, metrics.max_drawdown <= 0.05, metrics.sharpe_ratio >= 2.5])
    if targets == 3:
        print("MEDALLION TARGETS ACHIEVED!")
    elif targets >= 2:
        print("STRONG PERFORMANCE - 2/3 targets met")
    elif targets >= 1:
        print("DECENT - 1/3 targets met, more tuning needed")
    else:
        print("BELOW TARGETS - significant tuning needed")
    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Medallion V2 Backtest (Architecture-Compliant)")
    parser.add_argument("--days", type=int, default=180, help="Backtest period in days")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--base-leverage", type=float, default=5.0, help="Base leverage for dynamic calc")
    parser.add_argument("--min-leverage", type=float, default=1.0, help="Min leverage (high-risk floor)")
    parser.add_argument("--max-leverage", type=float, default=10.0, help="Max leverage (favorable ceiling)")
    parser.add_argument("--min-conviction", type=float, default=50, help="Min ML conviction threshold")
    args = parser.parse_args()

    config = BacktestConfig(
        days=args.days,
        initial_balance=args.balance,
        base_leverage=args.base_leverage,
        min_leverage=args.min_leverage,
        max_leverage=args.max_leverage,
        min_conviction=args.min_conviction,
    )

    engine = MedallionV2Engine(config)
    metrics = await engine.run()
    print_results(config, metrics, engine.trades)


if __name__ == "__main__":
    asyncio.run(main())
