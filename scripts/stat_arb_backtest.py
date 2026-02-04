#!/usr/bin/env python3
"""
Statistical Arbitrage Backtest - Medallion-Style

Implements Jim Simons' core edge: statistical arbitrage using:
1. Pairs Trading - Trade spreads between correlated assets
2. Mean Reversion - Z-score based entry/exit on single assets
3. Momentum Capture - Short-term momentum with quick exits

Target metrics:
- CAGR: 66%+
- Max Drawdown: <5%
- Sharpe Ratio: 2.5+

Usage:
    python scripts/stat_arb_backtest.py --days 180
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Literal

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.strategy.simons_engine import SimonsStrategyEngine, SimonsPositionSizer, SimonsSignal


@dataclass
class BacktestTrade:
    """Record of a simulated trade."""
    id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    strategy: str  # "pairs", "mean_reversion", "momentum"
    conviction: float
    z_score: float
    holding_period: int


@dataclass
class BacktestConfig:
    """Configuration for the stat arb backtest."""
    # Assets to trade (Medallion-style diversified portfolio)
    assets: list[str] = field(default_factory=lambda: [
        # Crypto Major (25%) - liquid majors
        "BTC", "ETH", "BNB", "XRP",
        # L1/L2 (20%) - ecosystem plays
        "SOL", "AVAX", "NEAR", "APT",
        # DeFi (15%) - protocol tokens
        "AAVE", "UNI", "LINK",
        # AI (10%) - narrative driven
        "FET",
        # Meme (10%) - high volatility
        "DOGE",
        # Traditional (20%) - for diversification
        "PAXG", "SPX",
    ])

    # Time range
    days: int = 180

    # Portfolio settings
    initial_balance: float = 10000.0
    leverage: int = 5  # Moderate leverage for better returns

    # Position sizing
    max_portfolio_exposure: float = 0.50  # 50% max total exposure (tighter)
    max_single_position: float = 0.08  # 8% max per position

    # Trading costs
    slippage_bps: float = 5.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005

    # Signal thresholds
    min_conviction: float = 60.0  # Minimum conviction to trade


@dataclass
class StatArbMetrics:
    """Performance metrics for the backtest."""
    total_return: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    final_balance: float = 0.0

    # Strategy breakdown
    pairs_trades: int = 0
    pairs_pnl: float = 0.0
    pairs_win_rate: float = 0.0

    mean_reversion_trades: int = 0
    mean_reversion_pnl: float = 0.0
    mean_reversion_win_rate: float = 0.0

    momentum_trades: int = 0
    momentum_pnl: float = 0.0
    momentum_win_rate: float = 0.0


class StatArbBacktestEngine:
    """
    Statistical arbitrage backtest engine.

    Uses the SimonsStrategyEngine to generate signals based on:
    - Pairs trading (spread z-score)
    - Mean reversion (single-asset z-score)
    - Momentum (short-term continuation)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.positions: dict[str, dict] = {}  # symbol -> position info

        # Initialize Simons strategy engine
        self.simons_engine = SimonsStrategyEngine()
        self.position_sizer = SimonsPositionSizer()

        # Exchange client (initialized lazily)
        self.client = None

    async def _init_client(self):
        """Initialize exchange client."""
        if self.client is None:
            from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
            self.client = HyperliquidRESTClient()
            await self.client.initialize()

    async def fetch_all_candles(self) -> dict[str, list]:
        """Fetch candles for all assets."""
        from src.exchange.models import UnifiedCandle

        await self._init_client()

        logger.info(f"Fetching candles for {len(self.config.assets)} assets...")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=self.config.days)

        all_candles: dict[str, list[UnifiedCandle]] = {}

        for asset in self.config.assets:
            symbol = f"{asset}USDT"

            try:
                # Calculate limit based on days (hourly candles)
                limit = self.config.days * 24 + 150  # Extra for warmup
                candles = await self.client.get_candles(
                    symbol=symbol,
                    interval="1h",
                    limit=limit,
                    start_time=start_time,
                    end_time=end_time,
                )
                if candles:
                    all_candles[symbol] = candles
                    logger.info(f"  {symbol}: {len(candles)} candles")
                else:
                    logger.warning(f"  {symbol}: No candles returned")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed to fetch - {e}")

        return all_candles

    async def run(self) -> StatArbMetrics:
        """Run the full backtest."""
        # Fetch data
        all_candles = await self.fetch_all_candles()

        if not all_candles:
            logger.error("No candles fetched, cannot run backtest")
            return StatArbMetrics()

        logger.info(f"\nRunning statistical arbitrage backtest with {len(all_candles)} assets...")

        # Initialize state
        balance = self.config.initial_balance
        peak_balance = balance
        max_drawdown = 0.0
        trade_id = 0

        # Track positions
        positions: dict[str, dict] = {}  # symbol -> {side, size, entry_price, entry_time, strategy, z_score, holding_period_target}

        # Find common time range
        min_len = min(len(c) for c in all_candles.values())
        warmup = 150  # Need enough for mean reversion calculations

        if min_len < warmup + 20:
            logger.error(f"Insufficient data: {min_len} candles, need {warmup + 20}")
            return StatArbMetrics()

        logger.info(f"Starting simulation with {min_len} candles, warmup={warmup}")

        # Main loop - iterate through time
        returns = []

        for i in range(warmup, min_len):
            # Get current candles for all assets
            current_candles = {
                symbol: candles[:i+1]
                for symbol, candles in all_candles.items()
            }

            # Current timestamp
            timestamp = list(all_candles.values())[0][i].timestamp

            # Calculate current exposure
            current_exposure = sum(
                pos["size"] / balance
                for pos in positions.values()
            )

            # Check for exits first
            for symbol in list(positions.keys()):
                pos = positions[symbol]
                candles = current_candles.get(symbol, [])
                if not candles:
                    continue

                current_price = candles[-1].close
                entry_price = pos["entry_price"]
                side = pos["side"]
                holding_candles = i - pos["entry_candle_idx"]

                # Calculate unrealized PnL
                if side == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                # Exit conditions based on strategy
                should_exit = False

                # 1. Pairs exit: check if we should exit both legs together
                if pos["strategy"] == "pairs" and pos.get("pair_symbol"):
                    pair_symbol = pos["pair_symbol"]
                    # Only process the exit once (from one leg)
                    if pair_symbol in positions:
                        should_exit, _ = self.simons_engine.should_exit_position(
                            symbol=symbol,
                            entry_z_score=pos["z_score"],
                            current_candles=current_candles,
                            holding_candles=holding_candles,
                            max_holding=pos["holding_period_target"] * 2,
                            strategy=pos["strategy"],
                        )
                    else:
                        # Other leg already exited, exit this one too
                        should_exit = True

                # 2. Mean reversion exit: z-score reverted
                elif pos["strategy"] == "mean_reversion":
                    should_exit, _ = self.simons_engine.should_exit_position(
                        symbol=symbol,
                        entry_z_score=pos["z_score"],
                        current_candles=current_candles,
                        holding_candles=holding_candles,
                        max_holding=pos["holding_period_target"] * 2,
                        strategy=pos["strategy"],
                    )

                # 3. Risk management exits (wider for pairs since hedged)
                stop_loss_pct = -0.02 if pos["strategy"] == "pairs" else -0.015
                take_profit_pct = 0.03 if pos["strategy"] == "pairs" else 0.025

                if pnl_pct <= stop_loss_pct:
                    should_exit = True
                elif pnl_pct >= take_profit_pct:
                    should_exit = True

                if should_exit:
                    # Calculate PnL
                    size_usd = pos["size"]
                    pnl = size_usd * pnl_pct * self.config.leverage
                    fees = size_usd * (self.config.taker_fee * 2)  # Entry + exit
                    net_pnl = pnl - fees

                    balance += net_pnl

                    # Record trade
                    trade_id += 1
                    self.trades.append(BacktestTrade(
                        id=str(trade_id),
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=current_price,
                        size_usd=size_usd,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct * 100,
                        fees=fees,
                        strategy=pos["strategy"],
                        conviction=pos.get("conviction", 50),
                        z_score=pos["z_score"],
                        holding_period=holding_candles,
                    ))

                    del positions[symbol]

            # Generate new signals
            signals = self.simons_engine.generate_signals(
                all_candles=current_candles,
                current_positions=positions,
            )

            # Filter and execute top signals
            for signal in signals:
                if signal.direction == "HOLD":
                    continue

                if signal.symbol in positions:
                    continue

                if signal.conviction < self.config.min_conviction:
                    continue

                # Calculate position size
                size_usd = self.position_sizer.calculate_size(
                    signal=signal,
                    balance=balance,
                    current_exposure=current_exposure,
                )

                if size_usd < 100:  # Min position size
                    continue

                # Check if we have room for more exposure
                if (current_exposure + size_usd / balance) > self.config.max_portfolio_exposure:
                    continue

                # Get entry price
                candles = current_candles.get(signal.symbol, [])
                if not candles:
                    continue

                entry_price = candles[-1].close

                # Apply slippage
                if signal.direction == "LONG":
                    entry_price *= (1 + self.config.slippage_bps / 10000)
                else:
                    entry_price *= (1 - self.config.slippage_bps / 10000)

                # Open position
                positions[signal.symbol] = {
                    "side": signal.direction,
                    "size": size_usd,
                    "entry_price": entry_price,
                    "entry_time": timestamp,
                    "entry_candle_idx": i,
                    "strategy": signal.strategy,
                    "conviction": signal.conviction,
                    "z_score": signal.z_score,
                    "holding_period_target": signal.holding_period,
                    "pair_symbol": signal.pair_symbol if signal.strategy == "pairs" else "",
                }

                current_exposure += size_usd / balance

                logger.debug(
                    f"ENTRY: {signal.direction} {signal.symbol} @ ${entry_price:,.2f} | "
                    f"Size: ${size_usd:,.2f} | Strategy: {signal.strategy} | "
                    f"Conv: {signal.conviction:.0f} | Z: {signal.z_score:.2f}"
                )

            # Track equity
            self.equity_curve.append((timestamp, balance))

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            dd = (peak_balance - balance) / peak_balance
            if dd > max_drawdown:
                max_drawdown = dd

            # Track returns
            if len(self.equity_curve) > 1:
                prev_balance = self.equity_curve[-2][1]
                ret = (balance - prev_balance) / prev_balance
                returns.append(ret)

        # Close any remaining positions at end
        for symbol, pos in list(positions.items()):
            candles = all_candles.get(symbol, [])
            if candles:
                current_price = candles[-1].close
                entry_price = pos["entry_price"]
                side = pos["side"]

                if side == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                size_usd = pos["size"]
                pnl = size_usd * pnl_pct * self.config.leverage
                fees = size_usd * (self.config.taker_fee * 2)
                balance += pnl - fees

        # Calculate metrics
        metrics = self._calculate_metrics(
            initial_balance=self.config.initial_balance,
            final_balance=balance,
            max_drawdown=max_drawdown,
            returns=returns,
        )

        if self.client:
            await self.client.close()

        return metrics

    def _calculate_metrics(
        self,
        initial_balance: float,
        final_balance: float,
        max_drawdown: float,
        returns: list[float],
    ) -> StatArbMetrics:
        """Calculate performance metrics."""
        metrics = StatArbMetrics()

        # Basic returns
        metrics.total_return = (final_balance - initial_balance) / initial_balance
        metrics.final_balance = final_balance
        metrics.max_drawdown = max_drawdown
        metrics.total_trades = len(self.trades)

        # Annualized returns (assuming hourly data)
        hours = len(returns)
        years = hours / (24 * 365)
        if years > 0 and final_balance > 0:
            metrics.cagr = (final_balance / initial_balance) ** (1 / years) - 1

        # Sharpe ratio (annualized)
        if returns and np.std(returns) > 0:
            hourly_sharpe = np.mean(returns) / np.std(returns)
            metrics.sharpe_ratio = hourly_sharpe * np.sqrt(24 * 365)

        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and np.std(negative_returns) > 0:
            hourly_sortino = np.mean(returns) / np.std(negative_returns)
            metrics.sortino_ratio = hourly_sortino * np.sqrt(24 * 365)

        # Trade statistics
        if self.trades:
            wins = [t for t in self.trades if t.pnl > 0]
            losses = [t for t in self.trades if t.pnl <= 0]

            metrics.win_rate = len(wins) / len(self.trades) if self.trades else 0
            metrics.avg_trade_pnl = float(np.mean([t.pnl for t in self.trades]))
            metrics.avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            metrics.avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0

            # Profit factor
            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 1
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Strategy breakdown
            pairs_trades = [t for t in self.trades if t.strategy == "pairs"]
            mr_trades = [t for t in self.trades if t.strategy == "mean_reversion"]
            mom_trades = [t for t in self.trades if t.strategy == "momentum"]

            metrics.pairs_trades = len(pairs_trades)
            metrics.pairs_pnl = sum(t.pnl for t in pairs_trades)
            metrics.pairs_win_rate = (
                sum(1 for t in pairs_trades if t.pnl > 0) / len(pairs_trades)
                if pairs_trades else 0
            )

            metrics.mean_reversion_trades = len(mr_trades)
            metrics.mean_reversion_pnl = sum(t.pnl for t in mr_trades)
            metrics.mean_reversion_win_rate = (
                sum(1 for t in mr_trades if t.pnl > 0) / len(mr_trades)
                if mr_trades else 0
            )

            metrics.momentum_trades = len(mom_trades)
            metrics.momentum_pnl = sum(t.pnl for t in mom_trades)
            metrics.momentum_win_rate = (
                sum(1 for t in mom_trades if t.pnl > 0) / len(mom_trades)
                if mom_trades else 0
            )

        return metrics


def print_results(config: BacktestConfig, metrics: StatArbMetrics, trades: list[BacktestTrade]):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("STATISTICAL ARBITRAGE BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Assets: {len(config.assets)} ({', '.join(config.assets[:5])}...)")
    print(f"  Period: {config.days} days")
    print(f"  Initial Balance: ${config.initial_balance:,.0f}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Max Portfolio Exposure: {config.max_portfolio_exposure:.0%}")

    print(f"\n{'=' * 70}")
    print("PERFORMANCE METRICS")
    print("=" * 70)

    # Compare to targets
    cagr_status = "OK" if metrics.cagr >= 0.66 else "X"
    dd_status = "OK" if metrics.max_drawdown <= 0.05 else "X"
    sharpe_status = "OK" if metrics.sharpe_ratio >= 2.5 else "X"

    print(f"\n{'Metric':<30} {'Value':>15} {'Target':>15} {'Status':>8}")
    print("-" * 70)
    print(f"{'Annualized Return (CAGR)':<30} {metrics.cagr:>14.1%} {'>66%':>15} {cagr_status:>8}")
    print(f"{'Max Drawdown':<30} {metrics.max_drawdown:>14.1%} {'<5%':>15} {dd_status:>8}")
    print(f"{'Sharpe Ratio':<30} {metrics.sharpe_ratio:>14.2f} {'>2.5':>15} {sharpe_status:>8}")
    print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio:>14.2f}")
    print(f"{'Win Rate':<30} {metrics.win_rate:>14.1%}")
    print(f"{'Profit Factor':<30} {metrics.profit_factor:>14.2f}")
    print(f"{'Total Trades':<30} {metrics.total_trades:>14}")

    print(f"\n{'=' * 70}")
    print("STRATEGY BREAKDOWN")
    print("=" * 70)

    print(f"\n{'Strategy':<20} {'Trades':>10} {'P&L':>15} {'Win Rate':>12}")
    print("-" * 60)
    print(f"{'Pairs Trading':<20} {metrics.pairs_trades:>10} ${metrics.pairs_pnl:>+13,.2f} {metrics.pairs_win_rate:>11.1%}")
    print(f"{'Mean Reversion':<20} {metrics.mean_reversion_trades:>10} ${metrics.mean_reversion_pnl:>+13,.2f} {metrics.mean_reversion_win_rate:>11.1%}")
    print(f"{'Momentum':<20} {metrics.momentum_trades:>10} ${metrics.momentum_pnl:>+13,.2f} {metrics.momentum_win_rate:>11.1%}")

    print(f"\n{'=' * 70}")
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"\n  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Final Balance:   ${metrics.final_balance:,.2f}")
    print(f"  Total Return:    {metrics.total_return:+.1%}")

    if trades:
        print(f"\n{'=' * 70}")
        print("RECENT TRADES (last 20)")
        print("=" * 70)
        print(f"\n{'Time':<20} {'Symbol':<10} {'Side':<6} {'Strategy':<15} {'PnL':>12}")
        print("-" * 70)
        for trade in trades[-20:]:
            pnl_str = f"+${trade.pnl:,.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):,.2f}"
            print(f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<20} {trade.symbol:<10} {trade.side:<6} {trade.strategy:<15} {pnl_str:>12}")

    print("\n" + "=" * 70)

    # Overall assessment
    targets_met = sum([
        metrics.cagr >= 0.66,
        metrics.max_drawdown <= 0.05,
        metrics.sharpe_ratio >= 2.5,
    ])

    if targets_met == 3:
        print("MEDALLION TARGETS MET! Strategy performs at hedge fund level.")
    elif targets_met >= 2:
        print("STRONG PERFORMANCE - 2/3 targets met. Close to Medallion level.")
    elif targets_met >= 1:
        print("DECENT PERFORMANCE - 1/3 targets met. More tuning needed.")
    else:
        print("BELOW TARGETS - All targets missed. Significant tuning needed.")

    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Backtest"
    )
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--balance", type=float, default=10000, help="Initial balance")
    parser.add_argument("--leverage", type=int, default=3, help="Leverage (default 3x)")
    parser.add_argument("--exposure", type=float, default=0.60, help="Max portfolio exposure (default 60%%)")
    parser.add_argument("--min-conviction", type=float, default=60, help="Minimum conviction to trade (default 60)")

    args = parser.parse_args()

    config = BacktestConfig(
        days=args.days,
        initial_balance=args.balance,
        leverage=args.leverage,
        max_portfolio_exposure=args.exposure,
        min_conviction=args.min_conviction,
    )

    engine = StatArbBacktestEngine(config)
    metrics = await engine.run()

    print_results(config, metrics, engine.trades)


if __name__ == "__main__":
    asyncio.run(main())
