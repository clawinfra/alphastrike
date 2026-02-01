#!/usr/bin/env python3
"""
WEEX Historical Backtest Runner

Downloads historical data and runs a backtest simulation.
Uses the v2 API endpoints directly for maximum compatibility.

Usage:
    python scripts/run_backtest.py --symbol BTCUSDT --days 7
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# WEEX API Base URL (candles endpoint is public, no auth needed)
BASE_URL = "https://api-contract.weex.com"


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    """Simulated trade record."""
    entry_time: datetime
    exit_time: datetime | None
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float | None
    size: float
    pnl: float = 0.0
    status: str = "open"


async def fetch_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
) -> list[Candle]:
    """
    Fetch historical candles from WEEX v2 API.

    NOTE: WEEX v2 API only returns the most recent `limit` candles.
    No startTime/endTime parameters are supported.
    For longer history, use larger intervals (1h, 4h, 1d).

    Args:
        session: aiohttp session
        symbol: Trading symbol (e.g., "BTCUSDT")
        interval: Candle interval (1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w)
        limit: Number of candles (max 1000)
    """
    # WEEX symbol format
    weex_symbol = f"cmt_{symbol.lower()}"

    url = f"{BASE_URL}/capi/v2/market/candles"
    params = {
        "symbol": weex_symbol,
        "granularity": interval,
        "limit": str(min(limit, 1000)),  # Max 1000
    }

    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed to fetch candles: HTTP {response.status}: {text}")
                return []

            data = await response.json()

            # Unwrap WEEX response
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            candles = []
            if isinstance(data, list):
                for item in data:
                    # Format: [timestamp, open, high, low, close, base_volume, quote_volume]
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(int(item[0]) / 1000),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    )
                    candles.append(candle)

            # Sort by timestamp ascending
            candles.sort(key=lambda c: c.timestamp)
            logger.info(f"Downloaded {len(candles)} candles for {symbol} ({interval})")
            return candles

    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        return []


def calculate_rsi(prices: list[float], period: int = 14) -> float:
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50.0

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent_changes = changes[-period:]

    gains = [c if c > 0 else 0 for c in recent_changes]
    losses = [-c if c < 0 else 0 for c in recent_changes]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices: list[float], period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    return sum(prices[-period:]) / period


def run_simple_backtest(
    candles: list[Candle],
    initial_balance: float = 10000.0,
    position_size_pct: float = 0.10,
    leverage: int = 5,
    slippage_bps: int = 5,
) -> dict:
    """
    Run a simple moving average crossover backtest.

    Strategy:
    - Long when price > SMA20 and RSI < 70
    - Short when price < SMA20 and RSI > 30
    - Exit on opposite signal or SL/TP hit
    """
    if len(candles) < 30:
        return {"error": "Not enough candles for backtest"}

    balance = initial_balance
    equity_curve = [balance]
    trades: list[Trade] = []
    open_trade: Trade | None = None

    # Parameters
    sma_period = 20
    rsi_period = 14
    stop_loss_pct = 0.02  # 2%
    take_profit_pct = 0.03  # 3%

    prices = [c.close for c in candles]

    for i in range(sma_period, len(candles)):
        candle = candles[i]
        current_price = candle.close

        # Calculate indicators
        recent_prices = prices[: i + 1]
        sma = calculate_sma(recent_prices, sma_period)
        rsi = calculate_rsi(recent_prices, rsi_period)

        # Check for SL/TP on open trade
        if open_trade:
            if open_trade.side == "long":
                pnl_pct = (current_price - open_trade.entry_price) / open_trade.entry_price
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    # Close trade
                    exit_price = current_price * (1 - slippage_bps / 10000)  # Slippage on exit
                    pnl = (exit_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size * leverage
                    open_trade.exit_price = exit_price
                    open_trade.exit_time = candle.timestamp
                    open_trade.pnl = pnl
                    open_trade.status = "closed"
                    balance += pnl
                    trades.append(open_trade)
                    open_trade = None

            elif open_trade.side == "short":
                pnl_pct = (open_trade.entry_price - current_price) / open_trade.entry_price
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    # Close trade
                    exit_price = current_price * (1 + slippage_bps / 10000)
                    pnl = (open_trade.entry_price - exit_price) / open_trade.entry_price * open_trade.size * leverage
                    open_trade.exit_price = exit_price
                    open_trade.exit_time = candle.timestamp
                    open_trade.pnl = pnl
                    open_trade.status = "closed"
                    balance += pnl
                    trades.append(open_trade)
                    open_trade = None

        # Generate signals (only if no open trade)
        if not open_trade:
            position_value = balance * position_size_pct

            # Long signal: price above SMA and RSI not overbought
            if current_price > sma and rsi < 70:
                entry_price = current_price * (1 + slippage_bps / 10000)  # Slippage on entry
                open_trade = Trade(
                    entry_time=candle.timestamp,
                    exit_time=None,
                    side="long",
                    entry_price=entry_price,
                    exit_price=None,
                    size=position_value,
                )

            # Short signal: price below SMA and RSI not oversold
            elif current_price < sma and rsi > 30:
                entry_price = current_price * (1 - slippage_bps / 10000)
                open_trade = Trade(
                    entry_time=candle.timestamp,
                    exit_time=None,
                    side="short",
                    entry_price=entry_price,
                    exit_price=None,
                    size=position_value,
                )

        # Track equity
        if open_trade:
            if open_trade.side == "long":
                unrealized = (current_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size * leverage
            else:
                unrealized = (open_trade.entry_price - current_price) / open_trade.entry_price * open_trade.size * leverage
            equity_curve.append(balance + unrealized)
        else:
            equity_curve.append(balance)

    # Close any remaining open trade at last price
    if open_trade:
        last_price = candles[-1].close
        if open_trade.side == "long":
            pnl = (last_price - open_trade.entry_price) / open_trade.entry_price * open_trade.size * leverage
        else:
            pnl = (open_trade.entry_price - last_price) / open_trade.entry_price * open_trade.size * leverage
        open_trade.exit_price = last_price
        open_trade.exit_time = candles[-1].timestamp
        open_trade.pnl = pnl
        open_trade.status = "closed"
        balance += pnl
        trades.append(open_trade)

    # Calculate metrics
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (simplified, assuming daily returns)
    if len(equity_curve) > 1:
        returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] for i in range(1, len(equity_curve))]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return) * (252**0.5) if std_return > 0 else 0
    else:
        sharpe = 0

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "total_return_pct": total_return,
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": win_rate,
        "max_drawdown_pct": max_dd * 100,
        "sharpe_ratio": sharpe,
        "leverage": leverage,
        "position_size_pct": position_size_pct * 100,
        "trades": trades,
    }


async def main():
    """Main backtest runner."""
    parser = argparse.ArgumentParser(description="Run WEEX historical backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="Candle interval (default: 1h). Use larger intervals for more history.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of candles (max: 1000)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance (default: 10000)")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage (default: 5)")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size as fraction (default: 0.10)")
    args = parser.parse_args()

    # Calculate approximate history based on interval
    interval_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "12h": 12, "1d": 24, "1w": 168}
    hours = interval_hours.get(args.interval, 1)
    approx_days = (hours * args.limit) / 24

    print("=" * 60)
    print("WEEX Historical Backtest")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Candles: {args.limit} (~{approx_days:.1f} days of history)")
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Leverage: {args.leverage}x")
    print(f"Position Size: {args.position_size * 100:.0f}%")
    print()
    print("NOTE: WEEX API returns most recent candles only (no date range support)")
    print()

    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Download candles
        candles = await fetch_candles(
            session,
            args.symbol,
            args.interval,
            args.limit,
        )

        if not candles:
            print("ERROR: No candles downloaded. Check API connectivity.")
            return 1

        print(f"\nDownloaded {len(candles)} candles")
        print(f"First candle: {candles[0].timestamp} - Close: ${candles[0].close:,.2f}")
        print(f"Last candle: {candles[-1].timestamp} - Close: ${candles[-1].close:,.2f}")
        print()

        # Run backtest
        print("Running backtest...")
        results = run_simple_backtest(
            candles,
            initial_balance=args.balance,
            position_size_pct=args.position_size,
            leverage=args.leverage,
        )

        if "error" in results:
            print(f"ERROR: {results['error']}")
            return 1

        # Print results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Balance:  ${results['initial_balance']:>12,.2f}")
        print(f"Final Balance:    ${results['final_balance']:>12,.2f}")
        print(f"Total Return:     {results['total_return_pct']:>12.2f}%")
        print()
        print(f"Total Trades:     {results['total_trades']:>12}")
        print(f"Winning Trades:   {results['winning_trades']:>12}")
        print(f"Losing Trades:    {results['losing_trades']:>12}")
        print(f"Win Rate:         {results['win_rate_pct']:>12.1f}%")
        print()
        print(f"Max Drawdown:     {results['max_drawdown_pct']:>12.2f}%")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:>12.2f}")
        print()

        # Print recent trades
        if results["trades"]:
            print("\nRecent Trades:")
            print("-" * 60)
            for trade in results["trades"][-10:]:
                pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
                print(f"  {trade.entry_time} | {trade.side:5} | Entry: ${trade.entry_price:,.2f} | PnL: {pnl_str}")

        print("\n" + "=" * 60)

        # Performance summary
        if results["total_return_pct"] > 0:
            print(f"Result: PROFITABLE (+{results['total_return_pct']:.2f}%)")
        else:
            print(f"Result: LOSS ({results['total_return_pct']:.2f}%)")

        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
