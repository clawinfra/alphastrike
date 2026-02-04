#!/usr/bin/env python3
"""
Hyperliquid Historical Data Collector

Fetches historical candles from Hyperliquid and saves to SQLite database.
This data is used for ML model training.

Per ARCHITECTURE.md Section 5.1:
- Minimum 1000 candles required for training
- Data stored in database for reproducibility

Usage:
    python scripts/hyperliquid_data_collector.py --days 90
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Asset universe for multi-asset strategy
ASSET_UNIVERSE = [
    {"symbol": "BTC", "class": "crypto_major"},
    {"symbol": "ETH", "class": "crypto_major"},
    {"symbol": "PAXG", "class": "gold_proxy"},  # Key diversifier
    {"symbol": "SOL", "class": "crypto_l1"},
    {"symbol": "AAVE", "class": "crypto_defi"},
]


async def fetch_hyperliquid_candles(
    adapter,
    symbol: str,
    interval: str,
    days: int,
) -> list:
    """
    Fetch historical candles from Hyperliquid.

    Args:
        adapter: Initialized HyperliquidAdapter
        symbol: Asset symbol (e.g., "BTC")
        interval: Candle interval (e.g., "1h")
        days: Number of days of history

    Returns:
        List of UnifiedCandle objects
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    unified_symbol = f"{symbol}USDT"

    logger.info(f"Fetching {symbol} candles from {start_time.date()} to {end_time.date()}...")

    candles = await adapter.rest.get_candles(
        symbol=unified_symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=days * 24,  # Hourly candles
    )

    return candles


async def save_candles_to_db(candles: list, symbol: str, interval: str) -> int:
    """
    Save candles to SQLite database.

    Args:
        candles: List of UnifiedCandle objects
        symbol: Asset symbol
        interval: Candle interval

    Returns:
        Number of candles saved
    """
    from src.data.database import Candle, Database

    db = Database()
    await db.initialize()

    try:
        # Convert UnifiedCandle to database Candle format
        # Use hyperliquid_{symbol} format to distinguish from WEEX data
        db_symbol = f"hyperliquid_{symbol.lower()}"

        db_candles = [
            Candle(
                symbol=db_symbol,
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                interval=interval,
            )
            for c in candles
        ]

        saved = await db.save_candles(db_candles)
        logger.info(f"  {symbol}: Saved {saved} candles to database (symbol={db_symbol})")
        return saved

    finally:
        await db.close()


def validate_candles(candles: list, symbol: str) -> bool:
    """
    Validate candle data quality.

    Checks:
    - Minimum count
    - No large gaps
    - Valid OHLCV
    """
    if len(candles) < 500:
        logger.warning(f"  {symbol}: Only {len(candles)} candles (need 500+)")
        return False

    # Check for gaps
    gap_count = 0
    for i in range(1, len(candles)):
        time_diff = (candles[i].timestamp - candles[i-1].timestamp).total_seconds()
        if time_diff > 2 * 3600:  # More than 2 hours gap
            gap_count += 1

    if gap_count > 10:
        logger.warning(f"  {symbol}: {gap_count} gaps > 2 hours detected")

    # Check OHLCV validity
    invalid_count = 0
    for c in candles:
        if not (c.low <= c.open <= c.high and c.low <= c.close <= c.high):
            invalid_count += 1
        if c.volume < 0:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(f"  {symbol}: {invalid_count} invalid OHLCV records")

    return True


async def main():
    parser = argparse.ArgumentParser(description="Hyperliquid Data Collector")
    parser.add_argument("--days", type=int, default=90, help="Days of history to fetch")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval")
    parser.add_argument("--symbols", type=str, nargs="+", help="Specific symbols (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("HYPERLIQUID HISTORICAL DATA COLLECTOR")
    print("=" * 70)
    print(f"Days: {args.days}")
    print(f"Interval: {args.interval}")
    print(f"Expected candles per asset: ~{args.days * 24}")
    print()

    # Initialize Hyperliquid adapter
    from src.exchange.adapters.hyperliquid import HyperliquidAdapter

    adapter = HyperliquidAdapter()
    await adapter.initialize()

    try:
        # Determine which symbols to fetch
        if args.symbols:
            symbols = [{"symbol": s, "class": "custom"} for s in args.symbols]
        else:
            symbols = ASSET_UNIVERSE

        print(f"Fetching data for {len(symbols)} assets...")
        print("-" * 40)

        total_saved = 0
        results = {}

        for asset in symbols:
            symbol = asset["symbol"]
            asset_class = asset["class"]

            try:
                # Fetch candles
                candles = await fetch_hyperliquid_candles(
                    adapter=adapter,
                    symbol=symbol,
                    interval=args.interval,
                    days=args.days,
                )

                if not candles:
                    logger.warning(f"  {symbol}: No candles fetched")
                    results[symbol] = {"status": "failed", "count": 0}
                    continue

                # Validate
                is_valid = validate_candles(candles, symbol)

                # Save to database
                saved = await save_candles_to_db(candles, symbol, args.interval)
                total_saved += saved

                results[symbol] = {
                    "status": "success" if is_valid else "warning",
                    "count": saved,
                    "class": asset_class,
                }

            except Exception as e:
                logger.error(f"  {symbol}: Error - {e}")
                results[symbol] = {"status": "error", "count": 0, "error": str(e)}

        # Print summary
        print()
        print("=" * 70)
        print("COLLECTION SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Symbol':<10} {'Class':<15} {'Status':<10} {'Candles':>10}")
        print("-" * 50)

        for symbol, result in results.items():
            status_icon = "✅" if result["status"] == "success" else ("⚠️" if result["status"] == "warning" else "❌")
            asset_class = result.get("class", "unknown")
            print(f"{symbol:<10} {asset_class:<15} {status_icon:<10} {result['count']:>10}")

        print("-" * 50)
        print(f"{'TOTAL':<10} {'':<15} {'':<10} {total_saved:>10}")
        print()

        # Check if we have enough data for training
        min_candles = 500
        ready_assets = sum(1 for r in results.values() if r["count"] >= min_candles)

        if ready_assets >= 3:
            print(f"✅ {ready_assets} assets ready for ML training (need 3+)")
            print()
            print("Next step: Run model training")
            print("  python scripts/train_hyperliquid_models.py")
        else:
            print(f"⚠️  Only {ready_assets} assets have enough data (need 3+)")
            print("   Consider fetching more history with --days 180")

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
