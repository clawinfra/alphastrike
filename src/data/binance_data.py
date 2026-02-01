"""
Binance Data Fetcher

Fetches historical candle data from Binance Futures API with proper pagination.
Supports fetching months/years of historical data.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)

# Binance Futures API
BINANCE_FUTURES_URL = "https://fapi.binance.com"
BINANCE_SPOT_URL = "https://api.binance.com"

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "candle_cache"


def get_cache_path(symbol: str, interval: str, source: str = "binance") -> Path:
    """Get path to cache file for a symbol/interval combination."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.lower()}_{interval}_{source}.json"


def interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    multipliers = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }
    return multipliers.get(interval, 60 * 60 * 1000)


async def fetch_binance_candles_batch(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
    use_futures: bool = True,
) -> list[dict]:
    """
    Fetch a single batch of candles from Binance API.

    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Number of candles (max 1500 for futures, 1000 for spot)
        use_futures: Use futures API (default) or spot API

    Returns:
        List of candle dictionaries
    """
    if use_futures:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        max_limit = 1500
    else:
        url = f"{BINANCE_SPOT_URL}/api/v3/klines"
        max_limit = 1000

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, max_limit),
    }

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Binance API error {response.status}: {text}")
                return []

            data = await response.json()

            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            candles = []
            for k in data:
                candles.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                })

            return candles

    except Exception as e:
        logger.error(f"Failed to fetch Binance candles: {e}")
        return []


async def fetch_binance_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int = 90,
    use_futures: bool = True,
) -> list[dict]:
    """
    Fetch historical candles from Binance with pagination.

    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        days: Number of days of history to fetch
        use_futures: Use futures API (default) or spot API

    Returns:
        List of candle dictionaries sorted by timestamp
    """
    all_candles = []
    interval_ms = interval_to_ms(interval)
    batch_size = 1000

    # Calculate time range
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current_start = start_time
    request_count = 0
    max_requests = 100  # Safety limit

    logger.info(f"Fetching {symbol} {interval} from Binance ({days} days)...")

    while current_start < end_time and request_count < max_requests:
        batch = await fetch_binance_candles_batch(
            session,
            symbol,
            interval,
            start_time=current_start,
            end_time=end_time,
            limit=batch_size,
            use_futures=use_futures,
        )

        if not batch:
            logger.warning(f"No more data from Binance at {datetime.fromtimestamp(current_start/1000)}")
            break

        all_candles.extend(batch)
        request_count += 1

        # Move start time forward
        last_timestamp = max(c["timestamp"] for c in batch)
        if last_timestamp <= current_start:
            # No progress, break to avoid infinite loop
            break

        current_start = last_timestamp + interval_ms

        if request_count % 5 == 0:
            logger.info(f"  Fetched {len(all_candles)} candles...")

        # Rate limiting - Binance allows 1200 requests/min
        await asyncio.sleep(0.1)

    # Remove duplicates and sort
    seen = set()
    unique_candles = []
    for c in sorted(all_candles, key=lambda x: x["timestamp"]):
        if c["timestamp"] not in seen:
            seen.add(c["timestamp"])
            unique_candles.append(c)

    logger.info(f"Fetched {len(unique_candles)} unique candles for {symbol}")
    return unique_candles


async def fetch_binance_with_cache(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int = 90,
    use_futures: bool = True,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch Binance candles with caching support.

    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        days: Number of days of history
        use_futures: Use futures API
        force_refresh: Ignore cache and fetch fresh data

    Returns:
        List of candle dictionaries
    """
    cache_path = get_cache_path(symbol, interval, "binance")

    # Try to load from cache
    cached_candles = []
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                cached_candles = data.get("candles", [])
                logger.info(f"Loaded {len(cached_candles)} cached Binance candles for {symbol}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache: {e}")

    if cached_candles:
        # Check if we have enough history
        oldest_cached = min(c["timestamp"] for c in cached_candles)
        target_start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        if oldest_cached <= target_start:
            # Cache has enough history, just fetch new candles
            newest_cached = max(c["timestamp"] for c in cached_candles)
            new_candles = await fetch_binance_candles_batch(
                session, symbol, interval,
                start_time=newest_cached + interval_to_ms(interval),
                use_futures=use_futures,
            )

            if new_candles:
                cached_candles.extend(new_candles)
                _save_cache(cache_path, symbol, interval, cached_candles)

            return sorted(cached_candles, key=lambda x: x["timestamp"])

        # Need to fetch older data
        logger.info(f"Cache missing older data, fetching from Binance...")

    # Fetch all data
    all_candles = await fetch_binance_candles(
        session, symbol, interval, days=days, use_futures=use_futures
    )

    # Merge with cache
    if cached_candles:
        seen = {c["timestamp"] for c in all_candles}
        for c in cached_candles:
            if c["timestamp"] not in seen:
                all_candles.append(c)

    # Sort and save
    all_candles = sorted(all_candles, key=lambda x: x["timestamp"])
    _save_cache(cache_path, symbol, interval, all_candles)

    return all_candles


def _save_cache(cache_path: Path, symbol: str, interval: str, candles: list[dict]) -> None:
    """Save candles to cache file."""
    # Remove duplicates
    seen = set()
    unique = []
    for c in sorted(candles, key=lambda x: x["timestamp"]):
        if c["timestamp"] not in seen:
            seen.add(c["timestamp"])
            unique.append(c)

    data = {
        "symbol": symbol,
        "interval": interval,
        "source": "binance",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(unique),
        "candles": unique,
    }

    with open(cache_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(unique)} Binance candles to cache for {symbol}")


def get_binance_cache_info(symbol: str, interval: str) -> dict:
    """Get information about cached Binance data."""
    cache_path = get_cache_path(symbol, interval, "binance")

    if not cache_path.exists():
        return {"cached": False, "count": 0, "source": "binance"}

    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
            candles = data.get("candles", [])

            if not candles:
                return {"cached": True, "count": 0, "source": "binance"}

            oldest = min(c["timestamp"] for c in candles)
            newest = max(c["timestamp"] for c in candles)

            return {
                "cached": True,
                "count": len(candles),
                "source": "binance",
                "oldest": datetime.fromtimestamp(oldest / 1000, tz=timezone.utc),
                "newest": datetime.fromtimestamp(newest / 1000, tz=timezone.utc),
                "updated_at": data.get("updated_at"),
                "days_of_data": (newest - oldest) / (24 * 60 * 60 * 1000),
            }
    except Exception as e:
        return {"cached": False, "error": str(e), "source": "binance"}


# CLI for testing
async def main():
    """Test Binance data fetching."""
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    print(f"Fetching {days} days of {symbol} 1h candles from Binance...")

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        candles = await fetch_binance_with_cache(session, symbol, "1h", days=days)

        if candles:
            oldest = datetime.fromtimestamp(candles[0]["timestamp"] / 1000)
            newest = datetime.fromtimestamp(candles[-1]["timestamp"] / 1000)
            print(f"Fetched {len(candles)} candles")
            print(f"Range: {oldest} to {newest}")
            print(f"Days: {(candles[-1]['timestamp'] - candles[0]['timestamp']) / (24*60*60*1000):.1f}")
        else:
            print("No candles fetched")


if __name__ == "__main__":
    asyncio.run(main())
