"""
Candle Data Cache

Caches historical candle data locally to reduce API calls.
Supports incremental updates - only fetches new data not already cached.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "candle_cache"
WEEX_BASE_URL = "https://api-contract.weex.com"


def get_cache_path(symbol: str, interval: str) -> Path:
    """Get path to cache file for a symbol/interval combination."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.lower()}_{interval}.json"


def load_cached_candles(symbol: str, interval: str) -> list[dict]:
    """Load candles from cache file."""
    cache_path = get_cache_path(symbol, interval)

    if not cache_path.exists():
        return []

    try:
        with open(cache_path) as f:
            data = json.load(f)
            candles = data.get("candles", [])
            logger.info(f"Loaded {len(candles)} cached candles for {symbol} {interval}")
            return candles
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load cache for {symbol}: {e}")
        return []


def save_cached_candles(symbol: str, interval: str, candles: list[dict]) -> None:
    """Save candles to cache file."""
    cache_path = get_cache_path(symbol, interval)

    # Sort by timestamp and remove duplicates
    seen = set()
    unique_candles = []
    for c in sorted(candles, key=lambda x: x["timestamp"]):
        if c["timestamp"] not in seen:
            seen.add(c["timestamp"])
            unique_candles.append(c)

    data = {
        "symbol": symbol,
        "interval": interval,
        "updated_at": datetime.now(UTC).isoformat(),
        "count": len(unique_candles),
        "candles": unique_candles,
    }

    with open(cache_path, "w") as f:
        json.dump(data, f)

    logger.info(f"Saved {len(unique_candles)} candles to cache for {symbol} {interval}")


async def fetch_candles_batch(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int = 1000,
    end_time: int | None = None,
) -> list[dict]:
    """Fetch a single batch of candles from WEEX API."""
    weex_symbol = f"cmt_{symbol.lower()}"
    url = f"{WEEX_BASE_URL}/capi/v2/market/candles"

    params = {
        "symbol": weex_symbol,
        "granularity": interval,
        "limit": str(min(limit, 1000)),
    }

    if end_time:
        params["endTime"] = str(end_time)

    try:
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"API error: {response.status}")
                return []

            data = await response.json()

            # Handle both list and dict responses
            if isinstance(data, dict):
                raw_candles = data.get("data", [])
            elif isinstance(data, list):
                raw_candles = data
            else:
                logger.error(f"Unexpected response type: {type(data)}")
                return []

            # Convert to dict format
            candles = []
            for c in raw_candles:
                if isinstance(c, list) and len(c) >= 6:
                    candles.append({
                        "timestamp": int(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })

            return candles

    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        return []


async def fetch_candles_with_cache(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    days: int = 90,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Fetch candles with caching support.

    Args:
        session: aiohttp session
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Candle interval (e.g., "1h")
        days: Number of days of history to fetch
        force_refresh: If True, ignore cache and fetch all data

    Returns:
        List of candle dictionaries sorted by timestamp
    """
    # Calculate how many candles we need
    interval_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}
    hours_per_candle = interval_hours.get(interval, 1)
    target_candles = int((days * 24) / hours_per_candle)

    # Load existing cache
    cached = [] if force_refresh else load_cached_candles(symbol, interval)

    if cached:
        # Find the oldest cached timestamp
        oldest_cached = min(c["timestamp"] for c in cached)
        newest_cached = max(c["timestamp"] for c in cached)

        # Calculate how far back we need to go
        target_start = datetime.now(UTC) - timedelta(days=days)
        target_start_ms = int(target_start.timestamp() * 1000)

        # Check if we have enough historical data
        if oldest_cached <= target_start_ms:
            logger.info(f"Cache has sufficient history for {symbol}")
            # Just need to fetch new candles since last cache
            new_candles = await _fetch_new_candles(session, symbol, interval, newest_cached)
            if new_candles:
                cached.extend(new_candles)
                save_cached_candles(symbol, interval, cached)
            return sorted(cached, key=lambda x: x["timestamp"])

        # Need to fetch older data
        logger.info(f"Cache missing older data for {symbol}, fetching...")
        older_candles = await _fetch_older_candles(
            session, symbol, interval, oldest_cached, target_start_ms
        )
        cached.extend(older_candles)

        # Also fetch new candles
        new_candles = await _fetch_new_candles(session, symbol, interval, newest_cached)
        cached.extend(new_candles)

        save_cached_candles(symbol, interval, cached)
        return sorted(cached, key=lambda x: x["timestamp"])

    # No cache - fetch all data
    logger.info(f"No cache for {symbol}, fetching {target_candles} candles...")
    all_candles = await _fetch_all_candles(session, symbol, interval, target_candles)

    if all_candles:
        save_cached_candles(symbol, interval, all_candles)

    return all_candles


async def _fetch_new_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    since_timestamp: int,
) -> list[dict]:
    """Fetch candles newer than the given timestamp."""
    new_candles = []

    # Fetch latest batch
    batch = await fetch_candles_batch(session, symbol, interval, limit=1000)

    for c in batch:
        if c["timestamp"] > since_timestamp:
            new_candles.append(c)

    if new_candles:
        logger.info(f"Fetched {len(new_candles)} new candles for {symbol}")

    return new_candles


async def _fetch_older_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    current_oldest: int,
    target_oldest: int,
) -> list[dict]:
    """Fetch candles older than current cache, going back to target."""
    older_candles = []
    end_time = current_oldest - 1

    while end_time > target_oldest:
        batch = await fetch_candles_batch(session, symbol, interval, limit=1000, end_time=end_time)

        if not batch:
            logger.warning(f"No more historical data available for {symbol}")
            break

        older_candles.extend(batch)

        # Update end_time to oldest in batch
        oldest_in_batch = min(c["timestamp"] for c in batch)

        if oldest_in_batch >= end_time:
            # No progress, API limit reached
            logger.warning(f"API historical limit reached for {symbol}")
            break

        end_time = oldest_in_batch - 1
        logger.info(f"Fetched batch: {len(batch)} candles, total older: {len(older_candles)}")

        # Safety limit
        if len(older_candles) > 10000:
            logger.warning("Safety limit reached, stopping fetch")
            break

    return older_candles


async def _fetch_all_candles(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    target_count: int,
) -> list[dict]:
    """Fetch all candles up to target count using pagination."""
    all_candles = []
    end_time = None

    while len(all_candles) < target_count:
        batch = await fetch_candles_batch(
            session, symbol, interval,
            limit=min(1000, target_count - len(all_candles)),
            end_time=end_time
        )

        if not batch:
            break

        all_candles.extend(batch)

        # Get oldest timestamp for next batch
        oldest = min(c["timestamp"] for c in batch)

        if end_time is not None and oldest >= end_time:
            # No progress, reached API limit
            logger.warning(f"Reached API historical limit at {len(all_candles)} candles")
            break

        end_time = oldest - 1
        logger.info(f"Fetched batch: {len(batch)}, total: {len(all_candles)}/{target_count}")

    return sorted(all_candles, key=lambda x: x["timestamp"])


def get_cache_info(symbol: str, interval: str) -> dict:
    """Get information about cached data for a symbol."""
    cache_path = get_cache_path(symbol, interval)

    if not cache_path.exists():
        return {"cached": False, "count": 0}

    try:
        with open(cache_path) as f:
            data = json.load(f)
            candles = data.get("candles", [])

            if not candles:
                return {"cached": True, "count": 0}

            oldest = min(c["timestamp"] for c in candles)
            newest = max(c["timestamp"] for c in candles)

            return {
                "cached": True,
                "count": len(candles),
                "oldest": datetime.fromtimestamp(oldest / 1000, tz=UTC),
                "newest": datetime.fromtimestamp(newest / 1000, tz=UTC),
                "updated_at": data.get("updated_at"),
            }
    except Exception as e:
        return {"cached": False, "error": str(e)}


def clear_cache(symbol: str | None = None, interval: str | None = None) -> None:
    """Clear cached data."""
    if symbol and interval:
        cache_path = get_cache_path(symbol, interval)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache for {symbol} {interval}")
    else:
        # Clear all cache
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
        logger.info("Cleared all cache")
