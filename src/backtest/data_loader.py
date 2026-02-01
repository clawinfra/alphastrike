"""
Historical Data Loader

Downloads and caches historical candle data from WEEX API.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from src.data.database import Candle, Database
from src.exchange.adapters.weex.adapter import WEEXRESTClient
from src.exchange.models import UnifiedCandle

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# WEEX API limits
MAX_CANDLES_PER_REQUEST = 1000


def unified_to_candle(unified: UnifiedCandle) -> Candle:
    """Convert UnifiedCandle to database Candle format."""
    return Candle(
        symbol=unified.symbol,
        timestamp=unified.timestamp,
        open=unified.open,
        high=unified.high,
        low=unified.low,
        close=unified.close,
        volume=unified.volume,
        interval=unified.interval,
    )


def candle_to_unified(candle: Candle) -> UnifiedCandle:
    """Convert database Candle to UnifiedCandle format."""
    return UnifiedCandle(
        symbol=candle.symbol,
        timestamp=candle.timestamp,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
        interval=candle.interval,
    )


class DataLoader:
    """
    Historical data loader with caching.

    Downloads candles from WEEX API and caches them in SQLite.
    Handles pagination for large date ranges.
    """

    def __init__(
        self,
        database: Database,
        rest_client: WEEXRESTClient,
    ):
        """
        Initialize data loader.

        Args:
            database: Database instance for caching
            rest_client: WEEX REST client for API calls
        """
        self.database = database
        self.rest_client = rest_client

    async def fetch_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        force_download: bool = False,
    ) -> list[UnifiedCandle]:
        """
        Fetch candles for a date range, using cache when available.

        Args:
            symbol: Trading pair (unified format, e.g., "BTCUSDT")
            start_time: Start of date range
            end_time: End of date range
            interval: Candle interval (default: "1m")
            force_download: If True, bypass cache and download fresh

        Returns:
            List of candles ordered by timestamp ascending
        """
        if not force_download:
            # Try to get from cache first
            cached = await self.get_cached_candles(symbol, start_time, end_time, interval)
            if cached:
                # Check if cache is complete (no large gaps)
                if self._is_cache_complete(cached, start_time, end_time, interval):
                    logger.info(f"Using cached data: {len(cached)} candles for {symbol}")
                    return cached
                else:
                    logger.info("Cache incomplete, downloading missing data")

        # Download from API
        candles = await self._download_candles(symbol, start_time, end_time, interval)

        # Cache the downloaded data
        if candles:
            db_candles = [unified_to_candle(c) for c in candles]
            saved = await self.database.save_candles(db_candles)
            logger.info(f"Cached {saved} candles for {symbol}")

        return candles

    async def get_cached_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[UnifiedCandle]:
        """
        Get candles from database cache.

        Args:
            symbol: Trading pair
            start_time: Start of date range
            end_time: End of date range
            interval: Candle interval

        Returns:
            List of cached candles (may be empty or incomplete)
        """
        db_candles = await self.database.get_candles(
            symbol=symbol,
            limit=1_000_000,  # Large limit to get all
            interval=interval,
            since=start_time,
            until=end_time,
        )

        # Convert and sort by timestamp ascending
        unified = [candle_to_unified(c) for c in db_candles]
        unified.sort(key=lambda c: c.timestamp)
        return unified

    async def _download_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[UnifiedCandle]:
        """
        Download candles from WEEX API with pagination.

        Args:
            symbol: Trading pair
            start_time: Start of date range
            end_time: End of date range
            interval: Candle interval

        Returns:
            List of candles ordered by timestamp ascending
        """
        all_candles: list[UnifiedCandle] = []
        current_start = start_time

        # Calculate interval duration for pagination
        interval_minutes = self._interval_to_minutes(interval)
        chunk_duration = timedelta(minutes=interval_minutes * MAX_CANDLES_PER_REQUEST)

        total_expected = int((end_time - start_time).total_seconds() / (interval_minutes * 60))
        logger.info(f"Downloading ~{total_expected} candles for {symbol} ({start_time} to {end_time})")

        request_count = 0
        while current_start < end_time:
            current_end = min(current_start + chunk_duration, end_time)

            try:
                candles = await self.rest_client.get_candles(
                    symbol=symbol,
                    interval=interval,
                    limit=MAX_CANDLES_PER_REQUEST,
                    start_time=current_start,
                    end_time=current_end,
                )

                if candles:
                    all_candles.extend(candles)
                    logger.debug(f"Downloaded {len(candles)} candles ({current_start} to {current_end})")

                request_count += 1

                # Rate limiting: 10 requests per second max
                if request_count % 10 == 0:
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Error downloading candles: {e}")
                # Continue to next chunk
                await asyncio.sleep(1.0)

            current_start = current_end

        # Sort by timestamp and remove duplicates
        all_candles.sort(key=lambda c: c.timestamp)
        unique_candles = self._deduplicate(all_candles)

        logger.info(f"Downloaded {len(unique_candles)} unique candles for {symbol}")
        return unique_candles

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }
        return mapping.get(interval, 1)

    def _is_cache_complete(
        self,
        candles: list[UnifiedCandle],
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> bool:
        """
        Check if cached data is complete (no significant gaps).

        A cache is considered complete if:
        - It has at least 90% of expected candles
        - No gaps larger than 2x the interval
        """
        if not candles:
            return False

        interval_minutes = self._interval_to_minutes(interval)
        expected_count = int((end_time - start_time).total_seconds() / (interval_minutes * 60))

        # Check count threshold
        if len(candles) < expected_count * 0.9:
            return False

        # Check for large gaps
        max_gap = timedelta(minutes=interval_minutes * 2)
        for i in range(1, len(candles)):
            gap = candles[i].timestamp - candles[i - 1].timestamp
            if gap > max_gap:
                return False

        return True

    def _deduplicate(self, candles: list[UnifiedCandle]) -> list[UnifiedCandle]:
        """Remove duplicate candles based on timestamp."""
        seen = set()
        unique = []
        for candle in candles:
            key = (candle.symbol, candle.timestamp, candle.interval)
            if key not in seen:
                seen.add(key)
                unique.append(candle)
        return unique
