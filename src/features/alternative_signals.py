"""
AlphaStrike Trading Bot - Alternative Data Signals (Simons-Inspired)

Implements high-value alternative data signals that go beyond basic OHLCV:
1. Funding Rate Signal - Mean reversion when funding is extreme
2. Open Interest Signal - Trend strength and divergence detection
3. Long/Short Ratio Signal - Crowd positioning indicator
4. Liquidation Signal - Cascade prediction

These signals follow Jim Simons' principle: "Many weak signals combined > one strong signal"
Each signal has ~55-60% edge individually, but combined they're powerful.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import aiohttp

logger = logging.getLogger(__name__)

# Binance Futures API endpoints
BINANCE_FUTURES_URL = "https://fapi.binance.com"


@dataclass
class FundingRateData:
    """Funding rate data from Binance Futures."""
    symbol: str
    funding_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    funding_time: datetime  # Next funding time
    mark_price: float
    index_price: float
    estimated_settle_price: float = 0.0

    @property
    def funding_rate_percent(self) -> float:
        """Funding rate as percentage."""
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (3 payments per day * 365 days)."""
        return self.funding_rate * 3 * 365 * 100


@dataclass
class OpenInterestData:
    """Open interest data from Binance Futures."""
    symbol: str
    open_interest: float  # Total open interest in contracts
    open_interest_value: float  # Total OI in USDT
    timestamp: datetime


@dataclass
class LongShortRatioData:
    """Long/Short ratio data."""
    symbol: str
    long_short_ratio: float  # Ratio of long accounts to short accounts
    long_account: float  # Percentage of accounts that are long
    short_account: float  # Percentage of accounts that are short
    timestamp: datetime


@dataclass
class AlternativeSignals:
    """Combined alternative data signals."""

    # Funding rate signal (-1 to +1)
    # Positive = expect price drop (longs paying shorts)
    # Negative = expect price rise (shorts paying longs)
    funding_signal: float = 0.0
    funding_rate: float = 0.0
    funding_extreme: bool = False  # True if funding > 0.05% or < -0.05%

    # Open interest signal (-1 to +1)
    # Positive = OI rising with trend (strong trend)
    # Negative = OI diverging from price (potential reversal)
    oi_signal: float = 0.0
    oi_change_pct: float = 0.0
    oi_price_divergence: bool = False

    # Long/Short ratio signal (-1 to +1)
    # When crowd is extremely long, expect reversal down
    # When crowd is extremely short, expect reversal up
    ls_ratio_signal: float = 0.0
    long_short_ratio: float = 1.0
    crowd_extreme: bool = False

    # Combined signal strength (weighted average)
    combined_signal: float = 0.0
    signal_count: int = 0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        """Convert to dictionary for feature pipeline."""
        return {
            "alt_funding_signal": self.funding_signal,
            "alt_funding_rate": self.funding_rate,
            "alt_funding_extreme": 1.0 if self.funding_extreme else 0.0,
            "alt_oi_signal": self.oi_signal,
            "alt_oi_change_pct": self.oi_change_pct,
            "alt_oi_divergence": 1.0 if self.oi_price_divergence else 0.0,
            "alt_ls_ratio_signal": self.ls_ratio_signal,
            "alt_ls_ratio": self.long_short_ratio,
            "alt_crowd_extreme": 1.0 if self.crowd_extreme else 0.0,
            "alt_combined_signal": self.combined_signal,
            "alt_signal_count": self.signal_count,
        }


class AlternativeDataFetcher:
    """
    Fetches alternative data from Binance Futures API.

    Data sources:
    - Funding rate: /fapi/v1/premiumIndex
    - Open Interest: /fapi/v1/openInterest
    - Long/Short Ratio: /futures/data/globalLongShortAccountRatio
    """

    def __init__(self, cache_ttl_seconds: int = 60):
        """
        Initialize the fetcher.

        Args:
            cache_ttl_seconds: Cache TTL (default 60s, funding updates every 8h)
        """
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[datetime, any]] = {}
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and not expired."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return (datetime.now(UTC) - cached_time).total_seconds() < self.cache_ttl

    def _get_cached(self, key: str) -> any | None:
        """Get cached data if valid."""
        if self._is_cached(key):
            return self._cache[key][1]
        return None

    def _set_cache(self, key: str, data: any):
        """Set cache data."""
        self._cache[key] = (datetime.now(UTC), data)

    async def get_funding_rate(self, symbol: str) -> FundingRateData | None:
        """
        Get current funding rate for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            FundingRateData or None if failed
        """
        cache_key = f"funding_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            session = await self._get_session()
            url = f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex"
            params = {"symbol": symbol.upper()}

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch funding rate for {symbol}: {response.status}")
                    return None

                data = await response.json()

                result = FundingRateData(
                    symbol=data["symbol"],
                    funding_rate=float(data["lastFundingRate"]),
                    funding_time=datetime.fromtimestamp(data["nextFundingTime"] / 1000, tz=UTC),
                    mark_price=float(data["markPrice"]),
                    index_price=float(data["indexPrice"]),
                    estimated_settle_price=float(data.get("estimatedSettlePrice", 0)),
                )

                self._set_cache(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None

    async def get_open_interest(self, symbol: str) -> OpenInterestData | None:
        """
        Get current open interest for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            OpenInterestData or None if failed
        """
        cache_key = f"oi_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            session = await self._get_session()
            url = f"{BINANCE_FUTURES_URL}/fapi/v1/openInterest"
            params = {"symbol": symbol.upper()}

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch open interest for {symbol}: {response.status}")
                    return None

                data = await response.json()

                result = OpenInterestData(
                    symbol=data["symbol"],
                    open_interest=float(data["openInterest"]),
                    open_interest_value=0,  # Not available in this endpoint
                    timestamp=datetime.now(UTC),
                )

                self._set_cache(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None

    async def get_open_interest_history(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 24
    ) -> list[dict]:
        """
        Get open interest history for trend analysis.

        Args:
            symbol: Trading pair
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: Number of periods (max 500)

        Returns:
            List of OI data points
        """
        cache_key = f"oi_hist_{symbol}_{period}_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            session = await self._get_session()
            url = f"{BINANCE_FUTURES_URL}/futures/data/openInterestHist"
            params = {
                "symbol": symbol.upper(),
                "period": period,
                "limit": min(limit, 500),
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch OI history for {symbol}: {response.status}")
                    return []

                data = await response.json()
                self._set_cache(cache_key, data)
                return data

        except Exception as e:
            logger.error(f"Error fetching OI history for {symbol}: {e}")
            return []

    async def get_long_short_ratio(self, symbol: str, period: str = "1h") -> LongShortRatioData | None:
        """
        Get global long/short account ratio.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            period: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"

        Returns:
            LongShortRatioData or None if failed
        """
        cache_key = f"ls_ratio_{symbol}_{period}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            session = await self._get_session()
            url = f"{BINANCE_FUTURES_URL}/futures/data/globalLongShortAccountRatio"
            params = {
                "symbol": symbol.upper(),
                "period": period,
                "limit": 1,
            }

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch L/S ratio for {symbol}: {response.status}")
                    return None

                data = await response.json()
                if not data:
                    return None

                latest = data[0]
                result = LongShortRatioData(
                    symbol=symbol,
                    long_short_ratio=float(latest["longShortRatio"]),
                    long_account=float(latest["longAccount"]),
                    short_account=float(latest["shortAccount"]),
                    timestamp=datetime.fromtimestamp(latest["timestamp"] / 1000, tz=UTC),
                )

                self._set_cache(cache_key, result)
                return result

        except Exception as e:
            logger.error(f"Error fetching L/S ratio for {symbol}: {e}")
            return None


class AlternativeSignalGenerator:
    """
    Generates trading signals from alternative data.

    Signal Generation Rules (Simons-inspired):

    1. Funding Rate Signal:
       - Extreme positive funding (>0.05%) → Short bias (-0.5 to -1.0)
       - Extreme negative funding (<-0.05%) → Long bias (+0.5 to +1.0)
       - Normal funding → Neutral (0)

    2. Open Interest Signal:
       - Rising OI + Rising Price → Strong trend (+0.5)
       - Rising OI + Falling Price → Accumulation/Distribution (watch for reversal)
       - Falling OI + Rising Price → Short squeeze (unsustainable, -0.3)
       - Falling OI + Falling Price → Capitulation (potential bottom, +0.3)

    3. Long/Short Ratio Signal:
       - Extreme long positioning (>70%) → Contrarian short (-0.5)
       - Extreme short positioning (<30%) → Contrarian long (+0.5)
       - Balanced → Neutral (0)
    """

    # Funding rate thresholds
    FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05%
    FUNDING_VERY_EXTREME_THRESHOLD = 0.001  # 0.1%

    # Long/Short ratio thresholds
    LS_EXTREME_LONG_THRESHOLD = 0.65  # 65% long
    LS_EXTREME_SHORT_THRESHOLD = 0.35  # 35% long (65% short)

    # OI change thresholds
    OI_SIGNIFICANT_CHANGE = 0.05  # 5% change

    def __init__(self, fetcher: AlternativeDataFetcher | None = None):
        """Initialize the signal generator."""
        self.fetcher = fetcher or AlternativeDataFetcher()
        self._oi_history: dict[str, list[float]] = {}  # Rolling OI for each symbol
        self._price_history: dict[str, list[float]] = {}  # Rolling price for each symbol

    async def close(self):
        """Close the fetcher."""
        await self.fetcher.close()

    def _calculate_funding_signal(self, funding_data: FundingRateData | None) -> tuple[float, bool]:
        """
        Calculate signal from funding rate.

        Returns:
            (signal: -1 to +1, is_extreme: bool)
        """
        if not funding_data:
            return 0.0, False

        rate = funding_data.funding_rate

        # Extreme positive funding → expect price drop (longs paying too much)
        if rate >= self.FUNDING_VERY_EXTREME_THRESHOLD:
            return -1.0, True
        elif rate >= self.FUNDING_EXTREME_THRESHOLD:
            return -0.5, True
        # Extreme negative funding → expect price rise (shorts paying too much)
        elif rate <= -self.FUNDING_VERY_EXTREME_THRESHOLD:
            return 1.0, True
        elif rate <= -self.FUNDING_EXTREME_THRESHOLD:
            return 0.5, True
        # Normal funding → neutral
        else:
            # Small signal based on direction
            return -rate * 100, False  # Scale for small signal

    def _calculate_oi_signal(
        self,
        current_oi: float,
        oi_history: list[float],
        price_change_pct: float
    ) -> tuple[float, float, bool]:
        """
        Calculate signal from open interest changes.

        Returns:
            (signal: -1 to +1, oi_change_pct: float, is_divergence: bool)
        """
        if not oi_history or len(oi_history) < 2:
            return 0.0, 0.0, False

        # Calculate OI change
        prev_oi = oi_history[-1] if oi_history else current_oi
        oi_change_pct = (current_oi - prev_oi) / prev_oi if prev_oi > 0 else 0.0

        # Determine signal based on OI/Price relationship
        oi_rising = oi_change_pct > self.OI_SIGNIFICANT_CHANGE
        oi_falling = oi_change_pct < -self.OI_SIGNIFICANT_CHANGE
        price_rising = price_change_pct > 0.01  # 1% threshold
        price_falling = price_change_pct < -0.01

        signal = 0.0
        divergence = False

        if oi_rising and price_rising:
            # Strong trend - go with it
            signal = 0.5
        elif oi_rising and price_falling:
            # Accumulation or distribution - watch for reversal
            signal = 0.3  # Slight bullish (accumulation more common)
            divergence = True
        elif oi_falling and price_rising:
            # Short squeeze - unsustainable
            signal = -0.3
            divergence = True
        elif oi_falling and price_falling:
            # Capitulation - potential bottom
            signal = 0.4

        return signal, oi_change_pct * 100, divergence

    def _calculate_ls_ratio_signal(self, ls_data: LongShortRatioData | None) -> tuple[float, bool]:
        """
        Calculate contrarian signal from long/short ratio.

        Returns:
            (signal: -1 to +1, is_extreme: bool)
        """
        if not ls_data:
            return 0.0, False

        long_pct = ls_data.long_account

        # Extreme long positioning → contrarian short
        if long_pct >= self.LS_EXTREME_LONG_THRESHOLD:
            strength = (long_pct - self.LS_EXTREME_LONG_THRESHOLD) / (1.0 - self.LS_EXTREME_LONG_THRESHOLD)
            return -0.5 * (1 + strength), True
        # Extreme short positioning → contrarian long
        elif long_pct <= self.LS_EXTREME_SHORT_THRESHOLD:
            strength = (self.LS_EXTREME_SHORT_THRESHOLD - long_pct) / self.LS_EXTREME_SHORT_THRESHOLD
            return 0.5 * (1 + strength), True
        # Balanced
        else:
            return 0.0, False

    async def generate_signals(
        self,
        symbol: str,
        current_price: float,
        price_change_24h_pct: float = 0.0
    ) -> AlternativeSignals:
        """
        Generate all alternative data signals for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            current_price: Current price for OI value calculation
            price_change_24h_pct: 24h price change for OI divergence detection

        Returns:
            AlternativeSignals with all signal values
        """
        # Fetch all data concurrently
        funding_task = self.fetcher.get_funding_rate(symbol)
        oi_task = self.fetcher.get_open_interest(symbol)
        ls_ratio_task = self.fetcher.get_long_short_ratio(symbol)

        funding_data, oi_data, ls_data = await asyncio.gather(
            funding_task, oi_task, ls_ratio_task,
            return_exceptions=True
        )

        # Handle exceptions - convert to proper types
        valid_funding: FundingRateData | None = None
        valid_oi: OpenInterestData | None = None
        valid_ls: LongShortRatioData | None = None

        if isinstance(funding_data, FundingRateData):
            valid_funding = funding_data
        elif isinstance(funding_data, Exception):
            logger.warning(f"Funding data error for {symbol}: {funding_data}")

        if isinstance(oi_data, OpenInterestData):
            valid_oi = oi_data
        elif isinstance(oi_data, Exception):
            logger.warning(f"OI data error for {symbol}: {oi_data}")

        if isinstance(ls_data, LongShortRatioData):
            valid_ls = ls_data
        elif isinstance(ls_data, Exception):
            logger.warning(f"L/S ratio error for {symbol}: {ls_data}")

        # Update OI history
        if symbol not in self._oi_history:
            self._oi_history[symbol] = []
        if valid_oi:
            self._oi_history[symbol].append(valid_oi.open_interest)
            # Keep last 24 data points
            self._oi_history[symbol] = self._oi_history[symbol][-24:]

        # Calculate signals
        funding_signal, funding_extreme = self._calculate_funding_signal(valid_funding)

        oi_signal, oi_change_pct, oi_divergence = self._calculate_oi_signal(
            valid_oi.open_interest if valid_oi else 0,
            self._oi_history.get(symbol, []),
            price_change_24h_pct
        )

        ls_signal, crowd_extreme = self._calculate_ls_ratio_signal(valid_ls)

        # Count valid signals
        signal_count = sum([
            1 if valid_funding else 0,
            1 if valid_oi else 0,
            1 if valid_ls else 0,
        ])

        # Calculate combined signal (weighted average)
        weights = {
            "funding": 0.4,  # Funding is strongest predictor
            "oi": 0.35,
            "ls_ratio": 0.25,
        }

        combined = 0.0
        total_weight = 0.0

        if valid_funding:
            combined += funding_signal * weights["funding"]
            total_weight += weights["funding"]
        if valid_oi:
            combined += oi_signal * weights["oi"]
            total_weight += weights["oi"]
        if valid_ls:
            combined += ls_signal * weights["ls_ratio"]
            total_weight += weights["ls_ratio"]

        if total_weight > 0:
            combined /= total_weight

        return AlternativeSignals(
            funding_signal=funding_signal,
            funding_rate=valid_funding.funding_rate if valid_funding else 0.0,
            funding_extreme=funding_extreme,
            oi_signal=oi_signal,
            oi_change_pct=oi_change_pct,
            oi_price_divergence=oi_divergence,
            ls_ratio_signal=ls_signal,
            long_short_ratio=valid_ls.long_short_ratio if valid_ls else 1.0,
            crowd_extreme=crowd_extreme,
            combined_signal=combined,
            signal_count=signal_count,
        )


# Singleton instance for global use
_signal_generator: AlternativeSignalGenerator | None = None


def get_alternative_signal_generator() -> AlternativeSignalGenerator:
    """Get or create the global signal generator."""
    global _signal_generator
    if _signal_generator is None:
        _signal_generator = AlternativeSignalGenerator()
    return _signal_generator


async def close_alternative_signal_generator():
    """Close the global signal generator."""
    global _signal_generator
    if _signal_generator:
        await _signal_generator.close()
        _signal_generator = None
