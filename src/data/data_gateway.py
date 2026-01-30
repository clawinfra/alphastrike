"""
AlphaStrike Trading Bot - Data Gateway

Filters bad/stale data before it reaches the feature layer.
Implements quality gates, circuit breaker, and fallback provider.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.core.config import get_settings
from src.data.database import Candle

logger = logging.getLogger(__name__)


class GateResult(str, Enum):
    """Result of a quality gate check."""
    PASS = "pass"
    FAIL = "fail"
    DEGRADED = "degraded"


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Rejecting all data
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ValidationResult:
    """Result of validating data through all gates."""
    passed: bool
    gate_results: dict[str, GateResult] = field(default_factory=dict)
    confidence_adjustment: float = 1.0
    rejection_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class GatewayMetrics:
    """Metrics for data gateway monitoring."""
    total_processed: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_degraded: int = 0
    rejections_by_gate: dict[str, int] = field(default_factory=dict)
    avg_staleness_seconds: float = 0.0
    circuit_breaker_trips: int = 0
    fallback_activations: int = 0


class StalenessChecker:
    """
    Check if data is stale based on timestamp age.

    Stale data indicates connectivity issues or exchange delays.
    """

    def __init__(self, threshold_seconds: float = 5.0):
        """
        Args:
            threshold_seconds: Maximum age in seconds before data is stale
        """
        self.threshold_seconds = threshold_seconds

    def check(self, timestamp: datetime) -> tuple[GateResult, float]:
        """
        Check if timestamp is within freshness threshold.

        Args:
            timestamp: Data timestamp to check

        Returns:
            (GateResult, age_in_seconds)
        """
        now = datetime.utcnow()
        age = (now - timestamp).total_seconds()

        if age < 0:
            # Future timestamp - likely clock skew
            logger.warning(f"Future timestamp detected: {age:.1f}s ahead")
            return GateResult.DEGRADED, abs(age)

        if age > self.threshold_seconds:
            return GateResult.FAIL, age

        return GateResult.PASS, age

    def get_age_seconds(self, timestamp: datetime) -> float:
        """Get age of timestamp in seconds."""
        return (datetime.utcnow() - timestamp).total_seconds()


class SequenceChecker:
    """
    Check for gaps in sequential data (e.g., candle timestamps).

    Detects missing candles that could affect indicator calculations.
    """

    def __init__(self, expected_interval_seconds: int = 60):
        """
        Args:
            expected_interval_seconds: Expected gap between sequential data points
        """
        self.expected_interval = expected_interval_seconds
        self._last_timestamps: dict[str, datetime] = {}

    def check(self, symbol: str, timestamp: datetime) -> tuple[GateResult, int]:
        """
        Check for sequence gaps.

        Args:
            symbol: Trading symbol
            timestamp: Current data timestamp

        Returns:
            (GateResult, gap_count) where gap_count is number of missing intervals
        """
        last_ts = self._last_timestamps.get(symbol)
        self._last_timestamps[symbol] = timestamp

        if last_ts is None:
            # First data point for this symbol
            return GateResult.PASS, 0

        gap_seconds = (timestamp - last_ts).total_seconds()

        # Allow 10% tolerance
        tolerance = self.expected_interval * 0.1
        expected_max = self.expected_interval + tolerance

        if gap_seconds < 0:
            # Out of order data
            logger.warning(f"Out of order data for {symbol}: {gap_seconds:.1f}s")
            return GateResult.DEGRADED, 0

        if gap_seconds > expected_max * 2:
            # Significant gap
            gap_count = int(gap_seconds / self.expected_interval) - 1
            return GateResult.FAIL, gap_count

        return GateResult.PASS, 0

    def reset(self, symbol: str | None = None) -> None:
        """Reset sequence tracking."""
        if symbol:
            self._last_timestamps.pop(symbol, None)
        else:
            self._last_timestamps.clear()


class PriceRangeValidator:
    """
    Validate prices are within reasonable range of recent history.

    Detects anomalous prices that could be data errors or flash crashes.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Maximum deviation from 24h range (0.5 = 50%)
        """
        self.threshold = threshold
        self._24h_ranges: dict[str, tuple[float, float]] = {}

    def update_24h_range(self, symbol: str, high: float, low: float) -> None:
        """Update the 24h range for a symbol."""
        self._24h_ranges[symbol] = (low, high)

    def check(self, symbol: str, price: float) -> GateResult:
        """
        Check if price is within acceptable range.

        Args:
            symbol: Trading symbol
            price: Price to validate

        Returns:
            GateResult
        """
        range_data = self._24h_ranges.get(symbol)
        if not range_data:
            # No range data yet - accept but log
            logger.debug(f"No 24h range for {symbol}, accepting price {price}")
            return GateResult.PASS

        low_24h, high_24h = range_data
        range_size = high_24h - low_24h

        if range_size <= 0:
            return GateResult.PASS

        # Calculate acceptable bounds
        margin = range_size * self.threshold
        acceptable_low = low_24h - margin
        acceptable_high = high_24h + margin

        if price < acceptable_low or price > acceptable_high:
            logger.warning(
                f"Price {price} outside range for {symbol}: "
                f"[{acceptable_low:.2f}, {acceptable_high:.2f}]"
            )
            return GateResult.FAIL

        return GateResult.PASS


class OHLCLogicValidator:
    """
    Validate OHLC candle data follows logical constraints.

    Catches malformed or corrupted candle data.
    """

    def check(self, candle: Candle) -> tuple[GateResult, str | None]:
        """
        Validate OHLC logic.

        Args:
            candle: Candle to validate

        Returns:
            (GateResult, error_message)
        """
        errors = []

        # High must be >= Low
        if candle.high < candle.low:
            errors.append(f"High ({candle.high}) < Low ({candle.low})")

        # High must be >= Open and Close
        if candle.high < candle.open:
            errors.append(f"High ({candle.high}) < Open ({candle.open})")
        if candle.high < candle.close:
            errors.append(f"High ({candle.high}) < Close ({candle.close})")

        # Low must be <= Open and Close
        if candle.low > candle.open:
            errors.append(f"Low ({candle.low}) > Open ({candle.open})")
        if candle.low > candle.close:
            errors.append(f"Low ({candle.low}) > Close ({candle.close})")

        # Prices must be positive
        if any(p <= 0 for p in [candle.open, candle.high, candle.low, candle.close]):
            errors.append("Non-positive price detected")

        if errors:
            return GateResult.FAIL, "; ".join(errors)

        return GateResult.PASS, None


class VolumeValidator:
    """
    Validate volume data for anomalies.

    Detects volume spikes that could indicate data errors.
    """

    def __init__(self, spike_multiplier: float = 100.0, window_size: int = 100):
        """
        Args:
            spike_multiplier: Max multiple of average volume before flagging
            window_size: Number of candles for rolling average
        """
        self.spike_multiplier = spike_multiplier
        self.window_size = window_size
        self._volume_history: dict[str, deque[float]] = {}

    def check(self, symbol: str, volume: float) -> tuple[GateResult, float | None]:
        """
        Check volume for anomalies.

        Args:
            symbol: Trading symbol
            volume: Volume to check

        Returns:
            (GateResult, volume_ratio) where ratio is volume/avg
        """
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.window_size)

        history = self._volume_history[symbol]

        # Volume must be non-negative
        if volume < 0:
            return GateResult.FAIL, None

        # Calculate average
        if len(history) > 10:  # Need minimum history
            avg_volume = sum(history) / len(history)
            if avg_volume > 0:
                ratio = volume / avg_volume
                if ratio > self.spike_multiplier:
                    logger.warning(
                        f"Volume spike for {symbol}: {volume:.2f} "
                        f"({ratio:.1f}x average)"
                    )
                    # Don't fail, just degrade confidence
                    history.append(volume)
                    return GateResult.DEGRADED, ratio

        history.append(volume)
        return GateResult.PASS, None


class SpreadChecker:
    """
    Check bid-ask spread for liquidity issues.
    """

    def __init__(self, max_spread_pct: float = 0.05):
        """
        Args:
            max_spread_pct: Maximum acceptable spread as percentage (0.05 = 5%)
        """
        self.max_spread_pct = max_spread_pct

    def check(self, bid: float, ask: float) -> tuple[GateResult, float]:
        """
        Check spread.

        Args:
            bid: Best bid price
            ask: Best ask price

        Returns:
            (GateResult, spread_percentage)
        """
        if bid <= 0 or ask <= 0:
            return GateResult.FAIL, 0.0

        if bid >= ask:
            # Crossed market
            return GateResult.FAIL, 0.0

        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid

        if spread_pct > self.max_spread_pct:
            return GateResult.DEGRADED, spread_pct

        return GateResult.PASS, spread_pct


class AnomalyDetector:
    """
    Detect various anomalies in data.
    """

    def __init__(self, spike_threshold: float = 3.0, gap_threshold: float = 0.1):
        """
        Args:
            spike_threshold: Std deviations for spike detection
            gap_threshold: Minimum gap percentage to flag
        """
        self.spike_threshold = spike_threshold
        self.gap_threshold = gap_threshold
        self._price_history: dict[str, deque[float]] = {}

    def detect_spike(self, symbol: str, value: float) -> bool:
        """
        Detect if value is a spike compared to recent history.

        Uses z-score to detect outliers.
        """
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=50)

        history = self._price_history[symbol]

        if len(history) > 10:
            import statistics
            mean = statistics.mean(history)
            std = statistics.stdev(history)
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > self.spike_threshold:
                    return True

        history.append(value)
        return False

    def detect_gap(self, current: float, previous: float) -> bool:
        """Detect if there's a significant price gap."""
        if previous <= 0:
            return False

        gap_pct = abs(current - previous) / previous
        return gap_pct > self.gap_threshold


class DataCircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    Opens after consecutive failures and requires recovery period.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
        half_open_successes: int = 3,
    ):
        """
        Args:
            failure_threshold: Failures before opening circuit
            reset_timeout_seconds: Time before attempting recovery
            half_open_successes: Successes needed in half-open to close
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout_seconds
        self.half_open_successes = half_open_successes

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_success_count = 0
                logger.info("Circuit breaker entering half-open state")
        return self._state

    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.half_open_successes:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed after recovery")
        else:
            self._failure_count = 0

    def record_failure(self, gate: str | None = None) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if gate:
            logger.warning(f"Data quality failure on gate: {gate}")

        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker OPENED after {self._failure_count} failures"
                )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_success_count = 0
        logger.info("Circuit breaker reset")


class FallbackDataProvider:
    """
    Provides fallback data when primary source fails.

    Caches last known good data for each symbol.
    """

    def __init__(self, max_age_seconds: float = 300.0):
        """
        Args:
            max_age_seconds: Maximum age of cached data before expiry
        """
        self.max_age = max_age_seconds
        self._cache: dict[str, tuple[Candle, float]] = {}  # symbol -> (candle, timestamp)

    def cache_valid_data(self, symbol: str, candle: Candle) -> None:
        """Store valid candle data in cache."""
        self._cache[symbol] = (candle, time.time())

    async def get_fallback(self, symbol: str) -> Candle | None:
        """
        Get fallback candle data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Cached candle if available and fresh, None otherwise
        """
        cached = self._cache.get(symbol)
        if cached is None:
            return None

        candle, cache_time = cached
        age = time.time() - cache_time

        if age > self.max_age:
            logger.warning(f"Fallback data for {symbol} expired ({age:.1f}s old)")
            return None

        logger.info(f"Using fallback data for {symbol} ({age:.1f}s old)")
        return candle

    def clear(self, symbol: str | None = None) -> None:
        """Clear cache."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()


class DataGateway:
    """
    Main data gateway that orchestrates all quality checks.

    Usage:
        gateway = DataGateway()
        result = await gateway.process_candle(candle)
        if result.passed:
            # Use the data
            pass
        else:
            # Handle rejection
            fallback = await gateway.get_fallback(symbol)
    """

    def __init__(self):
        """Initialize data gateway with configured thresholds."""
        settings = get_settings()
        config = settings.data_gateway

        # Initialize checkers
        self.staleness_checker = StalenessChecker(config.staleness_threshold_seconds)
        self.sequence_checker = SequenceChecker(expected_interval_seconds=60)
        self.price_range_validator = PriceRangeValidator(config.price_range_threshold)
        self.ohlc_validator = OHLCLogicValidator()
        self.volume_validator = VolumeValidator(config.volume_spike_multiplier)
        self.spread_checker = SpreadChecker()
        self.anomaly_detector = AnomalyDetector()

        # Circuit breaker and fallback
        self.circuit_breaker = DataCircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            reset_timeout_seconds=config.circuit_breaker_reset_seconds,
        )
        self.fallback_provider = FallbackDataProvider()

        # Metrics
        self.metrics = GatewayMetrics()
        self._staleness_samples: deque[float] = deque(maxlen=100)

    async def process_candle(self, candle: Candle) -> ValidationResult:
        """
        Process a candle through all quality gates.

        Args:
            candle: Candle data to validate

        Returns:
            ValidationResult with pass/fail status and details
        """
        self.metrics.total_processed += 1

        # Check circuit breaker first
        if self.circuit_breaker.is_open():
            self.metrics.total_failed += 1
            return ValidationResult(
                passed=False,
                rejection_reason="circuit_breaker_open",
            )

        result = ValidationResult(passed=True)
        confidence_factors: list[float] = []

        # Gate 1: Staleness
        stale_result, age = self.staleness_checker.check(candle.timestamp)
        result.gate_results["staleness"] = stale_result
        self._staleness_samples.append(age)

        if stale_result == GateResult.FAIL:
            result.passed = False
            result.rejection_reason = f"stale_data ({age:.1f}s)"
            self._record_failure("staleness")
            return result
        elif stale_result == GateResult.DEGRADED:
            confidence_factors.append(0.8)
            result.warnings.append(f"clock_skew ({age:.1f}s)")

        # Gate 2: Sequence
        seq_result, gap_count = self.sequence_checker.check(candle.symbol, candle.timestamp)
        result.gate_results["sequence"] = seq_result

        if seq_result == GateResult.FAIL:
            result.warnings.append(f"sequence_gap ({gap_count} missing)")
            confidence_factors.append(0.7)  # Degrade but don't fail
        elif seq_result == GateResult.DEGRADED:
            result.warnings.append("out_of_order_data")
            confidence_factors.append(0.9)

        # Gate 3: Price Range
        price_result = self.price_range_validator.check(candle.symbol, candle.close)
        result.gate_results["price_range"] = price_result

        if price_result == GateResult.FAIL:
            result.passed = False
            result.rejection_reason = "price_out_of_range"
            self._record_failure("price_range")
            return result

        # Gate 4: OHLC Logic
        ohlc_result, ohlc_error = self.ohlc_validator.check(candle)
        result.gate_results["ohlc_logic"] = ohlc_result

        if ohlc_result == GateResult.FAIL:
            result.passed = False
            result.rejection_reason = f"ohlc_invalid: {ohlc_error}"
            self._record_failure("ohlc_logic")
            return result

        # Gate 5: Volume
        vol_result, vol_ratio = self.volume_validator.check(candle.symbol, candle.volume)
        result.gate_results["volume"] = vol_result

        if vol_result == GateResult.FAIL:
            result.passed = False
            result.rejection_reason = "negative_volume"
            self._record_failure("volume")
            return result
        elif vol_result == GateResult.DEGRADED:
            result.warnings.append(f"volume_spike ({vol_ratio:.1f}x)")
            confidence_factors.append(0.85)

        # Gate 6: Anomaly Detection
        is_spike = self.anomaly_detector.detect_spike(candle.symbol, candle.close)
        if is_spike:
            result.gate_results["anomaly"] = GateResult.DEGRADED
            result.warnings.append("price_spike_detected")
            confidence_factors.append(0.75)
        else:
            result.gate_results["anomaly"] = GateResult.PASS

        # Calculate final confidence adjustment
        if confidence_factors:
            # Multiply all factors
            confidence = 1.0
            for factor in confidence_factors:
                confidence *= factor
            result.confidence_adjustment = confidence
            self.metrics.total_degraded += 1

        # Record success
        if result.passed:
            self.circuit_breaker.record_success()
            self.fallback_provider.cache_valid_data(candle.symbol, candle)
            self.metrics.total_passed += 1

        return result

    def _record_failure(self, gate: str) -> None:
        """Record a gate failure."""
        self.metrics.total_failed += 1
        self.metrics.rejections_by_gate[gate] = \
            self.metrics.rejections_by_gate.get(gate, 0) + 1
        self.circuit_breaker.record_failure(gate)

    async def get_fallback(self, symbol: str) -> Candle | None:
        """Get fallback data for a symbol."""
        candle = await self.fallback_provider.get_fallback(symbol)
        if candle:
            self.metrics.fallback_activations += 1
        return candle

    def update_24h_range(self, symbol: str, high: float, low: float) -> None:
        """Update 24h price range for a symbol."""
        self.price_range_validator.update_24h_range(symbol, high, low)

    def check_spread(self, bid: float, ask: float) -> tuple[GateResult, float]:
        """Check bid-ask spread."""
        return self.spread_checker.check(bid, ask)

    def get_metrics(self) -> dict[str, Any]:
        """Get current gateway metrics."""
        # Update average staleness
        if self._staleness_samples:
            self.metrics.avg_staleness_seconds = \
                sum(self._staleness_samples) / len(self._staleness_samples)

        pass_rate = (
            self.metrics.total_passed / self.metrics.total_processed * 100
            if self.metrics.total_processed > 0 else 0
        )

        return {
            "total_processed": self.metrics.total_processed,
            "total_passed": self.metrics.total_passed,
            "total_failed": self.metrics.total_failed,
            "total_degraded": self.metrics.total_degraded,
            "pass_rate_pct": round(pass_rate, 2),
            "rejections_by_gate": self.metrics.rejections_by_gate,
            "avg_staleness_seconds": round(self.metrics.avg_staleness_seconds, 3),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "fallback_activations": self.metrics.fallback_activations,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = GatewayMetrics()
        self._staleness_samples.clear()


# Module-level singleton
_gateway_instance: DataGateway | None = None


def get_data_gateway() -> DataGateway:
    """Get the data gateway singleton instance."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = DataGateway()
    return _gateway_instance
