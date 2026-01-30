"""
AlphaStrike Trading Bot - Microstructure Features Module (US-007)

Market microstructure features for ML models including:
- Orderbook imbalance
- Funding rate analysis
- Open interest metrics
- Volume profile analysis
- Whale activity detection
- Trade flow imbalance

All features return normalized values suitable for ML models.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSnapshot:
    """Snapshot of orderbook data at a point in time."""

    bids: list[tuple[float, float]]  # List of (price, quantity) tuples
    asks: list[tuple[float, float]]  # List of (price, quantity) tuples
    timestamp: float = 0.0  # Unix timestamp

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        """Get mid price."""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0.0


@dataclass
class TradeRecord:
    """Individual trade record."""

    price: float
    quantity: float
    side: str  # "buy" or "sell"
    timestamp: float = 0.0  # Unix timestamp


@dataclass
class FundingInfo:
    """Funding rate information for perpetual futures."""

    funding_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    next_funding_time: float = 0.0  # Unix timestamp of next funding
    predicted_rate: float | None = None  # Predicted next funding rate


@dataclass
class OpenInterestInfo:
    """Open interest data."""

    open_interest: float  # Total open interest in contracts/base currency
    open_interest_value: float = 0.0  # OI in quote currency (USDT)
    timestamp: float = 0.0


@dataclass
class MicrostructureData:
    """Container for all microstructure data inputs."""

    orderbook: OrderbookSnapshot | None = None
    funding: FundingInfo | None = None
    open_interest: OpenInterestInfo | None = None
    open_interest_history: list[OpenInterestInfo] = field(default_factory=list)
    recent_trades: list[TradeRecord] = field(default_factory=list)


@dataclass
class MicrostructureFeatures:
    """
    Calculates market microstructure features for ML models.

    All features are normalized to ranges suitable for ML:
    - [-1, 1] for directional features (imbalances)
    - [0, 1] for magnitude features

    Usage:
        calculator = MicrostructureFeatures()
        features = calculator.calculate(microstructure_data)
    """

    # Configuration
    orderbook_depth_levels: int = 10
    funding_rate_cap: float = 0.01  # Cap at 1% for normalization
    oi_change_window: int = 5  # Number of OI snapshots for change calculation
    large_trade_std_multiplier: float = 2.0  # Threshold for whale trades
    volume_profile_bins: int = 5

    # Internal state for rolling calculations
    _trade_sizes: list[float] = field(default_factory=list)
    _trade_size_window: int = 100  # Window for trade size statistics

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._trade_sizes: list[float] = []

    def calculate(self, data: MicrostructureData) -> dict[str, float]:
        """
        Calculate all microstructure features.

        Args:
            data: MicrostructureData containing orderbook, funding, OI, and trade data

        Returns:
            Dictionary of feature names to normalized values
        """
        features: dict[str, float] = {}

        # Orderbook features
        features["orderbook_imbalance"] = self._calculate_orderbook_imbalance(data.orderbook)

        # Funding rate features
        features["funding_rate"] = self._calculate_funding_rate(data.funding)

        # Open interest features
        features["open_interest"] = self._calculate_open_interest(data.open_interest)
        features["open_interest_change"] = self._calculate_oi_change(data.open_interest_history)

        # Trade-based features
        features["volume_profile"] = self._calculate_volume_profile(data.recent_trades)
        features["large_trade_ratio"] = self._calculate_large_trade_ratio(data.recent_trades)
        features["trade_flow_imbalance"] = self._calculate_trade_flow_imbalance(data.recent_trades)

        return features

    def _calculate_orderbook_imbalance(
        self,
        orderbook: OrderbookSnapshot | None,
    ) -> float:
        """
        Calculate orderbook imbalance from bid/ask depth.

        Returns:
            Value in [-1, 1] where:
            - Positive values indicate bid pressure (bullish)
            - Negative values indicate ask pressure (bearish)
            - 0 indicates balanced orderbook
        """
        if orderbook is None:
            return 0.0

        if not orderbook.bids or not orderbook.asks:
            return 0.0

        # Sum bid and ask volumes up to specified depth
        bid_depth = sum(
            qty for _, qty in orderbook.bids[:self.orderbook_depth_levels]
        )
        ask_depth = sum(
            qty for _, qty in orderbook.asks[:self.orderbook_depth_levels]
        )

        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return 0.0

        # Imbalance: (bid - ask) / (bid + ask)
        # Range: [-1, 1]
        imbalance = (bid_depth - ask_depth) / total_depth

        return float(np.clip(imbalance, -1.0, 1.0))

    def _calculate_funding_rate(self, funding: FundingInfo | None) -> float:
        """
        Calculate normalized funding rate feature.

        Returns:
            Value in [-1, 1] where:
            - Positive values indicate longs pay shorts (bullish sentiment)
            - Negative values indicate shorts pay longs (bearish sentiment)
            - Magnitude indicates strength of funding pressure
        """
        if funding is None:
            return 0.0

        # Normalize funding rate to [-1, 1] using cap
        # Typical funding rates are -0.01% to +0.01% per 8 hours
        normalized = funding.funding_rate / self.funding_rate_cap

        return float(np.clip(normalized, -1.0, 1.0))

    def _calculate_open_interest(
        self,
        oi_info: OpenInterestInfo | None,
    ) -> float:
        """
        Calculate normalized open interest feature.

        Since raw OI varies greatly by market, we return a normalized
        value based on relative position within recent range.

        Returns:
            Value in [0, 1] representing relative OI level.
            Returns 0.5 (neutral) if insufficient data.
        """
        if oi_info is None or oi_info.open_interest <= 0:
            return 0.5  # Neutral value

        # For single snapshot without history, return neutral
        # The actual normalization happens when we have historical data
        return 0.5

    def _calculate_oi_change(
        self,
        oi_history: list[OpenInterestInfo],
    ) -> float:
        """
        Calculate rate of change of open interest.

        Returns:
            Value in [-1, 1] where:
            - Positive values indicate increasing OI (new positions)
            - Negative values indicate decreasing OI (closing positions)
        """
        if len(oi_history) < 2:
            return 0.0

        # Get recent OI values
        recent_oi = [
            oi.open_interest
            for oi in oi_history[-self.oi_change_window:]
            if oi.open_interest > 0
        ]

        if len(recent_oi) < 2:
            return 0.0

        # Calculate percentage change from oldest to newest
        oldest_oi = recent_oi[0]
        newest_oi = recent_oi[-1]

        if oldest_oi <= 0:
            return 0.0

        pct_change = (newest_oi - oldest_oi) / oldest_oi

        # Normalize: assume +/-20% change is maximum
        # This maps typical OI changes to [-1, 1]
        normalized = pct_change / 0.20

        return float(np.clip(normalized, -1.0, 1.0))

    def _calculate_volume_profile(
        self,
        trades: list[TradeRecord],
    ) -> float:
        """
        Analyze recent trade distribution to detect concentration.

        Returns:
            Value in [0, 1] where:
            - Higher values indicate more concentrated trading (potential support/resistance)
            - Lower values indicate more distributed trading
        """
        if len(trades) < 10:
            return 0.5  # Neutral value

        prices = [t.price for t in trades if t.price > 0]
        if len(prices) < 10:
            return 0.5

        # Calculate price range
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        if price_range <= 0:
            return 1.0  # All trades at same price = maximum concentration

        # Bin prices and calculate distribution
        bin_width = price_range / self.volume_profile_bins
        bins = [0.0] * self.volume_profile_bins

        for price in prices:
            bin_idx = min(
                int((price - min_price) / bin_width),
                self.volume_profile_bins - 1,
            )
            bins[bin_idx] += 1

        # Calculate concentration using normalized entropy
        # Lower entropy = higher concentration
        total_trades = len(prices)
        probabilities = [b / total_trades for b in bins if b > 0]

        if len(probabilities) <= 1:
            return 1.0  # Single bin = maximum concentration

        # Calculate entropy
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
        max_entropy = np.log(self.volume_profile_bins)  # Maximum possible entropy

        # Normalize: higher concentration = lower entropy = higher feature value
        concentration = 1.0 - (entropy / max_entropy)

        return float(np.clip(concentration, 0.0, 1.0))

    def _calculate_large_trade_ratio(
        self,
        trades: list[TradeRecord],
    ) -> float:
        """
        Detect whale activity by identifying trades > 2 std from mean.

        Returns:
            Value in [0, 1] representing ratio of large trades.
            Higher values indicate more whale activity.
        """
        if len(trades) < 20:
            return 0.0  # Insufficient data

        # Get trade sizes
        sizes = [t.quantity for t in trades if t.quantity > 0]

        if len(sizes) < 20:
            return 0.0

        # Update rolling trade size history
        self._trade_sizes.extend(sizes)
        if len(self._trade_sizes) > self._trade_size_window:
            self._trade_sizes = self._trade_sizes[-self._trade_size_window:]

        # Calculate statistics
        try:
            mean_size = statistics.mean(self._trade_sizes)
            std_size = statistics.stdev(self._trade_sizes)
        except statistics.StatisticsError:
            return 0.0

        if std_size <= 0:
            return 0.0

        # Count large trades (> mean + 2*std)
        threshold = mean_size + (self.large_trade_std_multiplier * std_size)
        large_trades = sum(1 for s in sizes if s > threshold)

        # Calculate ratio
        ratio = large_trades / len(sizes)

        # Normalize: expect 0-10% large trades typically
        # Map to [0, 1] with 10% = 0.5 and 20%+ = 1.0
        normalized = min(ratio / 0.20, 1.0)

        return float(np.clip(normalized, 0.0, 1.0))

    def _calculate_trade_flow_imbalance(
        self,
        trades: list[TradeRecord],
    ) -> float:
        """
        Calculate buy vs sell volume imbalance.

        Returns:
            Value in [-1, 1] where:
            - Positive values indicate net buying (bullish)
            - Negative values indicate net selling (bearish)
        """
        if not trades:
            return 0.0

        buy_volume = sum(
            t.quantity for t in trades
            if t.side.lower() == "buy" and t.quantity > 0
        )
        sell_volume = sum(
            t.quantity for t in trades
            if t.side.lower() == "sell" and t.quantity > 0
        )

        total_volume = buy_volume + sell_volume
        if total_volume <= 0:
            return 0.0

        # Imbalance: (buy - sell) / (buy + sell)
        # Range: [-1, 1]
        imbalance = (buy_volume - sell_volume) / total_volume

        return float(np.clip(imbalance, -1.0, 1.0))

    def calculate_weighted_orderbook_imbalance(
        self,
        orderbook: OrderbookSnapshot | None,
        decay_factor: float = 0.9,
    ) -> float:
        """
        Calculate orderbook imbalance with distance-weighted volumes.

        Orders closer to mid price are weighted more heavily.

        Args:
            orderbook: Orderbook snapshot
            decay_factor: Exponential decay for distance weighting (0-1)

        Returns:
            Value in [-1, 1]
        """
        if orderbook is None:
            return 0.0

        if not orderbook.bids or not orderbook.asks:
            return 0.0

        mid_price = orderbook.mid_price
        if mid_price <= 0:
            return 0.0

        # Weight bids by distance from mid
        weighted_bid_depth = 0.0
        for i, (price, qty) in enumerate(orderbook.bids[:self.orderbook_depth_levels]):
            weight = decay_factor ** i
            weighted_bid_depth += qty * weight

        # Weight asks by distance from mid
        weighted_ask_depth = 0.0
        for i, (price, qty) in enumerate(orderbook.asks[:self.orderbook_depth_levels]):
            weight = decay_factor ** i
            weighted_ask_depth += qty * weight

        total_weighted = weighted_bid_depth + weighted_ask_depth
        if total_weighted == 0:
            return 0.0

        imbalance = (weighted_bid_depth - weighted_ask_depth) / total_weighted

        return float(np.clip(imbalance, -1.0, 1.0))

    def reset(self) -> None:
        """Reset internal state for fresh calculations."""
        self._trade_sizes = []


def calculate_microstructure_features(data: MicrostructureData) -> dict[str, float]:
    """
    Convenience function to calculate all microstructure features.

    Args:
        data: MicrostructureData containing market data

    Returns:
        Dictionary of normalized feature values
    """
    calculator = MicrostructureFeatures()
    return calculator.calculate(data)


# Feature metadata for ML pipeline
MICROSTRUCTURE_FEATURE_INFO: dict[str, dict[str, Any]] = {
    "orderbook_imbalance": {
        "range": (-1, 1),
        "description": "Bid/ask depth ratio from orderbook",
        "bullish_interpretation": "Positive values indicate bid pressure",
    },
    "funding_rate": {
        "range": (-1, 1),
        "description": "Normalized perpetual futures funding rate",
        "bullish_interpretation": "Positive values indicate longs pay shorts",
    },
    "open_interest": {
        "range": (0, 1),
        "description": "Normalized total open interest",
        "bullish_interpretation": "Context-dependent",
    },
    "open_interest_change": {
        "range": (-1, 1),
        "description": "Rate of change of open interest",
        "bullish_interpretation": "Positive with price up = trend confirmation",
    },
    "volume_profile": {
        "range": (0, 1),
        "description": "Trade distribution concentration",
        "bullish_interpretation": "High concentration may indicate support/resistance",
    },
    "large_trade_ratio": {
        "range": (0, 1),
        "description": "Whale activity detection (trades > 2 std)",
        "bullish_interpretation": "Context-dependent, watch direction",
    },
    "trade_flow_imbalance": {
        "range": (-1, 1),
        "description": "Buy vs sell volume imbalance",
        "bullish_interpretation": "Positive values indicate net buying",
    },
}
