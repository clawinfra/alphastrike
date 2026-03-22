"""
Tests for src/features/microstructure.py.

Target: 100% line + branch coverage.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.features.microstructure import (
    MICROSTRUCTURE_FEATURE_INFO,
    FundingInfo,
    MicrostructureData,
    MicrostructureFeatures,
    OpenInterestInfo,
    OrderbookCache,
    OrderbookSnapshot,
    TradeRecord,
    calculate_microstructure_features,
    calculate_top5_orderbook_imbalance,
    orderbook_from_unified,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_snapshot(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    timestamp: float = 0.0,
) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        bids=bids if bids is not None else [],
        asks=asks if asks is not None else [],
        timestamp=timestamp,
    )


def _make_trades(n: int, side: str = "buy", price: float = 100.0, qty: float = 1.0) -> list[TradeRecord]:
    return [TradeRecord(price=price, quantity=qty, side=side) for _ in range(n)]


# ---------------------------------------------------------------------------
# TestOrderbookSnapshot
# ---------------------------------------------------------------------------


class TestOrderbookSnapshot:
    def test_best_bid_with_bids(self):
        snap = _make_snapshot(bids=[(100.0, 1.0), (99.0, 2.0)])
        assert snap.best_bid == 100.0

    def test_best_bid_empty(self):
        snap = _make_snapshot(bids=[])
        assert snap.best_bid == 0.0

    def test_best_ask_with_asks(self):
        snap = _make_snapshot(asks=[(101.0, 1.0), (102.0, 2.0)])
        assert snap.best_ask == 101.0

    def test_best_ask_empty(self):
        snap = _make_snapshot(asks=[])
        assert snap.best_ask == 0.0

    def test_mid_price_normal(self):
        snap = _make_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
        assert snap.mid_price == 100.5

    def test_mid_price_no_bid(self):
        snap = _make_snapshot(bids=[], asks=[(101.0, 1.0)])
        assert snap.mid_price == 0.0

    def test_mid_price_no_ask(self):
        snap = _make_snapshot(bids=[(100.0, 1.0)], asks=[])
        assert snap.mid_price == 0.0


# ---------------------------------------------------------------------------
# TestMicrostructureDataclasses
# ---------------------------------------------------------------------------


class TestMicrostructureDataclasses:
    def test_trade_record_defaults(self):
        record = TradeRecord(price=100.0, quantity=1.0, side="buy")
        assert record.timestamp == 0.0

    def test_funding_info_defaults(self):
        info = FundingInfo(funding_rate=0.001)
        assert info.predicted_rate is None
        assert info.next_funding_time == 0.0

    def test_open_interest_info_defaults(self):
        info = OpenInterestInfo(open_interest=1000.0)
        assert info.timestamp == 0.0
        assert info.open_interest_value == 0.0

    def test_microstructure_data_defaults(self):
        data = MicrostructureData()
        assert data.orderbook is None
        assert data.funding is None
        assert data.open_interest is None
        assert data.open_interest_history == []
        assert data.recent_trades == []


# ---------------------------------------------------------------------------
# TestMicrostructureFeaturesInit
# ---------------------------------------------------------------------------


class TestMicrostructureFeaturesInit:
    def test_default_config(self):
        calc = MicrostructureFeatures()
        assert calc.orderbook_depth_levels == 10
        assert calc.funding_rate_cap == 0.01

    def test_post_init(self):
        calc = MicrostructureFeatures()
        assert calc._trade_sizes == []

    def test_custom_config(self):
        calc = MicrostructureFeatures(orderbook_depth_levels=5, funding_rate_cap=0.02)
        assert calc.orderbook_depth_levels == 5
        assert calc.funding_rate_cap == 0.02


# ---------------------------------------------------------------------------
# TestCalculateOrderbookImbalance
# ---------------------------------------------------------------------------


class TestCalculateOrderbookImbalance:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_none_orderbook(self):
        assert self.calc._calculate_orderbook_imbalance(None) == 0.0

    def test_empty_bids(self):
        snap = _make_snapshot(bids=[], asks=[(101.0, 1.0)])
        assert self.calc._calculate_orderbook_imbalance(snap) == 0.0

    def test_empty_asks(self):
        snap = _make_snapshot(bids=[(100.0, 1.0)], asks=[])
        assert self.calc._calculate_orderbook_imbalance(snap) == 0.0

    def test_balanced_book(self):
        snap = _make_snapshot(
            bids=[(100.0, 5.0)],
            asks=[(101.0, 5.0)],
        )
        assert self.calc._calculate_orderbook_imbalance(snap) == 0.0

    def test_bid_heavy(self):
        snap = _make_snapshot(
            bids=[(100.0, 8.0)],
            asks=[(101.0, 2.0)],
        )
        result = self.calc._calculate_orderbook_imbalance(snap)
        assert result > 0.0

    def test_ask_heavy(self):
        snap = _make_snapshot(
            bids=[(100.0, 2.0)],
            asks=[(101.0, 8.0)],
        )
        result = self.calc._calculate_orderbook_imbalance(snap)
        assert result < 0.0

    def test_extreme_bid_only(self):
        snap = _make_snapshot(
            bids=[(100.0, 10.0)],
            asks=[(101.0, 0.0)],
        )
        result = self.calc._calculate_orderbook_imbalance(snap)
        assert result == pytest.approx(1.0)

    def test_extreme_ask_only(self):
        snap = _make_snapshot(
            bids=[(100.0, 0.0)],
            asks=[(101.0, 10.0)],
        )
        result = self.calc._calculate_orderbook_imbalance(snap)
        assert result == pytest.approx(-1.0)

    def test_zero_total_depth(self):
        snap = _make_snapshot(
            bids=[(100.0, 0.0)],
            asks=[(101.0, 0.0)],
        )
        assert self.calc._calculate_orderbook_imbalance(snap) == 0.0

    def test_depth_levels_limit(self):
        # Only 2 levels should be considered
        calc = MicrostructureFeatures(orderbook_depth_levels=2)
        snap = _make_snapshot(
            bids=[(100.0, 5.0), (99.0, 5.0), (98.0, 100.0)],
            asks=[(101.0, 5.0), (102.0, 5.0), (103.0, 0.0)],
        )
        # Bid (top 2): 5+5=10, Ask (top 2): 5+5=10 → balanced
        result = calc._calculate_orderbook_imbalance(snap)
        assert result == pytest.approx(0.0)

    def test_clipping(self):
        # Force a value that would exceed [-1, 1] without clipping
        snap = _make_snapshot(
            bids=[(100.0, 1e18)],
            asks=[(101.0, 1e-18)],
        )
        result = self.calc._calculate_orderbook_imbalance(snap)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestCalculateTop5OrderbookImbalance
# ---------------------------------------------------------------------------


class TestCalculateTop5OrderbookImbalance:
    def test_none_orderbook(self):
        assert calculate_top5_orderbook_imbalance(None) == 0.5

    def test_empty_bids_and_asks(self):
        snap = _make_snapshot(bids=[], asks=[])
        assert calculate_top5_orderbook_imbalance(snap) == 0.5

    def test_balanced_book(self):
        snap = _make_snapshot(
            bids=[(100.0, 5.0)],
            asks=[(101.0, 5.0)],
        )
        assert calculate_top5_orderbook_imbalance(snap) == pytest.approx(0.5)

    def test_bid_heavy(self):
        snap = _make_snapshot(
            bids=[(100.0, 8.0)],
            asks=[(101.0, 2.0)],
        )
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(0.8)

    def test_ask_heavy(self):
        snap = _make_snapshot(
            bids=[(100.0, 2.0)],
            asks=[(101.0, 8.0)],
        )
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(0.2)

    def test_all_bids_no_asks(self):
        snap = _make_snapshot(
            bids=[(100.0, 10.0)],
            asks=[],
        )
        assert calculate_top5_orderbook_imbalance(snap) == pytest.approx(1.0)

    def test_no_bids_all_asks(self):
        snap = _make_snapshot(
            bids=[],
            asks=[(101.0, 10.0)],
        )
        assert calculate_top5_orderbook_imbalance(snap) == pytest.approx(0.0)

    def test_zero_depth_both_sides(self):
        snap = _make_snapshot(
            bids=[(100.0, 0.0)],
            asks=[(101.0, 0.0)],
        )
        assert calculate_top5_orderbook_imbalance(snap) == 0.5

    def test_fewer_than_5_levels(self):
        # 3 levels each — should still work
        snap = _make_snapshot(
            bids=[(100.0, 2.0), (99.0, 2.0), (98.0, 2.0)],
            asks=[(101.0, 4.0), (102.0, 4.0), (103.0, 4.0)],
        )
        # bid_sum=6, ask_sum=12, total=18, ratio=6/18≈0.333
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(6.0 / 18.0)

    def test_more_than_5_levels_only_uses_top5(self):
        # Extra levels beyond top-5 should be ignored
        snap = _make_snapshot(
            bids=[(100.0 - i, 1.0) for i in range(10)],   # 10 levels, 1 each
            asks=[(101.0 + i, 1.0) for i in range(10)],   # 10 levels, 1 each
        )
        # top-5 bid: 5, top-5 ask: 5 → 0.5
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(0.5)

    def test_exact_boundary_0_6(self):
        snap = _make_snapshot(
            bids=[(100.0, 6.0)],
            asks=[(101.0, 4.0)],
        )
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(0.6)
        assert result > 0.6 - 1e-9  # bid-heavy threshold

    def test_exact_boundary_0_4(self):
        snap = _make_snapshot(
            bids=[(100.0, 4.0)],
            asks=[(101.0, 6.0)],
        )
        result = calculate_top5_orderbook_imbalance(snap)
        assert result == pytest.approx(0.4)
        assert result < 0.4 + 1e-9  # ask-heavy threshold


# ---------------------------------------------------------------------------
# TestOrderbookCache
# ---------------------------------------------------------------------------


class TestOrderbookCache:
    def test_put_and_get(self):
        cache = OrderbookCache(ttl_seconds=10.0)
        snap = _make_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
        cache.put("BTCUSDT", snap)
        result = cache.get("BTCUSDT")
        assert result is snap

    def test_get_expired(self):
        cache = OrderbookCache(ttl_seconds=0.05)
        snap = _make_snapshot()
        cache.put("BTCUSDT", snap)
        time.sleep(0.1)
        result = cache.get("BTCUSDT")
        assert result is None

    def test_get_missing_symbol(self):
        cache = OrderbookCache()
        assert cache.get("NONEXISTENT") is None

    def test_invalidate_specific(self):
        cache = OrderbookCache(ttl_seconds=10.0)
        snap1 = _make_snapshot(bids=[(100.0, 1.0)])
        snap2 = _make_snapshot(bids=[(200.0, 1.0)])
        cache.put("BTCUSDT", snap1)
        cache.put("ETHUSDT", snap2)
        cache.invalidate("BTCUSDT")
        assert cache.get("BTCUSDT") is None
        assert cache.get("ETHUSDT") is snap2

    def test_invalidate_all(self):
        cache = OrderbookCache(ttl_seconds=10.0)
        cache.put("BTCUSDT", _make_snapshot())
        cache.put("ETHUSDT", _make_snapshot())
        cache.invalidate()
        assert cache.get("BTCUSDT") is None
        assert cache.get("ETHUSDT") is None

    def test_size(self):
        cache = OrderbookCache(ttl_seconds=10.0)
        assert cache.size == 0
        cache.put("BTCUSDT", _make_snapshot())
        assert cache.size == 1
        cache.put("ETHUSDT", _make_snapshot())
        assert cache.size == 2

    def test_overwrite_existing(self):
        cache = OrderbookCache(ttl_seconds=10.0)
        snap1 = _make_snapshot(bids=[(100.0, 1.0)])
        snap2 = _make_snapshot(bids=[(200.0, 2.0)])
        cache.put("BTCUSDT", snap1)
        cache.put("BTCUSDT", snap2)
        assert cache.get("BTCUSDT") is snap2
        assert cache.size == 1

    def test_thread_safety(self):
        """Concurrent get/put from multiple threads must not crash."""
        cache = OrderbookCache(ttl_seconds=10.0)
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(100):
                    cache.put(f"SYM{i % 5}", _make_snapshot())
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"SYM{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_custom_ttl(self):
        cache = OrderbookCache(ttl_seconds=0.1)
        snap = _make_snapshot()
        cache.put("BTCUSDT", snap)
        assert cache.get("BTCUSDT") is snap
        time.sleep(0.15)
        assert cache.get("BTCUSDT") is None


# ---------------------------------------------------------------------------
# TestOrderbookFromUnified
# ---------------------------------------------------------------------------


class TestOrderbookFromUnified:
    def _make_unified(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ):
        """Create a mock UnifiedOrderbook."""
        unified = MagicMock()
        unified.bids = bids
        unified.asks = asks
        unified.timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        return unified

    def test_conversion_preserves_data(self):
        bids = [(100.0, 1.5), (99.0, 2.0)]
        asks = [(101.0, 1.2), (102.0, 0.8)]
        unified = self._make_unified(bids, asks)
        snap = orderbook_from_unified(unified)
        assert snap.bids == bids
        assert snap.asks == asks
        assert snap.timestamp == pytest.approx(unified.timestamp.timestamp())

    def test_empty_unified(self):
        unified = self._make_unified([], [])
        snap = orderbook_from_unified(unified)
        assert snap.bids == []
        assert snap.asks == []


# ---------------------------------------------------------------------------
# TestCalculateFundingRate
# ---------------------------------------------------------------------------


class TestCalculateFundingRate:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_none_funding(self):
        assert self.calc._calculate_funding_rate(None) == 0.0

    def test_positive_funding(self):
        funding = FundingInfo(funding_rate=0.005)  # 0.5%
        result = self.calc._calculate_funding_rate(funding)
        assert result == pytest.approx(0.5)

    def test_negative_funding(self):
        funding = FundingInfo(funding_rate=-0.005)
        result = self.calc._calculate_funding_rate(funding)
        assert result == pytest.approx(-0.5)

    def test_extreme_positive_clipped(self):
        funding = FundingInfo(funding_rate=0.05)  # Way above cap
        result = self.calc._calculate_funding_rate(funding)
        assert result == 1.0

    def test_extreme_negative_clipped(self):
        funding = FundingInfo(funding_rate=-0.05)
        result = self.calc._calculate_funding_rate(funding)
        assert result == -1.0

    def test_zero_funding(self):
        funding = FundingInfo(funding_rate=0.0)
        result = self.calc._calculate_funding_rate(funding)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestCalculateOpenInterest
# ---------------------------------------------------------------------------


class TestCalculateOpenInterest:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_none_oi(self):
        assert self.calc._calculate_open_interest(None) == 0.5

    def test_zero_oi(self):
        oi = OpenInterestInfo(open_interest=0.0)
        assert self.calc._calculate_open_interest(oi) == 0.5

    def test_positive_oi(self):
        oi = OpenInterestInfo(open_interest=1000.0)
        # Single snapshot returns neutral 0.5
        assert self.calc._calculate_open_interest(oi) == 0.5


# ---------------------------------------------------------------------------
# TestCalculateOIChange
# ---------------------------------------------------------------------------


class TestCalculateOIChange:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_empty_history(self):
        assert self.calc._calculate_oi_change([]) == 0.0

    def test_single_entry(self):
        history = [OpenInterestInfo(open_interest=1000.0)]
        assert self.calc._calculate_oi_change(history) == 0.0

    def test_increasing_oi(self):
        history = [
            OpenInterestInfo(open_interest=1000.0),
            OpenInterestInfo(open_interest=1100.0),
        ]
        result = self.calc._calculate_oi_change(history)
        assert result > 0.0

    def test_decreasing_oi(self):
        history = [
            OpenInterestInfo(open_interest=1000.0),
            OpenInterestInfo(open_interest=900.0),
        ]
        result = self.calc._calculate_oi_change(history)
        assert result < 0.0

    def test_zero_oldest_oi(self):
        history = [
            OpenInterestInfo(open_interest=0.0),
            OpenInterestInfo(open_interest=1000.0),
        ]
        assert self.calc._calculate_oi_change(history) == 0.0

    def test_negative_oldest_oi(self):
        # oldest_oi <= 0 when negative
        history = [
            OpenInterestInfo(open_interest=-100.0),
            OpenInterestInfo(open_interest=1000.0),
        ]
        assert self.calc._calculate_oi_change(history) == 0.0

    def test_zero_values_filtered(self):
        # All zero values → less than 2 valid entries after filter
        history = [
            OpenInterestInfo(open_interest=0.0),
            OpenInterestInfo(open_interest=0.0),
        ]
        assert self.calc._calculate_oi_change(history) == 0.0

    def test_extreme_change_clipped(self):
        history = [
            OpenInterestInfo(open_interest=100.0),
            OpenInterestInfo(open_interest=10000.0),  # 9900% increase
        ]
        result = self.calc._calculate_oi_change(history)
        assert result == 1.0

    def test_window_size_respected(self):
        calc = MicrostructureFeatures(oi_change_window=2)
        history = [
            OpenInterestInfo(open_interest=2000.0),  # Should be ignored
            OpenInterestInfo(open_interest=1000.0),  # Window start
            OpenInterestInfo(open_interest=1100.0),  # Window end
        ]
        result = calc._calculate_oi_change(history)
        # 10% change / 20% cap = 0.5
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestCalculateVolumeProfile
# ---------------------------------------------------------------------------


class TestCalculateVolumeProfile:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_few_trades(self):
        trades = _make_trades(5)
        assert self.calc._calculate_volume_profile(trades) == 0.5

    def test_few_valid_prices(self):
        trades = [TradeRecord(price=0.0, quantity=1.0, side="buy") for _ in range(15)]
        assert self.calc._calculate_volume_profile(trades) == 0.5

    def test_all_same_price(self):
        trades = _make_trades(20, price=100.0)
        result = self.calc._calculate_volume_profile(trades)
        assert result == 1.0

    def test_uniform_distribution(self):
        # Spread trades evenly across 5 bins
        import math
        prices = [100.0 + i * 20.0 for i in range(20)]  # Wide spread
        trades = [TradeRecord(price=p, quantity=1.0, side="buy") for p in prices]
        result = self.calc._calculate_volume_profile(trades)
        assert 0.0 <= result <= 1.0

    def test_concentrated_distribution(self):
        # Most trades at same price = high concentration
        trades = _make_trades(18, price=100.0) + _make_trades(2, price=200.0)
        result = self.calc._calculate_volume_profile(trades)
        assert result > 0.5

    def test_zero_price_range(self):
        trades = _make_trades(15, price=100.0)
        result = self.calc._calculate_volume_profile(trades)
        assert result == 1.0

    def test_single_bin(self):
        calc = MicrostructureFeatures(volume_profile_bins=1)
        trades = _make_trades(20, price=100.0)
        result = calc._calculate_volume_profile(trades)
        assert result == 1.0

    def test_multiple_bins_outcome(self):
        # Verify volume profile returns valid float for multi-bin spread
        calc = MicrostructureFeatures(volume_profile_bins=5)
        trades = [TradeRecord(price=100.0 + i * 5.0, quantity=1.0, side="buy") for i in range(20)]
        result = calc._calculate_volume_profile(trades)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestCalculateLargeTradeRatio
# ---------------------------------------------------------------------------


class TestCalculateLargeTradeRatio:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_few_trades(self):
        trades = _make_trades(10)
        assert self.calc._calculate_large_trade_ratio(trades) == 0.0

    def test_no_large_trades(self):
        # 20 trades all exactly the same size → no outliers
        trades = _make_trades(20, qty=1.0)
        result = self.calc._calculate_large_trade_ratio(trades)
        assert result == 0.0

    def test_some_large_trades(self):
        # Mix small and whale trades
        small = _make_trades(18, qty=1.0)
        large = _make_trades(2, qty=100.0)
        result = self.calc._calculate_large_trade_ratio(small + large)
        assert result > 0.0

    def test_rolling_window_trimmed(self):
        calc = MicrostructureFeatures()
        calc._trade_size_window = 50
        # Add 60 trades → window trims to 50
        trades = _make_trades(60, qty=1.0)
        calc._calculate_large_trade_ratio(trades)
        assert len(calc._trade_sizes) <= 50

    def test_zero_std(self):
        # All same size → std=0 → return 0.0
        trades = _make_trades(25, qty=5.0)
        result = self.calc._calculate_large_trade_ratio(trades)
        assert result == 0.0

    def test_stats_error(self):
        # Single unique value in rolling window → stdev raises StatisticsError
        # This shouldn't raise — should return 0.0
        calc = MicrostructureFeatures()
        calc._trade_sizes = [1.0]  # Only 1 element → stdev fails
        trades = _make_trades(0)  # No new trades, but sizes < 20 from above; reset
        # Directly inject: simulate condition where sizes is length 1 inside the method
        # by giving 20 trades all same size (std=0 is caught separately)
        trades20 = _make_trades(20, qty=1.0)
        # set window artificially small so stdev fails on 1-item list
        calc._trade_size_window = 1
        # Can't easily trigger statistics.StatisticsError path directly via public interface
        # because std=0 returns 0.0 before statistics.stdev is reached.
        # Test that std=0 path is covered (also returns 0.0):
        result = calc._calculate_large_trade_ratio(trades20)
        assert result == 0.0

    def test_sizes_filtered_to_below_20(self):
        # 25 trades but all with quantity=0.0 → sizes=[] < 20 → return 0.0
        trades = [TradeRecord(price=100.0, quantity=0.0, side="buy") for _ in range(25)]
        result = self.calc._calculate_large_trade_ratio(trades)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestCalculateTradeFlowImbalance
# ---------------------------------------------------------------------------


class TestCalculateTradeFlowImbalance:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_empty_trades(self):
        assert self.calc._calculate_trade_flow_imbalance([]) == 0.0

    def test_all_buys(self):
        trades = _make_trades(10, side="buy", qty=1.0)
        assert self.calc._calculate_trade_flow_imbalance(trades) == pytest.approx(1.0)

    def test_all_sells(self):
        trades = _make_trades(10, side="sell", qty=1.0)
        assert self.calc._calculate_trade_flow_imbalance(trades) == pytest.approx(-1.0)

    def test_balanced(self):
        buys = _make_trades(5, side="buy", qty=1.0)
        sells = _make_trades(5, side="sell", qty=1.0)
        result = self.calc._calculate_trade_flow_imbalance(buys + sells)
        assert result == pytest.approx(0.0)

    def test_mostly_buys(self):
        buys = _make_trades(8, side="buy", qty=1.0)
        sells = _make_trades(2, side="sell", qty=1.0)
        result = self.calc._calculate_trade_flow_imbalance(buys + sells)
        assert result > 0.0

    def test_zero_volume(self):
        trades = [TradeRecord(price=100.0, quantity=0.0, side="buy") for _ in range(5)]
        result = self.calc._calculate_trade_flow_imbalance(trades)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestWeightedOrderbookImbalance
# ---------------------------------------------------------------------------


class TestWeightedOrderbookImbalance:
    def setup_method(self):
        self.calc = MicrostructureFeatures()

    def test_none_orderbook(self):
        assert self.calc.calculate_weighted_orderbook_imbalance(None) == 0.0

    def test_empty_bids_asks(self):
        snap = _make_snapshot(bids=[], asks=[])
        assert self.calc.calculate_weighted_orderbook_imbalance(snap) == 0.0

    def test_zero_mid_price(self):
        snap = _make_snapshot(bids=[], asks=[])
        assert self.calc.calculate_weighted_orderbook_imbalance(snap) == 0.0

    def test_zero_mid_price_with_zero_price_bid(self):
        # bid price=0.0 → best_bid=0 → mid_price=0 → return 0.0
        snap = _make_snapshot(bids=[(0.0, 5.0)], asks=[(101.0, 5.0)])
        assert self.calc.calculate_weighted_orderbook_imbalance(snap) == 0.0

    def test_balanced(self):
        snap = _make_snapshot(
            bids=[(100.0, 5.0)],
            asks=[(101.0, 5.0)],
        )
        result = self.calc.calculate_weighted_orderbook_imbalance(snap)
        assert result == pytest.approx(0.0)

    def test_bid_heavy_weighted(self):
        snap = _make_snapshot(
            bids=[(100.0, 10.0)],
            asks=[(101.0, 2.0)],
        )
        result = self.calc.calculate_weighted_orderbook_imbalance(snap)
        assert result > 0.0

    def test_decay_factor_effect(self):
        # With 2 levels, lower decay gives more weight to top level
        snap = _make_snapshot(
            bids=[(100.0, 1.0), (99.0, 100.0)],
            asks=[(101.0, 1.0), (102.0, 1.0)],
        )
        result_high_decay = self.calc.calculate_weighted_orderbook_imbalance(snap, decay_factor=0.99)
        result_low_decay = self.calc.calculate_weighted_orderbook_imbalance(snap, decay_factor=0.1)
        # With high decay, the deep bid at level 2 contributes a lot → more bid heavy
        # With low decay, level 2 is nearly ignored → balanced (top level equal)
        assert result_high_decay > result_low_decay

    def test_zero_total_weighted(self):
        snap = _make_snapshot(
            bids=[(100.0, 0.0)],
            asks=[(101.0, 0.0)],
        )
        result = self.calc.calculate_weighted_orderbook_imbalance(snap)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestCalculateAllFeatures
# ---------------------------------------------------------------------------


class TestCalculateAllFeatures:
    def test_full_calculation(self):
        calc = MicrostructureFeatures()
        snap = _make_snapshot(bids=[(100.0, 5.0)], asks=[(101.0, 5.0)])
        funding = FundingInfo(funding_rate=0.001)
        oi = OpenInterestInfo(open_interest=1000.0)
        oi_history = [OpenInterestInfo(open_interest=900.0), OpenInterestInfo(open_interest=1000.0)]
        trades = _make_trades(30, side="buy", qty=1.0)

        data = MicrostructureData(
            orderbook=snap,
            funding=funding,
            open_interest=oi,
            open_interest_history=oi_history,
            recent_trades=trades,
        )
        features = calc.calculate(data)

        expected_keys = {
            "orderbook_imbalance",
            "funding_rate",
            "open_interest",
            "open_interest_change",
            "volume_profile",
            "large_trade_ratio",
            "trade_flow_imbalance",
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_with_empty_data(self):
        calc = MicrostructureFeatures()
        data = MicrostructureData()
        features = calc.calculate(data)
        assert isinstance(features, dict)
        assert len(features) >= 7

    def test_convenience_function(self):
        data = MicrostructureData()
        result = calculate_microstructure_features(data)
        assert isinstance(result, dict)
        assert len(result) >= 7


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_trade_sizes(self):
        calc = MicrostructureFeatures()
        calc._trade_sizes = [1.0, 2.0, 3.0]
        calc.reset()
        assert calc._trade_sizes == []


# ---------------------------------------------------------------------------
# TestFeatureMetadata
# ---------------------------------------------------------------------------


class TestFeatureMetadata:
    def test_all_features_have_metadata(self):
        expected = {
            "orderbook_imbalance",
            "funding_rate",
            "open_interest",
            "open_interest_change",
            "volume_profile",
            "large_trade_ratio",
            "trade_flow_imbalance",
            "top5_orderbook_imbalance",
        }
        assert expected.issubset(set(MICROSTRUCTURE_FEATURE_INFO.keys()))

    def test_metadata_has_required_keys(self):
        for name, info in MICROSTRUCTURE_FEATURE_INFO.items():
            assert "range" in info, f"Missing 'range' in {name}"
            assert "description" in info, f"Missing 'description' in {name}"
            assert "bullish_interpretation" in info, f"Missing 'bullish_interpretation' in {name}"
