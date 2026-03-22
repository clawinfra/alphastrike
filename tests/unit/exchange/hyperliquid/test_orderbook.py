"""
Tests for HyperliquidRESTClient.get_orderbook() — symbol conversion, response
parsing, and error handling.

No real API calls are made; all HTTP calls are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.exchange.adapters.hyperliquid.adapter import HyperliquidRESTClient
from src.exchange.adapters.hyperliquid.mappers import HyperliquidMapper
from src.exchange.exceptions import ExchangeError
from src.exchange.models import UnifiedOrderbook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> HyperliquidRESTClient:
    """Create a client with mocked settings."""
    with patch("src.exchange.adapters.hyperliquid.adapter.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            exchange=MagicMock(
                wallet_private_key="0x" + "a" * 64,
                wallet_address=None,
            )
        )
        client = HyperliquidRESTClient(testnet=True)
        # Load known assets so symbol conversion works
        HyperliquidMapper.set_asset_meta([
            {"name": "BTC"},
            {"name": "ETH"},
            {"name": "SOL"},
        ])
        return client


def _l2book_response(bids: list[dict], asks: list[dict]) -> dict:
    """Build a realistic HL l2Book response."""
    return {
        "levels": [
            bids,   # index 0 = bids
            asks,   # index 1 = asks
        ]
    }


# ---------------------------------------------------------------------------
# TestGetOrderbookSuccess
# ---------------------------------------------------------------------------


class TestGetOrderbookSuccess:
    """Happy-path tests for get_orderbook()."""

    @pytest.fixture
    def client(self):
        return _make_client()

    @pytest.mark.asyncio
    async def test_basic_btc_orderbook(self, client):
        """Full round-trip: BTCUSDT → BTC coin → l2Book → UnifiedOrderbook."""
        mock_response = _l2book_response(
            bids=[{"px": "49990.0", "sz": "1.5"}, {"px": "49980.0", "sz": "2.0"}],
            asks=[{"px": "50010.0", "sz": "1.2"}, {"px": "50020.0", "sz": "0.8"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            result = await client.get_orderbook("BTCUSDT")

        assert isinstance(result, UnifiedOrderbook)
        assert result.symbol == "BTCUSDT"
        assert len(result.bids) == 2
        assert len(result.asks) == 2

        # Bids sorted descending
        assert result.bids[0][0] >= result.bids[1][0]
        # Asks sorted ascending
        assert result.asks[0][0] <= result.asks[1][0]

        # Verify the request used the right coin
        call_args = mock_req.call_args[0][0]
        assert call_args["type"] == "l2Book"
        assert call_args["coin"] == "BTC"

    @pytest.mark.asyncio
    async def test_eth_orderbook(self, client):
        """get_orderbook works for ETH symbol."""
        mock_response = _l2book_response(
            bids=[{"px": "2995.0", "sz": "5.0"}],
            asks=[{"px": "3005.0", "sz": "5.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response

            result = await client.get_orderbook("ETHUSDT")

        assert result.symbol == "ETHUSDT"
        assert result.best_bid == pytest.approx(2995.0)
        assert result.best_ask == pytest.approx(3005.0)

        call_args = mock_req.call_args[0][0]
        assert call_args["coin"] == "ETH"

    @pytest.mark.asyncio
    async def test_sol_orderbook(self, client):
        """get_orderbook works for SOL symbol."""
        mock_response = _l2book_response(
            bids=[{"px": "99.5", "sz": "10.0"}],
            asks=[{"px": "100.5", "sz": "8.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("SOLUSDT")

        assert result.symbol == "SOLUSDT"
        call_args = mock_req.call_args[0][0]
        assert call_args["coin"] == "SOL"

    @pytest.mark.asyncio
    async def test_nsigsigs_parameter_sent(self, client):
        """Verify nSigFigs is included in the request."""
        mock_response = _l2book_response(
            bids=[{"px": "50000.0", "sz": "1.0"}],
            asks=[{"px": "50001.0", "sz": "1.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.get_orderbook("BTCUSDT")

        call_args = mock_req.call_args[0][0]
        assert "nSigFigs" in call_args

    @pytest.mark.asyncio
    async def test_empty_bids(self, client):
        """Orderbook with empty bids is handled."""
        mock_response = _l2book_response(
            bids=[],
            asks=[{"px": "50010.0", "sz": "1.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.bids == []
        assert len(result.asks) == 1
        assert result.best_bid == 0.0

    @pytest.mark.asyncio
    async def test_empty_asks(self, client):
        """Orderbook with empty asks is handled."""
        mock_response = _l2book_response(
            bids=[{"px": "49990.0", "sz": "1.0"}],
            asks=[],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert len(result.bids) == 1
        assert result.asks == []
        assert result.best_ask == 0.0

    @pytest.mark.asyncio
    async def test_empty_book(self, client):
        """Empty orderbook (both sides empty) is handled."""
        mock_response = _l2book_response(bids=[], asks=[])

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.bids == []
        assert result.asks == []
        assert result.best_bid == 0.0
        assert result.best_ask == 0.0
        assert result.mid_price == 0.0

    @pytest.mark.asyncio
    async def test_single_level_book(self, client):
        """Single-level orderbook calculates spread correctly."""
        mock_response = _l2book_response(
            bids=[{"px": "100.0", "sz": "10.0"}],
            asks=[{"px": "101.0", "sz": "10.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.spread > 0.0
        assert result.mid_price == pytest.approx(100.5)

    @pytest.mark.asyncio
    async def test_large_orderbook(self, client):
        """Orderbook with many levels is handled."""
        bids = [{"px": str(50000.0 - i), "sz": "1.0"} for i in range(20)]
        asks = [{"px": str(50001.0 + i), "sz": "1.0"} for i in range(20)]
        mock_response = _l2book_response(bids, asks)

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert len(result.bids) == 20
        assert len(result.asks) == 20

    @pytest.mark.asyncio
    async def test_orderbook_bid_depth(self, client):
        """total_bid_depth sums correctly."""
        mock_response = _l2book_response(
            bids=[{"px": "100.0", "sz": "5.0"}, {"px": "99.0", "sz": "3.0"}],
            asks=[{"px": "101.0", "sz": "2.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.total_bid_depth() == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_orderbook_ask_depth(self, client):
        """total_ask_depth sums correctly."""
        mock_response = _l2book_response(
            bids=[{"px": "100.0", "sz": "1.0"}],
            asks=[{"px": "101.0", "sz": "2.0"}, {"px": "102.0", "sz": "3.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.total_ask_depth() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestGetOrderbookSymbolConversion
# ---------------------------------------------------------------------------


class TestGetOrderbookSymbolConversion:
    """Tests for symbol → HL coin conversion."""

    @pytest.fixture
    def client(self):
        return _make_client()

    @pytest.mark.asyncio
    async def test_usdt_suffix_stripped(self, client):
        """BTCUSDT → BTC coin in the request."""
        mock_response = _l2book_response(
            bids=[{"px": "50000.0", "sz": "1.0"}],
            asks=[{"px": "50001.0", "sz": "1.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await client.get_orderbook("BTCUSDT")

        call_payload = mock_req.call_args[0][0]
        assert call_payload["coin"] == "BTC"
        assert "USDT" not in call_payload["coin"]

    @pytest.mark.asyncio
    async def test_result_symbol_preserved(self, client):
        """Result UnifiedOrderbook.symbol should match the input symbol."""
        mock_response = _l2book_response(
            bids=[{"px": "50000.0", "sz": "1.0"}],
            asks=[{"px": "50001.0", "sz": "1.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await client.get_orderbook("BTCUSDT")

        assert result.symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# TestGetOrderbookErrorHandling
# ---------------------------------------------------------------------------


class TestGetOrderbookErrorHandling:
    """Tests for error handling in get_orderbook()."""

    @pytest.fixture
    def client(self):
        return _make_client()

    @pytest.mark.asyncio
    async def test_api_network_error_propagates(self, client):
        """Network errors from _info_request propagate."""
        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = ExchangeError("Connection refused")

            with pytest.raises(ExchangeError):
                await client.get_orderbook("BTCUSDT")

    @pytest.mark.asyncio
    async def test_api_generic_exception_propagates(self, client):
        """Generic exceptions from _info_request propagate."""
        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(RuntimeError):
                await client.get_orderbook("BTCUSDT")

    @pytest.mark.asyncio
    async def test_limit_param_ignored_by_hl(self, client):
        """HL doesn't use a limit param (nSigFigs instead). No crash."""
        mock_response = _l2book_response(
            bids=[{"px": "50000.0", "sz": "1.0"}],
            asks=[{"px": "50001.0", "sz": "1.0"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            # Even with a non-default limit, it should work
            result = await client.get_orderbook("BTCUSDT", limit=50)

        assert isinstance(result, UnifiedOrderbook)


# ---------------------------------------------------------------------------
# TestOrderbookFromUnifiedRoundtrip
# ---------------------------------------------------------------------------


class TestOrderbookFromUnifiedRoundtrip:
    """Verify that orderbook_from_unified() preserves data from get_orderbook()."""

    @pytest.fixture
    def client(self):
        return _make_client()

    @pytest.mark.asyncio
    async def test_roundtrip_to_snapshot(self, client):
        """get_orderbook() → orderbook_from_unified() → OrderbookSnapshot has correct data."""
        from src.features.microstructure import orderbook_from_unified

        mock_response = _l2book_response(
            bids=[{"px": "49990.0", "sz": "1.5"}],
            asks=[{"px": "50010.0", "sz": "1.2"}],
        )

        with patch.object(client, "_info_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            unified = await client.get_orderbook("BTCUSDT")

        snapshot = orderbook_from_unified(unified)

        assert snapshot.bids == unified.bids
        assert snapshot.asks == unified.asks
        assert snapshot.timestamp == pytest.approx(unified.timestamp.timestamp())
        assert snapshot.best_bid == pytest.approx(49990.0)
        assert snapshot.best_ask == pytest.approx(50010.0)
        assert snapshot.mid_price == pytest.approx(50000.0)
