"""
OpenAPI to Protocol Mapper

Maps OpenAPI endpoints to ExchangeRESTProtocol methods.
Helps developers understand how to build adapters from OpenAPI specs.

Usage:
    from src.exchange.openapi.parser import OpenAPIParser
    from src.exchange.openapi.mapper import ProtocolMapper

    parser = OpenAPIParser()
    spec = parser.parse_file("docs/weex_api.json")

    mapper = ProtocolMapper()
    mapping = mapper.generate_mapping(spec)

    for protocol_method, endpoint_info in mapping.items():
        print(f"{protocol_method}: {endpoint_info.path}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.exchange.openapi.parser import (
    APIEndpoint,
    HTTPMethod,
    ParsedOpenAPISpec,
)

logger = logging.getLogger(__name__)


@dataclass
class EndpointMapping:
    """Mapping from a protocol method to an API endpoint."""

    protocol_method: str
    endpoint: APIEndpoint
    confidence: float  # 0.0 to 1.0
    notes: str = ""

    # Field mappings (protocol field -> API field)
    request_mapping: dict[str, str] = field(default_factory=dict)
    response_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class ProtocolMappingResult:
    """Result of mapping an OpenAPI spec to the exchange protocol."""

    spec_title: str
    base_url: str
    mappings: dict[str, EndpointMapping]
    unmapped_endpoints: list[APIEndpoint]
    missing_protocol_methods: list[str]

    def get_mapping(self, protocol_method: str) -> EndpointMapping | None:
        """Get mapping for a specific protocol method."""
        return self.mappings.get(protocol_method)

    def get_coverage(self) -> float:
        """Get protocol coverage percentage."""
        total_methods = len(self.mappings) + len(self.missing_protocol_methods)
        if total_methods == 0:
            return 0.0
        return len(self.mappings) / total_methods

    def generate_adapter_skeleton(self) -> str:
        """Generate a skeleton adapter class based on mappings."""
        lines = [
            '"""',
            f"Auto-generated adapter skeleton for {self.spec_title}",
            '"""',
            "",
            "from src.exchange.protocols import ExchangeRESTProtocol",
            "from src.exchange.models import (",
            "    ExchangeCapabilities,",
            "    UnifiedAccountBalance,",
            "    UnifiedCandle,",
            "    UnifiedOrder,",
            "    UnifiedOrderbook,",
            "    UnifiedOrderResult,",
            "    UnifiedPosition,",
            "    UnifiedTicker,",
            "    UnifiedTrade,",
            ")",
            "",
            "",
            "class GeneratedRESTClient(ExchangeRESTProtocol):",
            '    """Generated REST client - customize as needed."""',
            "",
            f'    BASE_URL = "{self.base_url}"',
            "",
        ]

        # Generate method stubs for each mapping
        for method_name, mapping in self.mappings.items():
            lines.extend(
                [
                    f"    async def {method_name}(self, ...):",
                    '        """',
                    f"        Maps to: {mapping.endpoint.method.value} {mapping.endpoint.path}",
                    f"        Confidence: {mapping.confidence:.0%}",
                ]
            )
            if mapping.notes:
                lines.append(f"        Notes: {mapping.notes}")
            lines.extend(
                [
                    '        """',
                    "        # TODO: Implement",
                    "        raise NotImplementedError()",
                    "",
                ]
            )

        # Add stubs for missing methods
        for method_name in self.missing_protocol_methods:
            lines.extend(
                [
                    f"    async def {method_name}(self, ...):",
                    '        """No matching endpoint found in OpenAPI spec."""',
                    "        raise NotImplementedError()",
                    "",
                ]
            )

        return "\n".join(lines)


class ProtocolMapper:
    """
    Maps OpenAPI specifications to ExchangeRESTProtocol.

    Analyzes an OpenAPI spec and suggests how endpoints map to
    protocol methods, helping developers build new adapters.
    """

    # Protocol methods and their characteristics
    PROTOCOL_METHODS = {
        # Market data
        "get_ticker": {
            "keywords": ["ticker", "price", "quote"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_orderbook": {
            "keywords": ["orderbook", "depth", "book"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_candles": {
            "keywords": ["candle", "kline", "ohlc", "candlestick"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_recent_trades": {
            "keywords": ["trade", "fill", "execution"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_funding_rate": {
            "keywords": ["funding", "fundrate", "rate"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_symbol_info": {
            "keywords": ["contract", "symbol", "instrument", "info"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        "get_all_symbols": {
            "keywords": ["contracts", "symbols", "instruments", "all"],
            "method": HTTPMethod.GET,
            "category": "market",
        },
        # Account
        "get_account_balance": {
            "keywords": ["balance", "account", "asset", "equity"],
            "method": HTTPMethod.GET,
            "category": "account",
        },
        "get_positions": {
            "keywords": ["position", "positions", "all"],
            "method": HTTPMethod.GET,
            "category": "account",
        },
        "get_position": {
            "keywords": ["position", "single"],
            "method": HTTPMethod.GET,
            "category": "account",
        },
        # Orders
        "place_order": {
            "keywords": ["order", "place", "create", "new"],
            "method": HTTPMethod.POST,
            "category": "order",
        },
        "cancel_order": {
            "keywords": ["cancel", "order"],
            "method": HTTPMethod.POST,
            "category": "order",
        },
        "get_order": {
            "keywords": ["order", "detail", "query"],
            "method": HTTPMethod.GET,
            "category": "order",
        },
        "get_open_orders": {
            "keywords": ["order", "open", "current", "pending"],
            "method": HTTPMethod.GET,
            "category": "order",
        },
        # Leverage
        "set_leverage": {
            "keywords": ["leverage", "set"],
            "method": HTTPMethod.POST,
            "category": "leverage",
        },
        "get_leverage": {
            "keywords": ["leverage", "get"],
            "method": HTTPMethod.GET,
            "category": "leverage",
        },
        # Conditional orders
        "place_stop_loss": {
            "keywords": ["stop", "loss", "plan", "trigger"],
            "method": HTTPMethod.POST,
            "category": "order",
        },
        "place_take_profit": {
            "keywords": ["take", "profit", "plan", "trigger"],
            "method": HTTPMethod.POST,
            "category": "order",
        },
        "cancel_conditional_order": {
            "keywords": ["cancel", "plan", "conditional", "stop"],
            "method": HTTPMethod.POST,
            "category": "order",
        },
    }

    def generate_mapping(
        self,
        spec: ParsedOpenAPISpec,
    ) -> ProtocolMappingResult:
        """
        Generate mapping from OpenAPI spec to protocol methods.

        Args:
            spec: Parsed OpenAPI specification

        Returns:
            ProtocolMappingResult with suggested mappings
        """
        mappings: dict[str, EndpointMapping] = {}
        used_endpoints: set[str] = set()

        # Try to map each protocol method
        for method_name, characteristics in self.PROTOCOL_METHODS.items():
            best_match = self._find_best_match(
                spec.endpoints,
                method_name,
                characteristics,
                used_endpoints,
            )

            if best_match:
                endpoint, confidence, notes = best_match
                mappings[method_name] = EndpointMapping(
                    protocol_method=method_name,
                    endpoint=endpoint,
                    confidence=confidence,
                    notes=notes,
                )
                used_endpoints.add(endpoint.path + endpoint.method.value)

        # Find unmapped endpoints
        unmapped = [
            ep for ep in spec.endpoints if (ep.path + ep.method.value) not in used_endpoints
        ]

        # Find missing protocol methods
        missing = [method for method in self.PROTOCOL_METHODS if method not in mappings]

        return ProtocolMappingResult(
            spec_title=spec.title,
            base_url=spec.base_url,
            mappings=mappings,
            unmapped_endpoints=unmapped,
            missing_protocol_methods=missing,
        )

    def _find_best_match(
        self,
        endpoints: list[APIEndpoint],
        method_name: str,
        characteristics: dict[str, Any],
        used_endpoints: set[str],
    ) -> tuple[APIEndpoint, float, str] | None:
        """Find the best matching endpoint for a protocol method."""
        keywords = characteristics["keywords"]
        expected_method = characteristics["method"]

        best_match: tuple[APIEndpoint, float, str] | None = None
        best_score = 0.0

        for endpoint in endpoints:
            # Skip already used endpoints
            endpoint_key = endpoint.path + endpoint.method.value
            if endpoint_key in used_endpoints:
                continue

            # Calculate match score
            score, notes = self._calculate_match_score(
                endpoint,
                keywords,
                expected_method,
            )

            if score > best_score:
                best_score = score
                best_match = (endpoint, score, notes)

        # Only return if confidence is reasonable
        if best_match and best_match[1] >= 0.3:
            return best_match

        return None

    def _calculate_match_score(
        self,
        endpoint: APIEndpoint,
        keywords: list[str],
        expected_method: HTTPMethod,
    ) -> tuple[float, str]:
        """Calculate how well an endpoint matches a protocol method."""
        score = 0.0
        notes = []

        # Method match (25%)
        if endpoint.method == expected_method:
            score += 0.25
        else:
            notes.append(f"Method mismatch: expected {expected_method.value}")

        # Operation ID match (35%)
        op_id_lower = endpoint.operation_id.lower()
        keyword_matches = sum(1 for kw in keywords if kw in op_id_lower)
        if keyword_matches > 0:
            score += 0.35 * min(keyword_matches / len(keywords), 1.0)

        # Path match (25%)
        path_lower = endpoint.path.lower()
        path_matches = sum(1 for kw in keywords if kw in path_lower)
        if path_matches > 0:
            score += 0.25 * min(path_matches / len(keywords), 1.0)

        # Summary/description match (15%)
        text = (endpoint.summary + " " + endpoint.description).lower()
        text_matches = sum(1 for kw in keywords if kw in text)
        if text_matches > 0:
            score += 0.15 * min(text_matches / len(keywords), 1.0)

        # Bonus for exact operation ID patterns
        hint = endpoint.get_protocol_method_hint()
        if hint:
            score += 0.1

        return min(score, 1.0), "; ".join(notes) if notes else ""

    def print_mapping_report(
        self,
        result: ProtocolMappingResult,
    ) -> None:
        """Print a human-readable mapping report."""
        print(f"\n{'=' * 60}")
        print(f"Protocol Mapping Report for: {result.spec_title}")
        print(f"Base URL: {result.base_url}")
        print(f"Coverage: {result.get_coverage():.0%}")
        print(f"{'=' * 60}\n")

        print("Mapped Methods:")
        print("-" * 40)
        for method_name, mapping in sorted(result.mappings.items()):
            confidence_bar = "=" * int(mapping.confidence * 10)
            print(f"  {method_name}")
            print(f"    -> {mapping.endpoint.method.value} {mapping.endpoint.path}")
            print(f"    Confidence: [{confidence_bar:<10}] {mapping.confidence:.0%}")
            if mapping.notes:
                print(f"    Notes: {mapping.notes}")
            print()

        if result.missing_protocol_methods:
            print("\nMissing Protocol Methods:")
            print("-" * 40)
            for method in result.missing_protocol_methods:
                print(f"  - {method}")

        if result.unmapped_endpoints:
            print(f"\nUnmapped Endpoints ({len(result.unmapped_endpoints)}):")
            print("-" * 40)
            for ep in result.unmapped_endpoints[:10]:  # Show first 10
                print(f"  - {ep.method.value} {ep.path}")
            if len(result.unmapped_endpoints) > 10:
                print(f"  ... and {len(result.unmapped_endpoints) - 10} more")
