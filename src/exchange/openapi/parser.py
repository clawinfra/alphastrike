"""
OpenAPI Specification Parser

Parses OpenAPI 3.x specifications to extract exchange API information.
Used to help build adapters and understand exchange capabilities.

Usage:
    parser = OpenAPIParser()
    spec = parser.parse_file("docs/weex_api.json")
    print(spec.base_url)
    print(spec.endpoints)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class AuthenticationType(str, Enum):
    """Authentication types detected from OpenAPI spec."""

    NONE = "none"
    API_KEY = "api_key"
    HMAC = "hmac"
    OAUTH2 = "oauth2"
    BEARER = "bearer"


class EndpointCategory(str, Enum):
    """Categories of exchange API endpoints."""

    MARKET_DATA = "market_data"
    ACCOUNT = "account"
    ORDER = "order"
    POSITION = "position"
    LEVERAGE = "leverage"
    FUNDING = "funding"
    WEBSOCKET = "websocket"
    UNKNOWN = "unknown"


@dataclass
class APIParameter:
    """API endpoint parameter."""

    name: str
    location: str  # query, path, header, body
    required: bool
    param_type: str
    description: str = ""
    default: Any = None
    example: Any = None


@dataclass
class APIResponse:
    """API endpoint response schema."""

    status_code: int
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    example: dict[str, Any] | None = None


@dataclass
class APIEndpoint:
    """Parsed API endpoint."""

    path: str
    method: HTTPMethod
    operation_id: str
    summary: str
    description: str
    tags: list[str]
    parameters: list[APIParameter]
    request_body: dict[str, Any] | None
    responses: list[APIResponse]
    requires_auth: bool
    category: EndpointCategory = EndpointCategory.UNKNOWN

    @property
    def full_path(self) -> str:
        """Get the full path for this endpoint."""
        return self.path

    def get_protocol_method_hint(self) -> str | None:
        """
        Suggest which ExchangeRESTProtocol method this might map to.

        Based on operation ID and path patterns.
        """
        op_id = self.operation_id.lower()
        path = self.path.lower()

        # Market data
        if "ticker" in op_id or "ticker" in path:
            return "get_ticker"
        if "orderbook" in op_id or "depth" in path:
            return "get_orderbook"
        if "candle" in op_id or "kline" in path:
            return "get_candles"
        if "fundingrate" in op_id or "fundrate" in path or "funding" in path:
            return "get_funding_rate"
        if "contract" in op_id or "symbol" in op_id or "instrument" in op_id:
            return "get_all_symbols"
        if "trade" in op_id and ("recent" in op_id or "fills" in path):
            return "get_recent_trades"

        # Account
        if "balance" in op_id or "account" in path or "asset" in path:
            return "get_account_balance"
        if "position" in op_id or "position" in path:
            if "all" in op_id:
                return "get_positions"
            return "get_position"

        # Orders
        if "placeorder" in op_id or ("order" in path and self.method == HTTPMethod.POST):
            return "place_order"
        if "cancelorder" in op_id or "cancel" in op_id:
            return "cancel_order"
        if "orderdetail" in op_id or ("order" in path and "detail" in path):
            return "get_order"
        if "currentorder" in op_id or "openorder" in op_id:
            return "get_open_orders"

        # Leverage
        if "setleverage" in op_id:
            return "set_leverage"
        if "getleverage" in op_id or ("leverage" in path and self.method == HTTPMethod.GET):
            return "get_leverage"

        # Stop orders
        if "stoploss" in op_id or "lossplan" in op_id:
            return "place_stop_loss"
        if "takeprofit" in op_id or "profitplan" in op_id:
            return "place_take_profit"
        if "cancelplan" in op_id:
            return "cancel_conditional_order"

        return None


@dataclass
class AuthenticationInfo:
    """Authentication configuration from OpenAPI spec."""

    auth_type: AuthenticationType
    header_name: str | None = None
    description: str = ""


@dataclass
class ParsedOpenAPISpec:
    """Parsed OpenAPI specification."""

    title: str
    description: str
    version: str
    base_url: str
    endpoints: list[APIEndpoint]
    schemas: dict[str, dict[str, Any]]
    authentication: list[AuthenticationInfo]

    def get_endpoints_by_category(
        self,
        category: EndpointCategory,
    ) -> list[APIEndpoint]:
        """Get endpoints filtered by category."""
        return [ep for ep in self.endpoints if ep.category == category]

    def get_endpoint_by_operation_id(
        self,
        operation_id: str,
    ) -> APIEndpoint | None:
        """Find endpoint by operation ID."""
        for ep in self.endpoints:
            if ep.operation_id == operation_id:
                return ep
        return None

    def suggest_protocol_mapping(self) -> dict[str, APIEndpoint]:
        """
        Suggest mappings from ExchangeRESTProtocol methods to endpoints.

        Returns:
            Dict mapping protocol method names to suggested endpoints
        """
        mapping: dict[str, APIEndpoint] = {}

        for endpoint in self.endpoints:
            hint = endpoint.get_protocol_method_hint()
            if hint and hint not in mapping:
                mapping[hint] = endpoint

        return mapping


class OpenAPIParser:
    """
    Parser for OpenAPI 3.x specifications.

    Extracts exchange API information for adapter building.

    Example:
        parser = OpenAPIParser()
        spec = parser.parse_file("docs/weex_api.json")

        # Get suggested protocol mappings
        mappings = spec.suggest_protocol_mapping()
        for method, endpoint in mappings.items():
            print(f"{method} -> {endpoint.path}")
    """

    def parse_file(self, path: str | Path) -> ParsedOpenAPISpec:
        """
        Parse an OpenAPI specification from a file.

        Args:
            path: Path to OpenAPI JSON or YAML file

        Returns:
            ParsedOpenAPISpec with extracted information

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid OpenAPI
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"OpenAPI spec not found: {path}")

        content = path.read_text()

        if path.suffix in (".json", ".JSON"):
            data = json.loads(content)
        else:
            # Try YAML
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError:
                # Fallback to JSON parsing
                data = json.loads(content)

        return self.parse_dict(data)

    def parse_dict(self, data: dict[str, Any]) -> ParsedOpenAPISpec:
        """
        Parse an OpenAPI specification from a dictionary.

        Args:
            data: OpenAPI spec as a dictionary

        Returns:
            ParsedOpenAPISpec with extracted information
        """
        # Validate OpenAPI version
        openapi_version = data.get("openapi", "")
        if not openapi_version.startswith("3."):
            logger.warning(f"Untested OpenAPI version: {openapi_version}")

        # Extract basic info
        info = data.get("info", {})
        title = info.get("title", "Unknown")
        description = info.get("description", "")
        version = info.get("version", "1.0")

        # Extract base URL
        servers = data.get("servers", [])
        base_url = servers[0].get("url", "") if servers else ""

        # Extract schemas
        components = data.get("components", {})
        schemas = components.get("schemas", {})

        # Extract authentication
        security_schemes = components.get("securitySchemes", {})
        authentication = self._parse_authentication(security_schemes)

        # Extract endpoints
        paths = data.get("paths", {})
        endpoints = self._parse_paths(paths, data.get("security", []))

        return ParsedOpenAPISpec(
            title=title,
            description=description,
            version=version,
            base_url=base_url,
            endpoints=endpoints,
            schemas=schemas,
            authentication=authentication,
        )

    def _parse_authentication(
        self,
        security_schemes: dict[str, Any],
    ) -> list[AuthenticationInfo]:
        """Parse authentication schemes."""
        auth_list = []

        for name, scheme in security_schemes.items():
            scheme_type = scheme.get("type", "")
            header_name = scheme.get("name")
            description = scheme.get("description", "")

            if scheme_type == "apiKey":
                # Check if this might be HMAC based on name/description
                if "sign" in name.lower() or "sign" in description.lower():
                    auth_type = AuthenticationType.HMAC
                else:
                    auth_type = AuthenticationType.API_KEY
            elif scheme_type == "oauth2":
                auth_type = AuthenticationType.OAUTH2
            elif scheme_type == "http" and scheme.get("scheme") == "bearer":
                auth_type = AuthenticationType.BEARER
            else:
                auth_type = AuthenticationType.API_KEY

            auth_list.append(
                AuthenticationInfo(
                    auth_type=auth_type,
                    header_name=header_name,
                    description=description,
                )
            )

        return auth_list

    def _parse_paths(
        self,
        paths: dict[str, Any],
        global_security: list[dict],
    ) -> list[APIEndpoint]:
        """Parse API paths into endpoints."""
        endpoints = []

        for path, path_item in paths.items():
            for method_str, operation in path_item.items():
                if method_str.upper() not in [m.value for m in HTTPMethod]:
                    continue

                method = HTTPMethod(method_str.upper())

                # Parse parameters
                parameters = self._parse_parameters(operation.get("parameters", []))

                # Parse request body
                request_body = operation.get("requestBody")

                # Parse responses
                responses = self._parse_responses(operation.get("responses", {}))

                # Determine if auth required
                security = operation.get("security", global_security)
                requires_auth = bool(security)

                # Get tags and categorize
                tags = operation.get("tags", [])
                category = self._categorize_endpoint(path, tags, method)

                endpoint = APIEndpoint(
                    path=path,
                    method=method,
                    operation_id=operation.get("operationId", f"{method}_{path}"),
                    summary=operation.get("summary", ""),
                    description=operation.get("description", ""),
                    tags=tags,
                    parameters=parameters,
                    request_body=request_body,
                    responses=responses,
                    requires_auth=requires_auth,
                    category=category,
                )

                endpoints.append(endpoint)

        return endpoints

    def _parse_parameters(
        self,
        params: list[dict[str, Any]],
    ) -> list[APIParameter]:
        """Parse endpoint parameters."""
        parsed = []

        for param in params:
            schema = param.get("schema", {})
            parsed.append(
                APIParameter(
                    name=param.get("name", ""),
                    location=param.get("in", "query"),
                    required=param.get("required", False),
                    param_type=schema.get("type", "string"),
                    description=param.get("description", ""),
                    default=schema.get("default"),
                    example=schema.get("example"),
                )
            )

        return parsed

    def _parse_responses(
        self,
        responses: dict[str, Any],
    ) -> list[APIResponse]:
        """Parse endpoint responses."""
        parsed = []

        for status_code, response in responses.items():
            try:
                code = int(status_code)
            except ValueError:
                code = 0

            content = response.get("content", {})
            json_content = content.get("application/json", {})

            parsed.append(
                APIResponse(
                    status_code=code,
                    description=response.get("description", ""),
                    schema=json_content.get("schema", {}),
                    example=json_content.get("example"),
                )
            )

        return parsed

    def _categorize_endpoint(
        self,
        path: str,
        tags: list[str],
        method: HTTPMethod,
    ) -> EndpointCategory:
        """Categorize endpoint based on path and tags."""
        path_lower = path.lower()
        tags_lower = [t.lower() for t in tags]

        # Check tags first
        for tag in tags_lower:
            if "market" in tag:
                return EndpointCategory.MARKET_DATA
            if "account" in tag:
                return EndpointCategory.ACCOUNT
            if "order" in tag:
                return EndpointCategory.ORDER
            if "position" in tag:
                return EndpointCategory.POSITION

        # Check path patterns
        if "/market/" in path_lower:
            return EndpointCategory.MARKET_DATA
        if "/account/" in path_lower:
            return EndpointCategory.ACCOUNT
        if "/order/" in path_lower:
            return EndpointCategory.ORDER
        if "/position/" in path_lower:
            return EndpointCategory.POSITION
        if "/leverage" in path_lower:
            return EndpointCategory.LEVERAGE
        if "/funding" in path_lower or "/fundrate" in path_lower:
            return EndpointCategory.FUNDING
        if "/ws" in path_lower or "websocket" in path_lower:
            return EndpointCategory.WEBSOCKET

        return EndpointCategory.UNKNOWN
