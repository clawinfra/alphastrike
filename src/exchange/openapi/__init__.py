"""
OpenAPI Exchange Configuration Package

Provides tools for parsing OpenAPI specifications and mapping
endpoints to the unified exchange protocol methods.

Usage:
    from src.exchange.openapi import OpenAPIParser, ProtocolMapper

    # Parse an OpenAPI spec
    parser = OpenAPIParser()
    spec = parser.parse_file("docs/weex_api.json")

    # Generate protocol mapping suggestions
    mapper = ProtocolMapper()
    result = mapper.generate_mapping(spec)
    mapper.print_mapping_report(result)
"""

from __future__ import annotations

from src.exchange.openapi.mapper import (
    EndpointMapping,
    ProtocolMapper,
    ProtocolMappingResult,
)
from src.exchange.openapi.parser import (
    APIEndpoint,
    APIParameter,
    APIResponse,
    AuthenticationInfo,
    AuthenticationType,
    EndpointCategory,
    HTTPMethod,
    OpenAPIParser,
    ParsedOpenAPISpec,
)

__all__ = [
    # Parser
    "OpenAPIParser",
    "ParsedOpenAPISpec",
    "APIEndpoint",
    "APIParameter",
    "APIResponse",
    "AuthenticationInfo",
    "AuthenticationType",
    "EndpointCategory",
    "HTTPMethod",
    # Mapper
    "ProtocolMapper",
    "ProtocolMappingResult",
    "EndpointMapping",
]
