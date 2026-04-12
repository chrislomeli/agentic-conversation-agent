"""
infrastructure — Backwards-compatibility shim.

All reusable infrastructure has moved to the ``commons`` package.
This module re-exports everything so existing imports keep working.

Domain-specific tools (conversation_tools, project_graph_tools,
architectural_quiz) remain here because they depend on domain models.
"""

# ── Re-exports from commons ─────────────────────────────────────────
from commons import (
    CallHuman,
    CallLLM,
    CircuitBreakerMiddleware,
    ConfigMiddleware,
    ConsoleHuman,
    ErrorHandlingMiddleware,
    HumanRequest,
    HumanResponse,
    InstrumentedGraph,
    LLMRequest,
    LLMResponse,
    LocalToolClient,
    LoggingMiddleware,
    MetricsMiddleware,
    MockHuman,
    NodeError,
    NodeMetrics,
    NodeMiddleware,
    NodeResult,
    RetryMiddleware,
    ToolCallError,
    ToolClient,
    ToolContentBlock,
    ToolRegistry,
    ToolResultEnvelope,
    ToolResultMeta,
    ToolSpec,
    ValidationMiddleware,
    call_llm_stub,
    handle_error,
    validated_node,
)

# ── Legacy (deprecated) — kept for transition ───────────────────────
from conversation_engine.infrastructure.instrumented_graph import (
    Interceptor,
    Middleware,
)
from conversation_engine.infrastructure.interceptors import (
    LoggingInterceptor,
    MetricsInterceptor,
)
from conversation_engine.infrastructure.interceptors import (
    NodeMetrics as _LegacyNodeMetrics,
)

__all__ = [
    # Instrumented graph
    "InstrumentedGraph",
    # Middleware
    "NodeMiddleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "NodeMetrics",
    "ValidationMiddleware",
    "ErrorHandlingMiddleware",
    "RetryMiddleware",
    "CircuitBreakerMiddleware",
    "ConfigMiddleware",
    # Legacy (deprecated)
    "Interceptor",
    "Middleware",
    "LoggingInterceptor",
    "MetricsInterceptor",
    # Node validation
    "NodeError",
    "NodeResult",
    "validated_node",
    "handle_error",
    # Tool client
    "ToolSpec",
    "ToolRegistry",
    "ToolContentBlock",
    "ToolResultEnvelope",
    "ToolResultMeta",
    "ToolClient",
    "ToolCallError",
    "LocalToolClient",
    # LLM
    "CallLLM",
    "LLMRequest",
    "LLMResponse",
    "call_llm_stub",
    # Human
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
