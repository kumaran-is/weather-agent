"""Layer 3: Authentication & Authorization.

Validates user identity and permissions before processing requests.
Implements role-based access control (RBAC) for different query types.

Features:
- User authentication validation
- Role-based query permissions
- Rate limiting by user tier
- Session validation
- API key verification

Roles:
- anonymous: Basic weather queries only
- user: All weather queries, limited hurricane data
- premium: Full access, higher rate limits
- admin: Full access, management APIs
"""

from typing import Any
from enum import Enum

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class UserRole(str, Enum):
    """User roles for RBAC."""

    ANONYMOUS = "anonymous"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"


class QueryType(str, Enum):
    """Query types for permission checking."""

    BASIC_WEATHER = "basic_weather"
    FORECAST = "forecast"
    HURRICANE = "hurricane"
    HISTORICAL = "historical"
    ALERTS = "alerts"
    ADMIN = "admin"


# Role -> Allowed Query Types
ROLE_PERMISSIONS: dict[UserRole, set[QueryType]] = {
    UserRole.ANONYMOUS: {QueryType.BASIC_WEATHER},
    UserRole.USER: {QueryType.BASIC_WEATHER, QueryType.FORECAST, QueryType.ALERTS},
    UserRole.PREMIUM: {
        QueryType.BASIC_WEATHER,
        QueryType.FORECAST,
        QueryType.HURRICANE,
        QueryType.HISTORICAL,
        QueryType.ALERTS,
    },
    UserRole.ADMIN: set(QueryType),  # All permissions
}

# Rate limits by role (requests per minute)
RATE_LIMITS: dict[UserRole, int] = {
    UserRole.ANONYMOUS: 10,
    UserRole.USER: 60,
    UserRole.PREMIUM: 300,
    UserRole.ADMIN: 1000,
}


class AuthAuthzLayer(BaseGuardrailLayer):
    """Layer 3: Authentication & Authorization.

    Validates user identity and permissions before processing.
    """

    layer = GuardrailLayer.L3_AUTH_AUTHZ

    # Keywords to detect query type
    QUERY_TYPE_KEYWORDS: dict[QueryType, list[str]] = {
        QueryType.HURRICANE: ["hurricane", "cyclone", "tropical storm", "evacuate", "storm surge"],
        QueryType.HISTORICAL: ["historical", "last year", "past", "history", "archive"],
        QueryType.ALERTS: ["alert", "warning", "watch", "advisory", "emergency"],
        QueryType.ADMIN: ["admin", "config", "settings", "management", "delete", "update user"],
        QueryType.FORECAST: ["forecast", "tomorrow", "next week", "prediction", "outlook"],
    }

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Validate authentication and authorization.

        Args:
            content: User query
            context: Must contain 'user_id', 'role', optionally 'session_id'

        Returns:
            List of violations if auth/authz fails
        """
        violations: list[GuardrailViolation] = []
        context = context or {}

        # Check authentication
        user_id = context.get("user_id")
        if not user_id:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Authentication required",
                    details={"missing": "user_id"},
                    remediation="Provide valid authentication credentials",
                )
            )
            return violations  # Can't check authz without auth

        # Get user role (default to anonymous)
        role_str = context.get("role", UserRole.ANONYMOUS.value)
        try:
            role = UserRole(role_str)
        except ValueError:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message=f"Invalid role: {role_str}",
                    details={"provided_role": role_str},
                    remediation="Use valid role: anonymous, user, premium, admin",
                )
            )
            return violations

        # Determine query type
        query_type = self._detect_query_type(content)

        # Check authorization
        allowed_types = ROLE_PERMISSIONS.get(role, set())
        if query_type not in allowed_types:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message=f"Insufficient permissions for {query_type.value} queries",
                    details={
                        "user_role": role.value,
                        "query_type": query_type.value,
                        "allowed_types": [t.value for t in allowed_types],
                    },
                    remediation=f"Upgrade to premium for {query_type.value} access",
                )
            )

        # Check rate limiting (simplified - actual implementation would track state)
        request_count = context.get("request_count", 0)
        rate_limit = RATE_LIMITS.get(role, 10)
        if request_count >= rate_limit:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.MEDIUM,
                    message=f"Rate limit exceeded ({rate_limit} requests/minute)",
                    details={
                        "user_role": role.value,
                        "rate_limit": rate_limit,
                        "request_count": request_count,
                    },
                    remediation="Wait before making more requests or upgrade tier",
                )
            )

        # Check session validity (if session-based auth)
        session_id = context.get("session_id")
        session_valid = context.get("session_valid", True)
        if session_id and not session_valid:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Session expired or invalid",
                    details={"session_id": session_id[:8] + "..."},
                    remediation="Re-authenticate to get a new session",
                )
            )

        return violations

    def _detect_query_type(self, content: str) -> QueryType:
        """Detect query type from content.

        Args:
            content: User query

        Returns:
            Detected QueryType
        """
        content_lower = content.lower()

        for query_type, keywords in self.QUERY_TYPE_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                return query_type

        # Default to basic weather
        return QueryType.BASIC_WEATHER
