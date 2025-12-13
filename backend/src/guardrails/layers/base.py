"""Base class for all guardrail layers.

Provides common interface and utilities for implementing
guardrail checks.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


class BaseGuardrailLayer(ABC):
    """Abstract base class for guardrail layers.

    All 12 layers inherit from this class and implement
    the check() method for their specific validation.

    Attributes:
        layer: The GuardrailLayer enum value
        config: Guardrail configuration
        enabled: Whether this layer is active
    """

    layer: GuardrailLayer

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize layer with configuration.

        Args:
            config: Optional guardrail configuration
        """
        self.config = config or GuardrailConfig()
        self.enabled = self.layer in self.config.enabled_layers
        self._check_count = 0
        self._violation_count = 0

    @abstractmethod
    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Run guardrail check on content.

        Args:
            content: Text to check (input or output)
            context: Additional context (user_id, session, etc.)

        Returns:
            List of violations found (empty if none)
        """
        pass

    async def run(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[GuardrailViolation], float]:
        """Run check with timing and logging.

        Args:
            content: Text to check
            context: Additional context

        Returns:
            Tuple of (violations, execution_time_ms)
        """
        if not self.enabled:
            logger.debug(f"{self.layer.value} is disabled, skipping")
            return [], 0.0

        start_time = time.time()
        self._check_count += 1

        try:
            violations = await self.check(content, context)
            self._violation_count += len(violations)

            execution_time_ms = (time.time() - start_time) * 1000

            if violations:
                logger.warning(
                    f"{self.layer.value} found {len(violations)} violation(s) "
                    f"in {execution_time_ms:.1f}ms"
                )
            else:
                logger.debug(
                    f"{self.layer.value} passed in {execution_time_ms:.1f}ms"
                )

            return violations, execution_time_ms

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"{self.layer.value} error: {e}")

            # Return a violation for the error itself
            return [
                GuardrailViolation(
                    layer=self.layer,
                    severity=ViolationSeverity.HIGH,
                    message=f"Guardrail check failed: {str(e)}",
                    details={"error_type": type(e).__name__},
                    remediation="Review guardrail configuration and retry",
                )
            ], execution_time_ms

    def create_violation(
        self,
        severity: ViolationSeverity,
        message: str,
        details: dict[str, Any] | None = None,
        remediation: str = "",
    ) -> GuardrailViolation:
        """Helper to create a violation for this layer.

        Args:
            severity: Violation severity level
            message: Human-readable description
            details: Additional details
            remediation: Suggested fix

        Returns:
            GuardrailViolation instance
        """
        return GuardrailViolation(
            layer=self.layer,
            severity=severity,
            message=message,
            details=details or {},
            remediation=remediation,
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Get layer statistics."""
        return {
            "layer": self.layer.value,
            "enabled": self.enabled,
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "violation_rate": (
                self._violation_count / self._check_count
                if self._check_count > 0
                else 0.0
            ),
        }
