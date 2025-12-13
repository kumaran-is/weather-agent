"""12-Layer Enterprise Guardrails for Weather AI Agent.

This module implements a comprehensive 12-layer guardrail system for
production-grade safety and compliance.

Layers:
    L1: Input Validation - Schema, length, format checks
    L2: PII Detection - SSN, credit cards, phone numbers, emails
    L3: Auth/AuthZ - Role-based access control
    L4: Prompt Injection - Attack detection and blocking
    L5: Content Filtering - Prohibited content detection
    L6: Hallucination Detection - Fact verification
    L7: Bias Mitigation - Demographic and geographic bias
    L8: Output Validation - Response quality checks
    L9: Audit Logging - Comprehensive activity logging
    L10: Monitoring/Alerting - Real-time anomaly detection
    L11: Encryption - Data protection at rest and in transit
    L12: Compliance Reporting - Regulatory compliance tracking

Usage:
    >>> from backend.src.guardrails import GuardrailManager
    >>> manager = GuardrailManager()
    >>> result = await manager.check_input("What's the weather?", user_id="user123")
    >>> if result.passed:
    ...     # Process query
    ...     response = await agent.run(query)
    ...     output_result = await manager.check_output(response, query)
"""

from backend.src.guardrails.models import (
    GuardrailResult,
    GuardrailViolation,
    GuardrailLayer,
    GuardrailConfig,
    PIIType,
    ViolationSeverity,
    ComplianceFramework,
    AuditLogEntry,
)
from backend.src.guardrails.guardrail_manager import GuardrailManager

__all__ = [
    # Main manager
    "GuardrailManager",
    # Models
    "GuardrailResult",
    "GuardrailViolation",
    "GuardrailLayer",
    "GuardrailConfig",
    "PIIType",
    "ViolationSeverity",
    "ComplianceFramework",
    "AuditLogEntry",
]
