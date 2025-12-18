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

Level 6c Additions:
    - Constitutional AI - Principle-based response validation
    - Content Filtering - Harmful content detection
    - Output Validator - Response quality assurance

Usage:
    >>> from backend.src.guardrails import GuardrailManager
    >>> manager = GuardrailManager()
    >>> result = await manager.check_input("What's the weather?", user_id="user123")
    >>> if result.passed:
    ...     # Process query
    ...     response = await agent.run(query)
    ...     output_result = await manager.check_output(response, query)

Constitutional AI Usage:
    >>> from backend.src.guardrails import ConstitutionalAI, Constitution
    >>> constitution = Constitution.weather_domain()
    >>> constitutional_ai = ConstitutionalAI(constitution=constitution)
    >>> result = await constitutional_ai.validate_response(query, response)
    >>> print(f"Passes: {result.passes}, Score: {result.score}")
"""

# Level 6c: Constitutional AI and Enhanced Guardrails
from backend.src.guardrails.constitutional_ai import (
    Constitution,
    ConstitutionalAI,
    ConstitutionalResult,
    CritiqueResult,
    Principle,
    PrincipleCategory,
    RevisionResult,
)
from backend.src.guardrails.content_filter import (
    ContentFilter,
    ContentFilterResult,
    FilterCategory,
)
from backend.src.guardrails.guardrail_manager import GuardrailManager
from backend.src.guardrails.models import (
    AuditLogEntry,
    ComplianceFramework,
    GuardrailConfig,
    GuardrailLayer,
    GuardrailResult,
    GuardrailViolation,
    PIIType,
    ViolationSeverity,
)
from backend.src.guardrails.output_validator import (
    OutputValidator,
    RuleType,
    ValidationResult,
    ValidationRule,
)

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
    # Constitutional AI (Level 6c)
    "ConstitutionalAI",
    "Constitution",
    "Principle",
    "PrincipleCategory",
    "ConstitutionalResult",
    "CritiqueResult",
    "RevisionResult",
    # Content Filter (Level 6c)
    "ContentFilter",
    "ContentFilterResult",
    "FilterCategory",
    # Output Validator (Level 6c)
    "OutputValidator",
    "ValidationResult",
    "ValidationRule",
    "RuleType",
]
