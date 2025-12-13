"""Data models for 12-Layer Guardrails.

This module defines Pydantic v2 models for guardrail configuration,
results, and compliance tracking.

Models:
- GuardrailLayer: Enum of all 12 layers
- GuardrailViolation: Single violation record
- GuardrailResult: Result from guardrail check
- GuardrailConfig: Configuration for guardrail behavior
- AuditLogEntry: Audit trail entry
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailLayer(str, Enum):
    """Enum of all 12 guardrail layers."""

    L1_INPUT_VALIDATION = "L1_INPUT_VALIDATION"
    L2_PII_DETECTION = "L2_PII_DETECTION"
    L3_AUTH_AUTHZ = "L3_AUTH_AUTHZ"
    L4_PROMPT_INJECTION = "L4_PROMPT_INJECTION"
    L5_CONTENT_FILTERING = "L5_CONTENT_FILTERING"
    L6_HALLUCINATION_DETECTION = "L6_HALLUCINATION_DETECTION"
    L7_BIAS_MITIGATION = "L7_BIAS_MITIGATION"
    L8_OUTPUT_VALIDATION = "L8_OUTPUT_VALIDATION"
    L9_AUDIT_LOGGING = "L9_AUDIT_LOGGING"
    L10_MONITORING_ALERTING = "L10_MONITORING_ALERTING"
    L11_ENCRYPTION = "L11_ENCRYPTION"
    L12_COMPLIANCE_REPORTING = "L12_COMPLIANCE_REPORTING"


class ViolationSeverity(str, Enum):
    """Severity levels for guardrail violations."""

    LOW = "low"  # Warning, allow with logging
    MEDIUM = "medium"  # Allow with elevated monitoring
    HIGH = "high"  # Block and alert
    CRITICAL = "critical"  # Block, alert, and escalate


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVERS_LICENSE = "drivers_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"


class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""

    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    GDPR = "gdpr"
    CCPA = "ccpa"
    FERPA = "ferpa"


class GuardrailViolation(BaseModel):
    """A single guardrail violation.

    Records details of a specific policy violation for
    audit trails and compliance reporting.
    """

    layer: GuardrailLayer = Field(description="Which layer detected violation")
    severity: ViolationSeverity = Field(description="Severity level")
    message: str = Field(description="Human-readable violation description")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    remediation: str = Field(default="", description="Suggested fix")


class GuardrailResult(BaseModel):
    """Result from running guardrail checks.

    Aggregates results from all applicable guardrail layers
    and provides pass/fail determination.
    """

    passed: bool = Field(description="Overall pass/fail")
    violations: list[GuardrailViolation] = Field(
        default_factory=list, description="All violations found"
    )
    layers_checked: list[GuardrailLayer] = Field(
        default_factory=list, description="Layers that were checked"
    )
    execution_time_ms: float = Field(default=0.0, description="Total check time")
    blocked: bool = Field(default=False, description="Request should be blocked")
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Aggregate risk score"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def critical_violations(self) -> list[GuardrailViolation]:
        """Get critical severity violations only."""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]

    @property
    def high_violations(self) -> list[GuardrailViolation]:
        """Get high+ severity violations."""
        return [
            v
            for v in self.violations
            if v.severity in (ViolationSeverity.HIGH, ViolationSeverity.CRITICAL)
        ]


class GuardrailConfig(BaseModel):
    """Configuration for guardrail behavior.

    Allows customization of which layers are enabled,
    thresholds, and blocking behavior.
    """

    # Layer enablement
    enabled_layers: list[GuardrailLayer] = Field(
        default_factory=lambda: list(GuardrailLayer),
        description="Which layers to run",
    )

    # Blocking behavior
    block_on_high: bool = Field(default=True, description="Block on HIGH severity")
    block_on_critical: bool = Field(default=True, description="Block on CRITICAL severity")

    # PII configuration
    pii_types_to_detect: list[PIIType] = Field(
        default_factory=lambda: list(PIIType),
        description="PII types to scan for",
    )
    pii_redact: bool = Field(default=True, description="Redact detected PII")

    # Input validation
    max_input_length: int = Field(default=10000, description="Max query length")
    max_output_length: int = Field(default=50000, description="Max response length")

    # Content filtering
    prohibited_topics: list[str] = Field(
        default_factory=list, description="Topics to block"
    )

    # Compliance
    compliance_frameworks: list[ComplianceFramework] = Field(
        default_factory=list, description="Frameworks to enforce"
    )

    # Performance
    timeout_ms: float = Field(default=1000.0, description="Max guardrail check time")
    parallel_execution: bool = Field(default=True, description="Run layers in parallel")


class AuditLogEntry(BaseModel):
    """Audit trail entry for compliance.

    Records all guardrail activity for compliance reporting
    and forensic analysis.
    """

    id: str = Field(description="Unique entry ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str = Field(default="", description="User who triggered check")
    session_id: str = Field(default="", description="Session identifier")
    request_id: str = Field(default="", description="Request identifier")

    # What was checked
    input_hash: str = Field(default="", description="Hash of input (not raw)")
    output_hash: str = Field(default="", description="Hash of output (not raw)")
    check_type: str = Field(default="input", description="input or output")

    # Results
    result: GuardrailResult = Field(description="Guardrail check result")

    # Metadata
    ip_address: str = Field(default="", description="Client IP (if allowed)")
    user_agent: str = Field(default="", description="Client user agent")
    environment: str = Field(default="production", description="Environment name")

    # Compliance tracking
    compliance_frameworks: list[ComplianceFramework] = Field(
        default_factory=list, description="Applicable frameworks"
    )
    retention_days: int = Field(default=2555, description="7 years default retention")


class MonitoringAlert(BaseModel):
    """Alert generated by monitoring layer."""

    id: str = Field(description="Alert ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    alert_type: str = Field(description="Type of alert")
    severity: ViolationSeverity = Field(description="Alert severity")
    message: str = Field(description="Alert message")
    source_layer: GuardrailLayer = Field(description="Layer that triggered alert")
    metrics: dict[str, float] = Field(default_factory=dict, description="Related metrics")
    acknowledged: bool = Field(default=False, description="Has been acknowledged")
    resolved: bool = Field(default=False, description="Has been resolved")
