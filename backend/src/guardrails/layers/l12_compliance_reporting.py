"""Layer 12: Compliance Reporting.

Generates compliance reports for regulatory frameworks.
Tracks violations, remediations, and compliance posture.

Supported Frameworks:
- HIPAA: Healthcare data protection
- PCI-DSS: Payment card security
- SOC2: Security, availability, processing integrity
- GDPR: EU data protection
- CCPA: California privacy rights

Reports Generated:
- Violation summaries
- Remediation tracking
- Compliance score cards
- Audit-ready exports
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    ComplianceFramework,
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class ComplianceReportingLayer(BaseGuardrailLayer):
    """Layer 12: Compliance Reporting.

    Generates compliance reports and tracks remediation.
    """

    layer = GuardrailLayer.L12_COMPLIANCE_REPORTING

    # Framework requirements mapping
    FRAMEWORK_REQUIREMENTS: dict[ComplianceFramework, dict[str, Any]] = {
        ComplianceFramework.HIPAA: {
            "name": "Health Insurance Portability and Accountability Act",
            "required_layers": [
                GuardrailLayer.L2_PII_DETECTION,
                GuardrailLayer.L9_AUDIT_LOGGING,
                GuardrailLayer.L11_ENCRYPTION,
            ],
            "retention_years": 6,
            "breach_notification_hours": 72,
            "critical_violations": ["pii_leak", "unauthorized_access", "encryption_failure"],
        },
        ComplianceFramework.PCI_DSS: {
            "name": "Payment Card Industry Data Security Standard",
            "required_layers": [
                GuardrailLayer.L2_PII_DETECTION,
                GuardrailLayer.L3_AUTH_AUTHZ,
                GuardrailLayer.L9_AUDIT_LOGGING,
                GuardrailLayer.L11_ENCRYPTION,
            ],
            "retention_years": 1,
            "critical_violations": ["credit_card_exposure", "unauthorized_access"],
        },
        ComplianceFramework.SOC2: {
            "name": "Service Organization Control 2",
            "required_layers": [
                GuardrailLayer.L1_INPUT_VALIDATION,
                GuardrailLayer.L3_AUTH_AUTHZ,
                GuardrailLayer.L9_AUDIT_LOGGING,
                GuardrailLayer.L10_MONITORING_ALERTING,
            ],
            "retention_years": 1,
            "critical_violations": ["security_control_failure", "availability_breach"],
        },
        ComplianceFramework.GDPR: {
            "name": "General Data Protection Regulation",
            "required_layers": [
                GuardrailLayer.L2_PII_DETECTION,
                GuardrailLayer.L9_AUDIT_LOGGING,
                GuardrailLayer.L11_ENCRYPTION,
            ],
            "retention_years": 3,
            "breach_notification_hours": 72,
            "critical_violations": ["pii_leak", "consent_violation", "data_retention_violation"],
        },
        ComplianceFramework.CCPA: {
            "name": "California Consumer Privacy Act",
            "required_layers": [
                GuardrailLayer.L2_PII_DETECTION,
                GuardrailLayer.L9_AUDIT_LOGGING,
            ],
            "retention_years": 2,
            "critical_violations": ["pii_leak", "opt_out_violation"],
        },
    }

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)

        # Violation tracking
        self._violations_by_framework: dict[ComplianceFramework, list[dict]] = defaultdict(list)
        self._remediations: list[dict] = []

        # Compliance scores (0-100)
        self._compliance_scores: dict[ComplianceFramework, float] = {}

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Track compliance-relevant violations.

        This layer doesn't block - it tracks for reporting.

        Args:
            content: Content checked
            context: Should contain 'guardrail_result' and violations

        Returns:
            Empty list (reporting layer doesn't block)
        """
        context = context or {}
        guardrail_result = context.get("guardrail_result")

        if guardrail_result and guardrail_result.violations:
            self._track_violations(guardrail_result.violations, context)

        return []  # Never blocks

    def _track_violations(
        self,
        violations: list[GuardrailViolation],
        context: dict[str, Any],
    ) -> None:
        """Track violations for compliance reporting."""
        timestamp = datetime.utcnow()

        for violation in violations:
            # Determine affected frameworks
            affected_frameworks = self._get_affected_frameworks(violation)

            for framework in affected_frameworks:
                self._violations_by_framework[framework].append({
                    "timestamp": timestamp.isoformat(),
                    "layer": violation.layer.value,
                    "severity": violation.severity.value,
                    "message": violation.message,
                    "remediation": violation.remediation,
                    "user_id": context.get("user_id", "unknown"),
                    "request_id": context.get("request_id", ""),
                })

    def _get_affected_frameworks(
        self,
        violation: GuardrailViolation,
    ) -> list[ComplianceFramework]:
        """Determine which frameworks are affected by a violation."""
        affected = []

        for framework in self.config.compliance_frameworks:
            requirements = self.FRAMEWORK_REQUIREMENTS.get(framework, {})
            required_layers = requirements.get("required_layers", [])

            # If violation is in a layer required by framework, it's affected
            if violation.layer in required_layers:
                affected.append(framework)

            # Check for critical violation types
            critical = requirements.get("critical_violations", [])
            for crit_type in critical:
                if crit_type in violation.message.lower():
                    if framework not in affected:
                        affected.append(framework)

        return affected

    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate compliance report for a specific framework.

        Args:
            framework: Compliance framework
            start_date: Report start (default: 30 days ago)
            end_date: Report end (default: now)

        Returns:
            Compliance report dictionary
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        requirements = self.FRAMEWORK_REQUIREMENTS.get(framework, {})

        # Filter violations by date
        violations = [
            v for v in self._violations_by_framework[framework]
            if start_date <= datetime.fromisoformat(v["timestamp"]) <= end_date
        ]

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(framework, violations)

        # Group violations by severity
        violations_by_severity = defaultdict(list)
        for v in violations:
            violations_by_severity[v["severity"]].append(v)

        # Get required vs enabled layers
        required_layers = requirements.get("required_layers", [])
        enabled_layers = self.config.enabled_layers

        return {
            "framework": framework.value,
            "framework_name": requirements.get("name", framework.value),
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "compliance_score": compliance_score,
            "status": self._get_compliance_status(compliance_score),
            "summary": {
                "total_violations": len(violations),
                "critical_violations": len(violations_by_severity.get("critical", [])),
                "high_violations": len(violations_by_severity.get("high", [])),
                "medium_violations": len(violations_by_severity.get("medium", [])),
                "low_violations": len(violations_by_severity.get("low", [])),
            },
            "layer_coverage": {
                "required": [l.value for l in required_layers],
                "enabled": [l.value for l in enabled_layers if l in required_layers],
                "missing": [l.value for l in required_layers if l not in enabled_layers],
                "coverage_percent": (
                    len([l for l in required_layers if l in enabled_layers])
                    / max(len(required_layers), 1)
                    * 100
                ),
            },
            "violations_by_severity": dict(violations_by_severity),
            "recommendations": self._get_recommendations(framework, violations),
            "generated_at": datetime.utcnow().isoformat(),
            "retention_requirements": {
                "years": requirements.get("retention_years", 1),
                "breach_notification_hours": requirements.get("breach_notification_hours"),
            },
        }

    def _calculate_compliance_score(
        self,
        framework: ComplianceFramework,
        violations: list[dict],
    ) -> float:
        """Calculate compliance score (0-100)."""
        # Base score starts at 100
        score = 100.0

        # Deduct points for violations
        severity_deductions = {
            "critical": 20,
            "high": 10,
            "medium": 5,
            "low": 2,
        }

        for v in violations:
            severity = v.get("severity", "medium")
            deduction = severity_deductions.get(severity, 5)
            score -= deduction

        # Check layer coverage
        requirements = self.FRAMEWORK_REQUIREMENTS.get(framework, {})
        required_layers = requirements.get("required_layers", [])
        enabled_layers = self.config.enabled_layers

        missing_layers = [l for l in required_layers if l not in enabled_layers]
        score -= len(missing_layers) * 10  # 10 points per missing layer

        # Ensure score is in valid range
        return max(0.0, min(100.0, score))

    def _get_compliance_status(self, score: float) -> str:
        """Get compliance status from score."""
        if score >= 90:
            return "COMPLIANT"
        elif score >= 70:
            return "NEEDS_IMPROVEMENT"
        elif score >= 50:
            return "AT_RISK"
        else:
            return "NON_COMPLIANT"

    def _get_recommendations(
        self,
        framework: ComplianceFramework,
        violations: list[dict],
    ) -> list[str]:
        """Generate recommendations for improving compliance."""
        recommendations = []

        requirements = self.FRAMEWORK_REQUIREMENTS.get(framework, {})
        required_layers = requirements.get("required_layers", [])
        enabled_layers = self.config.enabled_layers

        # Recommend enabling missing layers
        for layer in required_layers:
            if layer not in enabled_layers:
                recommendations.append(
                    f"Enable {layer.value} guardrail layer (required for {framework.value})"
                )

        # Recommend based on violation patterns
        violation_layers = [v["layer"] for v in violations]
        layer_counts = defaultdict(int)
        for layer in violation_layers:
            layer_counts[layer] += 1

        # Layers with most violations need attention
        for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:3]:
            if count > 5:
                recommendations.append(
                    f"Review {layer} configuration - {count} violations in report period"
                )

        # Critical violations need immediate attention
        critical_count = sum(1 for v in violations if v["severity"] == "critical")
        if critical_count > 0:
            recommendations.insert(
                0,
                f"URGENT: Address {critical_count} critical violation(s) immediately"
            )

        return recommendations or ["No immediate recommendations - maintain current posture"]

    def generate_all_reports(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate reports for all configured frameworks."""
        reports = {}

        for framework in self.config.compliance_frameworks:
            reports[framework.value] = self.generate_compliance_report(
                framework, start_date, end_date
            )

        # Calculate overall compliance
        scores = [r["compliance_score"] for r in reports.values()]
        overall_score = sum(scores) / max(len(scores), 1)

        return {
            "overall_compliance_score": overall_score,
            "overall_status": self._get_compliance_status(overall_score),
            "frameworks": reports,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def add_remediation(
        self,
        violation_id: str,
        remediation_action: str,
        remediated_by: str,
    ) -> None:
        """Record remediation action for a violation."""
        self._remediations.append({
            "violation_id": violation_id,
            "action": remediation_action,
            "remediated_by": remediated_by,
            "timestamp": datetime.utcnow().isoformat(),
        })
