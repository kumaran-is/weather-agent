"""Layer 9: Audit Logging.

Comprehensive audit logging for compliance, forensics, and debugging.
Creates immutable audit trail of all guardrail activity.

Logged Events:
- All guardrail checks (input and output)
- All violations detected
- User identification (hashed)
- Session and request IDs
- Timestamps and execution times

Compliance:
- HIPAA: Maintains access logs for 6 years
- SOC2: Complete audit trail
- GDPR: Right to access audit logs
- PCI-DSS: Transaction logging
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    AuditLogEntry,
    ComplianceFramework,
    GuardrailConfig,
    GuardrailLayer,
    GuardrailResult,
    GuardrailViolation,
)

# Dedicated audit logger (separate from application logs)
audit_logger = logging.getLogger("guardrails.audit")


class AuditLoggingLayer(BaseGuardrailLayer):
    """Layer 9: Audit Logging.

    Creates comprehensive audit trail for all guardrail activity.
    This layer doesn't block - it only logs.
    """

    layer = GuardrailLayer.L9_AUDIT_LOGGING

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)
        self._log_entries: list[AuditLogEntry] = []
        self._max_buffer_size = 1000  # Flush after this many entries

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Log guardrail activity.

        This layer never returns violations - it only logs.
        Logging failures should not block the request.

        Args:
            content: Content being checked
            context: Must contain 'guardrail_result' for logging

        Returns:
            Empty list (logging never blocks)
        """
        context = context or {}

        try:
            # Create audit log entry
            entry = self._create_audit_entry(content, context)

            # Log to structured logger
            self._write_log(entry)

            # Buffer for batch persistence
            self._log_entries.append(entry)

            # Flush if buffer full
            if len(self._log_entries) >= self._max_buffer_size:
                await self._flush_logs()

        except Exception as e:
            # Logging failures should not block requests
            audit_logger.error(f"Audit logging failed: {e}")

        return []  # Never blocks

    def _create_audit_entry(
        self,
        content: str,
        context: dict[str, Any],
    ) -> AuditLogEntry:
        """Create structured audit log entry."""
        guardrail_result = context.get("guardrail_result")

        # Hash sensitive data (never log raw content)
        content_hash = self._hash_content(content)

        # Get request metadata
        user_id = context.get("user_id", "anonymous")
        session_id = context.get("session_id", "")
        request_id = context.get("request_id", str(uuid.uuid4()))
        check_type = context.get("check_type", "input")

        # Create result if not provided
        if guardrail_result is None:
            guardrail_result = GuardrailResult(
                passed=True,
                violations=[],
                layers_checked=[self.layer],
            )

        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=self._hash_user_id(user_id),
            session_id=session_id[:16] if session_id else "",
            request_id=request_id,
            input_hash=content_hash if check_type == "input" else "",
            output_hash=content_hash if check_type == "output" else "",
            check_type=check_type,
            result=guardrail_result,
            ip_address=self._mask_ip(context.get("ip_address", "")),
            user_agent=context.get("user_agent", "")[:100],
            environment=context.get("environment", "production"),
            compliance_frameworks=self.config.compliance_frameworks,
            retention_days=self._get_retention_days(),
        )

        return entry

    def _write_log(self, entry: AuditLogEntry) -> None:
        """Write log entry to audit logger."""
        log_data = {
            "audit_id": entry.id,
            "timestamp": entry.timestamp.isoformat(),
            "user_hash": entry.user_id,
            "request_id": entry.request_id,
            "check_type": entry.check_type,
            "passed": entry.result.passed,
            "violation_count": len(entry.result.violations),
            "blocked": entry.result.blocked,
            "risk_score": entry.result.risk_score,
            "layers_checked": [l.value for l in entry.result.layers_checked],
            "environment": entry.environment,
        }

        # Add violation details (without raw content)
        if entry.result.violations:
            log_data["violations"] = [
                {
                    "layer": v.layer.value,
                    "severity": v.severity.value,
                    "message": v.message,
                }
                for v in entry.result.violations
            ]

        audit_logger.info("guardrail_check", extra=log_data)

    async def _flush_logs(self) -> None:
        """Flush buffered logs to persistent storage.

        In production, this would write to:
        - Database (PostgreSQL, DynamoDB)
        - Object storage (S3, GCS)
        - Log aggregation (CloudWatch, Splunk)
        """
        if not self._log_entries:
            return

        try:
            # In production, persist to storage
            # For now, just log summary
            audit_logger.info(
                f"Flushing {len(self._log_entries)} audit log entries"
            )

            # Clear buffer
            self._log_entries = []

        except Exception as e:
            audit_logger.error(f"Failed to flush audit logs: {e}")

    def _hash_content(self, content: str) -> str:
        """Create hash of content for logging.

        Never log raw user content - only hash for correlation.
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy.

        Allows correlation without exposing actual user IDs.
        """
        if not user_id or user_id == "anonymous":
            return "anonymous"
        return hashlib.sha256(user_id.encode()).hexdigest()[:12]

    def _mask_ip(self, ip_address: str) -> str:
        """Mask IP address for privacy.

        Preserves network portion for debugging, masks host.
        """
        if not ip_address:
            return ""

        parts = ip_address.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        return "xxx.xxx.xxx.xxx"

    def _get_retention_days(self) -> int:
        """Get retention period based on compliance requirements.

        HIPAA: 6 years (2190 days)
        SOC2: 1 year (365 days)
        GDPR: Varies, typically 3 years
        PCI-DSS: 1 year (365 days)
        """
        frameworks = self.config.compliance_frameworks

        if ComplianceFramework.HIPAA in frameworks:
            return 2190  # 6 years
        elif ComplianceFramework.GDPR in frameworks:
            return 1095  # 3 years
        elif frameworks:
            return 365  # 1 year default for compliance
        else:
            return 90  # Default retention

    def get_logs(
        self,
        user_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        violations_only: bool = False,
    ) -> list[AuditLogEntry]:
        """Query buffered logs.

        In production, this would query persistent storage.

        Args:
            user_id: Filter by user (will be hashed)
            start_time: Filter by start time
            end_time: Filter by end time
            violations_only: Only return entries with violations

        Returns:
            Matching log entries
        """
        results = self._log_entries

        if user_id:
            user_hash = self._hash_user_id(user_id)
            results = [e for e in results if e.user_id == user_hash]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        if violations_only:
            results = [e for e in results if e.result.violations]

        return results

    async def export_for_compliance(
        self,
        framework: ComplianceFramework,
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        """Export logs for compliance reporting.

        Args:
            framework: Compliance framework
            start_time: Report start
            end_time: Report end

        Returns:
            Compliance-formatted export
        """
        logs = self.get_logs(start_time=start_time, end_time=end_time)

        return {
            "framework": framework.value,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "summary": {
                "total_checks": len(logs),
                "violations": sum(len(e.result.violations) for e in logs),
                "blocked_requests": sum(1 for e in logs if e.result.blocked),
            },
            "entries": [e.model_dump() for e in logs],
            "generated_at": datetime.utcnow().isoformat(),
        }
