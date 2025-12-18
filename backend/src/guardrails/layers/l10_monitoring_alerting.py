"""Layer 10: Monitoring & Alerting.

Real-time monitoring of guardrail activity and anomaly detection.
Triggers alerts when thresholds are exceeded.

Metrics Tracked:
- Violation rate (per layer, per minute)
- Block rate (requests blocked)
- Latency (guardrail check time)
- Error rate (guardrail failures)
- PII detection rate
- Injection attempt rate

Alert Types:
- Rate anomaly (sudden spike in violations)
- Pattern anomaly (new attack patterns)
- Performance degradation
- Safety threshold breach
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    MonitoringAlert,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Time-windowed metric storage."""

    window_size_seconds: int = 60
    values: list[tuple[float, float]] = None  # (timestamp, value)

    def __post_init__(self):
        if self.values is None:
            self.values = []

    def add(self, value: float) -> None:
        """Add value with current timestamp."""
        now = time.time()
        self.values.append((now, value))
        self._cleanup(now)

    def _cleanup(self, now: float) -> None:
        """Remove values outside window."""
        cutoff = now - self.window_size_seconds
        self.values = [(t, v) for t, v in self.values if t >= cutoff]

    def sum(self) -> float:
        """Sum of values in window."""
        self._cleanup(time.time())
        return sum(v for _, v in self.values)

    def count(self) -> int:
        """Count of values in window."""
        self._cleanup(time.time())
        return len(self.values)

    def rate(self) -> float:
        """Rate per second."""
        self._cleanup(time.time())
        if not self.values:
            return 0.0
        duration = time.time() - self.values[0][0] if self.values else 1.0
        return self.sum() / max(duration, 1.0)


class MonitoringAlertingLayer(BaseGuardrailLayer):
    """Layer 10: Monitoring & Alerting.

    Tracks metrics and generates alerts for anomalies.
    """

    layer = GuardrailLayer.L10_MONITORING_ALERTING

    # Alert thresholds
    THRESHOLDS = {
        "violation_rate_per_min": 100,  # Max violations per minute
        "block_rate_per_min": 50,  # Max blocks per minute
        "pii_detection_rate": 10,  # Max PII detections per minute
        "injection_rate": 5,  # Max injection attempts per minute
        "latency_p95_ms": 100,  # P95 latency threshold
        "error_rate": 0.01,  # Max error rate (1%)
    }

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize with configuration."""
        super().__init__(config)

        # Metric storage (per layer, windowed)
        self._violations: dict[str, MetricWindow] = defaultdict(MetricWindow)
        self._blocks: MetricWindow = MetricWindow()
        self._latencies: MetricWindow = MetricWindow()
        self._errors: MetricWindow = MetricWindow()
        self._requests: MetricWindow = MetricWindow()

        # Alert storage
        self._alerts: list[MonitoringAlert] = []
        self._last_alert_time: dict[str, float] = {}  # Debouncing
        self._alert_cooldown_seconds = 300  # 5 minute cooldown

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Monitor guardrail activity and generate alerts.

        This layer primarily monitors - violations are only returned
        for severe anomalies that should block requests.

        Args:
            content: Content being checked (used for metrics)
            context: Should contain 'guardrail_result', 'latency_ms'

        Returns:
            Violations for severe anomalies only
        """
        violations: list[GuardrailViolation] = []
        context = context or {}

        try:
            # Record metrics
            self._record_metrics(context)

            # Check for anomalies
            alerts = self._check_anomalies()

            # Generate violations for critical alerts
            for alert in alerts:
                if alert.severity in (ViolationSeverity.CRITICAL, ViolationSeverity.HIGH):
                    violations.append(
                        self.create_violation(
                            severity=alert.severity,
                            message=f"Monitoring alert: {alert.message}",
                            details={
                                "alert_type": alert.alert_type,
                                "metrics": alert.metrics,
                            },
                            remediation="Review recent activity for anomalies",
                        )
                    )

                # Store alert
                self._alerts.append(alert)

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self._errors.add(1)

        return violations

    def _record_metrics(self, context: dict[str, Any]) -> None:
        """Record metrics from guardrail result."""
        guardrail_result = context.get("guardrail_result")
        latency_ms = context.get("latency_ms", 0)

        # Request count
        self._requests.add(1)

        # Latency
        if latency_ms > 0:
            self._latencies.add(latency_ms)

        if guardrail_result:
            # Violations by layer
            for violation in guardrail_result.violations:
                layer_key = violation.layer.value
                self._violations[layer_key].add(1)

            # Blocks
            if guardrail_result.blocked:
                self._blocks.add(1)

    def _check_anomalies(self) -> list[MonitoringAlert]:
        """Check for anomalies and generate alerts."""
        alerts = []
        now = time.time()

        # Check violation rate
        total_violations = sum(m.count() for m in self._violations.values())
        if total_violations > self.THRESHOLDS["violation_rate_per_min"]:
            alert = self._create_alert_if_not_debounced(
                "violation_rate_spike",
                ViolationSeverity.HIGH,
                f"High violation rate: {total_violations}/min",
                {"violation_count": total_violations},
                now,
            )
            if alert:
                alerts.append(alert)

        # Check block rate
        block_count = self._blocks.count()
        if block_count > self.THRESHOLDS["block_rate_per_min"]:
            alert = self._create_alert_if_not_debounced(
                "block_rate_spike",
                ViolationSeverity.HIGH,
                f"High block rate: {block_count}/min",
                {"block_count": block_count},
                now,
            )
            if alert:
                alerts.append(alert)

        # Check PII detection rate
        pii_violations = self._violations.get("L2_PII_DETECTION", MetricWindow()).count()
        if pii_violations > self.THRESHOLDS["pii_detection_rate"]:
            alert = self._create_alert_if_not_debounced(
                "pii_detection_spike",
                ViolationSeverity.CRITICAL,
                f"High PII detection rate: {pii_violations}/min",
                {"pii_count": pii_violations},
                now,
            )
            if alert:
                alerts.append(alert)

        # Check injection rate
        injection_violations = self._violations.get("L4_PROMPT_INJECTION", MetricWindow()).count()
        if injection_violations > self.THRESHOLDS["injection_rate"]:
            alert = self._create_alert_if_not_debounced(
                "injection_attack_spike",
                ViolationSeverity.CRITICAL,
                f"High injection attempt rate: {injection_violations}/min",
                {"injection_count": injection_violations},
                now,
            )
            if alert:
                alerts.append(alert)

        # Check error rate
        request_count = self._requests.count()
        error_count = self._errors.count()
        if request_count > 0:
            error_rate = error_count / request_count
            if error_rate > self.THRESHOLDS["error_rate"]:
                alert = self._create_alert_if_not_debounced(
                    "high_error_rate",
                    ViolationSeverity.HIGH,
                    f"High error rate: {error_rate:.1%}",
                    {"error_rate": error_rate, "error_count": error_count},
                    now,
                )
                if alert:
                    alerts.append(alert)

        return alerts

    def _create_alert_if_not_debounced(
        self,
        alert_type: str,
        severity: ViolationSeverity,
        message: str,
        metrics: dict[str, float],
        now: float,
    ) -> MonitoringAlert | None:
        """Create alert if not in cooldown period."""
        last_alert = self._last_alert_time.get(alert_type, 0)

        if now - last_alert < self._alert_cooldown_seconds:
            return None  # Debounced

        self._last_alert_time[alert_type] = now

        return MonitoringAlert(
            id=f"{alert_type}_{int(now)}",
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            source_layer=self.layer,
            metrics=metrics,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for dashboard/API."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_seconds": 60,
            "requests": {
                "count": self._requests.count(),
                "rate_per_sec": self._requests.rate(),
            },
            "violations": {
                "total": sum(m.count() for m in self._violations.values()),
                "by_layer": {
                    layer: m.count() for layer, m in self._violations.items()
                },
            },
            "blocks": {
                "count": self._blocks.count(),
                "rate_per_min": self._blocks.count(),
            },
            "latency_ms": {
                "count": self._latencies.count(),
                "sum": self._latencies.sum(),
            },
            "errors": {
                "count": self._errors.count(),
                "rate": (
                    self._errors.count() / max(self._requests.count(), 1)
                ),
            },
            "thresholds": self.THRESHOLDS,
        }

    def get_alerts(
        self,
        severity: ViolationSeverity | None = None,
        unacknowledged_only: bool = False,
    ) -> list[MonitoringAlert]:
        """Get alerts, optionally filtered."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
