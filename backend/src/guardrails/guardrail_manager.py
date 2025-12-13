"""12-Layer Guardrail Manager.

Orchestrates all 12 guardrail layers for comprehensive
input and output validation.

Usage:
    >>> from backend.src.guardrails import GuardrailManager
    >>> manager = GuardrailManager()
    >>>
    >>> # Check input before processing
    >>> input_result = await manager.check_input(
    ...     query="What's the weather in Miami?",
    ...     user_id="user123",
    ...     session_id="session456"
    ... )
    >>>
    >>> if input_result.passed:
    ...     response = await agent.run(query)
    ...     output_result = await manager.check_output(response, query)
    ...     if output_result.passed:
    ...         return response
    ...     else:
    ...         return "Response blocked for safety"
    ... else:
    ...     return "Request blocked"

Execution Order:
    Input:  L1 → L2 → L3 → L4 → L5 → [L9, L10]
    Output: L6 → L7 → L8 → [L9, L10, L12]

Blocking Behavior:
    - CRITICAL violations: Instant block, stop processing
    - HIGH violations: Block if config.block_on_high = True
    - MEDIUM/LOW: Allow with logging
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailResult,
    GuardrailViolation,
    ViolationSeverity,
)
from backend.src.guardrails.layers import (
    InputValidationLayer,
    PIIDetectionLayer,
    AuthAuthzLayer,
    PromptInjectionLayer,
    ContentFilteringLayer,
    HallucinationDetectionLayer,
    BiasMitigationLayer,
    OutputValidationLayer,
    AuditLoggingLayer,
    MonitoringAlertingLayer,
    EncryptionLayer,
    ComplianceReportingLayer,
)

logger = logging.getLogger(__name__)


class GuardrailManager:
    """Orchestrates all 12 guardrail layers.

    Provides high-level API for checking inputs and outputs
    with comprehensive safety validation.

    Attributes:
        config: Guardrail configuration
        layers: Dictionary of initialized layer instances
    """

    # Layer execution order for input checking
    INPUT_LAYER_ORDER = [
        GuardrailLayer.L1_INPUT_VALIDATION,
        GuardrailLayer.L2_PII_DETECTION,
        GuardrailLayer.L3_AUTH_AUTHZ,
        GuardrailLayer.L4_PROMPT_INJECTION,
        GuardrailLayer.L5_CONTENT_FILTERING,
    ]

    # Layer execution order for output checking
    OUTPUT_LAYER_ORDER = [
        GuardrailLayer.L6_HALLUCINATION_DETECTION,
        GuardrailLayer.L7_BIAS_MITIGATION,
        GuardrailLayer.L8_OUTPUT_VALIDATION,
    ]

    # Layers that run for both input and output (async, non-blocking)
    COMMON_LAYERS = [
        GuardrailLayer.L9_AUDIT_LOGGING,
        GuardrailLayer.L10_MONITORING_ALERTING,
    ]

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        """Initialize manager with configuration.

        Args:
            config: Optional guardrail configuration
        """
        self.config = config or GuardrailConfig()

        # Initialize all layers
        self.layers = {
            GuardrailLayer.L1_INPUT_VALIDATION: InputValidationLayer(self.config),
            GuardrailLayer.L2_PII_DETECTION: PIIDetectionLayer(self.config),
            GuardrailLayer.L3_AUTH_AUTHZ: AuthAuthzLayer(self.config),
            GuardrailLayer.L4_PROMPT_INJECTION: PromptInjectionLayer(self.config),
            GuardrailLayer.L5_CONTENT_FILTERING: ContentFilteringLayer(self.config),
            GuardrailLayer.L6_HALLUCINATION_DETECTION: HallucinationDetectionLayer(self.config),
            GuardrailLayer.L7_BIAS_MITIGATION: BiasMitigationLayer(self.config),
            GuardrailLayer.L8_OUTPUT_VALIDATION: OutputValidationLayer(self.config),
            GuardrailLayer.L9_AUDIT_LOGGING: AuditLoggingLayer(self.config),
            GuardrailLayer.L10_MONITORING_ALERTING: MonitoringAlertingLayer(self.config),
            GuardrailLayer.L11_ENCRYPTION: EncryptionLayer(self.config),
            GuardrailLayer.L12_COMPLIANCE_REPORTING: ComplianceReportingLayer(self.config),
        }

        logger.info(
            f"GuardrailManager initialized with {len(self.config.enabled_layers)} "
            f"enabled layers"
        )

    async def check_input(
        self,
        query: str,
        user_id: str = "",
        session_id: str = "",
        role: str = "user",
        **extra_context: Any,
    ) -> GuardrailResult:
        """Check input query through all input guardrail layers.

        Args:
            query: User input query
            user_id: User identifier
            session_id: Session identifier
            role: User role for authorization
            **extra_context: Additional context

        Returns:
            GuardrailResult with pass/fail and violations
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        context = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "request_id": request_id,
            "check_type": "input",
            **extra_context,
        }

        return await self._run_layers(
            content=query,
            context=context,
            layer_order=self.INPUT_LAYER_ORDER,
            start_time=start_time,
        )

    async def check_output(
        self,
        response: str,
        original_query: str,
        trajectory: list[dict[str, Any]] | None = None,
        user_id: str = "",
        session_id: str = "",
        **extra_context: Any,
    ) -> GuardrailResult:
        """Check agent output through all output guardrail layers.

        Args:
            response: Agent's response
            original_query: Original user query (for context)
            trajectory: Agent execution trajectory (for hallucination check)
            user_id: User identifier
            session_id: Session identifier
            **extra_context: Additional context

        Returns:
            GuardrailResult with pass/fail and violations
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        context = {
            "query": original_query,
            "trajectory": trajectory or [],
            "user_id": user_id,
            "session_id": session_id,
            "request_id": request_id,
            "check_type": "output",
            **extra_context,
        }

        return await self._run_layers(
            content=response,
            context=context,
            layer_order=self.OUTPUT_LAYER_ORDER,
            start_time=start_time,
        )

    async def _run_layers(
        self,
        content: str,
        context: dict[str, Any],
        layer_order: list[GuardrailLayer],
        start_time: float,
    ) -> GuardrailResult:
        """Run guardrail layers in order.

        Stops early if critical violation found and blocking is enabled.

        Args:
            content: Content to check
            context: Execution context
            layer_order: Order to run layers
            start_time: Start timestamp

        Returns:
            Aggregated result from all layers
        """
        all_violations: list[GuardrailViolation] = []
        layers_checked: list[GuardrailLayer] = []
        total_latency_ms = 0.0
        blocked = False

        # Run primary layers in order
        for layer_type in layer_order:
            if layer_type not in self.config.enabled_layers:
                continue

            layer = self.layers[layer_type]
            violations, latency_ms = await layer.run(content, context)

            layers_checked.append(layer_type)
            all_violations.extend(violations)
            total_latency_ms += latency_ms

            # Check for blocking violations
            has_critical = any(
                v.severity == ViolationSeverity.CRITICAL for v in violations
            )
            has_high = any(
                v.severity == ViolationSeverity.HIGH for v in violations
            )

            if has_critical or (has_high and self.config.block_on_high):
                blocked = True
                logger.warning(
                    f"Request blocked by {layer_type.value}: "
                    f"{len(violations)} violation(s)"
                )
                # Don't stop - continue checking for full audit trail
                # But mark as blocked

        # Run common layers (audit, monitoring) - don't block on these
        if self.config.parallel_execution:
            common_tasks = []
            for layer_type in self.COMMON_LAYERS:
                if layer_type in self.config.enabled_layers:
                    layer = self.layers[layer_type]
                    # Include current result in context for logging
                    log_context = {
                        **context,
                        "guardrail_result": GuardrailResult(
                            passed=not blocked,
                            violations=all_violations,
                            layers_checked=layers_checked,
                        ),
                        "latency_ms": total_latency_ms,
                    }
                    common_tasks.append(layer.run(content, log_context))

            if common_tasks:
                common_results = await asyncio.gather(*common_tasks, return_exceptions=True)
                for i, result in enumerate(common_results):
                    if isinstance(result, tuple):
                        violations, latency = result
                        layers_checked.append(self.COMMON_LAYERS[i])
                        # Monitoring layer can add violations
                        all_violations.extend(violations)
        else:
            # Sequential execution
            for layer_type in self.COMMON_LAYERS:
                if layer_type in self.config.enabled_layers:
                    layer = self.layers[layer_type]
                    log_context = {
                        **context,
                        "guardrail_result": GuardrailResult(
                            passed=not blocked,
                            violations=all_violations,
                            layers_checked=layers_checked,
                        ),
                    }
                    violations, latency = await layer.run(content, log_context)
                    layers_checked.append(layer_type)
                    all_violations.extend(violations)

        # Calculate risk score
        risk_score = self._calculate_risk_score(all_violations)

        # Build final result
        execution_time_ms = (time.time() - start_time) * 1000

        result = GuardrailResult(
            passed=not blocked and len(all_violations) == 0,
            violations=all_violations,
            layers_checked=layers_checked,
            execution_time_ms=execution_time_ms,
            blocked=blocked,
            risk_score=risk_score,
            metadata={
                "request_id": context.get("request_id"),
                "check_type": context.get("check_type"),
            },
        )

        logger.info(
            f"Guardrail check complete: passed={result.passed}, "
            f"violations={len(all_violations)}, "
            f"blocked={blocked}, "
            f"time={execution_time_ms:.1f}ms"
        )

        return result

    def _calculate_risk_score(self, violations: list[GuardrailViolation]) -> float:
        """Calculate aggregate risk score from violations."""
        if not violations:
            return 0.0

        severity_weights = {
            ViolationSeverity.LOW: 0.1,
            ViolationSeverity.MEDIUM: 0.3,
            ViolationSeverity.HIGH: 0.6,
            ViolationSeverity.CRITICAL: 1.0,
        }

        total_weight = sum(
            severity_weights.get(v.severity, 0.3) for v in violations
        )

        # Normalize to 0-1 range (cap at 1.0)
        return min(1.0, total_weight / 2.0)

    async def check_both(
        self,
        query: str,
        response: str,
        trajectory: list[dict[str, Any]] | None = None,
        user_id: str = "",
        session_id: str = "",
        **extra_context: Any,
    ) -> tuple[GuardrailResult, GuardrailResult]:
        """Check both input and output in one call.

        Useful for batch processing or post-hoc validation.

        Args:
            query: User input
            response: Agent response
            trajectory: Agent execution trajectory
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Tuple of (input_result, output_result)
        """
        input_result = await self.check_input(
            query=query,
            user_id=user_id,
            session_id=session_id,
            **extra_context,
        )

        output_result = await self.check_output(
            response=response,
            original_query=query,
            trajectory=trajectory,
            user_id=user_id,
            session_id=session_id,
            **extra_context,
        )

        return input_result, output_result

    def get_layer_stats(self) -> dict[str, Any]:
        """Get statistics from all layers."""
        stats = {}
        for layer_type, layer in self.layers.items():
            stats[layer_type.value] = layer.stats
        return stats

    def get_metrics(self) -> dict[str, Any]:
        """Get monitoring metrics."""
        monitoring_layer = self.layers.get(GuardrailLayer.L10_MONITORING_ALERTING)
        if monitoring_layer and hasattr(monitoring_layer, "get_metrics"):
            return monitoring_layer.get_metrics()
        return {}

    def generate_compliance_report(
        self,
        framework: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate compliance report for specified framework."""
        compliance_layer = self.layers.get(GuardrailLayer.L12_COMPLIANCE_REPORTING)
        if compliance_layer and hasattr(compliance_layer, "generate_compliance_report"):
            from backend.src.guardrails.models import ComplianceFramework
            try:
                fw = ComplianceFramework(framework)
                return compliance_layer.generate_compliance_report(fw, **kwargs)
            except ValueError:
                return {"error": f"Unknown framework: {framework}"}
        return {"error": "Compliance reporting layer not available"}

    def redact_pii(self, content: str) -> str:
        """Redact PII from content using PII detection layer."""
        pii_layer = self.layers.get(GuardrailLayer.L2_PII_DETECTION)
        if pii_layer and hasattr(pii_layer, "redact_pii"):
            return pii_layer.redact_pii(content)
        return content

    def encrypt(self, plaintext: str) -> str:
        """Encrypt sensitive data using encryption layer."""
        encryption_layer = self.layers.get(GuardrailLayer.L11_ENCRYPTION)
        if encryption_layer and hasattr(encryption_layer, "encrypt"):
            return encryption_layer.encrypt(plaintext)
        raise RuntimeError("Encryption layer not available")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt encrypted data."""
        encryption_layer = self.layers.get(GuardrailLayer.L11_ENCRYPTION)
        if encryption_layer and hasattr(encryption_layer, "decrypt"):
            return encryption_layer.decrypt(ciphertext)
        raise RuntimeError("Encryption layer not available")
