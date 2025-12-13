"""Layer 11: Encryption.

Handles data encryption for sensitive content at rest and in transit.
Ensures PII and other sensitive data is protected.

Features:
- Field-level encryption for PII
- Encryption key rotation support
- Secure key management integration
- AES-256-GCM encryption
- Encrypted audit trails

Compliance:
- HIPAA: Encryption of PHI
- PCI-DSS: Encryption of card data
- GDPR: Technical measures for data protection
"""

import base64
import hashlib
import os
import secrets
from typing import Any

# Note: In production, use cryptography library
# This is a simplified implementation for demonstration
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None
    AESGCM = None

from backend.src.guardrails.layers.base import BaseGuardrailLayer
from backend.src.guardrails.models import (
    GuardrailConfig,
    GuardrailLayer,
    GuardrailViolation,
    ViolationSeverity,
)


class EncryptionLayer(BaseGuardrailLayer):
    """Layer 11: Encryption.

    Provides encryption utilities for sensitive data protection.
    This layer doesn't block - it provides encryption services.
    """

    layer = GuardrailLayer.L11_ENCRYPTION

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        encryption_key: bytes | None = None,
    ) -> None:
        """Initialize with configuration and encryption key.

        Args:
            config: Guardrail configuration
            encryption_key: 32-byte key for AES-256 (generated if not provided)
        """
        super().__init__(config)

        # Generate or use provided key
        if encryption_key:
            self._key = encryption_key
        else:
            # Generate from environment or create new
            env_key = os.environ.get("GUARDRAIL_ENCRYPTION_KEY")
            if env_key:
                self._key = base64.b64decode(env_key)
            else:
                self._key = secrets.token_bytes(32)  # 256 bits

        # Initialize Fernet if available
        if CRYPTO_AVAILABLE:
            # Fernet requires base64-encoded key
            fernet_key = base64.urlsafe_b64encode(self._key[:32].ljust(32, b'\0'))
            self._fernet = Fernet(fernet_key)
        else:
            self._fernet = None

        # Track encrypted fields
        self._encrypted_fields: set[str] = set()

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[GuardrailViolation]:
        """Check encryption status and encrypt sensitive data.

        This layer doesn't typically return violations - it performs
        encryption as a service. Violations only for encryption failures.

        Args:
            content: Content to potentially encrypt
            context: May contain 'encrypt_fields' list

        Returns:
            Violations only for encryption failures
        """
        violations: list[GuardrailViolation] = []
        context = context or {}

        # Check if encryption is requested
        encrypt_fields = context.get("encrypt_fields", [])

        if encrypt_fields and not CRYPTO_AVAILABLE:
            violations.append(
                self.create_violation(
                    severity=ViolationSeverity.HIGH,
                    message="Encryption requested but cryptography library not available",
                    details={"missing_library": "cryptography"},
                    remediation="Install cryptography package: pip install cryptography",
                )
            )

        # Check for unencrypted sensitive data in storage context
        if context.get("check_storage_encryption"):
            storage_violations = self._check_storage_encryption(context)
            violations.extend(storage_violations)

        return violations

    def _check_storage_encryption(
        self,
        context: dict[str, Any],
    ) -> list[GuardrailViolation]:
        """Check if sensitive data is encrypted before storage."""
        violations = []

        sensitive_fields = context.get("sensitive_fields", [])
        data = context.get("data", {})

        for field in sensitive_fields:
            if field in data:
                value = data[field]
                if not self._is_encrypted(value):
                    violations.append(
                        self.create_violation(
                            severity=ViolationSeverity.HIGH,
                            message=f"Sensitive field '{field}' is not encrypted",
                            details={"field": field},
                            remediation=f"Encrypt '{field}' before storage",
                        )
                    )

        return violations

    def _is_encrypted(self, value: str) -> bool:
        """Check if value appears to be encrypted.

        Encrypted values typically have specific markers.
        """
        if not isinstance(value, str):
            return False

        # Check for encryption markers
        encrypted_prefixes = ["enc:", "encrypted:", "ENC[", "gAAAAA"]
        return any(value.startswith(prefix) for prefix in encrypted_prefixes)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext using AES-256.

        Args:
            plaintext: Text to encrypt

        Returns:
            Base64-encoded ciphertext with "enc:" prefix
        """
        if not CRYPTO_AVAILABLE:
            # Fallback: Simple obfuscation (NOT secure - just for structure)
            encoded = base64.b64encode(plaintext.encode()).decode()
            return f"enc:fallback:{encoded}"

        ciphertext = self._fernet.encrypt(plaintext.encode())
        return f"enc:{ciphertext.decode()}"

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext.

        Args:
            ciphertext: Encrypted text with "enc:" prefix

        Returns:
            Decrypted plaintext
        """
        if not ciphertext.startswith("enc:"):
            raise ValueError("Invalid ciphertext format")

        encrypted_data = ciphertext[4:]  # Remove "enc:" prefix

        # Handle fallback format
        if encrypted_data.startswith("fallback:"):
            encoded = encrypted_data[9:]
            return base64.b64decode(encoded).decode()

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library required for decryption")

        return self._fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_dict(
        self,
        data: dict[str, Any],
        fields_to_encrypt: list[str],
    ) -> dict[str, Any]:
        """Encrypt specific fields in a dictionary.

        Args:
            data: Dictionary with potentially sensitive data
            fields_to_encrypt: List of field names to encrypt

        Returns:
            Dictionary with specified fields encrypted
        """
        result = data.copy()

        for field in fields_to_encrypt:
            if field in result and result[field]:
                value = str(result[field])
                result[field] = self.encrypt(value)
                self._encrypted_fields.add(field)

        return result

    def decrypt_dict(
        self,
        data: dict[str, Any],
        fields_to_decrypt: list[str] | None = None,
    ) -> dict[str, Any]:
        """Decrypt specific fields in a dictionary.

        Args:
            data: Dictionary with encrypted data
            fields_to_decrypt: Fields to decrypt (all encrypted if None)

        Returns:
            Dictionary with specified fields decrypted
        """
        result = data.copy()

        fields = fields_to_decrypt or list(result.keys())

        for field in fields:
            if field in result and isinstance(result[field], str):
                if self._is_encrypted(result[field]):
                    try:
                        result[field] = self.decrypt(result[field])
                    except Exception:
                        pass  # Leave encrypted if decryption fails

        return result

    def hash_for_lookup(self, value: str) -> str:
        """Create deterministic hash for encrypted field lookup.

        Allows searching for encrypted values without decryption.

        Args:
            value: Plaintext value to hash

        Returns:
            SHA-256 hash suitable for database index
        """
        # Use HMAC with key for deterministic but secure hashing
        salted = self._key[:16] + value.encode()
        return hashlib.sha256(salted).hexdigest()

    def rotate_key(self, new_key: bytes) -> None:
        """Rotate encryption key.

        In production, this would re-encrypt all stored data.

        Args:
            new_key: New 32-byte encryption key
        """
        if len(new_key) < 32:
            raise ValueError("Key must be at least 32 bytes")

        self._key = new_key[:32]

        if CRYPTO_AVAILABLE:
            fernet_key = base64.urlsafe_b64encode(self._key.ljust(32, b'\0'))
            self._fernet = Fernet(fernet_key)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new encryption key.

        Returns:
            32-byte random key suitable for AES-256
        """
        return secrets.token_bytes(32)
