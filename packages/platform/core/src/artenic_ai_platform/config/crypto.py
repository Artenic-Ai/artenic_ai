"""Secret encryption and masking utilities.

Uses Fernet symmetric encryption from the ``cryptography`` library.
The dependency is imported lazily so the module can be loaded even when
``cryptography`` is not installed (albeit with a clear error on use).
"""

from __future__ import annotations

import base64
import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

_DECRYPTION_SENTINEL = "***DECRYPTION_FAILED***"


class SecretManager:
    """Encrypt, decrypt, and mask sensitive strings.

    Parameters
    ----------
    passphrase:
        A passphrase used to derive a deterministic Fernet key via
        SHA-256.  If empty, a random key is generated (suitable only
        for development / testing).
    """

    def __init__(self, passphrase: str = "") -> None:
        self._fernet = self._build_fernet(passphrase)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fernet(passphrase: str) -> Fernet:
        from cryptography.fernet import Fernet as _Fernet

        if not passphrase:
            logger.warning(
                "No passphrase supplied — generating a random Fernet "
                "key.  Do NOT use this in production."
            )
            return _Fernet(_Fernet.generate_key())

        digest = hashlib.sha256(passphrase.encode()).digest()
        key = base64.urlsafe_b64encode(digest)
        return _Fernet(key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """Encrypt *plaintext* and return a base-64 ciphertext string."""
        token: bytes = self._fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(token).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt *ciphertext* and return the original plaintext.

        Returns ``***DECRYPTION_FAILED***`` if decryption fails for
        any reason (wrong key, corrupted data, etc.).
        """
        try:
            token = base64.urlsafe_b64decode(ciphertext.encode())
            return self._fernet.decrypt(token).decode()
        except Exception:
            logger.debug(
                "Decryption failed for ciphertext (length=%d)",
                len(ciphertext),
            )
            return _DECRYPTION_SENTINEL

    @staticmethod
    def mask(value: str, visible_chars: int = 4) -> str:
        """Return a masked version of *value*.

        The last *visible_chars* characters are shown; the rest are
        replaced with ``***``.  If the value is too short to expose
        that many characters, the entire value is masked.
        """
        if len(value) <= visible_chars:
            return "***"
        return f"***{value[-visible_chars:]}"
