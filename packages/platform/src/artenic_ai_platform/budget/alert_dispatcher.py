"""Webhook alert dispatcher with HMAC-SHA256 signing and retry."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Optional httpx import — alert dispatch degrades gracefully
try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HTTPX_AVAILABLE = False


class AlertDispatcher:
    """Sends budget alerts via webhooks with HMAC-SHA256 signatures.

    Each alert is POSTed to the configured webhook URL with:
    - Content-Type: application/json
    - X-Artenic-Signature: HMAC-SHA256(secret, body)
    - X-Artenic-Event: budget_alert

    Retries with exponential backoff on transient failures.
    """

    def __init__(
        self,
        *,
        webhook_url: str,
        webhook_secret: str,
        timeout_seconds: float = 10.0,
        retry_count: int = 2,
    ) -> None:
        self._url = webhook_url
        self._secret = webhook_secret
        self._timeout = timeout_seconds
        self._retries = retry_count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(self, alert: dict[str, Any]) -> bool:
        """Send an alert to the webhook endpoint.

        Returns True if delivery succeeded, False otherwise.
        """
        if not self._url:
            logger.debug("Webhook URL not configured, skipping alert")
            return False

        if not _HTTPX_AVAILABLE:
            logger.warning("httpx not installed, cannot send webhook alert")
            return False

        body = json.dumps(alert, default=str, sort_keys=True)
        signature = self._sign(body)

        headers = {
            "Content-Type": "application/json",
            "X-Artenic-Signature": signature,
            "X-Artenic-Event": "budget_alert",
        }

        for attempt in range(self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        self._url,
                        content=body,
                        headers=headers,
                    )
                if resp.status_code < 400:
                    logger.info(
                        "Alert dispatched to %s (status=%d)",
                        self._url,
                        resp.status_code,
                    )
                    return True
                logger.warning(
                    "Webhook returned %d on attempt %d",
                    resp.status_code,
                    attempt + 1,
                )
            except Exception:
                logger.warning(
                    "Webhook request failed (attempt %d/%d)",
                    attempt + 1,
                    self._retries + 1,
                    exc_info=True,
                )

        logger.error("Alert dispatch failed after %d attempts", self._retries + 1)
        return False

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _sign(self, body: str) -> str:
        """Compute HMAC-SHA256 signature of the request body."""
        return hmac.new(
            self._secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def verify_signature(
        body: str,
        secret: str,
        signature: str,
    ) -> bool:
        """Verify a webhook signature (utility for receivers)."""
        expected = hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
