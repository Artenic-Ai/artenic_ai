"""Tests for artenic_ai_platform.budget.alert_dispatcher."""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock, patch

from artenic_ai_platform.budget.alert_dispatcher import AlertDispatcher

# ======================================================================
# dispatch
# ======================================================================


class TestDispatchSuccess:
    """dispatch sends a POST and returns True on success."""

    async def test_dispatch_success(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/hook",
            webhook_secret="test-secret",
        )
        alert = {"level": "warning", "message": "Budget at 80%"}

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("artenic_ai_platform.budget.alert_dispatcher.httpx") as mock_httpx:
            mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
            result = await dispatcher.dispatch(alert)

        assert result is True
        mock_client.post.assert_called_once()

        # Verify the call arguments include proper headers
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Artenic-Event"] == "budget_alert"
        assert "X-Artenic-Signature" in headers


class TestDispatchFailureRetry:
    """dispatch retries on error and eventually returns False."""

    async def test_dispatch_failure_retry(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/hook",
            webhook_secret="test-secret",
            retry_count=2,
        )
        alert = {"level": "critical", "message": "Over budget"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("artenic_ai_platform.budget.alert_dispatcher.httpx") as mock_httpx:
            mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
            result = await dispatcher.dispatch(alert)

        assert result is False
        # retry_count=2 means 3 total attempts (initial + 2 retries)
        assert mock_client.post.call_count == 3


class TestDispatchNoUrl:
    """dispatch returns False when the webhook URL is empty."""

    async def test_dispatch_no_url(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="",
            webhook_secret="test-secret",
        )
        result = await dispatcher.dispatch({"message": "test"})
        assert result is False


# ======================================================================
# HMAC signature
# ======================================================================


class TestHmacSignature:
    """_sign produces a valid HMAC-SHA256 hex digest."""

    def test_hmac_signature(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/hook",
            webhook_secret="my-secret-key",
        )
        body = '{"alert": "test"}'
        sig = dispatcher._sign(body)

        # Compute expected signature
        expected = hmac.new(
            b"my-secret-key",
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert sig == expected


class TestVerifySignatureValid:
    """verify_signature returns True for a valid body+secret+signature."""

    def test_verify_signature_valid(self) -> None:
        secret = "webhook-secret"
        body = '{"event": "budget_alert", "amount": 100}'
        signature = hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert AlertDispatcher.verify_signature(body, secret, signature) is True


class TestVerifySignatureInvalid:
    """verify_signature returns False for a tampered body."""

    def test_verify_signature_invalid(self) -> None:
        secret = "webhook-secret"
        body = '{"event": "budget_alert", "amount": 100}'
        # Use a wrong signature
        wrong_sig = "0" * 64

        assert AlertDispatcher.verify_signature(body, secret, wrong_sig) is False


class TestDispatchHttpxUnavailable:
    """dispatch returns False when httpx is not available (lines 60-61)."""

    async def test_dispatch_httpx_unavailable(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/hook",
            webhook_secret="test-secret",
        )

        with patch("artenic_ai_platform.budget.alert_dispatcher._HTTPX_AVAILABLE", False):
            result = await dispatcher.dispatch({"message": "test"})

        assert result is False


class TestDispatchNon2xxResponse:
    """dispatch retries and returns False on non-2xx webhook responses (line 89)."""

    async def test_dispatch_non_2xx_response(self) -> None:
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/hook",
            webhook_secret="test-secret",
            retry_count=1,
        )
        alert = {"level": "warning", "message": "test"}

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("artenic_ai_platform.budget.alert_dispatcher.httpx") as mock_httpx:
            mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
            result = await dispatcher.dispatch(alert)

        assert result is False
        # retry_count=1 means 2 total attempts
        assert mock_client.post.call_count == 2
