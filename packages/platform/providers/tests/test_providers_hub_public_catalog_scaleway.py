"""Tests for Scaleway public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from artenic_ai_platform_providers.hub.public_catalog.scaleway import ScalewayCatalogFetcher


class TestScalewayCatalogComputeStatic:
    async def test_returns_static_fallback(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        # Force static mode by failing live fetch
        with patch.object(fetcher, "_try_live_compute", return_value=None):
            result = await fetcher.fetch_compute_catalog()
        assert len(result) >= 8
        assert result[0].provider_id == "scaleway"
        assert result[0].currency == "EUR"
        assert fetcher.supports_live_catalog() is False

    async def test_static_includes_gpu(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        with patch.object(fetcher, "_try_live_compute", return_value=None):
            result = await fetcher.fetch_compute_catalog()
        gpu_flavors = [f for f in result if f.gpu_count > 0]
        assert len(gpu_flavors) >= 1


class TestScalewayCatalogComputeLive:
    async def test_live_success(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "servers": {
                "DEV1-S": {
                    "ncpus": 2,
                    "ram": 2147483648,
                    "gpu": 0,
                    "hourly_price": 1000,
                },
                "GPU-3070-S": {
                    "ncpus": 8,
                    "ram": 17179869184,
                    "gpu": 1,
                    "arch": "ampere",
                    "hourly_price": 65000,
                },
            },
        }
        with patch("artenic_ai_platform_providers.hub.public_catalog.scaleway.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher.fetch_compute_catalog()

        assert fetcher.supports_live_catalog() is True
        assert len(result) == 2
        dev = next(r for r in result if r.name == "DEV1-S")
        assert dev.vcpus == 2
        assert dev.memory_gb == 2.0
        assert dev.gpu_count == 0

        gpu = next(r for r in result if r.name == "GPU-3070-S")
        assert gpu.gpu_count == 1
        assert gpu.gpu_type == "AMPERE"
        assert gpu.category == "gpu"

    async def test_live_skips_non_dict_server(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "servers": {
                "DEV1-S": "not-a-dict",
                "GP1-S": {
                    "ncpus": 8,
                    "ram": 34359738368,
                    "gpu": 0,
                    "hourly_price": 11000,
                },
            },
        }
        with patch(
            "artenic_ai_platform_providers.hub.public_catalog.scaleway.httpx",
        ) as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher.fetch_compute_catalog()
        assert len(result) == 1
        assert result[0].name == "GP1-S"

    async def test_live_401_falls_back(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("artenic_ai_platform_providers.hub.public_catalog.scaleway.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher.fetch_compute_catalog()
        assert fetcher.supports_live_catalog() is False
        assert len(result) >= 8

    async def test_live_exception_falls_back(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        with patch("artenic_ai_platform_providers.hub.public_catalog.scaleway.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("network error")
            result = await fetcher.fetch_compute_catalog()
        assert fetcher.supports_live_catalog() is False
        assert len(result) >= 8

    async def test_live_no_httpx(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        with patch("artenic_ai_platform_providers.hub.public_catalog.scaleway.httpx", None):
            result = await fetcher.fetch_compute_catalog()
        assert fetcher.supports_live_catalog() is False
        assert len(result) >= 8


class TestScalewayCatalogStorage:
    async def test_returns_storage(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        result = await fetcher.fetch_storage_catalog()
        assert len(result) >= 1
        assert result[0].provider_id == "scaleway"
        assert result[0].currency == "EUR"


class TestScalewaySupportsLive:
    def test_initial_state(self) -> None:
        fetcher = ScalewayCatalogFetcher()
        # Before any fetch, _is_live is None → supports_live returns False
        assert fetcher.supports_live_catalog() is False
