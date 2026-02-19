"""Tests for GCP public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform.providers_hub.public_catalog.gcp import (
    GcpCatalogFetcher,
    _extract_machine_type,
    _get_family,
    _gpu_count_from_name,
    _parse_machine_specs,
)


class TestExtractMachineType:
    def test_standard(self) -> None:
        key = "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4"
        assert _extract_machine_type(key) == "n1-standard-4"

    def test_no_vmimage(self) -> None:
        assert _extract_machine_type("CP-SOMETHING-ELSE") == ""


class TestGetFamily:
    def test_standard(self) -> None:
        assert _get_family("n1-standard-4") == "n1-standard"

    def test_single_part(self) -> None:
        assert _get_family("custom") == "custom"


class TestParseMachineSpecs:
    def test_n1_standard_4(self) -> None:
        vcpus, mem = _parse_machine_specs("n1-standard-4")
        assert vcpus == 4
        assert mem == 15.0  # 4 * 3.75

    def test_highmem(self) -> None:
        vcpus, mem = _parse_machine_specs("n1-highmem-8")
        assert vcpus == 8
        assert mem == 52.0  # 8 * 6.5

    def test_highcpu(self) -> None:
        vcpus, mem = _parse_machine_specs("n1-highcpu-4")
        assert vcpus == 4
        assert mem == 3.6  # 4 * 0.9

    def test_no_trailing_number(self) -> None:
        vcpus, mem = _parse_machine_specs("custom")
        assert vcpus == 0
        assert mem == 0.0


class TestGpuCountFromName:
    def test_with_count(self) -> None:
        assert _gpu_count_from_name("a2-highgpu-4g") == 4

    def test_no_count(self) -> None:
        assert _gpu_count_from_name("custom") == 1


class TestGcpCatalogCompute:
    async def test_parses_price_list(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4": {"us": 0.19},
                "CP-COMPUTEENGINE-VMIMAGE-A2-HIGHGPU-1G": {"us": 3.67},
                "some-other-key": {"us": 1.0},
                "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4-PREEMPTIBLE": {"us": 0.04},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()

        names = [r.name for r in result]
        assert "n1-standard-4" in names
        assert "a2-highgpu-1g" in names

        n1 = next(r for r in result if r.name == "n1-standard-4")
        assert n1.vcpus == 4
        assert n1.price_per_hour == 0.19
        assert n1.currency == "USD"
        assert n1.category == "general"

        a2 = next(r for r in result if r.name == "a2-highgpu-1g")
        assert a2.gpu_type == "A100"
        assert a2.category == "gpu"

    async def test_empty_price_list(self) -> None:
        fetcher = GcpCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={}):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_skips_no_us_price(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4": {"eu": 0.20},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_skips_non_dict_values(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4": "not-a-dict",
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_deduplicates_machine_types(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                # Different prefixes but same machine type after extraction
                "CP-COMPUTEENGINE-VMIMAGE-N1-STANDARD-4": {"us": 0.19},
                "CP-PREEMPTIBLE-VMIMAGE-N1-STANDARD-4": {"us": 0.04},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        # Both keys resolve to n1-standard-4 → deduplicated
        names = [r.name for r in result]
        assert names.count("n1-standard-4") == 1


class TestGcpCatalogStorage:
    async def test_parses_storage(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-CLOUD-STORAGE-STANDARD-CLASS-A": {"us": 0.026},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert len(result) == 1
        assert result[0].name == "Cloud Storage — Standard"
        assert result[0].price_per_gb_month == 0.026

    async def test_empty(self) -> None:
        fetcher = GcpCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={}):
            result = await fetcher.fetch_storage_catalog()
        assert result == []

    async def test_skips_non_dict_storage_value(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-CLOUD-STORAGE-STANDARD-CLASS-A": "not-a-dict",
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert result == []

    async def test_skips_non_storage_key(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-COMPUTEENGINE-SOMETHING": {"us": 1.0},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert result == []

    async def test_skips_non_cloud_storage_key(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-STORAGE-STANDARD-SOMETHING": {"us": 1.0},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert result == []

    async def test_skips_storage_without_us_price(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-CLOUD-STORAGE-STANDARD-CLASS-A": {"eu": 0.026},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert result == []

    async def test_nearline_storage(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_data = {
            "gcp_price_list": {
                "CP-CLOUD-STORAGE-NEARLINE-CLASS-A": {"us": 0.01},
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_storage_catalog()
        assert len(result) == 1
        assert result[0].name == "Cloud Storage — Nearline"


class TestGcpGetRequiresHttpx:
    async def test_raises_import_error(self) -> None:
        fetcher = GcpCatalogFetcher()
        with (
            patch("artenic_ai_platform.providers_hub.public_catalog.gcp.httpx", None),
            pytest.raises(ImportError, match="httpx"),
        ):
            await fetcher._get()

    async def test_get_success(self) -> None:
        fetcher = GcpCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"gcp_price_list": {}}
        with patch("artenic_ai_platform.providers_hub.public_catalog.gcp.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher._get()
        assert result == {"gcp_price_list": {}}
        mock_resp.raise_for_status.assert_called_once()
