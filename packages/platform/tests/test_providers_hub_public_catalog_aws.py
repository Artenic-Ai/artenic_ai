"""Tests for AWS public catalog fetcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform.providers_hub.public_catalog.aws import (
    AwsCatalogFetcher,
    _category_from_family,
    _extract_on_demand_price,
    _gpu_for_family,
    _parse_instance_family,
    _parse_storage,
)


class TestParseInstanceFamily:
    def test_standard(self) -> None:
        assert _parse_instance_family("m5.xlarge") == "m5"

    def test_gpu(self) -> None:
        assert _parse_instance_family("p4d.24xlarge") == "p4d"

    def test_no_dot(self) -> None:
        assert _parse_instance_family("m5") == "m5"


class TestGpuForFamily:
    def test_p3(self) -> None:
        assert _gpu_for_family("p3") == ("V100", 1)

    def test_g4dn(self) -> None:
        assert _gpu_for_family("g4dn") == ("T4", 1)

    def test_no_gpu(self) -> None:
        assert _gpu_for_family("m5") == (None, 0)


class TestCategoryFromFamily:
    def test_gpu(self) -> None:
        assert _category_from_family("p3") == "gpu"

    def test_compute(self) -> None:
        assert _category_from_family("c5") == "compute"

    def test_memory(self) -> None:
        assert _category_from_family("r5") == "memory"

    def test_storage(self) -> None:
        assert _category_from_family("i3") == "storage"

    def test_general(self) -> None:
        assert _category_from_family("m5") == "general"

    def test_hpc(self) -> None:
        assert _category_from_family("hpc") == "compute"


class TestParseStorage:
    def test_ebs_only(self) -> None:
        assert _parse_storage("EBS only") == 0.0

    def test_ebs_dash(self) -> None:
        assert _parse_storage("EBS-only") == 0.0

    def test_empty(self) -> None:
        assert _parse_storage("") == 0.0

    def test_single_disk(self) -> None:
        assert _parse_storage("900 GB") == 900.0

    def test_multi_disk(self) -> None:
        assert _parse_storage("2 x 900 NVMe SSD") == 1800.0


class TestExtractOnDemandPrice:
    def test_finds_price(self) -> None:
        on_demand = {
            "SKU1": {
                "offer1": {
                    "priceDimensions": {
                        "dim1": {"pricePerUnit": {"USD": "0.096"}},
                    },
                },
            },
        }
        assert _extract_on_demand_price("SKU1", on_demand) == 0.096

    def test_missing_sku(self) -> None:
        assert _extract_on_demand_price("missing", {}) is None

    def test_empty_price(self) -> None:
        on_demand = {
            "SKU1": {
                "offer1": {
                    "priceDimensions": {
                        "dim1": {"pricePerUnit": {}},
                    },
                },
            },
        }
        assert _extract_on_demand_price("SKU1", on_demand) is None

    def test_non_dict_offer_skipped(self) -> None:
        on_demand: dict[str, dict[str, object]] = {
            "SKU1": {"offer1": "bad-value"},
        }
        assert _extract_on_demand_price("SKU1", on_demand) is None

    def test_non_dict_price_per_unit(self) -> None:
        on_demand = {
            "SKU1": {
                "offer1": {
                    "priceDimensions": {
                        "dim1": {"pricePerUnit": "bad"},
                    },
                },
            },
        }
        assert _extract_on_demand_price("SKU1", on_demand) is None


class TestAwsCatalogCompute:
    async def test_parses_products(self) -> None:
        fetcher = AwsCatalogFetcher()
        mock_data = {
            "products": {
                "SKU1": {
                    "attributes": {
                        "instanceType": "m5.large",
                        "operatingSystem": "Linux",
                        "tenancy": "Shared",
                        "vcpu": "2",
                        "memory": "8 GiB",
                        "storage": "EBS only",
                        "location": "EU (Ireland)",
                    },
                },
                "SKU2": {
                    "attributes": {
                        "instanceType": "p3.2xlarge",
                        "operatingSystem": "Linux",
                        "tenancy": "Shared",
                        "vcpu": "8",
                        "memory": "61 GiB",
                        "storage": "EBS only",
                        "location": "EU (Ireland)",
                    },
                },
                "SKU3": {
                    "attributes": {
                        "instanceType": "m5.large",
                        "operatingSystem": "Windows",
                        "tenancy": "Shared",
                        "vcpu": "2",
                        "memory": "8 GiB",
                    },
                },
            },
            "terms": {
                "OnDemand": {
                    "SKU1": {
                        "offer1": {
                            "priceDimensions": {
                                "dim1": {"pricePerUnit": {"USD": "0.096"}},
                            },
                        },
                    },
                    "SKU2": {
                        "offer1": {
                            "priceDimensions": {
                                "dim1": {"pricePerUnit": {"USD": "3.06"}},
                            },
                        },
                    },
                },
            },
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()

        assert len(result) == 2
        m5 = next(r for r in result if r.name == "m5.large")
        assert m5.vcpus == 2
        assert m5.memory_gb == 8.0
        assert m5.price_per_hour == 0.096
        assert m5.currency == "USD"
        assert m5.gpu_count == 0
        assert m5.category == "general"

        p3 = next(r for r in result if r.name == "p3.2xlarge")
        assert p3.gpu_type == "V100"
        assert p3.category == "gpu"

    async def test_empty_data(self) -> None:
        fetcher = AwsCatalogFetcher()
        with patch.object(fetcher, "_get", return_value={"products": {}, "terms": {}}):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_skips_non_linux(self) -> None:
        fetcher = AwsCatalogFetcher()
        mock_data = {
            "products": {
                "SKU1": {
                    "attributes": {
                        "instanceType": "m5.xlarge",
                        "operatingSystem": "Windows",
                        "tenancy": "Shared",
                        "vcpu": "4",
                        "memory": "16 GiB",
                    },
                },
            },
            "terms": {"OnDemand": {}},
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        assert result == []

    async def test_skips_non_shared(self) -> None:
        fetcher = AwsCatalogFetcher()
        mock_data = {
            "products": {
                "SKU1": {
                    "attributes": {
                        "instanceType": "m5.large",
                        "operatingSystem": "Linux",
                        "tenancy": "Dedicated",
                        "vcpu": "2",
                        "memory": "8 GiB",
                    },
                },
            },
            "terms": {"OnDemand": {}},
        }
        with patch.object(fetcher, "_get", return_value=mock_data):
            result = await fetcher.fetch_compute_catalog()
        assert result == []


class TestAwsCatalogStorage:
    async def test_returns_empty(self) -> None:
        fetcher = AwsCatalogFetcher()
        result = await fetcher.fetch_storage_catalog()
        assert result == []


class TestAwsGetRequiresHttpx:
    async def test_raises_import_error(self) -> None:
        fetcher = AwsCatalogFetcher()
        with (
            patch("artenic_ai_platform.providers_hub.public_catalog.aws.httpx", None),
            pytest.raises(ImportError, match="httpx"),
        ):
            await fetcher._get()

    async def test_get_success(self) -> None:
        fetcher = AwsCatalogFetcher()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"products": {}}
        with patch("artenic_ai_platform.providers_hub.public_catalog.aws.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            result = await fetcher._get()
        assert result == {"products": {}}
        mock_resp.raise_for_status.assert_called_once()
