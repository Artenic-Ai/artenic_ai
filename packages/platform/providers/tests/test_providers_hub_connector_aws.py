"""Tests for the AWS connector."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from artenic_ai_platform_providers.hub.connectors.aws import (
    _GPU_FAMILIES,
    AwsConnector,
    _require_boto3,
)
from artenic_ai_platform_providers.hub.connectors.base import ConnectorContext

CTX = ConnectorContext(
    credentials={
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    },
    config={"region": "eu-west-1"},
)


# ======================================================================
# _require_boto3
# ======================================================================


class TestRequireBoto3:
    def test_raises_import_error(self) -> None:
        with pytest.raises(ImportError, match="boto3"):
            _require_boto3()


# ======================================================================
# test_connection
# ======================================================================


def _make_mock_session(
    *,
    identity: dict[str, Any] | None = None,
    buckets: list[dict[str, Any]] | None = None,
    instance_types: list[dict[str, Any]] | None = None,
    regions: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock boto3.Session."""
    session = MagicMock()

    # STS client
    sts = MagicMock()
    sts.get_caller_identity.return_value = identity or {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/test",
    }
    # S3 client
    s3 = MagicMock()
    s3.list_buckets.return_value = {"Buckets": buckets or []}

    # EC2 client
    ec2 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"InstanceTypes": instance_types or []}]
    ec2.get_paginator.return_value = paginator
    ec2.describe_regions.return_value = {"Regions": regions or []}

    def _client(service: str) -> MagicMock:
        if service == "sts":
            return sts
        if service == "s3":
            return s3
        return ec2

    session.client = _client
    return session


class TestAwsTestConnection:
    async def test_success(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session()
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.test_connection(CTX)
        assert result.success is True
        assert "123456789012" in result.message
        assert result.latency_ms is not None

    async def test_failure(self) -> None:
        connector = AwsConnector(provider_id="aws")
        with patch.object(
            connector,
            "_session",
            side_effect=Exception("invalid credentials"),
        ):
            result = await connector.test_connection(CTX)
        assert result.success is False
        assert "invalid credentials" in result.message


# ======================================================================
# list_storage_options
# ======================================================================


class TestAwsListStorage:
    async def test_returns_buckets(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            buckets=[
                {"Name": "my-data-bucket"},
                {"Name": "my-models-bucket"},
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_storage_options(CTX)
        assert len(result) == 2
        assert result[0].name == "my-data-bucket"
        assert result[0].provider_id == "aws"
        assert result[0].type == "s3"

    async def test_empty(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(buckets=[])
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_storage_options(CTX)
        assert result == []

    async def test_error_returns_empty(self) -> None:
        connector = AwsConnector(provider_id="aws")
        with patch.object(connector, "_session", side_effect=Exception("boom")):
            result = await connector.list_storage_options(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Session OK but S3 API call raises."""
        connector = AwsConnector(provider_id="aws")
        mock_session = MagicMock()
        s3 = MagicMock()
        s3.list_buckets.side_effect = Exception("API error")
        mock_session.client.return_value = s3
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_storage_options(CTX)
        assert result == []


# ======================================================================
# list_compute_instances
# ======================================================================


class TestAwsListCompute:
    async def test_returns_instances(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            instance_types=[
                {
                    "InstanceType": "m5.xlarge",
                    "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384},
                },
                {
                    "InstanceType": "p3.2xlarge",
                    "VCpuInfo": {"DefaultVCpus": 8},
                    "MemoryInfo": {"SizeInMiB": 61440},
                    "GpuInfo": {
                        "Gpus": [{"Name": "V100", "Count": 1}],
                    },
                },
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_compute_instances(CTX)
        assert len(result) == 2

        m5 = next(i for i in result if i.name == "m5.xlarge")
        assert m5.vcpus == 4
        assert m5.memory_gb == 16.0
        assert m5.gpu_count == 0

        p3 = next(i for i in result if i.name == "p3.2xlarge")
        assert p3.gpu_type == "V100"
        assert p3.gpu_count == 1

    async def test_gpu_only_filter(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            instance_types=[
                {
                    "InstanceType": "m5.xlarge",
                    "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384},
                },
                {
                    "InstanceType": "p3.2xlarge",
                    "VCpuInfo": {"DefaultVCpus": 8},
                    "MemoryInfo": {"SizeInMiB": 61440},
                    "GpuInfo": {
                        "Gpus": [{"Name": "V100", "Count": 1}],
                    },
                },
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_compute_instances(CTX, gpu_only=True)
        assert len(result) == 1
        assert result[0].gpu_count > 0

    async def test_region_override(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            instance_types=[
                {
                    "InstanceType": "t3.micro",
                    "VCpuInfo": {"DefaultVCpus": 2},
                    "MemoryInfo": {"SizeInMiB": 1024},
                },
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_compute_instances(CTX, region="us-east-1")
        assert result[0].region == "us-east-1"

    async def test_error_returns_empty(self) -> None:
        connector = AwsConnector(provider_id="aws")
        with patch.object(connector, "_session", side_effect=Exception("boom")):
            result = await connector.list_compute_instances(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Session OK but EC2 paginator raises."""
        connector = AwsConnector(provider_id="aws")
        mock_session = MagicMock()
        ec2 = MagicMock()
        ec2.get_paginator.side_effect = Exception("API error")
        mock_session.client.return_value = ec2
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_compute_instances(CTX)
        assert result == []

    async def test_gpu_family_fallback(self) -> None:
        """GPU detected from instance family name when GpuInfo is absent."""
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            instance_types=[
                {
                    "InstanceType": "g4dn.xlarge",
                    "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384},
                },
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_compute_instances(CTX)
        assert result[0].gpu_type == "T4"
        assert result[0].gpu_count == 1


# ======================================================================
# list_regions
# ======================================================================


class TestAwsListRegions:
    async def test_returns_regions(self) -> None:
        connector = AwsConnector(provider_id="aws")
        mock_session = _make_mock_session(
            regions=[
                {"RegionName": "eu-west-1"},
                {"RegionName": "us-east-1"},
            ],
        )
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_regions(CTX)
        assert len(result) == 2
        assert result[0].id == "eu-west-1"

    async def test_error_returns_empty(self) -> None:
        connector = AwsConnector(provider_id="aws")
        with patch.object(connector, "_session", side_effect=Exception("boom")):
            result = await connector.list_regions(CTX)
        assert result == []

    async def test_api_error_returns_empty(self) -> None:
        """Session OK but EC2 describe_regions raises."""
        connector = AwsConnector(provider_id="aws")
        mock_session = MagicMock()
        ec2 = MagicMock()
        ec2.describe_regions.side_effect = Exception("API error")
        mock_session.client.return_value = ec2
        with patch.object(connector, "_session", return_value=mock_session):
            result = await connector.list_regions(CTX)
        assert result == []


# ======================================================================
# GPU families dict
# ======================================================================


class TestGpuFamilies:
    def test_known_families(self) -> None:
        assert "p3" in _GPU_FAMILIES
        assert "g4dn" in _GPU_FAMILIES
        assert _GPU_FAMILIES["p3"] == "V100"


# ======================================================================
# _session requires boto3
# ======================================================================


class TestSessionRequiresBoto3:
    def test_raises_import_error(self) -> None:
        connector = AwsConnector()
        with pytest.raises(ImportError, match="boto3"):
            connector._session(CTX)
