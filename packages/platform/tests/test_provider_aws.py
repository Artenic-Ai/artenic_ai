"""Tests for artenic_ai_platform.providers.aws — AWSProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    JobStatus,
    TrainingSpec,
)

# ---------------------------------------------------------------------------
# Module path and shared patches
# ---------------------------------------------------------------------------

_MODULE = "artenic_ai_platform.providers.aws"


def _to_thread_patch():
    """Create a fresh asyncio.to_thread patch for each fixture call."""
    return patch(
        "asyncio.to_thread",
        new=AsyncMock(
            side_effect=lambda fn, *a, **kw: fn(*a, **kw),
        ),
    )


def _make_provider(**overrides):
    """Create an AWSProvider with mocked SDK availability."""
    from artenic_ai_platform.providers.aws import AWSProvider

    defaults = {
        "access_key_id": "AKIATEST",
        "secret_access_key": "secret",
        "region": "eu-west-1",
        "bucket_name": "test-bucket",
        "default_instance_type": "p3.2xlarge",
        "ami_id": "ami-test123",
        "subnet_id": "subnet-abc",
        "security_group_id": "sg-xyz",
        "key_name": "artenic-key",
    }
    defaults.update(overrides)
    return AWSProvider(**defaults)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_boto3():
    return MagicMock()


@pytest.fixture
def aws_patches(mock_boto3):
    """Apply all module-level patches for the AWS SDK."""
    # Build a real-looking ClientError class the provider can use
    client_error_cls = type(
        "ClientError",
        (Exception,),
        {
            "__init__": lambda self, response, operation_name: (
                Exception.__init__(self, str(response)),
                setattr(self, "response", response),
            )[-1],
            "response": {"Error": {}},
        },
    )
    patches = [
        patch(f"{_MODULE}._HAS_BOTO3", True),
        patch(f"{_MODULE}.boto3", mock_boto3, create=True),
        patch(f"{_MODULE}.BotoCoreError", Exception, create=True),
        patch(f"{_MODULE}.ClientError", client_error_cls, create=True),
        _to_thread_patch(),
    ]
    for p in patches:
        p.start()
    yield {
        "boto3": mock_boto3,
        "ClientError": client_error_cls,
    }
    for p in patches:
        p.stop()


# ======================================================================
# Tests
# ======================================================================


class TestAWSInit:
    def test_init_defaults(self, aws_patches):
        provider = _make_provider()
        assert provider.provider_name == "aws"
        assert provider._connected is False
        assert provider._access_key_id == "AKIATEST"
        assert provider._bucket_name == "test-bucket"

    def test_init_raises_without_boto3(self):
        with patch(f"{_MODULE}._HAS_BOTO3", False), pytest.raises(RuntimeError, match="boto3"):
            _make_provider()


class TestAWSProviderName:
    def test_provider_name(self, aws_patches):
        provider = _make_provider()
        assert provider.provider_name == "aws"


class TestAWSConnect:
    async def test_connect_creates_clients(self, aws_patches):
        mock_session = MagicMock()
        ec2 = MagicMock()
        s3 = MagicMock()
        pricing = MagicMock()
        mock_session.client.side_effect = [ec2, s3, pricing]
        aws_patches["boto3"].Session.return_value = mock_session

        provider = _make_provider()
        await provider._connect()

        assert provider._ec2 is ec2
        assert provider._s3 is s3
        assert provider._pricing is pricing


class TestAWSDisconnect:
    async def test_disconnect_clears_clients(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._pricing = MagicMock()

        await provider._disconnect()

        assert provider._ec2 is None
        assert provider._s3 is None
        assert provider._pricing is None


class TestAWSListInstances:
    async def test_list_instances(self, aws_patches):
        provider = _make_provider()

        # Mock ec2 paginator
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "p3.2xlarge",
                        "VCpuInfo": {"DefaultVCpus": 8},
                        "MemoryInfo": {"SizeInMiB": 61440},
                        "GpuInfo": {
                            "Gpus": [
                                {"Name": "V100", "Count": 1},
                            ]
                        },
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator

        # Mock pricing
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {"PriceList": []}

        # Mock spot pricing
        provider._ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}

        instances = await provider._list_instances()

        assert len(instances) == 1
        assert instances[0].name == "p3.2xlarge"
        assert instances[0].gpu_type == "V100"
        assert instances[0].gpu_count == 1

    async def test_list_instances_gpu_only(self, aws_patches):
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "p3.2xlarge",
                        "VCpuInfo": {"DefaultVCpus": 8},
                        "MemoryInfo": {"SizeInMiB": 61440},
                        "GpuInfo": {
                            "Gpus": [
                                {"Name": "V100", "Count": 1},
                            ]
                        },
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {"PriceList": []}
        provider._ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}

        instances = await provider._list_instances(gpu_only=True)
        assert len(instances) == 1


class TestAWSUploadCode:
    async def test_upload_code(self, aws_patches, tmp_path):
        provider = _make_provider()
        provider._s3 = MagicMock()

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="aws",
            config={"source_dir": str(tmp_path)},
        )

        # Create a dummy file so tar has something
        (tmp_path / "train.py").write_text("print('train')")

        uri = await provider._upload_code(spec)

        assert uri.startswith("s3://test-bucket/training/")
        assert uri.endswith("code.tar.gz")
        provider._s3.upload_file.assert_called_once()


class TestAWSProvisionAndStart:
    async def test_provision_on_demand(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-abc123"}]}

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="aws",
        )
        job_id = await provider._provision_and_start(spec)

        assert job_id.startswith("aws-")
        assert provider._instances[job_id] == "i-abc123"
        provider._ec2.run_instances.assert_called_once()

        # Verify spot options were not set
        call_kwargs = provider._ec2.run_instances.call_args[1]
        assert "InstanceMarketOptions" not in call_kwargs

    async def test_provision_spot_instance(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-spot123"}]}
        provider._ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-spot123",
                            "SpotInstanceRequestId": "sir-abc",
                        }
                    ]
                }
            ]
        }

        spec = TrainingSpec(
            service="nlp",
            model="bert",
            provider="aws",
            is_spot=True,
        )
        job_id = await provider._provision_and_start(spec)

        assert job_id.startswith("aws-")
        assert provider._instances[job_id] == "i-spot123"
        assert provider._spot_requests[job_id] == "sir-abc"

        call_kwargs = provider._ec2.run_instances.call_args[1]
        assert "InstanceMarketOptions" in call_kwargs


class TestAWSPollProvider:
    async def test_poll_running_from_ec2(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["job-1"] = "i-111"

        provider._ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-111",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        }

        # No status.json in S3 yet
        exc_cls = aws_patches["ClientError"]
        provider._s3.get_object.side_effect = exc_cls(
            {"Error": {"Code": "NoSuchKey"}},
            "GetObject",
        )
        # Also mock the exceptions attribute on the mock s3 client
        provider._s3.exceptions = MagicMock()
        provider._s3.exceptions.NoSuchKey = exc_cls

        status = await provider._poll_provider("job-1")
        assert status.status == JobStatus.RUNNING

    async def test_poll_completed_from_s3(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["job-2"] = "i-222"

        provider._ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-222",
                            "State": {"Name": "running"},
                        }
                    ]
                }
            ]
        }

        # S3 status.json says completed
        status_body = MagicMock()
        status_body.read.return_value = json.dumps({"status": "completed"}).encode()
        provider._s3.get_object.return_value = {"Body": status_body}
        provider._s3.exceptions = MagicMock()

        status = await provider._poll_provider("job-2")
        assert status.status == JobStatus.COMPLETED
        assert "output" in (status.artifacts_uri or "")

    async def test_poll_no_instance_tracked(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()

        status = await provider._poll_provider("unknown-job")
        assert status.status == JobStatus.FAILED
        assert "No instance" in (status.error or "")

    async def test_poll_spot_preempted(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["spot-job"] = "i-spot"
        provider._spot_requests["spot-job"] = "sir-123"

        provider._ec2.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-123",
                    "State": "active",
                    "Status": {
                        "Code": "instance-terminated-by-price",
                    },
                }
            ]
        }

        status = await provider._poll_provider("spot-job")
        assert status.status == JobStatus.PREEMPTED


class TestAWSCollectArtifacts:
    async def test_collect_artifacts(self, aws_patches, tmp_path):
        provider = _make_provider()
        provider._s3 = MagicMock()

        prefix = "artifacts/job-a/output/"
        provider._s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": f"{prefix}model.pt"},
            ]
        }

        dummy_status = CloudJobStatus(
            provider_job_id="job-a",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-a", dummy_status)

        assert result is not None
        provider._s3.download_file.assert_called_once()

    async def test_collect_artifacts_empty(self, aws_patches):
        provider = _make_provider()
        provider._s3 = MagicMock()

        provider._s3.list_objects_v2.return_value = {"Contents": []}

        dummy_status = CloudJobStatus(
            provider_job_id="job-b",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-b", dummy_status)
        assert result is None


class TestAWSCleanupCompute:
    async def test_cleanup_compute(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-c"] = "i-333"
        provider._spot_requests["job-c"] = "sir-333"

        await provider._cleanup_compute("job-c")

        provider._ec2.terminate_instances.assert_called_once_with(
            InstanceIds=["i-333"],
        )
        assert "job-c" not in provider._instances
        assert "job-c" not in provider._spot_requests

    async def test_cleanup_no_instance(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()

        # Should not raise
        await provider._cleanup_compute("missing-job")
        provider._ec2.terminate_instances.assert_not_called()

    async def test_cleanup_already_terminated(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-d"] = "i-444"

        error_cls = aws_patches["ClientError"]
        exc = error_cls(
            {"Error": {"Code": "InvalidInstanceID.NotFound"}},
            "TerminateInstances",
        )
        provider._ec2.terminate_instances.side_effect = exc

        # Should not raise — NotFound is swallowed
        await provider._cleanup_compute("job-d")


class TestAWSCancelProviderJob:
    async def test_cancel_with_spot(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-e"] = "i-555"
        provider._spot_requests["job-e"] = "sir-555"

        await provider._cancel_provider_job("job-e")

        provider._ec2.cancel_spot_instance_requests.assert_called_once_with(
            SpotInstanceRequestIds=["sir-555"],
        )
        provider._ec2.terminate_instances.assert_called_once_with(
            InstanceIds=["i-555"],
        )

    async def test_cancel_no_spot(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-f"] = "i-666"

        await provider._cancel_provider_job("job-f")

        provider._ec2.cancel_spot_instance_requests.assert_not_called()
        provider._ec2.terminate_instances.assert_called_once()

    async def test_cancel_no_instance(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()

        # Should not raise
        await provider._cancel_provider_job("ghost-job")
        provider._ec2.terminate_instances.assert_not_called()


# ======================================================================
# Additional tests for 100 % coverage
# ======================================================================


class TestAWSOnDemandPricingParsing:
    """Cover lines 246-257: on-demand pricing JSON parsing."""

    async def test_on_demand_pricing_parsed(self, aws_patches):
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "m5.large",
                        "VCpuInfo": {"DefaultVCpus": 2},
                        "MemoryInfo": {"SizeInMiB": 8192},
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator

        # Return a real pricing JSON structure (as a string, to exercise the
        # isinstance(price_json, str) branch on line 247).
        price_json = json.dumps(
            {
                "terms": {
                    "OnDemand": {
                        "term1": {
                            "priceDimensions": {
                                "dim1": {
                                    "pricePerUnit": {"USD": "0.096"},
                                }
                            }
                        }
                    }
                }
            }
        )
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {
            "PriceList": [price_json],
        }

        provider._ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}

        instances = await provider._list_instances()
        assert len(instances) == 1
        # Price should be 0.096 * 0.92 = 0.08832
        assert instances[0].price_per_hour_eur == round(0.096 * 0.92, 6)

    async def test_on_demand_pricing_as_dict(self, aws_patches):
        """Cover the else branch when price_json is already a dict (line 247)."""
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "m5.large",
                        "VCpuInfo": {"DefaultVCpus": 2},
                        "MemoryInfo": {"SizeInMiB": 8192},
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator

        # Return a dict directly (not a string)
        price_dict = {
            "terms": {
                "OnDemand": {
                    "term1": {
                        "priceDimensions": {
                            "dim1": {
                                "pricePerUnit": {"USD": "0.50"},
                            }
                        }
                    }
                }
            }
        }
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {
            "PriceList": [price_dict],
        }
        provider._ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}

        instances = await provider._list_instances()
        assert instances[0].price_per_hour_eur == round(0.50 * 0.92, 6)

    async def test_on_demand_pricing_lookup_failure(self, aws_patches):
        """Cover line 256-257: BotoCoreError during pricing lookup."""
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "m5.large",
                        "VCpuInfo": {"DefaultVCpus": 2},
                        "MemoryInfo": {"SizeInMiB": 8192},
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator
        provider._pricing = MagicMock()
        provider._pricing.get_products.side_effect = Exception("pricing error")
        provider._ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}

        instances = await provider._list_instances()
        assert len(instances) == 1
        # Price should default to 0
        assert instances[0].price_per_hour_eur == 0.0


class TestAWSSpotPricingParsing:
    """Cover lines 273-279: spot pricing parsing with lowest-price logic."""

    async def test_spot_pricing_parsed(self, aws_patches):
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "p3.2xlarge",
                        "VCpuInfo": {"DefaultVCpus": 8},
                        "MemoryInfo": {"SizeInMiB": 61440},
                        "GpuInfo": {"Gpus": [{"Name": "V100", "Count": 1}]},
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {"PriceList": []}

        # Return multiple spot prices — should keep the lowest
        provider._ec2.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [
                {"InstanceType": "p3.2xlarge", "SpotPrice": "2.50"},
                {"InstanceType": "p3.2xlarge", "SpotPrice": "1.80"},
                {"InstanceType": "p3.2xlarge", "SpotPrice": "2.10"},
            ]
        }

        instances = await provider._list_instances()
        assert len(instances) == 1
        expected_spot = round(1.80 * 0.92, 6)
        assert instances[0].spot_price_per_hour_eur == expected_spot

    async def test_spot_pricing_lookup_failure(self, aws_patches):
        """Cover line 278-279: BotoCoreError during spot pricing lookup."""
        provider = _make_provider()

        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "InstanceTypes": [
                    {
                        "InstanceType": "m5.large",
                        "VCpuInfo": {"DefaultVCpus": 2},
                        "MemoryInfo": {"SizeInMiB": 8192},
                    }
                ]
            }
        ]
        provider._ec2 = MagicMock()
        provider._ec2.get_paginator.return_value = mock_paginator
        provider._pricing = MagicMock()
        provider._pricing.get_products.return_value = {"PriceList": []}
        provider._ec2.describe_spot_price_history.side_effect = Exception("spot error")

        instances = await provider._list_instances()
        assert len(instances) == 1
        assert instances[0].spot_price_per_hour_eur is None


class TestAWSRecordSpotRequestNoSIR:
    """Cover line 404: _record_spot_request when no SpotInstanceRequestId found."""

    async def test_record_spot_request_no_sir(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()

        # Instance exists but has no SpotInstanceRequestId
        provider._ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-nosir",
                        }
                    ]
                }
            ]
        }

        await provider._record_spot_request("job-nosir", "i-nosir")
        # Should not store anything since sir_id is None
        assert "job-nosir" not in provider._spot_requests


class TestAWSPollDescribeEmpty:
    """Cover line 496: _describe returning empty dict (no reservations)."""

    async def test_poll_empty_describe(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["job-empty"] = "i-gone"

        # describe_instances returns no reservations
        provider._ec2.describe_instances.return_value = {"Reservations": []}

        # No status in S3 either
        exc_cls = aws_patches["ClientError"]
        provider._s3.get_object.side_effect = exc_cls(
            {"Error": {"Code": "NoSuchKey"}},
            "GetObject",
        )
        provider._s3.exceptions = MagicMock()
        provider._s3.exceptions.NoSuchKey = exc_cls

        status = await provider._poll_provider("job-empty")
        # ec2_state defaults to "unknown", mapped to PENDING
        assert status.status == JobStatus.PENDING


class TestAWSPollFailedAndRunningFromS3:
    """Cover lines 512-519: S3 status 'failed' and 'running' branches."""

    async def test_poll_failed_from_s3(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["job-fail"] = "i-fail"

        provider._ec2.describe_instances.return_value = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-fail", "State": {"Name": "running"}}]}
            ]
        }

        status_body = MagicMock()
        status_body.read.return_value = json.dumps({"status": "failed", "exit_code": 1}).encode()
        provider._s3.get_object.return_value = {"Body": status_body}
        provider._s3.exceptions = MagicMock()

        status = await provider._poll_provider("job-fail")
        assert status.status == JobStatus.FAILED
        assert "exit_code" in (status.error or "").lower() or "1" in (status.error or "")

    async def test_poll_running_from_s3(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._s3 = MagicMock()
        provider._instances["job-run"] = "i-run"

        provider._ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"InstanceId": "i-run", "State": {"Name": "pending"}}]}]
        }

        status_body = MagicMock()
        status_body.read.return_value = json.dumps({"status": "running"}).encode()
        provider._s3.get_object.return_value = {"Body": status_body}
        provider._s3.exceptions = MagicMock()

        status = await provider._poll_provider("job-run")
        assert status.status == JobStatus.RUNNING


class TestAWSCheckSpotStatusNoSIR:
    """Cover line 544: _check_spot_status returning None when sir_id missing."""

    async def test_check_spot_no_sir_id(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        # job is in spot_requests but the value is empty/falsy
        provider._spot_requests["job-no-sir"] = ""

        result = await provider._check_spot_status("job-no-sir")
        assert result is None


class TestAWSCheckSpotCancelled:
    """Cover lines 569-575: spot state 'cancelled' branch."""

    async def test_check_spot_cancelled(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-cancel-spot"] = "i-cancel-spot"
        provider._spot_requests["job-cancel-spot"] = "sir-cancel"

        provider._ec2.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-cancel",
                    "State": "cancelled",
                    "Status": {"Code": "request-canceled-and-instances-running"},
                }
            ]
        }

        # Call via _poll_provider so all branches are exercised
        provider._s3 = MagicMock()
        status = await provider._poll_provider("job-cancel-spot")
        assert status.status == JobStatus.CANCELLED


class TestAWSCheckSpotActive:
    """Cover line 575: spot request active (no preemption/cancellation)."""

    async def test_check_spot_active_returns_none(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._spot_requests["job-active-spot"] = "sir-active"

        provider._ec2.describe_spot_instance_requests.return_value = {
            "SpotInstanceRequests": [
                {
                    "SpotInstanceRequestId": "sir-active",
                    "State": "active",
                    "Status": {"Code": "fulfilled"},
                }
            ]
        }

        result = await provider._check_spot_status("job-active-spot")
        assert result is None


class TestAWSCollectArtifactsEmptyRelative:
    """Cover line 622: skipping artifact with empty relative path."""

    async def test_collect_artifacts_skips_prefix_only_key(self, aws_patches):
        provider = _make_provider()
        provider._s3 = MagicMock()

        prefix = "artifacts/job-skip/output/"
        provider._s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": prefix},  # This has empty relative path
                {"Key": f"{prefix}model.pt"},
            ]
        }

        dummy_status = CloudJobStatus(
            provider_job_id="job-skip",
            status=JobStatus.COMPLETED,
        )
        result = await provider._collect_artifacts("job-skip", dummy_status)

        assert result is not None
        # Only the model.pt file should be downloaded, not the prefix-only key
        assert provider._s3.download_file.call_count == 1


class TestAWSCleanupNonNotFoundRaises:
    """Cover line 654: cleanup with non-NotFound ClientError should re-raise."""

    async def test_cleanup_non_notfound_raises(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-err"] = "i-err"

        error_cls = aws_patches["ClientError"]
        exc = error_cls(
            {"Error": {"Code": "UnauthorizedOperation"}},
            "TerminateInstances",
        )
        provider._ec2.terminate_instances.side_effect = exc

        with pytest.raises(error_cls):
            await provider._cleanup_compute("job-err")


class TestAWSCancelSpotClientError:
    """Cover lines 677-678: cancel_spot_instance_requests ClientError is logged."""

    async def test_cancel_spot_client_error_logged(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-sp-err"] = "i-sp-err"
        provider._spot_requests["job-sp-err"] = "sir-sp-err"

        error_cls = aws_patches["ClientError"]
        provider._ec2.cancel_spot_instance_requests.side_effect = error_cls(
            {"Error": {"Code": "InvalidSpotInstanceRequestID.NotFound"}},
            "CancelSpotInstanceRequests",
        )

        # Should not raise — error is logged and continues to terminate
        await provider._cancel_provider_job("job-sp-err")
        provider._ec2.terminate_instances.assert_called_once()


class TestAWSCancelTerminateNonNotFoundRaises:
    """Cover lines 693-696: cancel terminate with non-NotFound ClientError."""

    async def test_cancel_terminate_non_notfound_raises(self, aws_patches):
        provider = _make_provider()
        provider._ec2 = MagicMock()
        provider._instances["job-t-err"] = "i-t-err"

        error_cls = aws_patches["ClientError"]
        exc = error_cls(
            {"Error": {"Code": "UnauthorizedOperation"}},
            "TerminateInstances",
        )
        provider._ec2.terminate_instances.side_effect = exc

        with pytest.raises(error_cls):
            await provider._cancel_provider_job("job-t-err")
