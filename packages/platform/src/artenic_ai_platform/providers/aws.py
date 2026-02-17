"""Amazon Web Services (AWS) cloud training provider.

Uses the boto3 SDK to provision EC2 instances, upload/download artifacts
via S3, and query live on-demand and spot pricing through the Pricing API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from typing import Any

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider

try:
    import boto3  # pragma: no cover
    from botocore.exceptions import BotoCoreError, ClientError  # pragma: no cover

    _HAS_BOTO3 = True  # pragma: no cover
except ImportError:
    _HAS_BOTO3 = False

logger = logging.getLogger(__name__)

# Approximate EUR/USD conversion factor used when the Pricing API
# returns USD values and we need to expose EUR to the platform.
_USD_TO_EUR = 0.92


class AWSProvider(CloudProvider):
    """AWS EC2 + S3 training provider.

    Parameters
    ----------
    access_key_id:
        AWS access key ID for authentication.
    secret_access_key:
        AWS secret access key for authentication.
    region:
        AWS region for compute resources (default ``eu-west-1``).
    bucket_name:
        S3 bucket used for code upload and artifact storage.
    default_instance_type:
        EC2 instance type to use when the training spec does not specify one.
    ami_id:
        Amazon Machine Image ID used to launch training instances.
    subnet_id:
        VPC subnet ID for the launched instances.
    security_group_id:
        Security group ID attached to launched instances.
    key_name:
        EC2 key-pair name for SSH access to instances.
    """

    def __init__(
        self,
        *,
        access_key_id: str,
        secret_access_key: str,
        region: str = "eu-west-1",
        bucket_name: str,
        default_instance_type: str,
        ami_id: str,
        subnet_id: str,
        security_group_id: str,
        key_name: str,
    ) -> None:
        super().__init__()
        if not _HAS_BOTO3:
            raise RuntimeError(
                "boto3 is required for the AWS provider. Install it with:  pip install boto3"
            )
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._region = region
        self._bucket_name = bucket_name
        self._default_instance_type = default_instance_type
        self._ami_id = ami_id
        self._subnet_id = subnet_id
        self._security_group_id = security_group_id
        self._key_name = key_name

        # SDK clients - initialised in _connect()
        self._ec2: Any = None
        self._s3: Any = None
        self._pricing: Any = None

        # Track launched instances for cleanup.  Maps provider_job_id -> EC2 instance id.
        self._instances: dict[str, str] = {}
        # Track spot request IDs separately.
        self._spot_requests: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "aws"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Create boto3 EC2, S3, and Pricing clients."""

        def _create_clients() -> tuple[Any, Any, Any]:
            session = boto3.Session(
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name=self._region,
            )
            ec2 = session.client("ec2")
            s3 = session.client("s3")
            # The Pricing API is only available in us-east-1 and ap-south-1.
            pricing = session.client("pricing", region_name="us-east-1")
            return ec2, s3, pricing

        self._ec2, self._s3, self._pricing = await asyncio.to_thread(_create_clients)
        logger.info("[aws] Connected to AWS in region %s", self._region)

    async def _disconnect(self) -> None:
        """Release SDK clients."""
        self._ec2 = None
        self._s3 = None
        self._pricing = None
        logger.info("[aws] Disconnected from AWS")

    # ------------------------------------------------------------------
    # Instance listing & pricing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query EC2 instance types and live pricing.

        Uses ``describe_instance_types`` for hardware specs and the
        Pricing ``get_products`` API for on-demand and spot pricing.
        """
        target_region = region or self._region

        def _fetch_instances() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            paginator = self._ec2.get_paginator("describe_instance_types")
            filters: list[dict[str, Any]] = []
            if gpu_only:
                filters.append({"Name": "accelerator-count", "Values": ["1", "2", "4", "8", "16"]})
            page_kwargs: dict[str, Any] = {}
            if filters:
                page_kwargs["Filters"] = filters
            for page in paginator.paginate(**page_kwargs):
                for it in page.get("InstanceTypes", []):
                    results.append(it)
            return results

        raw_instances = await asyncio.to_thread(_fetch_instances)
        logger.info("[aws] Fetched %d instance types from EC2", len(raw_instances))

        # Build a mapping for pricing lookups.
        instance_names = [it["InstanceType"] for it in raw_instances]
        pricing_map = await self._fetch_pricing(instance_names, target_region)

        instances: list[InstanceType] = []
        for it in raw_instances:
            name: str = it["InstanceType"]
            vcpus: int = it.get("VCpuInfo", {}).get("DefaultVCpus", 0)
            mem_mb: float = it.get("MemoryInfo", {}).get("SizeInMiB", 0)
            memory_gb = round(mem_mb / 1024.0, 2)

            # GPU information
            gpu_type: str | None = None
            gpu_count: int = 0
            gpus = it.get("GpuInfo", {}).get("Gpus", [])
            if gpus:
                gpu_type = gpus[0].get("Name")
                gpu_count = sum(g.get("Count", 0) for g in gpus)

            od_usd, spot_usd = pricing_map.get(name, (0.0, None))
            price_eur = round(od_usd * _USD_TO_EUR, 6)
            spot_eur = round(spot_usd * _USD_TO_EUR, 6) if spot_usd is not None else None

            instances.append(
                InstanceType(
                    name=name,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    price_per_hour_eur=price_eur,
                    spot_price_per_hour_eur=spot_eur,
                    region=target_region,
                    available=True,
                )
            )

        return instances

    async def _fetch_pricing(
        self,
        instance_names: list[str],
        region: str,
    ) -> dict[str, tuple[float, float | None]]:
        """Retrieve on-demand and spot prices for a list of instance types.

        Returns a dict mapping instance-type name to
        ``(on_demand_usd, spot_usd | None)``.
        """

        # AWS region code -> location label mapping (subset)
        region_label = _region_to_location(region)

        def _get_on_demand_prices() -> dict[str, float]:
            prices: dict[str, float] = {}
            for name in instance_names:
                try:
                    response = self._pricing.get_products(
                        ServiceCode="AmazonEC2",
                        Filters=[
                            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": name},
                            {"Type": "TERM_MATCH", "Field": "location", "Value": region_label},
                            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                        ],
                        MaxResults=1,
                    )
                    for price_json in response.get("PriceList", []):
                        price_data = (
                            json.loads(price_json) if isinstance(price_json, str) else price_json
                        )
                        on_demand = price_data.get("terms", {}).get("OnDemand", {})
                        for _term_key, term_val in on_demand.items():
                            for _dim_key, dim_val in term_val.get("priceDimensions", {}).items():
                                usd_str = dim_val.get("pricePerUnit", {}).get("USD", "0")
                                prices[name] = float(usd_str)
                                break
                            break
                except (BotoCoreError, ClientError) as exc:
                    logger.debug("[aws] Pricing lookup failed for %s: %s", name, exc)
            return prices

        def _get_spot_prices() -> dict[str, float]:
            prices: dict[str, float] = {}
            # describe_spot_price_history accepts up to ~100 instance types at once.
            batch_size = 50
            for i in range(0, len(instance_names), batch_size):
                batch = instance_names[i : i + batch_size]
                try:
                    response = self._ec2.describe_spot_price_history(
                        InstanceTypes=batch,
                        ProductDescriptions=["Linux/UNIX"],
                        MaxResults=len(batch),
                    )
                    for item in response.get("SpotPriceHistory", []):
                        itype = item.get("InstanceType", "")
                        price = float(item.get("SpotPrice", "0"))
                        # Keep the lowest observed price for each type.
                        if itype not in prices or price < prices[itype]:
                            prices[itype] = price
                except (BotoCoreError, ClientError) as exc:
                    logger.debug("[aws] Spot pricing lookup failed: %s", exc)
            return prices

        on_demand_map, spot_map = await asyncio.gather(
            asyncio.to_thread(_get_on_demand_prices),
            asyncio.to_thread(_get_spot_prices),
        )

        merged: dict[str, tuple[float, float | None]] = {}
        for name in instance_names:
            od = on_demand_map.get(name, 0.0)
            sp = spot_map.get(name)
            merged[name] = (od, sp)
        return merged

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Package the working directory and upload to S3.

        The tarball is uploaded to
        ``s3://<bucket>/training/<job_prefix>/<archive>.tar.gz``.

        Returns the S3 URI of the uploaded archive.
        """
        job_prefix = f"{spec.service}/{spec.model}/{uuid.uuid4().hex[:8]}"
        s3_key = f"training/{job_prefix}/code.tar.gz"

        source_dir = spec.config.get("source_dir", ".")

        def _package_and_upload() -> str:
            tmp_dir = tempfile.mkdtemp(prefix="artenic_aws_")
            try:
                archive_path = os.path.join(tmp_dir, "code.tar.gz")
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(source_dir, arcname=".")
                self._s3.upload_file(archive_path, self._bucket_name, s3_key)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return f"s3://{self._bucket_name}/{s3_key}"

        uri = await asyncio.to_thread(_package_and_upload)
        logger.info("[aws] Uploaded code to %s", uri)
        return uri

    # ------------------------------------------------------------------
    # Provision & start
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Launch an EC2 instance and begin training.

        Returns a provider job ID (format: ``aws-<short_uuid>``).
        The EC2 instance ID is stored internally for later polling and cleanup.
        """
        provider_job_id = f"aws-{uuid.uuid4().hex[:8]}"
        instance_type = spec.instance_type or self._default_instance_type

        # Build a user_data bootstrap script.
        user_data_script = self._build_user_data(spec, provider_job_id)
        user_data_b64 = base64.b64encode(user_data_script.encode()).decode()

        def _launch() -> str:
            run_kwargs: dict[str, Any] = {
                "ImageId": self._ami_id,
                "InstanceType": instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "KeyName": self._key_name,
                "SecurityGroupIds": [self._security_group_id],
                "SubnetId": self._subnet_id,
                "UserData": user_data_b64,
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Name", "Value": f"artenic-{provider_job_id}"},
                            {"Key": "artenic:job_id", "Value": provider_job_id},
                            {"Key": "artenic:service", "Value": spec.service},
                            {"Key": "artenic:model", "Value": spec.model},
                        ],
                    }
                ],
            }

            if spec.is_spot:
                run_kwargs["InstanceMarketOptions"] = {
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                }

            response = self._ec2.run_instances(**run_kwargs)
            instance_id: str = response["Instances"][0]["InstanceId"]
            return instance_id

        instance_id = await asyncio.to_thread(_launch)
        self._instances[provider_job_id] = instance_id

        # If spot, record the spot request ID for later polling.
        if spec.is_spot:
            await self._record_spot_request(provider_job_id, instance_id)

        logger.info(
            "[aws] Launched instance %s (%s) for job %s",
            instance_id,
            instance_type,
            provider_job_id,
        )
        return provider_job_id

    async def _record_spot_request(self, provider_job_id: str, instance_id: str) -> None:
        """Look up the spot instance request associated with an instance."""

        def _lookup() -> str | None:
            resp = self._ec2.describe_instances(InstanceIds=[instance_id])
            for res in resp.get("Reservations", []):
                for inst in res.get("Instances", []):
                    sir_id = inst.get("SpotInstanceRequestId")
                    if sir_id:
                        return str(sir_id)
            return None

        sir_id = await asyncio.to_thread(_lookup)
        if sir_id:
            self._spot_requests[provider_job_id] = sir_id

    def _build_user_data(self, spec: TrainingSpec, job_id: str) -> str:
        """Generate a cloud-init user_data bash script.

        The script downloads the training code from S3, runs the training
        command, and uploads artifacts back to S3 when finished.
        """
        s3_code_uri = f"s3://{self._bucket_name}/training/{spec.service}/{spec.model}"
        s3_artifacts = f"s3://{self._bucket_name}/artifacts/{job_id}"
        train_command = spec.config.get("train_command", "python train.py")
        env_vars = spec.config.get("env", {})

        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items())

        return f"""#!/bin/bash
set -euo pipefail

echo "=== Artenic AI Training Bootstrap ==="
echo "Job ID: {job_id}"

# Install AWS CLI if not present
if ! command -v aws &>/dev/null; then
    pip install awscli --quiet
fi

{env_lines}
export ARTENIC_JOB_ID="{job_id}"

# Download training code
mkdir -p /opt/artenic/training
cd /opt/artenic/training
aws s3 cp {s3_code_uri}/code.tar.gz code.tar.gz
tar xzf code.tar.gz

# Signal that training is starting
aws s3 cp /dev/stdin {s3_artifacts}/status.json <<STATUSEOF
{{"status": "running", "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
STATUSEOF

# Run training
{train_command} 2>&1 | tee /opt/artenic/training.log
TRAIN_EXIT=$?

# Upload artifacts
aws s3 sync /opt/artenic/training/output/ {s3_artifacts}/output/ || true
aws s3 cp /opt/artenic/training.log {s3_artifacts}/training.log || true

if [ $TRAIN_EXIT -eq 0 ]; then
    STATUS="completed"
else
    STATUS="failed"
fi

aws s3 cp /dev/stdin {s3_artifacts}/status.json <<STATUSEOF
{{"status": "$STATUS", "finished_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "exit_code": $TRAIN_EXIT}}
STATUSEOF

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region {self._region} || true
"""

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check the EC2 instance state and S3 status file."""
        instance_id = self._instances.get(provider_job_id)
        if not instance_id:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"No instance tracked for job {provider_job_id}",
            )

        # Check if this is a spot instance and whether it was interrupted.
        if provider_job_id in self._spot_requests:
            spot_status = await self._check_spot_status(provider_job_id)
            if spot_status is not None:
                return spot_status

        def _describe() -> dict[str, Any]:
            resp = self._ec2.describe_instances(InstanceIds=[instance_id])
            for reservation in resp.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    return dict(inst)
            return {}

        instance_info = await asyncio.to_thread(_describe)
        ec2_state = instance_info.get("State", {}).get("Name", "unknown")

        # Try to read the status file from S3 for richer information.
        s3_status = await self._read_s3_status(provider_job_id)

        if s3_status:
            raw_status = s3_status.get("status", "")
            if raw_status == "completed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.COMPLETED,
                    artifacts_uri=f"s3://{self._bucket_name}/artifacts/{provider_job_id}/output/",
                )
            if raw_status == "failed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.FAILED,
                    error=f"Training exited with code {s3_status.get('exit_code', '?')}",
                )
            if raw_status == "running":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.RUNNING,
                )

        # Fall back to EC2 instance state.
        status_map: dict[str, JobStatus] = {
            "pending": JobStatus.PENDING,
            "running": JobStatus.RUNNING,
            "shutting-down": JobStatus.RUNNING,
            "terminated": JobStatus.FAILED,
            "stopping": JobStatus.RUNNING,
            "stopped": JobStatus.FAILED,
        }
        mapped = status_map.get(ec2_state, JobStatus.PENDING)

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=mapped,
        )

    async def _check_spot_status(self, provider_job_id: str) -> CloudJobStatus | None:
        """Return a preempted status if the spot request was interrupted."""
        sir_id = self._spot_requests.get(provider_job_id)
        if not sir_id:
            return None

        def _describe_spot() -> dict[str, Any]:
            resp = self._ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[sir_id])
            requests = resp.get("SpotInstanceRequests", [])
            return requests[0] if requests else {}

        spot_info = await asyncio.to_thread(_describe_spot)
        spot_state = spot_info.get("State", "")
        spot_status_code = spot_info.get("Status", {}).get("Code", "")

        if spot_status_code in (
            "instance-terminated-by-price",
            "instance-terminated-capacity-oversubscribed",
            "instance-terminated-no-capacity",
            "instance-terminated-by-service",
            "instance-stopped-by-price",
            "instance-stopped-no-capacity",
        ):
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.PREEMPTED,
                error=f"Spot instance interrupted: {spot_status_code}",
            )

        if spot_state == "cancelled":
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.CANCELLED,
            )

        return None

    async def _read_s3_status(self, provider_job_id: str) -> dict[str, Any] | None:
        """Attempt to read the status.json file from the artifacts prefix."""

        def _read() -> dict[str, Any] | None:
            key = f"artifacts/{provider_job_id}/status.json"
            try:
                resp = self._s3.get_object(Bucket=self._bucket_name, Key=key)
                body = resp["Body"].read().decode()
                result: dict[str, Any] = json.loads(body)
                return result
            except (self._s3.exceptions.NoSuchKey, ClientError):
                return None

        return await asyncio.to_thread(_read)

    # ------------------------------------------------------------------
    # Artifact collection
    # ------------------------------------------------------------------

    async def _collect_artifacts(
        self,
        provider_job_id: str,
        status: CloudJobStatus,
    ) -> str | None:
        """Download training artifacts from S3 to a local directory.

        Returns the local directory path containing downloaded artifacts,
        or ``None`` if nothing was found.
        """
        s3_prefix = f"artifacts/{provider_job_id}/output/"

        def _download() -> str | None:
            # List objects under the prefix.
            resp = self._s3.list_objects_v2(Bucket=self._bucket_name, Prefix=s3_prefix)
            contents = resp.get("Contents", [])
            if not contents:
                return None

            local_dir = os.path.join(tempfile.gettempdir(), "artenic_artifacts", provider_job_id)
            os.makedirs(local_dir, exist_ok=True)

            for obj in contents:
                key: str = obj["Key"]
                relative = key[len(s3_prefix) :]
                if not relative:
                    continue
                local_path = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self._s3.download_file(self._bucket_name, key, local_path)

            return local_dir

        local_dir = await asyncio.to_thread(_download)
        if local_dir:
            logger.info("[aws] Artifacts downloaded to %s", local_dir)
        else:
            logger.info("[aws] No artifacts found for job %s", provider_job_id)
        return local_dir

    # ------------------------------------------------------------------
    # Cleanup & cancellation
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Terminate the EC2 instance associated with a job."""
        instance_id = self._instances.pop(provider_job_id, None)
        if not instance_id:
            logger.debug("[aws] No instance to clean up for job %s", provider_job_id)
            return

        def _terminate() -> None:
            try:
                self._ec2.terminate_instances(InstanceIds=[instance_id])
            except ClientError as exc:
                # Instance may already be terminated.
                error_code = exc.response.get("Error", {}).get("Code", "")
                if error_code != "InvalidInstanceID.NotFound":
                    raise

        await asyncio.to_thread(_terminate)
        self._spot_requests.pop(provider_job_id, None)
        logger.info(
            "[aws] Terminated instance %s for job %s",
            instance_id,
            provider_job_id,
        )

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by terminating the underlying EC2 instance.

        If the job was launched as a spot instance, the spot request is also
        cancelled to prevent AWS from relaunching the instance.
        """
        # Cancel the spot request first (if applicable).
        sir_id = self._spot_requests.get(provider_job_id)
        if sir_id:

            def _cancel_spot() -> None:
                try:
                    self._ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[sir_id])
                except ClientError as exc:
                    logger.debug(
                        "[aws] Failed to cancel spot request %s: %s",
                        sir_id,
                        exc,
                    )

            await asyncio.to_thread(_cancel_spot)

        # Terminate the instance (the cleanup method handles the actual call).
        instance_id = self._instances.get(provider_job_id)
        if instance_id:

            def _terminate() -> None:
                try:
                    self._ec2.terminate_instances(InstanceIds=[instance_id])
                except ClientError as exc:
                    error_code = exc.response.get("Error", {}).get("Code", "")
                    if error_code != "InvalidInstanceID.NotFound":
                        raise

            await asyncio.to_thread(_terminate)
            logger.info(
                "[aws] Cancelled job %s (instance %s)",
                provider_job_id,
                instance_id,
            )
        else:
            logger.warning(
                "[aws] No instance found for job %s during cancellation",
                provider_job_id,
            )


# ======================================================================
# Helpers
# ======================================================================

_REGION_LOCATION_MAP: dict[str, str] = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-central-2": "EU (Zurich)",
    "eu-north-1": "EU (Stockholm)",
    "eu-south-1": "EU (Milan)",
    "eu-south-2": "EU (Spain)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "sa-east-1": "South America (Sao Paulo)",
    "ca-central-1": "Canada (Central)",
    "me-south-1": "Middle East (Bahrain)",
    "af-south-1": "Africa (Cape Town)",
}


def _region_to_location(region: str) -> str:
    """Convert an AWS region code to the human-readable location label
    expected by the Pricing API.

    Falls back to the region code itself if no mapping is found.
    """
    return _REGION_LOCATION_MAP.get(region, region)
