"""Microsoft Azure cloud training provider.

Uses the Azure SDK (``azure-identity``, ``azure-mgmt-compute``,
``azure-mgmt-storage``, ``azure-mgmt-network``) to provision GPU-enabled
Virtual Machines, upload training code to Azure Blob Storage, monitor job
execution, and collect artifacts.  All blocking SDK calls are dispatched
via :func:`asyncio.to_thread` so the provider remains fully async.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pathlib
import shutil
import tarfile
import tempfile
import time
import uuid
from typing import Any

import httpx

from artenic_ai_platform.providers.base import (
    CloudJobStatus,
    InstanceType,
    JobStatus,
    TrainingSpec,
)
from artenic_ai_platform.providers.cloud_base import CloudProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional Azure SDK packages
# ---------------------------------------------------------------------------

try:
    from azure.identity import ClientSecretCredential, DefaultAzureCredential  # pragma: no cover

    _HAS_IDENTITY = True  # pragma: no cover
except ImportError:
    _HAS_IDENTITY = False

try:
    from azure.mgmt.compute import ComputeManagementClient  # pragma: no cover

    _HAS_COMPUTE = True  # pragma: no cover
except ImportError:
    _HAS_COMPUTE = False

try:
    import azure.mgmt.storage  # noqa: F401 — required dependency  # pragma: no cover
    from azure.storage.blob import BlobServiceClient  # pragma: no cover

    _HAS_STORAGE = True  # pragma: no cover
except ImportError:
    _HAS_STORAGE = False

try:
    from azure.mgmt.network import NetworkManagementClient  # pragma: no cover

    _HAS_NETWORK = True  # pragma: no cover
except ImportError:
    _HAS_NETWORK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USD_TO_EUR = 0.92

_DEFAULT_VM_IMAGE = {
    "publisher": "Canonical",
    "offer": "0001-com-ubuntu-server-jammy",
    "sku": "22_04-lts-gen2",
    "version": "latest",
}

_AZURE_RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"

_STARTUP_SCRIPT_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

echo "===== Artenic AI Training Bootstrap ====="
echo "Job ID: {job_id}"

# Install Azure CLI if not present
if ! command -v az &>/dev/null; then
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash
fi

{env_lines}
export ARTENIC_JOB_ID="{job_id}"

# Download training code from Azure Blob Storage
mkdir -p /opt/artenic/training
cd /opt/artenic/training
az storage blob download-batch \\
    --destination . \\
    --source {container_name} \\
    --pattern "{code_prefix}/*" \\
    --account-name {storage_account} \\
    --auth-mode login \\
    --no-progress 2>/dev/null || true

# Signal that training is starting
echo '{{"status": "running", "started_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}}' > /tmp/status.json
az storage blob upload \\
    --container-name {container_name} \\
    --name "artifacts/{job_id}/status.json" \\
    --file /tmp/status.json \\
    --account-name {storage_account} \\
    --auth-mode login \\
    --overwrite \\
    --no-progress 2>/dev/null || true

# Run training
{train_command} 2>&1 | tee /opt/artenic/training.log
TRAIN_EXIT=$?

# Upload artifacts
if [ -d /opt/artenic/training/output ]; then
    az storage blob upload-batch \\
        --destination {container_name} \\
        --source /opt/artenic/training/output \\
        --destination-path "artifacts/{job_id}/output" \\
        --account-name {storage_account} \\
        --auth-mode login \\
        --no-progress 2>/dev/null || true
fi

az storage blob upload \\
    --container-name {container_name} \\
    --name "artifacts/{job_id}/training.log" \\
    --file /opt/artenic/training.log \\
    --account-name {storage_account} \\
    --auth-mode login \\
    --overwrite \\
    --no-progress 2>/dev/null || true

if [ $TRAIN_EXIT -eq 0 ]; then
    STATUS="completed"
else
    STATUS="failed"
fi

FINISHED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > /tmp/status.json <<EOJSON
{{"status":"$STATUS","finished_at":"$FINISHED","exit_code":$TRAIN_EXIT}}
EOJSON
az storage blob upload \\
    --container-name {container_name} \\
    --name "artifacts/{job_id}/status.json" \\
    --file /tmp/status.json \\
    --account-name {storage_account} \\
    --auth-mode login \\
    --overwrite \\
    --no-progress 2>/dev/null || true
"""


class _AzureJobState:
    """Tracks the state of a training job running on Azure."""

    __slots__ = (
        "code_prefix",
        "created_at",
        "ip_name",
        "nic_name",
        "os_disk_name",
        "resource_group",
        "spec",
        "vm_name",
    )

    def __init__(
        self,
        *,
        vm_name: str,
        nic_name: str,
        ip_name: str,
        os_disk_name: str,
        resource_group: str,
        created_at: float,
        spec: TrainingSpec,
        code_prefix: str,
    ) -> None:
        self.vm_name = vm_name
        self.nic_name = nic_name
        self.ip_name = ip_name
        self.os_disk_name = os_disk_name
        self.resource_group = resource_group
        self.created_at = created_at
        self.spec = spec
        self.code_prefix = code_prefix


class AzureProvider(CloudProvider):
    """Microsoft Azure training provider.

    Provisions Azure Virtual Machines with optional GPU accelerators,
    uploads training code to Azure Blob Storage, monitors jobs via the
    Compute SDK, and collects artifacts from Blob Storage.

    Parameters
    ----------
    subscription_id:
        Azure subscription ID.
    tenant_id:
        Azure Active Directory tenant ID.  Required when using
        ``client_id`` / ``client_secret`` authentication.
    client_id:
        Service principal application (client) ID.  When omitted the
        provider falls back to ``DefaultAzureCredential``.
    client_secret:
        Service principal client secret.
    resource_group:
        Azure resource group for all created resources.
    region:
        Azure region / location for compute resources (default
        ``westeurope``).
    storage_account:
        Azure Storage account name used for Blob Storage.
    container_name:
        Blob container used for code upload and artifact storage.
    vm_size:
        Default VM size when the training spec does not specify one.
    admin_username:
        Linux admin username on provisioned VMs.
    ssh_public_key_path:
        Path to an SSH public key file injected into the VM for debug access.
    vm_image:
        Override for the VM OS image (dict with publisher / offer / sku /
        version keys).
    """

    def __init__(
        self,
        *,
        subscription_id: str,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        resource_group: str,
        region: str = "westeurope",
        storage_account: str,
        container_name: str = "artenic-training",
        vm_size: str = "Standard_NC6s_v3",
        admin_username: str = "artenic",
        ssh_public_key_path: str | None = None,
        vm_image: dict[str, str] | None = None,
    ) -> None:
        super().__init__()

        # Validate SDK availability eagerly so callers get clear errors.
        if not _HAS_IDENTITY:
            raise RuntimeError(
                "azure-identity is required for the Azure provider. "
                "Install it with:  pip install azure-identity"
            )
        if not _HAS_COMPUTE:
            raise RuntimeError(
                "azure-mgmt-compute is required for the Azure provider. "
                "Install it with:  pip install azure-mgmt-compute"
            )
        if not _HAS_STORAGE:
            raise RuntimeError(
                "azure-mgmt-storage and azure-storage-blob are required "
                "for the Azure provider. Install them with:  "
                "pip install azure-mgmt-storage azure-storage-blob"
            )
        if not _HAS_NETWORK:
            raise RuntimeError(
                "azure-mgmt-network is required for the Azure provider. "
                "Install it with:  pip install azure-mgmt-network"
            )

        self._subscription_id = subscription_id
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._resource_group = resource_group
        self._region = region
        self._storage_account = storage_account
        self._container_name = container_name
        self._vm_size = vm_size
        self._admin_username = admin_username
        self._ssh_public_key_path = ssh_public_key_path
        self._vm_image = vm_image or dict(_DEFAULT_VM_IMAGE)

        # SDK clients -- initialised in _connect()
        self._credential: Any = None
        self._compute_client: Any = None
        self._network_client: Any = None
        self._blob_service_client: Any = None
        self._container_client: Any = None

        # Job tracking: provider_job_id -> _AzureJobState
        self._jobs: dict[str, _AzureJobState] = {}

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "azure"

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Authenticate and initialise Azure SDK management clients."""

        def _create_clients() -> tuple[Any, Any, Any, Any, Any]:
            # Choose credential type based on supplied parameters.
            if self._client_id and self._client_secret and self._tenant_id:
                credential = ClientSecretCredential(
                    tenant_id=self._tenant_id,
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                )
                logger.info(
                    "[azure] Using ClientSecretCredential (tenant=%s, client=%s)",
                    self._tenant_id,
                    self._client_id,
                )
            else:
                credential = DefaultAzureCredential()
                logger.info("[azure] Using DefaultAzureCredential")

            compute_client = ComputeManagementClient(credential, self._subscription_id)
            network_client = NetworkManagementClient(credential, self._subscription_id)

            blob_account_url = f"https://{self._storage_account}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(
                account_url=blob_account_url,
                credential=credential,
            )

            # Ensure the container exists.
            container_client = blob_service_client.get_container_client(self._container_name)
            try:
                container_client.get_container_properties()
            except Exception:
                container_client.create_container()
                logger.info("[azure] Created blob container %s", self._container_name)

            return (
                credential,
                compute_client,
                network_client,
                blob_service_client,
                container_client,
            )

        (
            self._credential,
            self._compute_client,
            self._network_client,
            self._blob_service_client,
            self._container_client,
        ) = await asyncio.to_thread(_create_clients)

        logger.info(
            "[azure] Connected to subscription %s in region %s",
            self._subscription_id,
            self._region,
        )

    async def _disconnect(self) -> None:
        """Release SDK clients and credential."""
        try:
            if self._blob_service_client is not None:
                self._blob_service_client.close()
        except Exception:
            logger.debug("[azure] Error closing blob client", exc_info=True)
        finally:
            self._credential = None
            self._compute_client = None
            self._network_client = None
            self._blob_service_client = None
            self._container_client = None
            logger.info("[azure] Disconnected")

    # ------------------------------------------------------------------
    # Instance listing & pricing
    # ------------------------------------------------------------------

    async def _list_instances(
        self,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[InstanceType]:
        """Query Azure for available VM sizes and live retail pricing.

        Uses ``compute_client.virtual_machine_sizes.list(location)`` for
        hardware specs and the public Azure Retail Prices REST API for
        on-demand pricing.
        """
        target_region = region or self._region

        def _fetch_vm_sizes() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for size in self._compute_client.virtual_machine_sizes.list(
                location=target_region,
            ):
                results.append(
                    {
                        "name": size.name,
                        "vcpus": size.number_of_cores,
                        "memory_mb": size.memory_in_mb,
                        "max_data_disk_count": size.max_data_disk_count,
                        "os_disk_size_mb": size.os_disk_size_in_mb,
                        "resource_disk_size_mb": size.resource_disk_size_in_mb,
                    }
                )
            return results

        raw_sizes = await asyncio.to_thread(_fetch_vm_sizes)
        logger.info(
            "[azure] Fetched %d VM sizes from region %s",
            len(raw_sizes),
            target_region,
        )

        # Fetch pricing from the public Azure Retail Prices API
        pricing_map = await self._fetch_retail_pricing(
            [s["name"] for s in raw_sizes],
            target_region,
        )

        instances: list[InstanceType] = []
        for size in raw_sizes:
            name: str = size["name"]
            vcpus: int = size["vcpus"]
            memory_gb = round(size["memory_mb"] / 1024.0, 2)

            # Infer GPU info from the VM size name.
            gpu_type, gpu_count = _parse_gpu_from_vm_name(name)

            if gpu_only and gpu_count == 0:
                continue

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

        logger.info(
            "[azure] Listed %d instance types (gpu_only=%s)",
            len(instances),
            gpu_only,
        )
        return instances

    async def _fetch_retail_pricing(
        self,
        vm_names: list[str],
        region: str,
    ) -> dict[str, tuple[float, float | None]]:
        """Retrieve on-demand and spot prices from the Azure Retail Prices API.

        The API is public and requires no authentication.  We page through
        results filtering by ``serviceName eq 'Virtual Machines'`` and the
        target ARM region.

        Returns a dict mapping VM size name to
        ``(on_demand_usd, spot_usd | None)``.
        """
        name_set = set(vm_names)
        on_demand: dict[str, float] = {}
        spot: dict[str, float] = {}

        # Build OData filter -- the API supports $filter on serviceName
        # and armRegionName.  We fetch all Virtual Machines prices for the
        # region and post-filter by VM name.
        odata_filter = (
            f"serviceName eq 'Virtual Machines' "
            f"and armRegionName eq '{region}' "
            f"and priceType eq 'Consumption'"
        )

        async def _fetch_pages() -> None:
            url: str | None = f"{_AZURE_RETAIL_PRICES_URL}?$filter={odata_filter}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                while url:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()

                    for item in data.get("Items", []):
                        arm_sku: str = item.get("armSkuName", "")
                        if arm_sku not in name_set:
                            continue

                        unit_price: float = item.get("retailPrice", 0.0)
                        sku_name_lower: str = item.get("skuName", "").lower()
                        meter_name_lower: str = item.get("meterName", "").lower()

                        # Skip non-Linux / Windows entries
                        if "windows" in sku_name_lower:
                            continue

                        is_spot = "spot" in sku_name_lower or "spot" in meter_name_lower
                        is_low_priority = "low priority" in sku_name_lower

                        if is_spot:
                            if arm_sku not in spot or unit_price < spot[arm_sku]:
                                spot[arm_sku] = unit_price
                        elif not is_low_priority and (
                            arm_sku not in on_demand or unit_price < on_demand[arm_sku]
                        ):
                            on_demand[arm_sku] = unit_price

                    url = data.get("NextPageLink")

        try:
            await _fetch_pages()
        except Exception as exc:
            logger.warning("[azure] Failed to fetch retail pricing: %s", exc)

        merged: dict[str, tuple[float, float | None]] = {}
        for name in vm_names:
            od = on_demand.get(name, 0.0)
            sp = spot.get(name)
            merged[name] = (od, sp)
        return merged

    # ------------------------------------------------------------------
    # Code upload
    # ------------------------------------------------------------------

    async def _upload_code(self, spec: TrainingSpec) -> str:
        """Package and upload training code to Azure Blob Storage.

        The code is tarred and uploaded to
        ``<container>/training/<service>/<model>/<short_uuid>/code.tar.gz``.

        Returns the blob URI prefix.
        """
        job_prefix = f"{spec.service}/{spec.model}/{uuid.uuid4().hex[:8]}"
        blob_prefix = f"training/{job_prefix}"
        blob_name = f"{blob_prefix}/code.tar.gz"
        source_dir = spec.config.get("source_dir", ".")

        def _package_and_upload() -> str:
            tmp_dir = tempfile.mkdtemp(prefix="artenic_azure_")
            try:
                archive_path = os.path.join(tmp_dir, "code.tar.gz")
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(source_dir, arcname=".")

                blob_client = self._container_client.get_blob_client(blob_name)
                with open(archive_path, "rb") as fh:
                    blob_client.upload_blob(fh, overwrite=True)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return blob_prefix

        code_prefix = await asyncio.to_thread(_package_and_upload)
        uri = (
            f"https://{self._storage_account}.blob.core.windows.net/"
            f"{self._container_name}/{blob_name}"
        )
        logger.info("[azure] Uploaded code to %s", uri)
        return code_prefix

    # ------------------------------------------------------------------
    # Provision & start
    # ------------------------------------------------------------------

    async def _provision_and_start(self, spec: TrainingSpec) -> str:
        """Create an Azure VM with a startup script and begin training.

        Creates a public IP, NIC, and VM.  Returns a provider job ID
        (format: ``azure-<short_uuid>``).
        """
        provider_job_id = f"azure-{uuid.uuid4().hex[:8]}"
        vm_size = spec.instance_type or self._vm_size
        target_region = spec.region or self._region
        resource_group = self._resource_group

        vm_name = f"artenic-{provider_job_id}"
        ip_name = f"{vm_name}-ip"
        nic_name = f"{vm_name}-nic"

        # Build the startup script.
        code_prefix = spec.config.get("_code_prefix", f"training/{spec.service}/{spec.model}")
        env_vars = spec.config.get("env", {})
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items())
        train_command = spec.config.get("train_command", "python train.py")

        startup_script = _STARTUP_SCRIPT_TEMPLATE.format(
            job_id=provider_job_id,
            env_lines=env_lines,
            container_name=self._container_name,
            code_prefix=code_prefix,
            storage_account=self._storage_account,
            train_command=train_command,
        )

        # Step 1: Create public IP address
        def _create_ip() -> Any:
            poller = self._network_client.public_ip_addresses.begin_create_or_update(
                resource_group,
                ip_name,
                {
                    "location": target_region,
                    "sku": {"name": "Standard"},
                    "public_ip_allocation_method": "Static",
                    "public_ip_address_version": "IPv4",
                    "tags": {
                        "artenic:job_id": provider_job_id,
                        "managed-by": "artenic-ai-platform",
                    },
                },
            )
            return poller.result()

        ip_resource = await asyncio.to_thread(_create_ip)
        logger.info("[azure] Created public IP %s", ip_name)

        # Step 2: Create NIC (using default VNet/subnet)
        def _create_nic() -> Any:
            # Look up the default subnet.
            vnets = list(self._network_client.virtual_networks.list(resource_group))
            if not vnets:
                raise RuntimeError(
                    f"No virtual networks found in resource group "
                    f"{resource_group!r}.  Create a VNet + subnet first."
                )
            vnet = vnets[0]
            subnets = list(self._network_client.subnets.list(resource_group, vnet.name))
            if not subnets:
                raise RuntimeError(f"No subnets found in VNet {vnet.name!r}.")
            subnet = subnets[0]

            poller = self._network_client.network_interfaces.begin_create_or_update(
                resource_group,
                nic_name,
                {
                    "location": target_region,
                    "ip_configurations": [
                        {
                            "name": f"{nic_name}-ipconfig",
                            "subnet": {"id": subnet.id},
                            "public_ip_address": {"id": ip_resource.id},
                        }
                    ],
                    "tags": {
                        "artenic:job_id": provider_job_id,
                        "managed-by": "artenic-ai-platform",
                    },
                },
            )
            return poller.result()

        nic_resource = await asyncio.to_thread(_create_nic)
        logger.info("[azure] Created NIC %s", nic_name)

        # Step 3: Create the VM
        custom_data_b64 = base64.b64encode(startup_script.encode()).decode()

        # Prepare SSH key configuration if provided.
        ssh_config: dict[str, Any] | None = None
        if self._ssh_public_key_path:
            ssh_key_path = pathlib.Path(self._ssh_public_key_path)
            if ssh_key_path.exists():
                ssh_key_data = ssh_key_path.read_text().strip()
                ssh_config = {
                    "public_keys": [
                        {
                            "path": f"/home/{self._admin_username}/.ssh/authorized_keys",
                            "key_data": ssh_key_data,
                        }
                    ]
                }

        os_profile: dict[str, Any] = {
            "computer_name": vm_name,
            "admin_username": self._admin_username,
            "custom_data": custom_data_b64,
        }

        # SSH key auth is preferred; fall back to password if no key.
        if ssh_config:
            os_profile["linux_configuration"] = {
                "disable_password_authentication": True,
                "ssh": ssh_config,
            }
        else:
            # Generate a random password so the VM can be created without
            # SSH key.  The password is not stored or returned; access is
            # intended to be through the startup script only.
            random_password = f"Art3n1c!{uuid.uuid4().hex[:16]}"
            os_profile["admin_password"] = random_password
            os_profile["linux_configuration"] = {
                "disable_password_authentication": False,
            }

        vm_parameters: dict[str, Any] = {
            "location": target_region,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": self._vm_image["publisher"],
                    "offer": self._vm_image["offer"],
                    "sku": self._vm_image["sku"],
                    "version": self._vm_image["version"],
                },
                "os_disk": {
                    "name": f"{vm_name}-osdisk",
                    "caching": "ReadWrite",
                    "create_option": "FromImage",
                    "managed_disk": {
                        "storage_account_type": "Premium_LRS",
                    },
                },
            },
            "os_profile": os_profile,
            "network_profile": {"network_interfaces": [{"id": nic_resource.id}]},
            "tags": {
                "artenic:job_id": provider_job_id,
                "artenic:service": spec.service,
                "artenic:model": spec.model,
                "managed-by": "artenic-ai-platform",
            },
        }

        # Enable spot / low-priority if requested.
        if spec.is_spot:
            vm_parameters["priority"] = "Spot"
            vm_parameters["eviction_policy"] = "Deallocate"
            vm_parameters["billing_profile"] = {"max_price": -1}

        # Assign a system-managed identity so the VM can authenticate to
        # Blob Storage via ``az ... --auth-mode login``.
        vm_parameters["identity"] = {"type": "SystemAssigned"}

        def _create_vm() -> str:
            poller = self._compute_client.virtual_machines.begin_create_or_update(
                resource_group,
                vm_name,
                vm_parameters,
            )
            result = poller.result()
            return str(result.storage_profile.os_disk.name)

        os_disk_name = await asyncio.to_thread(_create_vm)

        logger.info(
            "[azure] Provisioned VM %s (size=%s, spot=%s) for job %s",
            vm_name,
            vm_size,
            spec.is_spot,
            provider_job_id,
        )

        # Store job state for later polling / cleanup.
        self._jobs[provider_job_id] = _AzureJobState(
            vm_name=vm_name,
            nic_name=nic_name,
            ip_name=ip_name,
            os_disk_name=os_disk_name,
            resource_group=resource_group,
            created_at=time.time(),
            spec=spec,
            code_prefix=code_prefix,
        )

        return provider_job_id

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_provider(self, provider_job_id: str) -> CloudJobStatus:
        """Check VM power state and read the status file from Blob Storage."""
        state = self._jobs.get(provider_job_id)
        if state is None:
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.FAILED,
                error=f"No state tracked for job {provider_job_id}",
            )

        # Fetch VM instance view for power state.
        def _get_instance_view() -> str:
            vm = self._compute_client.virtual_machines.get(
                state.resource_group,
                state.vm_name,
                expand="instanceView",
            )
            statuses = vm.instance_view.statuses if vm.instance_view else []
            power_state = "unknown"
            for s in statuses:
                if s.code and s.code.startswith("PowerState/"):
                    power_state = s.code.split("/", 1)[1]
                    break
            return power_state

        try:
            power_state = await asyncio.to_thread(_get_instance_view)
        except Exception as exc:
            error_msg = str(exc)
            if "ResourceNotFound" in error_msg or "not found" in error_msg.lower():
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.FAILED,
                    error="VM no longer exists",
                )
            logger.warning(
                "[azure] Failed to get VM status for %s: %s",
                provider_job_id,
                exc,
            )
            return CloudJobStatus(
                provider_job_id=provider_job_id,
                status=JobStatus.RUNNING,
                error=f"Could not query VM: {exc}",
            )

        # Try to read the status.json blob for richer information.
        blob_status = await self._read_blob_status(provider_job_id)
        elapsed = time.time() - state.created_at

        if blob_status:
            raw_status = blob_status.get("status", "")
            if raw_status == "completed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.COMPLETED,
                    artifacts_uri=(
                        f"https://{self._storage_account}.blob.core.windows.net/"
                        f"{self._container_name}/artifacts/{provider_job_id}/output/"
                    ),
                    duration_seconds=elapsed,
                )
            if raw_status == "failed":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.FAILED,
                    error=f"Training exited with code {blob_status.get('exit_code', '?')}",
                    duration_seconds=elapsed,
                )
            if raw_status == "running":
                return CloudJobStatus(
                    provider_job_id=provider_job_id,
                    status=JobStatus.RUNNING,
                    duration_seconds=elapsed,
                )

        # Fall back to VM power state mapping.
        _power_state_map: dict[str, JobStatus] = {
            "starting": JobStatus.PENDING,
            "running": JobStatus.RUNNING,
            "stopping": JobStatus.RUNNING,
            "stopped": JobStatus.FAILED,
            "deallocating": JobStatus.RUNNING,
            "deallocated": JobStatus.FAILED,
        }

        # A deallocated spot VM is likely preempted.
        if power_state == "deallocated" and state.spec.is_spot:
            job_status = JobStatus.PREEMPTED
        else:
            job_status = _power_state_map.get(power_state, JobStatus.PENDING)

        return CloudJobStatus(
            provider_job_id=provider_job_id,
            status=job_status,
            duration_seconds=elapsed,
        )

    async def _read_blob_status(
        self,
        provider_job_id: str,
    ) -> dict[str, Any] | None:
        """Attempt to read ``artifacts/<job_id>/status.json`` from Blob Storage."""

        def _read() -> dict[str, Any] | None:
            blob_name = f"artifacts/{provider_job_id}/status.json"
            blob_client = self._container_client.get_blob_client(blob_name)
            try:
                data = blob_client.download_blob().readall()
                result: dict[str, Any] = json.loads(data.decode())
                return result
            except Exception:
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
        """Download training artifacts from Azure Blob Storage.

        Returns the local directory path containing downloaded artifacts,
        or ``None`` if nothing was found.
        """
        prefix = f"artifacts/{provider_job_id}/output/"

        def _download() -> str | None:
            blobs = list(self._container_client.list_blobs(name_starts_with=prefix))
            if not blobs:
                logger.info("[azure] No artifacts found for job %s", provider_job_id)
                return None

            local_dir = os.path.join(
                tempfile.gettempdir(),
                "artenic_artifacts",
                provider_job_id,
            )
            os.makedirs(local_dir, exist_ok=True)

            for blob in blobs:
                relative = blob.name[len(prefix) :]
                if not relative:
                    continue
                local_path = os.path.join(local_dir, relative)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                blob_client = self._container_client.get_blob_client(blob.name)
                with open(local_path, "wb") as fh:
                    stream = blob_client.download_blob()
                    fh.write(stream.readall())

            return local_dir

        local_dir = await asyncio.to_thread(_download)
        if local_dir:
            logger.info("[azure] Artifacts downloaded to %s", local_dir)
        return local_dir

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup_compute(self, provider_job_id: str) -> None:
        """Delete VM, NIC, public IP, and OS disk for a completed job.

        Resources are deleted in dependency order: VM first, then NIC,
        then public IP, and finally the OS disk.
        """
        state = self._jobs.pop(provider_job_id, None)
        if state is None:
            logger.debug("[azure] No state to clean up for job %s", provider_job_id)
            return

        rg = state.resource_group

        # 1. Delete VM
        await self._delete_resource(
            "VM",
            lambda: self._compute_client.virtual_machines.begin_delete(rg, state.vm_name),
        )

        # 2. Delete NIC (depends on VM being deleted first)
        await self._delete_resource(
            "NIC",
            lambda: self._network_client.network_interfaces.begin_delete(rg, state.nic_name),
        )

        # 3. Delete public IP (depends on NIC being deleted first)
        await self._delete_resource(
            "PublicIP",
            lambda: self._network_client.public_ip_addresses.begin_delete(rg, state.ip_name),
        )

        # 4. Delete OS disk (depends on VM being deleted first)
        await self._delete_resource(
            "OSDisk",
            lambda: self._compute_client.disks.begin_delete(rg, state.os_disk_name),
        )

        logger.info("[azure] Cleaned up all resources for job %s", provider_job_id)

    async def _delete_resource(
        self,
        resource_type: str,
        delete_fn: Any,
    ) -> None:
        """Execute a delete operation and wait for completion.

        Swallows ``ResourceNotFound`` errors (resource already deleted).
        """

        def _do_delete() -> None:
            try:
                poller = delete_fn()
                poller.result()
            except Exception as exc:
                error_str = str(exc)
                if "ResourceNotFound" in error_str or "not found" in error_str.lower():
                    logger.debug("[azure] %s already deleted", resource_type)
                else:
                    raise

        try:
            await asyncio.to_thread(_do_delete)
            logger.debug("[azure] Deleted %s", resource_type)
        except Exception as exc:
            logger.warning("[azure] Failed to delete %s: %s", resource_type, exc)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _cancel_provider_job(self, provider_job_id: str) -> None:
        """Cancel a running job by deallocating and then deleting the VM.

        Deallocating first ensures that billing stops immediately, even
        if the delete operation takes a moment.
        """
        state = self._jobs.get(provider_job_id)
        if state is None:
            logger.warning("[azure] Cannot cancel -- unknown job %s", provider_job_id)
            return

        rg = state.resource_group

        # Attempt deallocation first (stops billing).
        def _deallocate() -> None:
            try:
                poller = self._compute_client.virtual_machines.begin_deallocate(rg, state.vm_name)
                poller.result()
                logger.info(
                    "[azure] Deallocated VM %s for job %s",
                    state.vm_name,
                    provider_job_id,
                )
            except Exception as exc:
                error_str = str(exc)
                if "ResourceNotFound" in error_str or "not found" in error_str.lower():
                    logger.info("[azure] VM %s already gone", state.vm_name)
                else:
                    logger.warning(
                        "[azure] Deallocation failed for %s: %s",
                        state.vm_name,
                        exc,
                    )

        await asyncio.to_thread(_deallocate)

        # Now delete the VM (full cleanup happens in _cleanup_compute).
        def _delete_vm() -> None:
            try:
                poller = self._compute_client.virtual_machines.begin_delete(rg, state.vm_name)
                poller.result()
                logger.info(
                    "[azure] Deleted VM %s for job %s",
                    state.vm_name,
                    provider_job_id,
                )
            except Exception as exc:
                error_str = str(exc)
                if "ResourceNotFound" in error_str or "not found" in error_str.lower():
                    pass
                else:
                    logger.warning(
                        "[azure] VM deletion failed for %s: %s",
                        state.vm_name,
                        exc,
                    )

        await asyncio.to_thread(_delete_vm)


# ======================================================================
# Helpers (module-level)
# ======================================================================

# GPU family patterns in Azure VM size names.  The mapping is:
# substring found in the *normalised* (lowercase) VM name -> (gpu_type, gpus_per_unit).
# The actual count is scaled by the numeric suffix (e.g. NC24 = 4x K80).
_GPU_FAMILIES: list[tuple[str, str, int]] = [
    # H100 series
    ("standard_nd96isr_h100_v5", "H100", 8),
    ("standard_nd96is_h100_v5", "H100", 8),
    ("standard_nc40ads_h100_v5", "H100", 1),
    ("standard_nc80adis_h100_v5", "H100", 2),
    # A100 series
    ("standard_nd96asr_v4", "A100", 8),
    ("standard_nd96amsr_a100_v4", "A100-80GB", 8),
    ("standard_nc24ads_a100_v4", "A100", 1),
    ("standard_nc48ads_a100_v4", "A100", 2),
    ("standard_nc96ads_a100_v4", "A100", 4),
    # A10 series
    ("standard_nc8ads_a10_v4", "A10", 1),
    ("standard_nc16ads_a10_v4", "A10", 1),
    ("standard_nc32ads_a10_v4", "A10", 2),
    # T4 series
    ("standard_nc4as_t4_v3", "T4", 1),
    ("standard_nc8as_t4_v3", "T4", 1),
    ("standard_nc16as_t4_v3", "T4", 1),
    ("standard_nc64as_t4_v3", "T4", 4),
    # V100 series
    ("standard_nc6s_v3", "V100", 1),
    ("standard_nc12s_v3", "V100", 2),
    ("standard_nc24s_v3", "V100", 4),
    ("standard_nc24rs_v3", "V100", 4),
    # M60 (NV series)
    ("standard_nv6", "M60", 1),
    ("standard_nv12", "M60", 2),
    ("standard_nv24", "M60", 4),
    ("standard_nv12s_v3", "M60", 1),
    ("standard_nv24s_v3", "M60", 2),
    ("standard_nv48s_v3", "M60", 4),
    # A10G / NVads A10 v5
    ("standard_nv6ads_a10_v5", "A10", 1),
    ("standard_nv12ads_a10_v5", "A10", 1),
    ("standard_nv18ads_a10_v5", "A10", 1),
    ("standard_nv36ads_a10_v5", "A10", 1),
    ("standard_nv36adms_a10_v5", "A10", 1),
    ("standard_nv72ads_a10_v5", "A10", 2),
    # K80 (NC classic)
    ("standard_nc6", "K80", 1),
    ("standard_nc12", "K80", 2),
    ("standard_nc24", "K80", 4),
    ("standard_nc24r", "K80", 4),
    # P40 (NC v2)
    ("standard_nc6s_v2", "P40", 1),
    ("standard_nc12s_v2", "P40", 2),
    ("standard_nc24s_v2", "P40", 4),
    ("standard_nc24rs_v2", "P40", 4),
    # P100 (ND)
    ("standard_nd6s", "P40", 1),
    ("standard_nd12s", "P40", 2),
    ("standard_nd24s", "P40", 4),
    ("standard_nd24rs", "P40", 4),
]


def _parse_gpu_from_vm_name(name: str) -> tuple[str | None, int]:
    """Infer GPU type and count from an Azure VM size name.

    Returns ``(gpu_type, gpu_count)`` or ``(None, 0)`` when the size
    does not have GPUs.
    """
    lower = name.lower()

    # Check exact / prefix matches against known GPU families.
    for pattern, gpu_type, gpu_count in _GPU_FAMILIES:
        if lower == pattern:
            return gpu_type, gpu_count

    # Heuristic fallback: if the name starts with "standard_n" and
    # contains "c" or "d" (NC, ND, NV families) it is likely GPU-enabled,
    # but we cannot determine the exact type.
    if lower.startswith("standard_n") and any(seg in lower for seg in ("nc", "nd", "nv")):
        return "GPU", 1

    return None, 0
