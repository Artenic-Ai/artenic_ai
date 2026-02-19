"""Abstract base for provider connectors.

Each connector wraps an open-source SDK (openstacksdk, boto3, …) and
exposes a uniform interface to test connectivity, list storage options,
compute instances, and regions.

Credentials are passed through to the provider's own API and never
transmitted to any third party.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artenic_ai_platform.providers_hub.schemas import (
        ComputeInstance,
        ConnectionTestResult,
        ProviderRegion,
        StorageOption,
    )


@dataclass(frozen=True)
class ConnectorContext:
    """Holds decrypted credentials and config for a single call."""

    credentials: dict[str, str]
    config: dict[str, str]


class ProviderConnector(ABC):
    """Interface implemented by every provider connector.

    All methods are async so blocking SDK calls can be dispatched
    via ``asyncio.to_thread``.
    """

    @abstractmethod
    async def test_connection(self, ctx: ConnectorContext) -> ConnectionTestResult:
        """Verify that the credentials are valid.

        Should perform a lightweight API call (e.g. list flavors,
        list buckets) and return success / failure with a message.
        """

    @abstractmethod
    async def list_storage_options(
        self,
        ctx: ConnectorContext,
    ) -> list[StorageOption]:
        """Return real storage containers / buckets for this account."""

    @abstractmethod
    async def list_compute_instances(
        self,
        ctx: ConnectorContext,
        *,
        region: str | None = None,
        gpu_only: bool = False,
    ) -> list[ComputeInstance]:
        """Return real compute flavors / instance types for this account."""

    @abstractmethod
    async def list_regions(
        self,
        ctx: ConnectorContext,
    ) -> list[ProviderRegion]:
        """Return regions available to this account."""
