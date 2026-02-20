"""Dynamic cost estimation — queries live provider pricing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from artenic_ai_platform_providers.base import InstanceType, TrainingProvider

logger = logging.getLogger(__name__)


@dataclass
class InstanceEstimate:
    """Cost estimate for a single instance type."""

    provider: str
    instance_name: str
    vcpus: int
    memory_gb: float
    gpu_type: str | None
    gpu_count: int
    price_per_hour_eur: float
    spot_price_per_hour_eur: float | None
    estimated_cost_eur: float
    estimated_duration_hours: float
    region: str | None = None
    is_spot: bool = False


@dataclass
class CostPrediction:
    """Ranked cost estimates across providers."""

    estimates: list[InstanceEstimate] = field(default_factory=list)
    cheapest: InstanceEstimate | None = None
    fastest: InstanceEstimate | None = None


class CostPredictor:
    """Queries live pricing from providers to estimate training costs.

    Uses provider APIs (not static catalogs) for up-to-date pricing.
    Ranks instances by estimated cost and capability.
    """

    def __init__(
        self,
        providers: dict[str, TrainingProvider],
    ) -> None:
        self._providers = providers

    async def predict_cost(
        self,
        *,
        estimated_hours: float = 1.0,
        region: str | None = None,
        gpu_only: bool = False,
        use_spot: bool = False,
        provider_filter: list[str] | None = None,
        max_results: int = 20,
    ) -> CostPrediction:
        """Query all providers for live pricing and rank by cost.

        Parameters
        ----------
        estimated_hours:
            Expected training duration for cost calculation.
        region:
            Filter instances by region (provider-specific).
        gpu_only:
            Only return GPU-capable instances.
        use_spot:
            Use spot prices when available.
        provider_filter:
            Limit to specific providers. None = all enabled.
        max_results:
            Maximum estimates to return.
        """
        providers_to_query = (
            {k: v for k, v in self._providers.items() if k in provider_filter}
            if provider_filter
            else self._providers
        )

        estimates: list[InstanceEstimate] = []

        for name, provider in providers_to_query.items():
            try:
                instances = await provider.list_instance_types(region=region, gpu_only=gpu_only)
                for inst in instances:
                    if not inst.available:
                        continue
                    estimate = self._build_estimate(
                        provider_name=name,
                        instance=inst,
                        hours=estimated_hours,
                        use_spot=use_spot,
                    )
                    estimates.append(estimate)
            except Exception:
                logger.warning("Failed to query pricing from %s", name, exc_info=True)

        # Sort by estimated cost (cheapest first)
        estimates.sort(key=lambda e: e.estimated_cost_eur)

        # Trim to max_results
        estimates = estimates[:max_results]

        prediction = CostPrediction(estimates=estimates)
        if estimates:
            prediction.cheapest = estimates[0]
            # "Fastest" heuristic: most GPU/vCPU resources
            prediction.fastest = max(
                estimates,
                key=lambda e: (e.gpu_count, e.vcpus),
            )

        return prediction

    async def estimate_for_instance(
        self,
        provider_name: str,
        instance_name: str,
        *,
        estimated_hours: float = 1.0,
        region: str | None = None,
        use_spot: bool = False,
    ) -> InstanceEstimate | None:
        """Get cost estimate for a specific provider + instance type."""
        provider = self._providers.get(provider_name)
        if provider is None:
            return None

        try:
            instances = await provider.list_instance_types(region=region)
        except Exception:
            logger.warning("Failed to query %s", provider_name)
            return None

        match = next(
            (i for i in instances if i.name == instance_name),
            None,
        )
        if match is None:
            return None

        return self._build_estimate(
            provider_name=provider_name,
            instance=match,
            hours=estimated_hours,
            use_spot=use_spot,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_estimate(
        provider_name: str,
        instance: InstanceType,
        hours: float,
        use_spot: bool,
    ) -> InstanceEstimate:
        """Build an InstanceEstimate from a provider's InstanceType."""
        rate = instance.price_per_hour_eur
        is_spot = False
        spot_rate = instance.spot_price_per_hour_eur

        if use_spot and spot_rate is not None:
            rate = spot_rate
            is_spot = True

        return InstanceEstimate(
            provider=provider_name,
            instance_name=instance.name,
            vcpus=instance.vcpus,
            memory_gb=instance.memory_gb,
            gpu_type=instance.gpu_type,
            gpu_count=instance.gpu_count,
            price_per_hour_eur=instance.price_per_hour_eur,
            spot_price_per_hour_eur=spot_rate,
            estimated_cost_eur=rate * hours,
            estimated_duration_hours=hours,
            region=instance.region,
            is_spot=is_spot,
        )
