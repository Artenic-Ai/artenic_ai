"""Tests for artenic_ai_platform.training.cost_predictor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from artenic_ai_platform.providers.mock import MockProvider
from artenic_ai_platform.training.cost_predictor import CostPredictor

# ======================================================================
# predict_cost
# ======================================================================


class TestPredictCostSingleProvider:
    """MockProvider returns cost estimates for its three instance types."""

    async def test_predict_cost_single_provider(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        prediction = await predictor.predict_cost(estimated_hours=2.0)

        # MockProvider has 3 available instances
        assert len(prediction.estimates) == 3
        assert prediction.cheapest is not None
        assert prediction.fastest is not None

        # All estimates should have provider = "mock"
        for est in prediction.estimates:
            assert est.provider == "mock"
            assert est.estimated_duration_hours == 2.0


class TestPredictCostGpuOnly:
    """gpu_only=True filters to GPU instances only."""

    async def test_predict_cost_gpu_only(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        prediction = await predictor.predict_cost(gpu_only=True)

        # MockProvider has 1 GPU instance (mock-gpu-a100)
        assert len(prediction.estimates) == 1
        assert prediction.estimates[0].gpu_type == "A100"
        assert prediction.estimates[0].gpu_count == 1


class TestPredictCostSpot:
    """use_spot=True uses spot prices when available."""

    async def test_predict_cost_spot(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        prediction = await predictor.predict_cost(estimated_hours=1.0, use_spot=True)

        # Find the GPU instance — it has a spot price
        gpu_est = next(e for e in prediction.estimates if e.instance_name == "mock-gpu-a100")
        assert gpu_est.is_spot is True
        assert gpu_est.estimated_cost_eur == pytest.approx(0.75)  # spot_price * 1h
        assert gpu_est.spot_price_per_hour_eur == pytest.approx(0.75)

        # CPU instances have no spot price, so they should NOT be spot
        cpu_est = next(e for e in prediction.estimates if e.instance_name == "mock-cpu-small")
        assert cpu_est.is_spot is False


class TestPredictCostCheapestAndFastest:
    """Cheapest and fastest fields are populated correctly."""

    async def test_predict_cost_cheapest_and_fastest(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        prediction = await predictor.predict_cost(estimated_hours=1.0)

        # Cheapest should be mock-cpu-small at EUR 0.01/hr
        assert prediction.cheapest is not None
        assert prediction.cheapest.instance_name == "mock-cpu-small"
        assert prediction.cheapest.estimated_cost_eur == pytest.approx(0.01)

        # Fastest = most GPU/vCPU => mock-gpu-a100 (1 GPU, 12 vCPU)
        assert prediction.fastest is not None
        assert prediction.fastest.instance_name == "mock-gpu-a100"


class TestPredictCostProviderFilter:
    """provider_filter limits which providers are queried."""

    async def test_predict_cost_provider_filter(self) -> None:
        provider1 = MockProvider()
        provider2 = MockProvider()
        predictor = CostPredictor({"mock1": provider1, "mock2": provider2})

        # Only query mock1
        prediction = await predictor.predict_cost(provider_filter=["mock1"])
        for est in prediction.estimates:
            assert est.provider == "mock1"


class TestPredictCostProviderError:
    """When a provider raises, the predictor degrades gracefully."""

    async def test_predict_cost_provider_error(self) -> None:
        good_provider = MockProvider()
        bad_provider = AsyncMock()
        bad_provider.list_instance_types = AsyncMock(side_effect=RuntimeError("API down"))

        predictor = CostPredictor({"good": good_provider, "bad": bad_provider})
        prediction = await predictor.predict_cost()

        # Only the good provider's instances should appear
        assert len(prediction.estimates) == 3
        for est in prediction.estimates:
            assert est.provider == "good"


# ======================================================================
# estimate_for_instance
# ======================================================================


class TestEstimateForInstance:
    """Specific instance lookup returns a valid estimate."""

    async def test_estimate_for_instance(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        est = await predictor.estimate_for_instance("mock", "mock-gpu-a100", estimated_hours=3.0)
        assert est is not None
        assert est.instance_name == "mock-gpu-a100"
        assert est.estimated_cost_eur == pytest.approx(2.50 * 3.0)
        assert est.gpu_type == "A100"


class TestEstimateForInstanceNotFound:
    """Instance lookup returns None when the instance name doesn't exist."""

    async def test_estimate_for_instance_not_found(self) -> None:
        provider = MockProvider()
        predictor = CostPredictor({"mock": provider})

        est = await predictor.estimate_for_instance("mock", "nonexistent-instance")
        assert est is None


class TestEstimateForUnknownProvider:
    """Instance lookup returns None for an unknown provider name."""

    async def test_estimate_for_unknown_provider(self) -> None:
        predictor = CostPredictor({})

        est = await predictor.estimate_for_instance("does-not-exist", "mock-cpu-small")
        assert est is None


class TestPredictCostUnavailableInstance:
    """Unavailable instances are filtered out (line 97)."""

    async def test_predict_cost_unavailable_instance(self) -> None:
        from artenic_ai_platform.providers.base import InstanceType

        # Provider that returns a mix of available and unavailable instances
        provider = AsyncMock()
        provider.list_instance_types = AsyncMock(
            return_value=[
                InstanceType(
                    name="available-1",
                    vcpus=4,
                    memory_gb=16.0,
                    price_per_hour_eur=1.0,
                    available=True,
                ),
                InstanceType(
                    name="unavailable-1",
                    vcpus=8,
                    memory_gb=32.0,
                    price_per_hour_eur=2.0,
                    available=False,
                ),
            ]
        )

        predictor = CostPredictor({"test": provider})
        prediction = await predictor.predict_cost()

        # Only the available instance should appear
        assert len(prediction.estimates) == 1
        assert prediction.estimates[0].instance_name == "available-1"


class TestEstimateForInstanceProviderError:
    """estimate_for_instance returns None when provider raises (lines 143-145)."""

    async def test_estimate_for_instance_provider_error(self) -> None:
        bad_provider = AsyncMock()
        bad_provider.list_instance_types = AsyncMock(side_effect=RuntimeError("API down"))

        predictor = CostPredictor({"bad": bad_provider})
        est = await predictor.estimate_for_instance("bad", "some-instance")
        assert est is None
