"""Tests for artenic_ai_platform.entities.lineage.router — REST API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from artenic_ai_platform.app import _lifespan, create_app
from artenic_ai_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from fastapi import FastAPI


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
async def app_with_lifespan(tmp_path: Path) -> AsyncGenerator[FastAPI, None]:
    settings = PlatformSettings(
        database_url="sqlite+aiosqlite://",
        api_key="",
        secret_key="test-secret",
        otel_enabled=False,
        dataset={"storage": {"local_dir": str(tmp_path / "datasets")}},
    )
    app = create_app(settings)
    async with _lifespan(app):
        yield app


@pytest.fixture
async def client(app_with_lifespan: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_lifespan)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


BASE = "/api/v1/lineage"


# ======================================================================
# POST /api/v1/lineage
# ======================================================================


class TestAddLink:
    async def test_add_link_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post(
            BASE,
            json={
                "source_id": "ds_train_v1",
                "target_id": "mdl_output_v1",
                "relation_type": "trained_on",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["source_id"] == "ds_train_v1"
        assert data["target_id"] == "mdl_output_v1"
        assert data["relation_type"] == "trained_on"

    async def test_add_link_validation_error(self, client: AsyncClient) -> None:
        resp = await client.post(BASE, json={})
        assert resp.status_code == 422


# ======================================================================
# GET /api/v1/lineage/{entity_id}
# ======================================================================


class TestGetLineage:
    async def test_get_upstream_downstream(self, client: AsyncClient) -> None:
        # ds -> model (model trained_on ds)
        await client.post(
            BASE,
            json={
                "source_id": "ds_a_v1",
                "target_id": "mdl_b_v1",
                "relation_type": "trained_on",
            },
        )
        # model -> ensemble (ensemble uses model)
        await client.post(
            BASE,
            json={
                "source_id": "mdl_b_v1",
                "target_id": "ens_c_v1",
                "relation_type": "part_of",
            },
        )

        # Query from model — should see upstream (ds) and downstream (ensemble)
        resp = await client.get(f"{BASE}/mdl_b_v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_id"] == "mdl_b_v1"
        assert len(data["upstream"]) == 1
        assert data["upstream"][0]["source_id"] == "ds_a_v1"
        assert len(data["downstream"]) == 1
        assert data["downstream"][0]["target_id"] == "ens_c_v1"

    async def test_get_empty_lineage(self, client: AsyncClient) -> None:
        resp = await client.get(f"{BASE}/isolated_entity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["upstream"] == []
        assert data["downstream"] == []


# ======================================================================
# GET /api/v1/lineage/{entity_id}/graph
# ======================================================================


class TestGetGraph:
    async def test_graph_traversal(self, client: AsyncClient) -> None:
        # Create a chain: ds -> model -> ensemble
        await client.post(
            BASE,
            json={
                "source_id": "ds_g1_v1",
                "target_id": "mdl_g1_v1",
                "relation_type": "trained_on",
            },
        )
        await client.post(
            BASE,
            json={
                "source_id": "mdl_g1_v1",
                "target_id": "ens_g1_v1",
                "relation_type": "part_of",
            },
        )

        resp = await client.get(f"{BASE}/ds_g1_v1/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["root"] == "ds_g1_v1"
        assert set(data["nodes"]) == {"ds_g1_v1", "mdl_g1_v1", "ens_g1_v1"}
        assert len(data["edges"]) == 2


# ======================================================================
# DELETE /api/v1/lineage
# ======================================================================


class TestRemoveLink:
    async def test_remove_link(self, client: AsyncClient) -> None:
        await client.post(
            BASE,
            json={
                "source_id": "ds_rm_v1",
                "target_id": "mdl_rm_v1",
                "relation_type": "trained_on",
            },
        )
        resp = await client.delete(
            BASE,
            params={
                "source_id": "ds_rm_v1",
                "target_id": "mdl_rm_v1",
                "relation_type": "trained_on",
            },
        )
        assert resp.status_code == 200

    async def test_remove_link_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete(
            BASE,
            params={
                "source_id": "nonexistent",
                "target_id": "also_nonexistent",
                "relation_type": "nope",
            },
        )
        assert resp.status_code == 404


# ======================================================================
# Additional coverage tests
# ======================================================================


class TestGraphEdgeCases:
    async def test_graph_diamond_visits_node_once(self, client: AsyncClient) -> None:
        """BFS graph with diamond shape triggers revisit skip."""
        # Diamond: A -> B -> D, A -> C -> D
        for src, tgt in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
            await client.post(
                BASE,
                json={"source_id": src, "target_id": tgt, "relation_type": "depends_on"},
            )
        resp = await client.get(f"{BASE}/A/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["nodes"]) == {"A", "B", "C", "D"}
        assert len(data["edges"]) == 4
