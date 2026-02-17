"""Tests for artenic_ai_platform.budget.router — /api/v1/budgets/*."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from httpx import AsyncClient


# Helper to create a budget via the API and return the full response dict.
async def _create_budget(client: AsyncClient, **overrides: object) -> dict:
    payload = {
        "scope": "service",
        "scope_value": "test-svc",
        "period": "monthly",
        "limit_eur": 500.0,
    }
    payload.update(overrides)
    resp = await client.post("/api/v1/budgets", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ======================================================================
# GET /api/v1/budgets
# ======================================================================


class TestListBudgets:
    async def test_list_budgets_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/budgets")
        assert resp.status_code == 200
        assert resp.json() == []


# ======================================================================
# POST /api/v1/budgets
# ======================================================================


class TestCreateBudget:
    async def test_create_budget(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/budgets",
            json={
                "scope": "service",
                "scope_value": "fraud",
                "period": "monthly",
                "limit_eur": 1000.0,
                "alert_threshold_pct": 75.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope"] == "service"
        assert data["scope_value"] == "fraud"
        assert data["period"] == "monthly"
        assert data["limit_eur"] == 1000.0
        assert data["alert_threshold_pct"] == 75.0
        assert data["enabled"] is True
        assert "id" in data

    async def test_create_budget_invalid_period(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/budgets",
            json={
                "scope": "service",
                "scope_value": "test",
                "period": "biweekly",  # invalid
                "limit_eur": 100.0,
            },
        )
        # ValueError from service -> 500
        assert resp.status_code == 500


# ======================================================================
# PUT /api/v1/budgets/{budget_id}
# ======================================================================


class TestUpdateBudget:
    async def test_update_budget(self, client: AsyncClient) -> None:
        created = await _create_budget(client)
        budget_id = created["id"]

        resp = await client.put(
            f"/api/v1/budgets/{budget_id}",
            json={"limit_eur": 750.0, "alert_threshold_pct": 90.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit_eur"] == 750.0
        assert data["alert_threshold_pct"] == 90.0

    async def test_update_budget_not_found(self, client: AsyncClient) -> None:
        resp = await client.put(
            "/api/v1/budgets/99999",
            json={"limit_eur": 100.0},
        )
        # KeyError from service -> 500
        assert resp.status_code == 500


# ======================================================================
# GET /api/v1/budgets/spending
# ======================================================================


class TestGetSpending:
    async def test_get_spending(self, client: AsyncClient) -> None:
        # Create a budget first so the spending endpoint has something to
        # aggregate against.
        await _create_budget(client, scope="service", scope_value="test")

        resp = await client.get(
            "/api/v1/budgets/spending",
            params={"scope": "service", "scope_value": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        entry = data[0]
        assert entry["scope"] == "service"
        assert entry["scope_value"] == "test"
        assert "spent_eur" in entry
        assert "limit_eur" in entry
        assert "pct_used" in entry
        assert "alerts" in entry
