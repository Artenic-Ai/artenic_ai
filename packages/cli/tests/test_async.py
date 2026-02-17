"""Tests for _async module."""

from __future__ import annotations

import pytest

from artenic_ai_cli._async import run_async


class TestRunAsync:
    def test_returns_coroutine_result(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b

        assert run_async(add(2, 3)) == 5

    def test_propagates_exception(self) -> None:
        async def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_async(fail())

    def test_runs_without_existing_loop(self) -> None:
        async def check() -> str:
            return "ok"

        result = run_async(check())
        assert result == "ok"
