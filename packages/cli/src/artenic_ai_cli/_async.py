"""Async-to-sync bridge for Click commands."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

T = TypeVar("T")


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously.

    Each CLI invocation is a fresh process so ``asyncio.run``
    is safe — no existing event loop to conflict with.
    """
    return asyncio.run(coro)
