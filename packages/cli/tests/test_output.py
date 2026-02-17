"""Tests for _output module."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from rich.console import Console

from artenic_ai_cli._config import CliConfig
from artenic_ai_cli._context import CliContext
from artenic_ai_cli._output import print_error, print_result, print_success


def _make_ctx(*, json_mode: bool = False) -> tuple[CliContext, io.StringIO, io.StringIO]:
    out = io.StringIO()
    err = io.StringIO()
    return (
        CliContext(
            api=MagicMock(),
            console=Console(file=out, no_color=True, width=200),
            err_console=Console(file=err, no_color=True, width=200),
            json_mode=json_mode,
            config=CliConfig(),
        ),
        out,
        err,
    )


class TestPrintResult:
    def test_dict_table(self) -> None:
        ctx, out, _ = _make_ctx()
        print_result(ctx, {"key": "value", "num": 42}, title="Info")
        text = out.getvalue()
        assert "key" in text
        assert "value" in text

    def test_list_table(self) -> None:
        ctx, out, _ = _make_ctx()
        rows = [{"name": "a", "v": 1}, {"name": "b", "v": 2}]
        print_result(ctx, rows, columns=[("Name", "name"), ("Value", "v")])
        text = out.getvalue()
        assert "a" in text
        assert "b" in text

    def test_list_table_auto_columns(self) -> None:
        ctx, out, _ = _make_ctx()
        rows = [{"x": 1, "y": 2}]
        print_result(ctx, rows)
        text = out.getvalue()
        assert "1" in text

    def test_json_mode(self) -> None:
        ctx, out, _ = _make_ctx(json_mode=True)
        print_result(ctx, {"key": "val"})
        text = out.getvalue()
        assert '"key"' in text
        assert '"val"' in text

    def test_scalar_value(self) -> None:
        ctx, out, _ = _make_ctx()
        print_result(ctx, "hello world")
        assert "hello world" in out.getvalue()

    def test_empty_list(self) -> None:
        ctx, _, err = _make_ctx()
        print_result(ctx, [])
        assert "No results found" in err.getvalue()

    def test_empty_list_json(self) -> None:
        ctx, out, _ = _make_ctx(json_mode=True)
        print_result(ctx, [])
        assert "[]" in out.getvalue()

    def test_list_missing_key(self) -> None:
        ctx, out, _ = _make_ctx()
        rows = [{"name": "a"}]
        print_result(ctx, rows, columns=[("Name", "name"), ("Missing", "nope")])
        text = out.getvalue()
        assert "a" in text


class TestPrintSuccess:
    def test_writes_to_stderr(self) -> None:
        ctx, _, err = _make_ctx()
        print_success(ctx, "done!")
        assert "done!" in err.getvalue()


class TestPrintError:
    def test_writes_to_stderr(self) -> None:
        ctx, _, err = _make_ctx()
        print_error(ctx, "something failed")
        assert "something failed" in err.getvalue()
