"""Tests for config commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from artenic_ai_cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner


class TestConfigShow:
    def test_show(self, runner: CliRunner, monkeypatch: Any) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        with patch("artenic_ai_cli._config._load_toml", return_value={}):
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code == 0
            assert "url" in result.output

    def test_show_json(self, runner: CliRunner, monkeypatch: Any) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        result = runner.invoke(cli, ["--json", "config", "show"])
        assert result.exit_code == 0
        assert '"url"' in result.output


class TestConfigSet:
    def test_set_url(self, runner: CliRunner, tmp_path: Path) -> None:
        tmp_path / "config.toml"
        with patch("artenic_ai_cli.commands.config_cmd.save_config_value") as mock_save:
            result = runner.invoke(cli, ["config", "set", "url", "http://new:9000"])
            assert result.exit_code == 0
            mock_save.assert_called_once_with("url", "http://new:9000", profile="default")

    def test_set_with_profile(self, runner: CliRunner) -> None:
        with patch("artenic_ai_cli.commands.config_cmd.save_config_value") as mock_save:
            result = runner.invoke(
                cli, ["config", "set", "api_key", "my-key", "--profile", "staging"]
            )
            assert result.exit_code == 0
            mock_save.assert_called_once_with("api_key", "my-key", profile="staging")

    def test_set_masks_sensitive_value(self, runner: CliRunner) -> None:
        with patch("artenic_ai_cli.commands.config_cmd.save_config_value"):
            result = runner.invoke(cli, ["config", "set", "api_key", "sk-1234567890abcdef"])
            assert result.exit_code == 0
            assert "sk-1234567890abcdef" not in result.stderr
            assert "***" in result.stderr

    def test_set_shows_non_sensitive_value(self, runner: CliRunner) -> None:
        with patch("artenic_ai_cli.commands.config_cmd.save_config_value"):
            result = runner.invoke(cli, ["config", "set", "url", "http://new:9000"])
            assert result.exit_code == 0
            assert "http://new:9000" in result.stderr

    def test_set_invalid_key(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "set", "invalid_key", "value"])
        assert result.exit_code != 0


class TestConfigUseProfile:
    def test_use_profile(self, runner: CliRunner) -> None:
        with patch("artenic_ai_cli.commands.config_cmd.save_config_value") as mock_save:
            result = runner.invoke(cli, ["config", "use-profile", "staging"])
            assert result.exit_code == 0
            mock_save.assert_called_once_with("_active_profile", "staging")
            assert "staging" in result.stderr


class TestMask:
    def test_empty(self) -> None:
        from artenic_ai_cli.commands.config_cmd import _mask

        assert _mask("") == "(not set)"

    def test_short(self) -> None:
        from artenic_ai_cli.commands.config_cmd import _mask

        assert _mask("abc") == "***"

    def test_exactly_8_chars(self) -> None:
        from artenic_ai_cli.commands.config_cmd import _mask

        assert _mask("12345678") == "***"

    def test_long(self) -> None:
        from artenic_ai_cli.commands.config_cmd import _mask

        result = _mask("sk-1234567890abcdef")
        assert result == "***...cdef"
        assert "sk-1" not in result
