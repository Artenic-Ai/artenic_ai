"""Tests for _config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from artenic_ai_cli._config import load_config, save_config_value


class TestLoadConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg = load_config(config_path=Path("/nonexistent/config.toml"))
        assert cfg.url == "http://localhost:9000"
        assert cfg.api_key == ""
        assert cfg.timeout == 30.0
        assert cfg.profile == "default"

    def test_env_vars_override_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_URL", "http://env:8000")
        monkeypatch.setenv("ARTENIC_API_KEY", "env-key")
        monkeypatch.setenv("ARTENIC_TIMEOUT", "60")
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg = load_config(config_path=Path("/nonexistent/config.toml"))
        assert cfg.url == "http://env:8000"
        assert cfg.api_key == "env-key"
        assert cfg.timeout == 60.0

    def test_cli_flags_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_URL", "http://env:8000")
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)
        cfg = load_config(url="http://flag:7000", config_path=Path("/nonexistent/config.toml"))
        assert cfg.url == "http://flag:7000"

    def test_cli_timeout_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg = load_config(timeout=42.0, config_path=Path("/nonexistent/config.toml"))
        assert cfg.timeout == 42.0

    def test_toml_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[default]\nurl = "http://toml:5000"\napi_key = "toml-key"\ntimeout = 10\n',
            encoding="utf-8",
        )
        cfg = load_config(config_path=cfg_file)
        assert cfg.url == "http://toml:5000"
        assert cfg.api_key == "toml-key"
        assert cfg.timeout == 10.0

    def test_profile_selection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            '[default]\nurl = "http://default"\n\n[staging]\nurl = "http://staging"\n',
            encoding="utf-8",
        )
        cfg = load_config(profile="staging", config_path=cfg_file)
        assert cfg.url == "http://staging"
        assert cfg.profile == "staging"

    def test_env_profile(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.setenv("ARTENIC_PROFILE", "staging")

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[staging]\nurl = "http://staging"\n', encoding="utf-8")
        cfg = load_config(config_path=cfg_file)
        assert cfg.profile == "staging"
        assert cfg.url == "http://staging"

    def test_invalid_timeout_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARTENIC_TIMEOUT", "not-a-number")
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg = load_config(config_path=Path("/nonexistent/config.toml"))
        assert cfg.timeout == 30.0

    def test_malformed_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("this is not valid toml {{{", encoding="utf-8")
        with pytest.warns(UserWarning, match="Failed to parse"):
            cfg = load_config(config_path=cfg_file)
        assert cfg.url == "http://localhost:9000"

    def test_missing_profile_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[other]\nurl = "http://other"\n', encoding="utf-8")
        cfg = load_config(config_path=cfg_file)
        assert cfg.url == "http://localhost:9000"

    def test_non_string_file_value_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ARTENIC_URL", raising=False)
        monkeypatch.delenv("ARTENIC_API_KEY", raising=False)
        monkeypatch.delenv("ARTENIC_TIMEOUT", raising=False)
        monkeypatch.delenv("ARTENIC_PROFILE", raising=False)

        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("[default]\nurl = 123\n", encoding="utf-8")
        cfg = load_config(config_path=cfg_file)
        assert cfg.url == "http://localhost:9000"


class TestSaveConfigValue:
    def test_creates_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "sub" / "config.toml"
        save_config_value("url", "http://saved", config_path=cfg_file)
        assert cfg_file.exists()
        content = cfg_file.read_text(encoding="utf-8")
        assert 'url = "http://saved"' in content

    def test_updates_existing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('[default]\nurl = "http://old"\n', encoding="utf-8")
        save_config_value("url", "http://new", config_path=cfg_file)
        content = cfg_file.read_text(encoding="utf-8")
        assert 'url = "http://new"' in content
        assert "http://old" not in content

    def test_non_dict_section_skipped(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('title = "test"\n\n[default]\nurl = "http://old"\n', encoding="utf-8")
        save_config_value("api_key", "key1", config_path=cfg_file)
        content = cfg_file.read_text(encoding="utf-8")
        assert 'api_key = "key1"' in content
        assert "title" not in content

    def test_different_profile(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        save_config_value("url", "http://staging", profile="staging", config_path=cfg_file)
        content = cfg_file.read_text(encoding="utf-8")
        assert "[staging]" in content
        assert 'url = "http://staging"' in content

    def test_escapes_quotes_and_backslashes(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        save_config_value("api_key", 'val"with\\special', config_path=cfg_file)
        content = cfg_file.read_text(encoding="utf-8")
        assert 'api_key = "val\\"with\\\\special"' in content
        # Verify the file is still valid TOML
        import tomllib

        parsed = tomllib.loads(content)
        assert parsed["default"]["api_key"] == 'val"with\\special'

    def test_dir_created_on_save(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "nested" / "deep" / "config.toml"
        save_config_value("url", "http://test", config_path=cfg_file)
        assert cfg_file.parent.exists()

    def test_chmod_on_non_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("artenic_ai_cli._config.sys.platform", "linux")
        cfg_file = tmp_path / "sub" / "config.toml"
        save_config_value("url", "http://test", config_path=cfg_file)
        assert cfg_file.exists()
        # On a real Linux system, dir would have 0o700 permissions.
        # Here we just ensure the code path runs without error.
