"""CLI configuration loading with layered precedence."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path.home() / ".artenic"
_CONFIG_FILE = _CONFIG_DIR / "config.toml"

_DEFAULT_URL = "http://localhost:9000"
_DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class CliConfig:
    """Resolved CLI configuration."""

    url: str = _DEFAULT_URL
    api_key: str = ""
    timeout: float = _DEFAULT_TIMEOUT
    profile: str = "default"


def _load_toml(path: Path | None = None) -> dict[str, Any]:
    """Read a TOML config file, returning {} if absent or malformed."""
    target = path or _CONFIG_FILE
    if not target.is_file():
        return {}
    try:
        return tomllib.loads(target.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, PermissionError) as e:
        import warnings

        warnings.warn(f"Failed to parse {target}: {e}", stacklevel=2)
        return {}


def _profile_values(data: dict[str, Any], profile: str) -> dict[str, Any]:
    """Extract a profile section from parsed TOML data."""
    section = data.get(profile)
    if isinstance(section, dict):
        return section
    return {}


def load_config(
    *,
    url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    profile: str | None = None,
    config_path: Path | None = None,
) -> CliConfig:
    """Build config with precedence: CLI flags > env vars > TOML file > defaults."""
    effective_profile = profile or os.environ.get("ARTENIC_PROFILE", "default")

    toml_data = _load_toml(config_path)
    file_vals = _profile_values(toml_data, effective_profile)

    def _resolve_str(flag: str | None, env_key: str, file_key: str, default: str) -> str:
        if flag is not None:
            return flag
        env = os.environ.get(env_key)
        if env is not None:
            return env
        file_val = file_vals.get(file_key)
        if isinstance(file_val, str):
            return file_val
        return default

    def _resolve_float(flag: float | None, env_key: str, file_key: str, default: float) -> float:
        if flag is not None:
            return flag
        env = os.environ.get(env_key)
        if env is not None:
            try:
                return float(env)
            except ValueError:
                pass
        file_val = file_vals.get(file_key)
        if isinstance(file_val, (int, float)):
            return float(file_val)
        return default

    return CliConfig(
        url=_resolve_str(url, "ARTENIC_URL", "url", _DEFAULT_URL),
        api_key=_resolve_str(api_key, "ARTENIC_API_KEY", "api_key", ""),
        timeout=_resolve_float(timeout, "ARTENIC_TIMEOUT", "timeout", _DEFAULT_TIMEOUT),
        profile=effective_profile,
    )


def save_config_value(
    key: str,
    value: str,
    *,
    profile: str = "default",
    config_path: Path | None = None,
) -> None:
    """Write a single key=value into ~/.artenic/config.toml under [profile]."""
    target_dir = (config_path or _CONFIG_FILE).parent if config_path else _CONFIG_DIR
    target_file = config_path or _CONFIG_FILE

    target_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        target_dir.chmod(0o700)

    data = _load_toml(target_file)
    if profile not in data or not isinstance(data[profile], dict):
        data[profile] = {}
    data[profile][key] = value

    lines: list[str] = []
    for section_name, section_vals in data.items():
        if not isinstance(section_vals, dict):
            continue
        lines.append(f"[{section_name}]")
        for k, v in section_vals.items():
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{escaped}"')
        lines.append("")

    target_file.write_text("\n".join(lines), encoding="utf-8")
