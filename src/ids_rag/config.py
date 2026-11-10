"""Configuration loader and schema validator."""

from pathlib import Path
from typing import Any
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if config_path is None:
        target = get_project_root() / "config.yaml"
    else:
        target = Path(config_path)
        if not target.is_absolute():
            target = get_project_root() / target

    if not target.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config
