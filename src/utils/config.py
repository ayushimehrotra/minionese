"""
Configuration loading utilities.

Loads YAML config files and provides a unified config object.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load all config files from the configs directory and merge into one dict.

    Args:
        config_dir: Path to the configs directory.

    Returns:
        Merged config dict with keys: models, languages, experiment, paths.
    """
    config_dir = Path(config_dir)
    config = {}
    for name in ["models", "languages", "experiment", "paths"]:
        cfg_path = config_dir / f"{name}.yaml"
        if cfg_path.exists():
            config[name] = load_yaml(str(cfg_path))
        else:
            config[name] = {}
    return config


def get_model_config(config: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """
    Retrieve model-specific configuration.

    Args:
        config: Merged config dict from load_config().
        model_key: One of "llama", "gemma", "qwen".

    Returns:
        Model config dict.
    """
    return config["models"]["target_models"][model_key]


def get_language_tier(config: Dict[str, Any], lang: str) -> Optional[str]:
    """
    Return the tier name for a given language code.

    Args:
        config: Merged config dict.
        lang: Language code (e.g. "en", "ar").

    Returns:
        Tier name (e.g. "tier1") or None if not found.
    """
    tiers = config["languages"].get("tiers", {})
    for tier_name, tier_data in tiers.items():
        if lang in tier_data.get("languages", []):
            return tier_name
    return None


def resolve_path(config: Dict[str, Any], key: str) -> str:
    """
    Resolve a path from the paths config, expanding relative paths
    relative to the repository root.

    Args:
        config: Merged config dict.
        key: Key in paths config (e.g. "activations_dir").

    Returns:
        Absolute path string.
    """
    paths = config.get("paths", {})
    raw = paths.get(key, key)
    return str(Path(raw).resolve())
