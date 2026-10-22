"""Configuration loader and storage resolver for the FEVER IR system."""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

_FALLBACK_BASE: Optional[Path] = None


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_storage_dir(relative_path: str = "artifacts", for_writing: bool = False) -> Path:
    """Resolves a storage directory, handling OS permissions and Windows Controlled Folder Access.

    Args:
        relative_path: Configured directory path (e.g. 'artifacts/indexes').
        for_writing: Whether the directory will be used for file creation.

    Returns:
        Resolved Path object.
    """
    global _FALLBACK_BASE
    root = get_project_root()
    target = Path(relative_path)
    if not target.is_absolute():
        target = root / target

    fallback = Path.home() / ".fever_cache" / relative_path

    if not for_writing:
        target_files = [f for f in target.iterdir() if f.name != ".gitkeep" and not f.name.endswith(".tmp")] if target.exists() else []
        fallback_files = [f for f in fallback.iterdir() if f.name != ".gitkeep" and not f.name.endswith(".tmp")] if fallback.exists() else []

        if target_files and not fallback_files:
            return target
        if fallback_files and not target_files:
            return fallback
        if target_files and fallback_files:
            t_mtime = max(f.stat().st_mtime for f in target_files)
            f_mtime = max(f.stat().st_mtime for f in fallback_files)
            return fallback if f_mtime > t_mtime else target
        return target

    # When writing, test write permission
    try:
        target.mkdir(parents=True, exist_ok=True)
        test_file = target / ".write_test.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        if test_file.exists():
            test_file.unlink()
        return target
    except (OSError, PermissionError, FileNotFoundError):
        if _FALLBACK_BASE is None:
            _FALLBACK_BASE = Path.home() / ".fever_cache"
            logger.info(
                f"Notice: Workspace path '{target}' is protected by OS access control. "
                f"Writable storage redirected to '{_FALLBACK_BASE}'"
            )
        fallback = _FALLBACK_BASE / relative_path
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path relative to project root or absolute path.

    Returns:
        Dictionary of configuration options.
    """
    root = get_project_root()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
