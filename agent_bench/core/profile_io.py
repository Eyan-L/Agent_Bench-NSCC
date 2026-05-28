from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - exercised only on Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

import yaml


class ProfileLoadError(ValueError):
    """Raised when a profile file cannot be loaded as a mapping."""


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser()
    if not profile_path.exists():
        raise ProfileLoadError(f"missing profile file: {profile_path}")

    suffix = profile_path.suffix.lower()
    if suffix == ".json":
        with profile_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh) if profile_path.stat().st_size else {}
    elif suffix == ".toml":
        with profile_path.open("rb") as fh:
            data = tomllib.load(fh)
    elif suffix in {".yaml", ".yml"}:
        with profile_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) if profile_path.stat().st_size else {}
    else:
        raise ProfileLoadError(f"unsupported profile file extension: {profile_path}")

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ProfileLoadError(f"profile file must contain a top-level mapping: {profile_path}")
    return data


def resolve_profile_ref(value: str | Path, base_dir: str | Path, suffixes: Sequence[str]) -> Path:
    """Resolve a profile ref like ``vllm/default`` against known suffixes."""
    suffix_choices = tuple(suffixes)
    if not suffix_choices:
        raise ProfileLoadError("profile suffix list must not be empty")

    path = Path(value).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((Path.cwd() / path).resolve())
        candidates.append((Path(base_dir) / path).expanduser().resolve())

    if path.suffix:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    for candidate in candidates:
        if candidate.exists():
            return candidate
        for suffix in suffix_choices:
            with_suffix = Path(str(candidate) + suffix)
            if with_suffix.exists():
                return with_suffix
    return Path(str(candidates[-1]) + suffix_choices[0])
