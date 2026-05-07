"""
Model profile loader.

Canonical source: data/unified/model_profile.json
Each entry is expected to look like:
  { "family": "<str|unknown>", "size": <float_in_B|unknown>, ... }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


_CACHED_PATH: Optional[Path] = None
_CACHED_MTIME: Optional[float] = None
_CACHED_PROFILE: Optional[Dict[str, Any]] = None


def _default_profile_path() -> Path:
    # module/utils/model_profile.py -> parents[2] is repo root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "unified" / "model_profile.json"


def load_model_profile(profile_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load and cache the model profile JSON.

    Cache invalidates if file mtime changes.
    """
    global _CACHED_PATH, _CACHED_MTIME, _CACHED_PROFILE

    p = Path(profile_path) if profile_path is not None else _default_profile_path()
    p = p.expanduser().resolve()

    try:
        mtime = p.stat().st_mtime
    except FileNotFoundError:
        return {}

    if _CACHED_PROFILE is not None and _CACHED_PATH == p and _CACHED_MTIME == mtime:
        return _CACHED_PROFILE

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"model_profile must be a JSON object (dict), got {type(data)} at {p}")

    _CACHED_PATH = p
    _CACHED_MTIME = mtime
    _CACHED_PROFILE = data
    return data


def get_profile_entry(model_name: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if profile is None:
        profile = load_model_profile()
    v = profile.get(model_name)
    return v if isinstance(v, dict) else {}


def get_model_family(model_name: str, profile: Optional[Dict[str, Any]] = None) -> str:
    v = get_profile_entry(model_name, profile)
    fam = v.get("family")
    if isinstance(fam, str) and fam.strip():
        return fam.strip().lower()
    return "unknown"


def get_model_size_b(model_name: str, profile: Optional[Dict[str, Any]] = None) -> float:
    """
    Return size in billions (B). Unknown -> NaN.
    """
    v = get_profile_entry(model_name, profile)
    size = v.get("size")
    try:
        if isinstance(size, str) and size.strip().lower() == "unknown":
            return float("nan")
        x = float(size)
        if x == 0.0:
            return float("nan")
        return x
    except Exception:
        return float("nan")


def build_model2size_from_profile(
    model_names: list[str],
    profile: Optional[Dict[str, Any]] = None,
    include_unknown: bool = True,
) -> Dict[str, Any]:
    """
    Build a model2size mapping from model_profile.json.

    Values are floats (billions) when known; otherwise "unknown" (if include_unknown)
    or omitted.
    """
    if profile is None:
        profile = load_model_profile()
    out: Dict[str, Any] = {}
    for name in model_names:
        x = get_model_size_b(name, profile)
        if not (x != x):  # not NaN
            out[name] = float(x)
        elif include_unknown:
            out[name] = "unknown"
    return out
