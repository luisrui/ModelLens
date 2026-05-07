# module/model/registry.py
from typing import Callable, Dict, Type

MODEL_REGISTRY: Dict[str, Type] = {}
MODEL_ALIASES: Dict[str, str] = {}  # alias -> canonical name

def register_model(name: str, aliases: list[str] | None = None):
    """
    Usage: @register_model("TwoTowerRecModel_ModelName", aliases=["name_only","tw_name"])
    """
    def deco(cls: Type):
        key = name.strip()
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model name duplicated: {key}")
        MODEL_REGISTRY[key] = cls
        if aliases:
            for a in aliases:
                MODEL_ALIASES[a.strip()] = key
        return cls
    return deco

def get_model_class(name: str) -> Type:
    """Case-insensitive; support aliases; throw error if not found."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # Case-insensitive
    for k in MODEL_REGISTRY:
        if k.lower() == name.lower():
            return MODEL_REGISTRY[k]
    # Aliases
    if name in MODEL_ALIASES:
        return MODEL_REGISTRY[MODEL_ALIASES[name]]
    for a, k in MODEL_ALIASES.items():
        if a.lower() == name.lower():
            return MODEL_REGISTRY[k]
    raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")