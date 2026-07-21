"""Shared helpers for resolving and upgrading local JARVIS checkpoints."""

import json
import os
from pathlib import Path


CHECKPOINT_ROOT = Path("checkpoints")
MODEL_BANK_DIR = CHECKPOINT_ROOT / "model_bank"
MODEL_REGISTRY_PATH = CHECKPOINT_ROOT / "model_registry.json"
DEFAULT_CHECKPOINT_NAME = "best_model.pt"


def resolve_active_checkpoint(default=None):
    env_path = os.environ.get("JARVIS_CHECKPOINT", "").strip()
    if env_path:
        return env_path

    if MODEL_REGISTRY_PATH.exists():
        try:
            with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
            if isinstance(registry, dict):
                active = registry.get("active_checkpoint") or registry.get("current_checkpoint")
                if active and os.path.exists(active):
                    return active
        except Exception:
            pass

    model_bank_active = MODEL_BANK_DIR / "current.pt"
    if model_bank_active.exists():
        return str(model_bank_active)

    fallback = default or str(CHECKPOINT_ROOT / DEFAULT_CHECKPOINT_NAME)
    return fallback


def load_registry():
    if not MODEL_REGISTRY_PATH.exists():
        return {"models": [], "active_checkpoint": resolve_active_checkpoint()}
    try:
        with open(MODEL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("models", [])
            data.setdefault("active_checkpoint", resolve_active_checkpoint())
            return data
    except Exception:
        pass
    return {"models": [], "active_checkpoint": resolve_active_checkpoint()}


def save_registry(registry):
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MODEL_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_checkpoint(name, checkpoint_path, notes="", metrics=None, activate=False):
    MODEL_BANK_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    entry = {
        "name": name,
        "checkpoint_path": checkpoint_path,
        "notes": notes,
        "metrics": metrics or {},
    }
    models = [m for m in registry.get("models", []) if m.get("name") != name]
    models.append(entry)
    registry["models"] = models
    if activate:
        registry["active_checkpoint"] = checkpoint_path
        registry["current_checkpoint"] = checkpoint_path
    save_registry(registry)
    return registry
