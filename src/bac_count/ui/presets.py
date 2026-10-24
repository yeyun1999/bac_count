"""参数预设：JSON 存取。"""
from __future__ import annotations

import json
from pathlib import Path

PRESET_DIR = Path.home() / ".bac_count" / "presets"


def save_preset(name: str, params: dict) -> Path:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    p = PRESET_DIR / f"{_safe(name)}.json"
    p.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_preset(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_presets() -> list[Path]:
    return sorted(PRESET_DIR.glob("*.json")) if PRESET_DIR.exists() else []


def _safe(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in "-_")
