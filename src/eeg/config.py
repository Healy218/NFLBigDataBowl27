"""Load physics assumptions from config/physics.yaml.

Each parameter block has a ``value`` and, if it is swept in the sensitivity
analysis, ``low`` and ``high`` bounds. Fixed parameters omit both bounds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_CONFIG = PROJECT_ROOT / "config" / "physics.yaml"


@dataclass(frozen=True)
class SweepParam:
    """A parameter with sensitivity bounds, addressed as ``section.name``."""

    section: str
    name: str
    value: float
    low: float
    high: float

    @property
    def key(self) -> str:
        return f"{self.section}.{self.name}"


def load_raw(path: Path = PHYSICS_CONFIG) -> dict[str, dict[str, dict[str, Any]]]:
    """Full config, including units/notes/bounds."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def values(cfg: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Collapse each parameter block to its ``value``: ``{section: {name: value}}``."""
    return {sec: {name: blk["value"] for name, blk in params.items()} for sec, params in cfg.items()}


def load(path: Path = PHYSICS_CONFIG) -> dict[str, dict[str, Any]]:
    """Default parameter values, e.g. ``load()["broad"]["theta_deg"] -> 45.0``."""
    return values(load_raw(path))


def sweep_params(cfg: dict[str, dict[str, dict[str, Any]]]) -> Iterator[SweepParam]:
    """Every parameter that has sensitivity bounds."""
    for sec, params in cfg.items():
        for name, blk in params.items():
            if "low" in blk:
                yield SweepParam(sec, name, blk["value"], blk["low"], blk["high"])


def with_override(
    cfg: dict[str, dict[str, dict[str, Any]]], key: str, value: Any
) -> dict[str, dict[str, dict[str, Any]]]:
    """Copy of ``cfg`` with ``section.name`` set to ``value`` (for one-at-a-time sweeps)."""
    sec, name = key.split(".", 1)
    out = copy.deepcopy(cfg)
    out[sec][name]["value"] = value
    return out


def validate(cfg: dict[str, dict[str, dict[str, Any]]]) -> None:
    for sec, params in cfg.items():
        for name, blk in params.items():
            key = f"{sec}.{name}"
            if not isinstance(blk, dict) or "value" not in blk:
                raise ValueError(f"{key}: parameter block must have a 'value'")
            has_low, has_high = "low" in blk, "high" in blk
            if has_low != has_high:
                raise ValueError(f"{key}: give both 'low' and 'high' or neither")
            if has_low and not blk["low"] <= blk["value"] <= blk["high"]:
                raise ValueError(
                    f"{key}: value {blk['value']} outside [{blk['low']}, {blk['high']}]"
                )
