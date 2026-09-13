"""Shared helpers for the synthetic generator tests: small, fast configs."""

from __future__ import annotations

from typing import Any

from dataset.generator.config import GeneratorConfig, config_from_dict

#: Small frames and 2x supersampling keep each sample well under a second.
SMALL: dict[str, Any] = {"image_sizes": [[320, 240], [400, 300]], "supersample": 2}


def small_config(**overrides: Any) -> GeneratorConfig:
    return config_from_dict({**SMALL, **overrides})


def force_effects(level: str, probabilities: dict[str, float], max_effects: int = 11) -> dict[str, Any]:
    """A ``difficulties`` override that forces effect probabilities."""
    return {level: {"effect_probability": probabilities, "max_effects": max_effects}}
