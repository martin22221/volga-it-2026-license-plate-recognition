"""Offline Russian license plate detection and recognition.

Volga-IT 2026 semifinal, "Artificial Intelligence and Data Analysis".

The package is organised as a chain of small, replaceable stages:

    image -> detector -> crop -> classifier -> ocr -> validator -> csv

Every stage is defined by a ``Protocol`` so a placeholder can be swapped for a
real model without touching :mod:`src.pipeline`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "classifier",
    "csv_writer",
    "dataset_meta",
    "detector",
    "external_audit",
    "ocr",
    "pipeline",
    "validator",
]
