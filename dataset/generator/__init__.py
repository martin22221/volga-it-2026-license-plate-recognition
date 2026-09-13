"""Synthetic Russian registration plate generator.

Renders ``type1``, ``type1a`` and ``type1b`` plates from explicit geometry
templates and a built-in stroke font, mounts them on procedural vehicle
panels, applies a seeded geometric and photometric degradation pipeline, and
writes images plus ``meta.csv``-compatible annotations.

Everything the generator draws is produced programmatically inside this
package: no font file, texture, template image or photograph is loaded.  That
keeps the output redistributable under the dataset's CC BY 4.0 license without
any third-party asset analysis.  See ``README.md`` in this directory.

Entry point::

    python -m dataset.generator --output data/synthetic_dev/run --count 100
"""

from __future__ import annotations

from typing import Final

GENERATOR_NAME: Final[str] = "volga-synthetic-plate-generator"
GENERATOR_VERSION: Final[str] = "2.0.0"

#: ``source`` written into every ``meta.csv`` row the generator produces.
#: Registered in ``docs/data_sources.md``; never an external dataset's id.
SOURCE_ID: Final[str] = "volga_synthetic_generator"

#: License of the generated images: our own work, released with the dataset.
OUTPUT_LICENSE: Final[str] = "CC BY 4.0"

__all__ = ["GENERATOR_NAME", "GENERATOR_VERSION", "OUTPUT_LICENSE", "SOURCE_ID"]
