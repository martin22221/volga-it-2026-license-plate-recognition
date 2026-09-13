"""Command-line entry point: generate a synthetic batch.

Examples::

    python -m dataset.generator --output data/synthetic_dev/demo --count 100
    python -m dataset.generator --output data/synthetic_dev/demo \\
        --per-class type1=20,type1a=20,type1b=20 --seed 20260913 --contact-sheet
    python -m dataset.generator --output out --count 50 --difficulty hard
    python -m dataset.generator --output out --config dataset/generator/configs/default.json

Exit codes: 0 success, 1 the batch was written but failed validation,
2 invalid arguments or configuration, 3 the output directory cannot be used.

The same configuration and seed always produce the same plate identities,
annotations and image bytes (with the same NumPy/Pillow versions, which the
manifest records).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

from . import GENERATOR_NAME, GENERATOR_VERSION
from .config import (
    DIFFICULTIES,
    PLATE_TYPES,
    ConfigError,
    GeneratorConfig,
    load_config,
    validate_config,
)
from .fonts import get_font_provider
from .sample import GeneratedSample, SamplePlan, generate_sample, plan_dataset
from .scene import ProceduralBackground
from .writer import DatasetWriter, OutputError, image_name, prepare_output

logger = logging.getLogger("dataset.generator")

EXIT_OK = 0
EXIT_INVALID_OUTPUT = 1
EXIT_USAGE = 2
EXIT_OUTPUT_DIR = 3

#: The competition dataset directory; a development batch must not land there.
DATASET_DIR = Path(__file__).resolve().parents[1]


def parse_mapping(text: str, allowed: Sequence[str], cast: Callable[[str], float]) -> dict:
    """Parse ``key=value,key=value`` against ``allowed`` keys."""
    result = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ConfigError(f"expected key=value, got {item!r}")
        key, value = (part.strip() for part in item.split("=", 1))
        if key not in allowed:
            raise ConfigError(f"unknown key {key!r}; allowed: {', '.join(allowed)}")
        try:
            result[key] = cast(value)
        except ValueError:
            raise ConfigError(f"invalid value {value!r} for {key!r}") from None
    if not result:
        raise ConfigError("empty mapping")
    return result


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ConfigError(f"image size must look like 640x480, got {text!r}") from None
    return width, height


def build_config(args: argparse.Namespace) -> GeneratorConfig:
    config = load_config(args.config) if args.config else GeneratorConfig()
    changes: dict = {}
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.per_class is not None:
        if args.count is not None or args.class_weights is not None:
            raise ConfigError("--per-class cannot be combined with --count or --class-weights")
        counts = parse_mapping(args.per_class, PLATE_TYPES, int)
        changes["class_counts"] = {name: int(counts.get(name, 0)) for name in PLATE_TYPES}
    else:
        if args.count is not None:
            changes["count"] = args.count
            changes["class_counts"] = None
        if args.class_weights is not None:
            changes["class_weights"] = parse_mapping(args.class_weights, PLATE_TYPES, float)
            changes["class_counts"] = None
    if args.difficulty is not None and args.difficulty_weights is not None:
        raise ConfigError("--difficulty cannot be combined with --difficulty-weights")
    if args.difficulty is not None:
        changes["difficulty_weights"] = {level: (1.0 if level == args.difficulty else 0.0) for level in DIFFICULTIES}
    if args.difficulty_weights is not None:
        changes["difficulty_weights"] = parse_mapping(args.difficulty_weights, DIFFICULTIES, float)
    if args.image_size:
        changes["image_sizes"] = tuple(parse_size(size) for size in args.image_size)
    return validate_config(replace(config, **changes))


def _inside(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def _generate_one(task: tuple[GeneratorConfig, SamplePlan, str]) -> GeneratedSample:
    """Worker entry point: one sample, without the (large) plate mask."""
    config, plan, name = task
    sample = generate_sample(config, plan, font=get_font_provider(config.font),
                             background=ProceduralBackground(), image_name=name)
    sample.plate_mask = sample.plate_mask[:0, :0]
    return sample


def generate_dataset(
    config: GeneratorConfig,
    output: Path,
    *,
    overwrite: bool = False,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Generate the whole batch into ``output`` and return the manifest.

    With ``workers > 1`` samples are rendered in parallel processes.  Every
    sample depends only on its own seed and results are written in index
    order, so the output is identical to a single-process run.
    """
    validate_config(config)
    if workers < 1:
        raise ConfigError("workers must be >= 1")
    prepare_output(output, overwrite=overwrite)
    plans = plan_dataset(config)
    tasks = [(config, plan, f"images/synthetic/{image_name(config.seed, plan.index)}") for plan in plans]
    writer = DatasetWriter(output, config)
    if workers == 1:
        results = map(_generate_one, tasks)
        pool = None
    else:
        pool = multiprocessing.get_context("spawn").Pool(workers)
        results = pool.imap(_generate_one, tasks, chunksize=2)
    try:
        for done, sample in enumerate(results, start=1):
            writer.add(sample)
            if progress is not None:
                progress(done, len(tasks))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return writer.finish()


def validate_output(output: Path) -> tuple[bool, str]:
    """Run the repository's dataset validator on the batch, if available."""
    try:
        from src.dataset_meta import validate_meta  # the repository's own checker
    except ImportError:
        return True, "repository validator not importable; skipped"
    report = validate_meta(output / "meta.csv", output)
    summary = f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    for issue in report.issues[:20]:
        logger.warning("validator: %s", issue)
    return report.is_valid, summary


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dataset.generator",
        description=f"{GENERATOR_NAME} {GENERATOR_VERSION}: synthetic Russian plate images with annotations.",
    )
    parser.add_argument("--output", "-o", type=Path, required=True, help="batch directory to create")
    parser.add_argument("--config", type=Path, default=None, help="JSON configuration overlaid on the defaults")
    parser.add_argument("--count", type=int, default=None, help="total number of images")
    parser.add_argument("--seed", type=int, default=None, help="master seed (default 20260913)")
    parser.add_argument("--per-class", default=None, metavar="type1=N,type1a=N,type1b=N", help="exact images per plate type")
    parser.add_argument("--class-weights", default=None, metavar="type1=W,...", help="class proportions for --count")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None, help="generate a single difficulty")
    parser.add_argument("--difficulty-weights", default=None, metavar="easy=W,medium=W,hard=W", help="difficulty mix")
    parser.add_argument("--image-size", action="append", default=None, metavar="WxH", help="output size; repeatable")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (output is identical)")
    parser.add_argument("--overwrite", action="store_true", help="replace a batch previously written here")
    parser.add_argument("--contact-sheet", action="store_true", help="also build review/contact_sheet.html")
    parser.add_argument("--no-validate", action="store_true", help="skip the dataset validator run")
    parser.add_argument(
        "--allow-dataset-dir", action="store_true",
        help="permit writing inside the competition dataset/ directory (final generation only)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print errors and the summary line")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exit_:  # argparse already printed the message
        return EXIT_USAGE if exit_.code not in (0, None) else EXIT_OK
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    try:
        config = build_config(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE

    if _inside(args.output, DATASET_DIR) and not args.allow_dataset_dir:
        print(
            f"error: {args.output} is inside {DATASET_DIR}; development batches belong elsewhere "
            "(e.g. data/synthetic_dev/). Pass --allow-dataset-dir for final generation.",
            file=sys.stderr,
        )
        return EXIT_OUTPUT_DIR

    started = time.perf_counter()
    step = max(1, config.total_count // 10)

    def progress(done: int, total: int) -> None:
        if not args.quiet and (done % step == 0 or done == total):
            logger.info("  %d / %d", done, total)

    try:
        manifest = generate_dataset(
            config, args.output, overwrite=args.overwrite, workers=args.workers, progress=progress
        )
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE
    except OutputError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_OUTPUT_DIR
    elapsed = time.perf_counter() - started

    valid, summary = (True, "skipped") if args.no_validate else validate_output(args.output)
    if args.contact_sheet:
        from .contact_sheet import build_contact_sheet

        sheet = build_contact_sheet(args.output)
        logger.info("contact sheet: %s", sheet)

    counts = manifest["counts"]
    print(
        f"{manifest['image_count']} images -> {args.output} in {elapsed:.1f}s "
        f"({1000 * elapsed / max(1, manifest['image_count']):.0f} ms/image); "
        f"types {counts['plate_type']}; difficulty {counts['difficulty']}; validator: {summary}"
    )
    return EXIT_OK if valid else EXIT_INVALID_OUTPUT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
