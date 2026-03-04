# core/data_loader.py
"""
Client-facing pipeline: raw sensor data directory → cleaned feature CSV.

Usage
-----
# Process a raw dataset and save artifacts:
python data_loader.py /path/to/Sensor_STWIN --out output_dir/processed

# Load and inspect previously saved artifacts:
python data_loader.py --load output_dir/processed --preview

# Filter while processing:
python data_loader.py /path/to/Sensor_STWIN --belt-status OK --sensor-type acc

All bag loading, filtering and grouping live in utils.py so that plotting.py
can reuse the same functions without importing anything from here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import BagFilters, fetch_bags, filter_bags


DEFAULT_OUTPUT_DIR = "output_dir/processed"


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def generate_features(
    bags: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Extract features from *bags* and return (feature_dicts, cleaned_df).

    Delegates entirely to feature_extraction; no statistical logic lives here.
    Import is deferred to avoid pulling in heavy ML dependencies at module load.
    """
    from feature_extraction import (
        extract_features_from_bags,
        prepare_combined_feature_dataframe,
    )
    feature_dicts = extract_features_from_bags(bags)
    cleaned_df    = prepare_combined_feature_dataframe(feature_dicts)
    return feature_dicts, cleaned_df


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_feature_artifacts(
    output_dir: str | Path,
    feature_dicts: list[dict[str, Any]],
    cleaned_df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Save feature artifacts under *output_dir*:

        output_dir/
            dictionary.json   raw feature dicts
            cleaned_df.csv    imputed feature matrix
            manifest.json     provenance + shape info

    Returns resolved absolute paths for each saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dictionary_path = output_path / "dictionary.json"
    dataframe_path  = output_path / "cleaned_df.csv"
    manifest_path   = output_path / "manifest.json"

    with dictionary_path.open("w", encoding="utf-8") as fh:
        json.dump(feature_dicts, fh, indent=2, ensure_ascii=True)

    cleaned_df.to_csv(dataframe_path, index=False)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_bags":       len(feature_dicts),
        "num_rows":       int(cleaned_df.shape[0]),
        "num_columns":    int(cleaned_df.shape[1]),
        "files": {
            "dictionary": dictionary_path.name,
            "dataframe":  dataframe_path.name,
        },
    }
    if metadata:
        manifest.update(metadata)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=True)

    return {
        "output_dir":  str(output_path.resolve()),
        "dictionary":  str(dictionary_path.resolve()),
        "cleaned_df":  str(dataframe_path.resolve()),
        "manifest":    str(manifest_path.resolve()),
    }


def load_feature_artifacts(
    path: str | Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """
    Load previously saved artifacts from *path*.

    *path* may point to:
    - the artifacts directory (containing ``cleaned_df.csv``), or
    - directly to ``cleaned_df.csv``.

    ``feature_dicts`` and ``manifest`` fall back to empty structures when
    their files are absent (dataframe-only mode is valid).
    """
    input_path = Path(path)

    if input_path.is_file():
        if input_path.name != "cleaned_df.csv":
            raise FileNotFoundError(
                "When pointing to a file, it must be named 'cleaned_df.csv'."
            )
        output_path    = input_path.parent
        dataframe_path = input_path
    else:
        output_path    = input_path
        dataframe_path = output_path / "cleaned_df.csv"

    if not dataframe_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataframe: {dataframe_path}")

    cleaned_df = pd.read_csv(dataframe_path)

    feature_dicts: list[dict[str, Any]] = []
    dictionary_path = output_path / "dictionary.json"
    if dictionary_path.exists():
        with dictionary_path.open("r", encoding="utf-8") as fh:
            feature_dicts = json.load(fh)

    manifest: dict[str, Any] = {}
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    return feature_dicts, cleaned_df, manifest


def filter_feature_dataframe(
    cleaned_df: pd.DataFrame,
    filters: BagFilters | None = None,
) -> pd.DataFrame:
    """
    Apply *filters* to an already-generated feature DataFrame.

    Mirrors ``filter_bags`` from utils.py but operates on DataFrame column
    values rather than bag dicts — both share the same BagFilters contract.
    """
    if filters is None:
        return cleaned_df

    out = cleaned_df.copy()
    for field in ("sensor_type", "sensor", "belt_status", "condition", "rpm"):
        value = getattr(filters, field, None)
        if value is not None and field in out.columns:
            out = out[out[field].astype(str) == value]
    return out


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    only_active: bool = True,
    verbose: bool = False,
    filters: BagFilters | None = None,
) -> dict[str, str]:
    """
    Full pipeline: fetch bags → (filter) → extract features → save artifacts.

    Parameters
    ----------
    root        : dataset root path
    output_dir  : where to write artifacts
    limit       : max bags to load (None = all)
    only_active : skip inactive sub-sensors
    verbose     : enable loader warnings
    filters     : optional bag-level filters applied before feature extraction

    Returns resolved file paths for all saved artifacts.
    """
    bags = fetch_bags(root=root, limit=limit, only_active=only_active, verbose=verbose)

    if filters is not None:
        bags = filter_bags(bags, filters=filters)

    if not bags:
        raise ValueError("No bags found after loading/filtering; cannot generate features.")

    feature_dicts, cleaned_df = generate_features(bags)

    return save_feature_artifacts(
        output_dir=output_dir,
        feature_dicts=feature_dicts,
        cleaned_df=cleaned_df,
        metadata={
            "source_root": str(Path(root).resolve()),
            "only_active": bool(only_active),
            "filters":     filters.__dict__ if filters else {},
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "STWIN sensor data pipeline.\n\n"
            "  Process mode: provide a raw dataset root path.\n"
            "  Load mode:    use --load to inspect saved artifacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Process vs Load are mutually exclusive — no more silent mode-switching
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Raw dataset root path (or set $STAT_AI_DATA). Triggers processing mode.",
    )
    mode.add_argument(
        "--load",
        metavar="PATH",
        help="Load existing artifacts from this directory or a cleaned_df.csv path.",
    )

    parser.add_argument("--out",     default=DEFAULT_OUTPUT_DIR,
                        help=f"Output folder (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max bags to load.")
    parser.add_argument("--include-inactive", action="store_true",
                        help="Include inactive sensors.")
    parser.add_argument("--quiet",   action="store_true",
                        help="Reduce loader verbosity.")
    parser.add_argument("--preview", action="store_true",
                        help="Print a short dataframe preview after processing.")

    flt = parser.add_argument_group("filters (apply in both modes)")
    flt.add_argument("--sensor-type",  default=None)
    flt.add_argument("--sensor",       default=None)
    flt.add_argument("--belt-status",  default=None)
    flt.add_argument("--condition",    default=None)
    flt.add_argument("--rpm",          default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    filters = BagFilters(
        sensor_type=args.sensor_type,
        sensor=     args.sensor,
        belt_status=args.belt_status,
        condition=  args.condition,
        rpm=        args.rpm,
    )
    has_filters = any(v is not None for v in filters.__dict__.values())

    # ── Load mode ─────────────────────────────────────────────────────────────
    if args.load:
        feature_dicts, cleaned_df, manifest = load_feature_artifacts(args.load)

        if has_filters:
            cleaned_df = filter_feature_dataframe(cleaned_df, filters)

        print(f"Loaded from  : {Path(args.load).resolve()}")
        print(f"Feature dicts: {len(feature_dicts)}")
        print(f"Dataframe    : {cleaned_df.shape}")
        if has_filters:
            print(f"Filters      : {filters.__dict__}")
        if manifest:
            print(f"Created at   : {manifest.get('created_at_utc', 'unknown')}")
        if cleaned_df.empty:
            raise SystemExit("No rows matched the selected filters.")
        if args.preview:
            print(cleaned_df.head())
        return

    # ── Process mode ──────────────────────────────────────────────────────────
    root = args.root or os.getenv("STAT_AI_DATA", "")
    if not root:
        raise SystemExit(
            "Provide a dataset root as a positional argument or set $STAT_AI_DATA."
        )

    paths = run_pipeline(
        root=root,
        output_dir=args.out,
        limit=args.limit,
        only_active=not args.include_inactive,
        verbose=not args.quiet,
        filters=filters if has_filters else None,
    )

    print("Saved artifacts:")
    for key, val in paths.items():
        print(f"  {key}: {val}")

    if args.preview:
        _, cleaned_df, _ = load_feature_artifacts(args.out)
        print(cleaned_df.head())


if __name__ == "__main__":
    main()