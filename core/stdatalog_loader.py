# stdatalog_hsd_loader.py
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import re
from pathlib import Path
from types import GeneratorType
from typing import Dict, Iterator, Optional, Iterable
from utils import get_odr_map, norm, get_sensor_type, normalize_sensor_columns
import pandas as pd

import sys as _sys
_here = Path(__file__).resolve()
for rel in ['..', '../..']:
    cand = (_here.parent / rel).resolve()
    if cand.exists() and str(cand) not in _sys.path:
        _sys.path.append(str(cand))

try:
    from stdatalog_core.HSD.HSDatalog import HSDatalog
except Exception as _e:
    HSDatalog = None
    _IMPORT_ERR = _e
else:
    _IMPORT_ERR = None

# ── path metadata helpers ────────────────────────────────────────────────────
KNOWN_CONDITIONS = {'vel-fissa', 'no-load-cycles', 'vel_fissa', 'no_load_cycles'}
RPM_RE = re.compile(r'^(PMI|PMS)[_-]?(\d+\s*rpm|\d+rpm|\d+)$', re.IGNORECASE)


def _find_token(parts: Iterable[str], pool: set[str]) -> Optional[str]:
    for p in parts:
        if p in pool:
            return p
    return None


def _find_status(parts: Iterable[str]) -> Optional[str]:
    for p in parts:
        up = p.upper()
        if up == 'OK' or up.startswith('KO'):
            return p
    return None


def _find_rpm(parts: Iterable[str]) -> Optional[str]:
    normalized = [p.replace('-', '_') for p in parts]
    for p in normalized:
        m = RPM_RE.match(p)
        if m:
            kind = m.group(1).upper()
            num = re.sub(r'[^0-9]', '', m.group(2))
            return f"{kind}_{num}rpm"
    for i in range(len(normalized) - 1):
        a, b = normalized[i], normalized[i + 1]
        if a.upper() in {'PMI', 'PMS'} and re.search(r'\d+', b):
            num = re.sub(r'[^0-9]', '', b)
            return f"{a.upper()}_{num}rpm"
    return None


def _infer_meta_from_path(acq_dir: Path) -> Dict[str, str]:
    parts = acq_dir.parts
    condition = _find_token(parts, KNOWN_CONDITIONS) or 'vel-fissa'
    status = _find_status(parts) or 'OK'
    rpm = _find_rpm(parts) or 'NO_RPM_VALUE'
    if condition == 'vel_fissa':
        condition = 'vel-fissa'
    return {'condition': condition, 'belt_status': status,
            'rpm': rpm, 'full_path': str(acq_dir)}


def _is_hsd_folder(dir_path: Path) -> bool:
    if not dir_path.is_dir():
        return False
    for name in ['deviceConfig.json', 'DeviceConfig.json', 'hsd.json']:
        if (dir_path / name).exists():
            return True
    return any(p.suffix.lower() == '.dat' for p in dir_path.iterdir()
               if p.is_file())


def _normalize_dataframe(obj) -> Optional[pd.DataFrame]:
    """
    Collapse whatever get_dataframe() returns into a single DataFrame.

    The HSD v1 SDK always returns a LIST of DataFrames (one per acquisition
    chunk / timestamp block).  Each chunk has identical columns.  We must
    ROW-concatenate them, not column-join them.

    Handled types
    -------------
    list / tuple   → row-concat all DataFrame elements (axis=0)
    GeneratorType  → same after exhausting
    dict           → col-join values (legacy path, kept for safety)
    DataFrame      → returned as-is
    """
    if obj is None:
        return None

    if isinstance(obj, GeneratorType):
        obj = list(obj)

    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, (list, tuple)):
        frames = [v for v in obj if isinstance(v, pd.DataFrame)]
        if not frames:
            return None
        # Row-concatenate: each element is a time-contiguous chunk
        return pd.concat(frames, axis=0, ignore_index=True)

    if isinstance(obj, dict):
        # Some future SDK version might return a dict — col-join as fallback
        frames = []
        for key, val in obj.items():
            if isinstance(val, pd.DataFrame):
                frames.append(val)
            elif hasattr(val, "__array__"):
                frames.append(pd.DataFrame(val, columns=[key]))
        if not frames:
            return None
        return pd.concat(frames, axis=1, join="outer").sort_index()

    return None


def _iter_acquisition_dirs(root: Path) -> Iterator[Path]:
    """Yield each unique HSD acquisition directory under root."""
    seen: set[Path] = set()
    for d in root.rglob('*'):
        candidate: Optional[Path] = None
        if d.is_dir() and _is_hsd_folder(d):
            candidate = d
        elif d.suffix.lower() == '.dat':
            candidate = d.parent
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            yield candidate


# ── main iterator ────────────────────────────────────────────────────────────

def _raw_col_fingerprint(df: pd.DataFrame) -> frozenset[str]:
    """
    Return a frozenset of the non-time raw column headers in a DataFrame.
    Used to detect when two virtual sensor names resolve to the same
    underlying .dat data (SDK bug for combined-dat chips).

    e.g. DataFrame(['Time','TEMP [Celsius]']) → frozenset({'TEMP [Celsius]'})
    """
    return frozenset(
        c for c in df.columns
        if "time" not in c.lower()
    )


def iter_hsd_items(
    root: str | Path,
    only_active: bool = True,
    verbose: bool = False,
) -> Iterator[Dict[str, object]]:
    """
    Yield one bag dict per active sub-sensor per acquisition folder.

    SDK behaviour (HSD v1 + stdatalog_core):
    - get_sensor_list()  returns a list of dicts, one per virtual sub-sensor
                         e.g. [{'hts221_temp': {...}}, {'hts221_hum': {...}}]
    - get_sensor()       accepts a virtual sub-sensor name string
    - get_dataframe()    ALWAYS returns a list[DataFrame] — one element per
                         acquisition chunk.  The list must be row-concatenated.
    - BUG: for combined-dat chips (hts221, lps22hh) the SDK returns the SAME
           data (sub-sensor[0]'s column) regardless of which virtual name is
           passed.  Detection: two virtual names whose concatenated DataFrames
           share identical non-time column headers → deduplicate by raw header.
    """
    if HSDatalog is None:
        raise ImportError(
            f"Could not import HSDatalog from stdatalog_core: {_IMPORT_ERR}"
        )

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    for acq_dir in _iter_acquisition_dirs(root):
        meta = _infer_meta_from_path(acq_dir)

        try:
            hsd = HSDatalog()
            try:
                _ = hsd.validate_hsd_folder(str(acq_dir))
            except Exception as e:
                if verbose:
                    print(f"⚠️  Invalid HSD folder {acq_dir}: {e}")
                continue

            hsd_instance = hsd.create_hsd(acquisition_folder=str(acq_dir))
            odr_map      = get_odr_map(acq_dir)

            # ── 1. Resolve sensor names ──────────────────────────────────────
            # get_sensor_list returns list[dict] in HSD v1:
            #   [{'hts221_temp': {odr, dim, ...}}, {'hts221_hum': {...}}, ...]
            # Extract the string key from each dict (or use as-is if already str).
            raw_list = hsd.get_sensor_list(hsd_instance, only_active=only_active)
            sensor_names: list[str] = []
            for item in raw_list:
                if isinstance(item, str):
                    sensor_names.append(norm(item))
                elif isinstance(item, dict):
                    # Each dict has exactly one key: the sensor name
                    sensor_names.extend(norm(k) for k in item.keys())
                else:
                    # Fallback: SDK object with a name attribute
                    try:
                        sensor_names.append(
                            norm(hsd.get_sensor_name(hsd_instance, item))
                        )
                    except Exception:
                        pass

            # ── 2. Fetch each virtual sub-sensor and deduplicate by raw data ─
            #
            # The SDK bug: for combined-dat chips, every virtual sub-sensor
            # name resolves to the SAME raw data (sub-sensor[0]'s column).
            # We detect this by comparing the frozenset of non-time column
            # headers across all virtual names for the same parent chip.
            #
            # Strategy: iterate all virtual names; for each one, fetch and
            # row-concat the list of chunk DataFrames.  Track which (parent_chip,
            # raw_col_fingerprint) pairs we have already yielded — if a new
            # virtual name returns an already-seen fingerprint for that chip,
            # skip it (it's a duplicate).  This handles both the broken SDK
            # case (same column returned twice) and a hypothetical fixed SDK
            # (different columns returned for each virtual name).

            # keyed by (parent_chip, frozenset_of_raw_cols) → already yielded
            seen_fingerprints: set[tuple[str, frozenset]] = set()

            for sensor_name in sensor_names:
                parent_chip = sensor_name.split("_")[0]

                try:
                    sensor_obj = hsd.get_sensor(hsd_instance, sensor_name)
                    df_obj     = hsd.get_dataframe(hsd_instance, sensor_obj)
                except Exception as e:
                    if verbose:
                        print(f"⚠️  get_dataframe failed for {sensor_name}: {e}")
                    continue

                # Row-concat the list of chunk DataFrames
                df = _normalize_dataframe(df_obj)
                if df is None or df.empty:
                    if verbose:
                        print(f"⚠️  Empty data for {sensor_name}")
                    continue

                # Deduplicate: has this exact raw data already been yielded
                # for this parent chip?
                fingerprint = (parent_chip, _raw_col_fingerprint(df))
                if fingerprint in seen_fingerprints:
                    if verbose:
                        print(
                            f"⚠️  Skipping {sensor_name}: SDK returned duplicate "
                            f"data (same columns as a previously processed "
                            f"sub-sensor of {parent_chip}). "
                            f"Raw cols: {_raw_col_fingerprint(df)}"
                        )
                    continue
                seen_fingerprints.add(fingerprint)

                # Normalize column names (raw SDK headers → 'time','prs','temp',…)
                # Pass the virtual sensor name so normalize_sensor_columns can
                # use sensor_type hints where needed (e.g. x/y/z for acc/gyro).
                df = normalize_sensor_columns(df.copy(), sensor_name)

                sensor_type = get_sensor_type(sensor_name)
                odr         = odr_map.get(sensor_name)

                yield {
                    'condition':   meta['condition'],
                    'belt_status': meta['belt_status'],
                    'sensor':      sensor_name,
                    'sensor_type': sensor_type,
                    'rpm':         meta['rpm'],
                    'data':        df,
                    'path':        meta['full_path'],
                    'odr':         odr,
                }

        except Exception as e:
            if verbose:
                print(f"⚠️  Skipping acquisition {acq_dir}: {e}")
            continue


__all__ = ['iter_hsd_items']