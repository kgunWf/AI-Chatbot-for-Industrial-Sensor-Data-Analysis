# core/utils.py
"""
Shared helpers used across the core pipeline modules.

Everything that is needed by MORE THAN ONE module lives here to avoid
duplication.  Currently this covers:

  Data structures
  ---------------
  BagFilters               filter spec shared by data_loader, plotting, feature_analysis

  Bag utilities  (used by data_loader AND plotting)
  ---------------
  fetch_bags()             load HSD bags from disk
  filter_bags()            filter a bag list by metadata fields
  group_by_sensor_name()   partition bags into a dict keyed by sensor name

  Sensor naming helpers  (used by stdatalog_loader AND feature_extraction)
  ----------------------
  norm()                   lowercase + underscore normalisation
  SUB                      verbose subtype → short abbreviation map
  get_sensor_type()        infer "acc" / "mic" / … from a full sensor name
  get_odr_map()            parse DeviceConfig.json → {sensor: odr_hz}
  normalize_sensor_columns()  standardise raw SDK column names, merge duplicates
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# BagFilters
# ---------------------------------------------------------------------------

@dataclass
class BagFilters:
    """
    Unified filter specification shared by bag-level and DataFrame-level filters.
    All fields default to None (= no filter applied for that dimension).
    """
    sensor_type: Optional[str] = None
    sensor:      Optional[str] = None
    belt_status: Optional[str] = None
    condition:   Optional[str] = None
    rpm:         Optional[str] = None


# ---------------------------------------------------------------------------
# Bag loading & filtering
# Used by: data_loader.py (client CLI), plotting.py, standalone_test.py
# ---------------------------------------------------------------------------

def fetch_bags(
    root: str | Path,
    limit: int | None = None,
    only_active: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Load HSD bags from *root* with an optional row limit.

    Parameters
    ----------
    root        : dataset root directory
    limit       : stop after this many bags (None = all); useful for quick tests
    only_active : skip inactive sub-sensors when True
    verbose     : print per-bag loader warnings

    Returns
    -------
    list[dict]  one bag per active sub-sensor, with keys:
                condition, belt_status, sensor, sensor_type, rpm, data, path, odr
    """
    from stdatalog_loader import iter_hsd_items  # deferred: heavy SDK import

    bags: list[dict[str, Any]] = []
    for index, bag in enumerate(
        iter_hsd_items(root, only_active=only_active, verbose=verbose),
        start=1,
    ):
        bags.append(bag)
        if limit is not None and index >= limit:
            break
    return bags


def filter_bags(
    bags: list[dict[str, Any]],
    filters: BagFilters | None = None,
    *,
    # Convenience keyword overrides — avoids constructing BagFilters for
    # simple one-off calls (both plotting.py and data_loader need this)
    sensor_type: str | None = None,
    sensor:      str | None = None,
    belt_status: str | None = None,
    condition:   str | None = None,
    rpm:         str | None = None,
) -> list[dict[str, Any]]:
    """
    Return bags matching every non-None filter field.

    Accepts a ``BagFilters`` instance *or* plain keyword arguments.
    When both are given, keywords win over the dataclass values.

    Examples
    --------
    >>> filter_bags(bags, BagFilters(sensor_type="acc", belt_status="OK"))
    >>> filter_bags(bags, sensor_type="acc", belt_status="OK")  # equivalent
    """
    f = filters or BagFilters()
    effective = BagFilters(
        sensor_type=sensor_type if sensor_type is not None else f.sensor_type,
        sensor=     sensor      if sensor      is not None else f.sensor,
        belt_status=belt_status if belt_status is not None else f.belt_status,
        condition=  condition   if condition   is not None else f.condition,
        rpm=        rpm         if rpm         is not None else f.rpm,
    )

    out: list[dict[str, Any]] = []
    for bag in bags:
        if effective.sensor_type and bag.get("sensor_type") != effective.sensor_type:
            continue
        if effective.sensor      and bag.get("sensor")      != effective.sensor:
            continue
        if effective.belt_status and str(bag.get("belt_status")) != effective.belt_status:
            continue
        if effective.condition   and str(bag.get("condition"))   != effective.condition:
            continue
        if effective.rpm         and str(bag.get("rpm"))         != effective.rpm:
            continue
        out.append(bag)
    return out


def group_by_sensor_name(
    bags: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Partition *bags* into a dict keyed by sensor name.

    Returns
    -------
    {"iis3dwb_acc": [bag, ...], "imp34dt05_mic": [...], ...}
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for bag in bags:
        grouped.setdefault(str(bag.get("sensor", "unknown")), []).append(bag)
    return grouped


# normalize names like "IIS3DWB_ACC" → "iis3dwb_acc"
def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

# map verbose subtype names to your suffix style
SUB = {
    "accelerometer":"acc","accel":"acc","acc":"acc",
    "gyroscope":"gyro","gyro":"gyro",
    "magnetometer":"mag","mag":"mag",
    "temperature":"temp","temp":"temp",
    "humidity":"hum","hum":"hum",
    "pressure":"prs","press":"prs",
    "microphone":"mic","audio":"mic","mic":"mic",
}
def normalize_sensor_columns(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """
    Standardize column names like 'MIC [Waveform]' to 'mic', etc.,
    and remove duplicate columns (e.g., prs/prs_1, time/time_1)
    by merging their values.
    """
    import re
    import pandas as pd

    df = df.copy()
    sensor_type = get_sensor_type(sensor)

    # --- Step 1: Normalize column names ---
    new_cols = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if sensor_type in {"acc", "gyro", "mag"} and "x" in col_lower:
            new_cols[col] = "x"
        elif sensor_type in {"acc", "gyro", "mag"} and "y" in col_lower:
            new_cols[col] = "y"
        elif sensor_type in {"acc", "gyro", "mag"} and "z" in col_lower:
            new_cols[col] = "z"
        elif "temp" in col_lower:
            new_cols[col] = "temp"
        elif "hum" in col_lower:
            new_cols[col] = "hum"
        elif "press" in col_lower or "prs" in col_lower:
            new_cols[col] = "prs"
        elif any(k in col_lower for k in ["mic", "audio", "waveform"]):
            new_cols[col] = "mic"
        elif "time" in col_lower:
            new_cols[col] = "time"
        else:
            new_cols[col] = col_lower
    df.rename(columns=new_cols, inplace=True)

   # --- Step 2: Merge duplicates safely ---
    merged = {}
    for col in df.columns:
        base = re.sub(r'_\d+$', '', col)
        col_data = df[col]

        # if accidentally multi-column, reduce to first column
        if isinstance(col_data, pd.DataFrame):
            if len(col_data.columns) == 1:
                col_data = col_data.iloc[:, 0]
            else:
                col_data = col_data.mean(axis=1, numeric_only=True)  # fallback

        if base not in merged:
            merged[base] = col_data
        else:
            left = merged[base]
            if isinstance(left, pd.DataFrame):
                # same safety fallback
                if len(left.columns) == 1:
                    left = left.iloc[:, 0]
                else:
                    left = left.mean(axis=1, numeric_only=True)
            merged[base] = left.combine_first(col_data)

    # --- Step 3: Build final dataframe ---
    df = pd.DataFrame(merged)

    # --- Step 4: Optional reorder: time first if present ---
    cols = df.columns.tolist()
    if "time" in cols:
        cols = ["time"] + [c for c in cols if c != "time"]
        df = df[cols]

    return df



def get_sensor_type(sensor_name: str) -> str:
    """
    Infers the sensor type (acc, gyro, mic, etc.) from a full sensor or sensor+subtype name.
    Examples:
        "IIS3DWB_ACC" → "acc"
        "HTS221_TEMP" → "temp"
    """
    parts = norm(sensor_name).split("_")
    for p in reversed(parts):
        if p in SUB:
            return SUB[p]
    return "unknown"

def get_odr_map(acq_dir: str | Path) -> dict[str, float]:
    """
    Return {'lps22hh_temp': 199.2, 'lps22hh_press': 199.2, ...}
    by directly matching sub-sensor names as used by stdatalog_loader.
    """
    cfg = Path(acq_dir) / "DeviceConfig.json"
    if not cfg.exists():
        print(f"❌ Missing DeviceConfig.json at {cfg}")
        return {}

    try:
        j = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return {}

    sensors = (j.get("device") or {}).get("sensor") or []
    out = {}

    for s in sensors:
        sensor_name = norm(s.get("name", ""))  # e.g., 'lps22hh'
        descriptors = (s.get("sensorDescriptor") or {}).get("subSensorDescriptor") or []
        statuses = (s.get("sensorStatus") or {}).get("subSensorStatus") or []

        for d, st in zip(descriptors, statuses):
            if not st.get("isActive", True):
                continue
            
            raw_sub = d.get("sensorType", "").strip().lower()
            short_sub = SUB.get(raw_sub, raw_sub)               # "press" → "prs"
            full_key = norm(f"{sensor_name}_{short_sub}")       # → "lps22hh_prs"
            
            odr = st.get("ODRMeasured") or st.get("ODR")
            if odr is not None:
                try:
                    out[full_key] = float(odr)
                except Exception as e:
                    print(f"⚠️ Could not assign ODR for {full_key}: {e}")

    return out
