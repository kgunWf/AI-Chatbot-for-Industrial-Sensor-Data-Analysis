# core/odr_from_cfg.py
import json
from pathlib import Path
import pandas as pd
import re

import re



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
