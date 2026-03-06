# core/plot_raw.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

from utils import fetch_bags, filter_bags, group_by_sensor_name



def plot_frequency_spectrum(bag: dict, axis: str | None = None):
    """
    Unified, robust FFT plot for ANY supported sensor bag.

    Returns:
        matplotlib.figure.Figure | None
    """

    sensor_type = bag.get("sensor_type")
    sensor_name = bag.get("sensor")
    df: pd.DataFrame = bag.get("data")
    odr = bag.get("odr", None)

    # FFT requires valid sampling rate
    if not odr or odr <= 0:
        print(f"⚠️ Skipping {sensor_name}: no valid ODR (needed for FFT).")
        return None

    # -------------------------
    # VECTOR SENSORS (acc/gyro/mag)
    # -------------------------
    if sensor_type in {"acc", "gyro", "mag"}:
        axes = ["x", "y", "z"]

        if not all(a in df.columns for a in axes):
            print(f"⚠️ Skipping {sensor_name}: missing x/y/z columns.")
            return None

        vec = df[axes].dropna()
        if vec.empty:
            print(f"⚠️ Skipping {sensor_name}: no x/y/z samples.")
            return None

        n = len(vec)
        freqs = rfftfreq(n, d=1.0 / odr)

        fig, ax = plt.subplots()

        if axis in axes:  # single axis FFT
            sig = vec[axis].to_numpy()
            fft_vals = np.abs(rfft(sig - sig.mean()))
            ax.plot(freqs, fft_vals, label=f"{sensor_type}_{axis}")
        else:             # all axes FFT
            for ax_name in axes:
                sig = vec[ax_name].to_numpy()
                fft_vals = np.abs(rfft(sig - sig.mean()))
                ax.plot(freqs, fft_vals, label=f"{sensor_type}_{ax_name}")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Frequency Spectrum – {sensor_name} [{sensor_type}]")
        ax.set_xlim(0, odr / 2)
        ax.legend()
        fig.tight_layout()

        return fig

    # -------------------------
    # MIC SENSORS
    # -------------------------
    if sensor_type == "mic":
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        mic_cols = list(dict.fromkeys(mic_cols))

        if not mic_cols:
            print(f"⚠️ Skipping {sensor_name}: no mic-like columns.")
            return None

        mic_col = mic_cols[0]
        series = df[mic_col].dropna()
        if series.empty:
            print(f"⚠️ Skipping {sensor_name}: mic data empty.")
            return None

        try:
            first = series.iloc[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                sig = np.concatenate([np.asarray(x).ravel() for x in series])
            else:
                sig = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        except Exception as e:
            print(f"⚠️ Skipping {sensor_name}: error flattening mic data ({e})")
            return None

        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: mic signal invalid.")
            return None

        n = sig.size
        freqs = rfftfreq(n, d=1.0 / odr)
        fft_vals = np.abs(rfft(sig - np.mean(sig)))

        fig, ax = plt.subplots()
        ax.plot(freqs, fft_vals)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Frequency Spectrum – {sensor_name} [mic]")
        ax.set_xlim(0, odr / 2)
        fig.tight_layout()

        return fig

    # -------------------------
    # TEMP/HUM/PRS (skip)
    # -------------------------
    print(f"⚠️ Skipping {sensor_name}: FFT not applicable to sensor_type '{sensor_type}'.")
    return None


def _build_time_axis(n: int, odr: float | None):
    """Return (t, label) for the time axis."""
    if odr and odr > 0:#if odr(sampling rate) is available and greater than 0, use it to build the time axis
        t = np.arange(n) / odr
        label = "Time (s)"
    else:
        t = np.arange(n)#if odr(sampling rate) is not available or less than 0, use sample index to build the time axis
        label = "Sample index"
    return t, label

def is_bag_plottable(bag: dict) -> bool:
    """
    Check if this bag has the columns needed to be plotted,
    based on its sensor_type.
    """
    sensor_type = bag["sensor_type"]
    df = bag["data"]

    # Vector sensors: need x, y, z
    if sensor_type in {"acc", "gyro", "mag"}:
        return all(c in df.columns for c in ["x", "y", "z"])

    # Scalar sensors
    if sensor_type == "temp":
        return "temp" in df.columns
    if sensor_type == "hum":
        return "hum" in df.columns
    if sensor_type == "prs":
        return "prs" in df.columns

    # Mic: need at least one mic-like column
    if sensor_type == "mic":
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        return len(mic_cols) > 0

    # Anything else: not supported (for now)
    return False

def plot_time_series(bag: dict, axis: str | None = None):
    """
    Unified, robust time-series plot for ANY supported sensor bag.

    Returns:
        matplotlib.figure.Figure | None
    """

    sensor_type = bag.get("sensor_type")
    sensor_name = bag.get("sensor")
    df: pd.DataFrame = bag.get("data")
    odr = bag.get("odr", None)

    if df is None:
        print(f"⚠️ Skipping {sensor_name}: no 'data' DataFrame in bag.")
        return None

    # -------------------------
    # VECTOR SENSORS: acc/gyro/mag
    # -------------------------
    if sensor_type in {"acc", "gyro", "mag"}:
        axes = ["x", "y", "z"]

        if not all(a in df.columns for a in axes):
            print(f"⚠️ Skipping {sensor_name}: missing x,y,z columns; cols={list(df.columns)}")
            return None

        vec = df[axes].dropna()
        if vec.empty:
            print(f"⚠️ Skipping {sensor_name}: no x/y/z samples after dropna().")
            return None

        n = len(vec)
        t, t_label = _build_time_axis(n, odr)

        fig, ax = plt.subplots()

        if axis in axes:  # single axis
            ax.plot(t, vec[axis].to_numpy(), label=f"{sensor_type}_{axis}")
        else:             # all axes
            for ax_name in axes:
                ax.plot(t, vec[ax_name].to_numpy(), label=f"{sensor_type}_{ax_name}")

        ax.set_xlabel(t_label)
        ax.set_ylabel(sensor_type)
        ax.set_title(f"Time series – {sensor_name} [{sensor_type}]")
        ax.legend()
        fig.tight_layout()

        return fig

    # -------------------------
    # SCALAR SENSORS: temp/hum/prs
    # -------------------------
    if sensor_type in {"temp", "hum", "prs"}:
        col = sensor_type  # same name mapping

        if col not in df.columns:
            print(f"⚠️ Skipping {sensor_name}: expected '{col}' not in {list(df.columns)}")
            return None

        sig = df[col].dropna().to_numpy()
        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: column '{col}' is empty after dropna().")
            return None

        n = sig.size
        t, t_label = _build_time_axis(n, odr)

        fig, ax = plt.subplots()
        ax.plot(t, sig)
        ax.set_xlabel(t_label)
        ax.set_ylabel(col)
        ax.set_title(f"Time series – {sensor_name} [{sensor_type}]")
        fig.tight_layout()

        return fig

    # -------------------------
    # MIC SENSORS
    # -------------------------
    if sensor_type == "mic":
        mic_cols = [
            c for c in df.columns
            if "mic" in c.lower() or "audio" in c.lower() or "waveform" in c.lower()
        ]
        mic_cols = list(dict.fromkeys(mic_cols))

        if not mic_cols:
            print(f"⚠️ Skipping {sensor_name}: no mic-like columns; cols={list(df.columns)}")
            return None

        # Choose most dynamic column if multiple
        if len(mic_cols) > 1:
            try:
                stats = {c: df[c].std(skipna=True) for c in mic_cols}
                mic_col = max(stats, key=stats.get)
            except Exception as e:
                print(f"⚠️ Skipping {sensor_name}: cannot choose mic column ({e})")
                return None
        else:
            mic_col = mic_cols[0]

        series = df[mic_col].dropna()
        if series.empty:
            print(f"⚠️ Skipping {sensor_name}: mic column '{mic_col}' empty.")
            return None

        try:
            first = series.iloc[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                sig = np.concatenate([np.asarray(x).ravel() for x in series])
            else:
                sig = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        except Exception as e:
            print(f"⚠️ Skipping {sensor_name}: error flattening mic data ({e})")
            return None

        if sig.size == 0:
            print(f"⚠️ Skipping {sensor_name}: mic signal empty.")
            return None

        n = sig.size
        t, t_label = _build_time_axis(n, odr)

        fig, ax = plt.subplots()
        ax.plot(t, sig)
        ax.set_xlabel(t_label)
        ax.set_ylabel(mic_col)
        ax.set_title(f"Time series – {sensor_name} [mic]")
        fig.tight_layout()

        return fig

    # -------------------------
    # UNSUPPORTED SENSOR TYPE
    # -------------------------
    print(f"⚠️ Skipping {sensor_name}: unsupported sensor_type '{sensor_type}'.")
    return None

#function that plots the bags according to the provided filters
#we assume that the bags are loaded somewhere else
def time_plot(
    bags: list[dict],
    axis: str | None = None,
) -> None:
    """
    Plot time-series for one representative bag per sensor.

    Assumes:
    - bags are already filtered
    - bags contain raw data
    """

    if not bags:
        print("⚠️ time_plot: empty bag list.")
        return

    grouped = group_by_sensor_name(bags)

    representatives = {
        sensor_name: cycles[0]
        for sensor_name, cycles in grouped.items()
    }

    for bag in representatives.values():
        plot_time_series(bag, axis=axis)

def freq_plot(
    bags: list[dict],
    axis: str | None = None,
) -> None:
    """
    Plot frequency spectrum for one representative bag per sensor.

    Assumes:
    - bags are already filtered
    - bags contain raw data
    """

    if not bags:
        print("⚠️ freq_plot: empty bag list.")
        return

    grouped = group_by_sensor_name(bags)

    representatives = {
        sensor_name: cycles[0]
        for sensor_name, cycles in grouped.items()
    }

    for bag in representatives.values():
        sensor_type = bag.get("sensor_type")

        if sensor_type in {"temp", "hum", "prs"}:
            print(
                f"⚠️ Skipping frequency plot for sensor "
                f"{bag.get('sensor')} (sensor_type={sensor_type})"
            )
            continue

        plot_frequency_spectrum(bag, axis=axis)

def resolve_experiment_path(
    base_root: str,
    condition: str,
    belt_status: str,
    rpm: str | None,
    stwin: str | None,
) -> tuple[str, str | None, str | None, str]:
    """
    Resolve valid rpm/stwin and build the experiment path.
    Returns: (path, rpm, stwin, belt_status)
    """

    if belt_status not in {"OK","KO_HIGH_2mm", "KO_LOW_2mm", "KO_HIGH_4mm", "KO_LOW_4mm"}:
        if not belt_status.startswith("KO"):
            print("Invalid belt_status. Expected 'OK', 'KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_HIGH_4mm', or 'KO_LOW_4mm'. Assigning default 'OK'.")
            belt_status = "OK"
        else:
            print("Assigning default 'KO' status 'KO_LOW_4mm'.")
            belt_status = "KO_LOW_4mm"

    if condition == "vel-fissa":
        rpm = rpm or "PMS_50rpm"
        stwin = None
        subdir = rpm

    elif condition == "no-load-cycles":
        stwin = stwin or "STWIN_00008"
        rpm = None
        subdir = stwin

    else:
        raise ValueError(
            f"Invalid condition='{condition}'. "
            "Expected 'vel-fissa' or 'no-load-cycles'."
        )

    path = f"{base_root}/{condition}/{belt_status}/{subdir}/"
    return path, rpm, stwin, belt_status

def plotting(
    base_root: str,
    plot_type: str,
    sensor_type: str | None = None,
    sensor: str | None = None,
    belt_status: str = "OK",
    condition: str = "vel-fissa",
    rpm: str | None = None,
    stwin: str | None = None,
    axis: str | None = None,
    limit: int | None = None,
) -> None:
    """
    Unified wrapper for time/frequency raw plots.

    plot_type: "time" | "frequency"
    """

    # -----------------------
    # 1) Resolve path + params
    # -----------------------
    try:
        path, rpm, stwin, belt_status = resolve_experiment_path(
            base_root=base_root,
            condition=condition,
            belt_status=belt_status,
            rpm=rpm,
            stwin=stwin,
        )
    except ValueError as e:
        print(f"⚠️ {e}")
        return

    # -----------------------
    # 2) Load raw bags
    # -----------------------
    bags = fetch_bags(root=path, limit=limit, only_active=True, verbose=False)

    if not bags:
        print(f"⚠️ No raw data found at path:\n{path}")
        return
    print("✅ Loaded", len(bags), "bags from", path)
    # -----------------------
    # 3) Filter
    # -----------------------
    filtered = filter_bags(
        bags=bags,
        sensor_type=sensor_type,
        sensor=sensor,
        belt_status=belt_status,
        condition=condition,
        rpm=rpm,
    )

    if not filtered:
        print("⚠️ No bags matched the given filters.")
        return

    grouped = group_by_sensor_name(filtered)
    representatives = {k: v[0] for k, v in grouped.items()}

    # -----------------------
    # 4) Dispatch plot
    # -----------------------
    figs = []
    for bag in representatives.values():
        if plot_type == "time":
            figs.append(plot_time_series(bag, axis=axis))

        elif plot_type == "frequency":
            if bag["sensor_type"] in {"temp", "hum", "prs"}:
                print(
                    f"⚠️ Skipping frequency plot for sensor_type="
                    f"{bag['sensor_type']}"
                )
                continue
            figs.append(plot_frequency_spectrum(bag, axis=axis))
            

        else:
            print(
                f"⚠️ Unknown plot_type='{plot_type}'. "
                "Use 'time' or 'frequency'."
            )
            return 
        
    print(f"✅ Generated {len(figs)} plots for plot_type='{plot_type}'.")
    return figs
