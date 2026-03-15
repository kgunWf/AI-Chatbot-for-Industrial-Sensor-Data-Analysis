import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis, entropy
from sklearn.impute import SimpleImputer
import python_speech_features as psf

def _to_float(val):
    """Safely convert numpy or array values to Python float."""
    if isinstance(val, (np.ndarray, list)):
        if len(np.shape(val)) == 0:
            return float(val)
        if len(val) == 1:
            return float(val[0])
        return float(np.mean(val))  # fallback for multi-element scalar signals
    try:
        return float(val)
    except Exception:
        return np.nan

def extract_vector_stats(signal: np.ndarray, prefix: str) -> dict:
    """Compute mean/std/rms/etc. over a 3D signal vector, flattening per-axis stats."""
    axes = ["x", "y", "z"]
    out = {}

    # Defensive: handle cases where signal doesn't have exactly 3 columns
    if signal.ndim != 2 or signal.shape[1] not in (2, 3):
        raise ValueError(f"Unexpected vector shape for {prefix}: {signal.shape}")

    # Compute stats
    mean_vals = np.mean(signal, axis=0)
    std_vals = np.std(signal, axis=0)
    rms_vals = np.sqrt(np.mean(signal**2, axis=0))
    ptp_vals = np.ptp(signal, axis=0)

    # Flatten each dimension explicitly
    for i, axis in enumerate(axes[: signal.shape[1]]):
        out[f"{prefix}_mean_{axis}"] = float(mean_vals[i])
        out[f"{prefix}_std_{axis}"] = float(std_vals[i])
        out[f"{prefix}_rms_{axis}"] = float(rms_vals[i])
        out[f"{prefix}_ptp_{axis}"] = float(ptp_vals[i])

    return out


def extract_scalar_stats(signal: np.ndarray, prefix: str) -> dict:
    """Compute time-domain features for 1D scalar signals."""
     # Avoid unstable skew/kurtosis on nearly constant data
    if np.allclose(signal, signal[0]):
        skew_val = np.nan
        kurt_val = np.nan
    else:
        try:
            skew_val = _to_float(skew(signal, bias=False, nan_policy="omit"))
        except Exception:
            skew_val = np.nan
        try:
            kurt_val = _to_float(kurtosis(signal, bias=False, nan_policy="omit"))
        except Exception:
            kurt_val = np.nan

    return {
        f"{prefix}_mean": _to_float(np.mean(signal)),
        f"{prefix}_std": _to_float(np.std(signal)),
        f"{prefix}_min": _to_float(np.min(signal)),
        f"{prefix}_max": _to_float(np.max(signal)),
        f"{prefix}_median": _to_float(np.median(signal)),
        f"{prefix}_rms": _to_float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}_skew": skew_val,
        f"{prefix}_kurt": kurt_val,
        # Rolling variance
        f"{prefix}_rolling_std" : float(pd.Series(signal).rolling(10, min_periods=1).std().mean()),
        # First derivative mean
        f"{prefix}_derivative_mean" : float(np.mean(np.diff(signal))),
        # IQR
        f"{prefix}_iqr" : float(np.percentile(signal, 75) - np.percentile(signal, 25))
    }

def spectral_centroid(freqs, magnitudes):
    return np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-12)

def spectral_rolloff(freqs, magnitudes, roll_percent=0.85):
    cumulative = np.cumsum(magnitudes)
    threshold = roll_percent * cumulative[-1]
    return freqs[np.searchsorted(cumulative, threshold)]

def spectral_flatness(magnitudes):
    gm = np.exp(np.mean(np.log(magnitudes + 1e-12)))
    am = np.mean(magnitudes)
    return gm / (am + 1e-12)

def band_energy(magnitudes, freqs, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(magnitudes[mask])

def extract_features_from_bag(bag: dict) -> dict:
    sensor_type = bag["sensor_type"]
    df = bag["data"]
    odr = bag.get("odr", None)

    out = {
        "condition": bag["condition"],
        "belt_status": bag["belt_status"],
        "sensor": bag["sensor"],
        "path": bag["path"],
        "sensor_type": sensor_type,
        "rpm": bag["rpm"],
    }

# === VECTOR SIGNALS ===
    if sensor_type in {"acc", "gyro", "mag"}:
        print(f"🪲 [scalar] Processing {sensor_type} from {bag['sensor']}")
        print(f"🪲 DataFrame columns: {list(df.columns)}")
        try:
            sig = df[["x", "y", "z"]].dropna().to_numpy()
            if sig.shape[0] < 50:
                print(f"⚠️ Too few samples in {bag['sensor']}")
                out['_valid'] = False
                return out

            prefix = sensor_type
            out.update(extract_vector_stats(sig, prefix))
            
            axes_present = ["x", "y", "z"][:sig.shape[1]]
            # Compute FFT spectrum
            if odr and odr > 100 and sig.shape[1] == 3:   # only run FFT for full 3-axis
                fft_vals = np.abs(rfft(sig - sig.mean(axis=0), axis=0))
                freqs = rfftfreq(sig.shape[0], d=1.0 / odr)
                for i, axis in enumerate(axes_present):
                    magn = fft_vals[:, i]

                    # Spectral centroid
                    out[f"{prefix}_spec_centroid_{axis}"] = spectral_centroid(freqs, magn)

                    # Spectral rolloff
                    out[f"{prefix}_spec_rolloff_{axis}"] = spectral_rolloff(freqs, magn)

                    # Spectral flatness
                    out[f"{prefix}_spec_flatness_{axis}"] = spectral_flatness(magn)

                    # Entropy
                    p = magn / (np.sum(magn) + 1e-12)
                    out[f"{prefix}_spec_entropy_{axis}"] = entropy(p)

                    # Band energies (domain-specific — example values)
                    out[f"{prefix}_band_energy_low_{axis}"]  = band_energy(magn, freqs, 0, odr/10)
                    out[f"{prefix}_band_energy_mid_{axis}"]  = band_energy(magn, freqs, odr/10, odr/5)
                    out[f"{prefix}_band_energy_high_{axis}"] = band_energy(magn, freqs, odr/5, odr/2)

                    # Harmonic ratio (2nd peak / 1st peak)
                    peak1 = np.argmax(magn)
                    magn2 = magn.copy()
                    magn2[peak1] = 0
                    peak2 = np.argmax(magn2)
                    out[f"{prefix}_harmonic_ratio_{axis}"] = magn2[peak2] / (magn[peak1] + 1e-12)

        except KeyError as e:
            print(f"⚠️ Missing expected axis in {bag['sensor']}: {e}")
            out['_valid'] = False
            return out

    # === AUDIO (mic) ===
    elif sensor_type == "mic":
        #debugging info
        print(f"🪲 [scalar] Processing {sensor_type} from {bag['sensor']}")
        print(f"🪲 DataFrame columns: {list(df.columns)}")
        
        mic_cols = [c for c in df.columns if "mic" in c.lower() or "waveform" in c.lower() or "audio" in c.lower()]
        mic_cols = list(dict.fromkeys(mic_cols))
        print(f"🪲 [mic] Found mic-like columns: {mic_cols} in {bag['sensor']}")

        if not mic_cols:
            print(f"⚠️ No mic column found in {bag['sensor']}")
            out['_valid'] = False
            return out

        if len(mic_cols) > 1:
            try:
                stats = {col: df[col].std(skipna=True) for col in mic_cols}
                mic_col = max(stats, key=stats.get)
                print(f"✅ Using mic column with highest std: {mic_col}")
            except Exception as e:
                print(f"⚠️ Failed to pick mic column by std: {e}")
                mic_col = mic_cols[0]
        else:
            mic_col = mic_cols[0]

        mic_series = df[mic_col].dropna()
        if mic_series.empty:
            print(f"⚠️ mic_series empty in {bag['sensor']}")
            out['_valid'] = False
            return out

        first_val = mic_series.iloc[0]
        if isinstance(first_val, (list, tuple, np.ndarray)):
            try:
                sig = np.concatenate([np.asarray(x).ravel() for x in mic_series])
            except Exception as e:
                print(f"⚠️ mic flattening error: {e}")
                out['_valid'] = False
                return out
        else:
            try:
                sig = pd.to_numeric(mic_series, errors="coerce").dropna().to_numpy()
            except Exception as e:
                print(f"⚠️ mic signal conversion error: {e}")
                out['_valid'] = False
                return out

        if sig.size < 100:
            print(f"⚠️ mic signal too short ({sig.size}) in {bag['sensor']}")
            out['_valid'] = False
            return out

        out.update(extract_scalar_stats(sig, "mic"))

        if odr and odr > 1000:
            fft_input = sig - np.mean(sig)
            fft_input *= np.hanning(len(fft_input))
            fft_vals = np.abs(rfft(fft_input))
            freqs = rfftfreq(len(fft_input), d=1.0 / odr)
            out["mic_dom_freq"] = _to_float(freqs[np.argmax(fft_vals)])
            psd = fft_vals ** 2
            p = psd / (np.sum(psd) + 1e-12)
            out["mic_freq_entropy"] = _to_float(entropy(p, base=2))

            # Compute MFCCs
            winlen = 0.025  # 25 ms
            winstep = 0.010 # 10 ms

            samplerate = int(odr)  # microphone ODR

            frame_length = int(winlen * samplerate)
            # Next power of 2 >= frame_length, robust for any ODR
            nfft = 1 << (frame_length - 1).bit_length()
            nfft = max(nfft, 512)   # enforce minimum sensible FFT size

            # Now extract MFCC safely
            mfccs = psf.mfcc(
                sig.astype(float),
                samplerate=samplerate,
                numcep=8,
                winlen=winlen,
                winstep=winstep,
                nfft=nfft
            )

            # Export mean MFCCs
            mfcc_mean = np.mean(mfccs, axis=0)
            for i, val in enumerate(mfcc_mean):
                out[f"mic_mfcc_{i+1}"] = float(val)



    # === SCALAR SIGNALS: TEMP, HUM, PRS ===
    elif sensor_type in {"temp", "hum", "prs"}:
        #debugging info
        print(f"🪲 [scalar] Processing {sensor_type} from {bag['sensor']}")
        print(f"🪲 DataFrame columns: {list(df.columns)}")
        expected = {"temp": ["temp"], "hum": ["hum"], "prs": ["prs"]}.get(sensor_type, [])
        print(f"🪲 Expected columns for {sensor_type}: {expected}")

        found_cols = []
        for col in df.columns:
            if col in expected:
                found_cols.append(col)
                sig = df[col].dropna().to_numpy()
                print(f"🪲 Found valid column '{col}' with {sig.size} samples")
                if sig.size < 10:
                    print(f"⚠️ Not enough samples in {col}")
                    continue
                out.update(extract_scalar_stats(sig, col))

        if not found_cols:
            print(f"⚠️ No expected column found for {sensor_type} in {bag['sensor']}")
            out['_valid'] = False
            return out

    else:
        print(f"⚠️ Unknown sensor_type: {sensor_type} for {bag['sensor']}")
        out['_valid'] = False
        return out

    out['_valid'] = True
    return out


def extract_features_from_bags(bags: list[dict]) -> list[dict]:
    rows = []
    for bag in bags:
        try:
            features = extract_features_from_bag(bag)
            rows.append(features)
        except Exception as e:
            print(f"⚠️ Error processing {bag.get('sensor')}: {e}")
    return rows


def prepare_combined_feature_dataframe(bags: list[dict]) -> pd.DataFrame:
    """
    Convert list of feature dicts → single combined DataFrame.
    Missing features are imputed using per-column median.
    """
    df = pd.DataFrame(bags)

    # Normalize KO labels
    df["belt_status"] = df["belt_status"].apply(
        lambda x: "KO" if str(x).upper().startswith("KO") else "OK"
    )

    # Keep metadata but treat all others as features
    meta_cols = ["condition", "belt_status", "sensor", "sensor_type", "rpm", "path"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Convert all features to numeric (non-numeric → NaN)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace infs
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing using median
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Reassemble full dataframe
    df_cleaned = pd.concat([df[meta_cols], X_imputed], axis=1)

    return df_cleaned
