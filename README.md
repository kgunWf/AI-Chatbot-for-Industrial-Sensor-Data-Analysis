# Statistical AI Agent for Dataset Analysis

A Streamlit-based AI agent for anomaly detection in belt regulation using STWIN sensor data. The agent supports natural language queries for time/frequency plots, feature importance analysis, and dataset inspection.

---

## Table of Contents

1. [Downloading the Dataset](#1-downloading-the-dataset)
2. [Setting Up the SDK](#2-setting-up-the-sdk)
3. [Loading Raw Sensor Data](#3-loading-raw-sensor-data)
4. [Known SDK Limitations](#4-known-sdk-limitations)
5. [Project Folder Structure](#5-project-folder-structure)
6. [Generating Features with `data_loader.py`](#6-generating-features-with-data_loaderpy)
7. [Running the Application](#7-running-the-application)
8. [Sample Prompts](#8-sample-prompts)

---

## 1. Downloading the Dataset

The raw sensor data (~4 GB) is not stored in this repository due to size constraints.

**Raw `.dat` format:**
[Download Sensor_STWIN (Google Drive)](https://drive.google.com/file/d/1kTVLvjYt6VXaEUNkX_yk3yqUrmNlWBDV/view?usp=sharing)

After downloading, unzip and place the folder so your project looks like:

```
Systems-Project/
└── data/
    └── Sensor_STWIN/
        ├── vel-fissa/
        │   ├── OK/
        │   │   ├── PMS_50rpm/
        │   │   ├── PMS_100rpm/
        │   │   └── ...
        │   ├── KO_HIGH_2mm/
        │   ├── KO_LOW_2mm/
        │   └── KO_LOW_4mm/
        └── no-load-cycles/
            ├── OK/
            └── KO_*/
```

**Pre-converted CSV format** (optional, skips SDK conversion step):
[Download converted_csv (Google Drive)](https://drive.google.com/file/d/1B5StbnPc7qC2LkN_QUzkrDD5vxLPIFg1/view?usp=drive_link)

This archive contains `.csv.gz` files that can be read directly with pandas:

```python
import pandas as pd
df = pd.read_csv("path/to/file.csv.gz")
```

---

## 2. Setting Up the SDK

The data loader depends on the **STMicroelectronics STDatalog Core SDK** (`stdatalog_core`), which is used to read binary `.dat` acquisition files produced by the STEVAL-STWINKT1B board.

### Why a virtual environment is required

The SDK has strict version pinning that can conflict with other installed packages (notably `numpy` and `pandas`). A dedicated virtual environment keeps it isolated.

### Step-by-step setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Systems-Project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
#    macOS / Linux:
source .venv/bin/activate
#    Windows:
.venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install the ST SDK
#    The SDK is distributed as a wheel — download it from ST's website:
#    https://www.st.com/en/embedded-software/fp-sns-datalog1.html
#    then install it:
pip install stdatalog_core-1.2.1-py3-none-any.whl

# 6. Install all remaining project dependencies
pip install -r requirements.txt
```

> **Note:** The SDK has been tested with `stdatalog_core==1.2.1`. Other versions may change the `get_dataframe()` behaviour described in the limitations section below.

### Verifying the SDK installation

```bash
python - <<'EOF'
from stdatalog_core.HSD.HSDatalog import HSDatalog
print("SDK import OK")
EOF
```

---

## 3. Loading Raw Sensor Data

Raw `.dat` files are read through `core/stdatalog_loader.py`, which wraps the SDK and yields one **bag** dict per active sub-sensor per acquisition folder. A bag contains:

| Key           | Type        | Description                                        |
| ------------- | ----------- | -------------------------------------------------- |
| `sensor`      | `str`       | Full sensor name, e.g. `iis3dwb_acc`               |
| `sensor_type` | `str`       | Short type: `acc`, `gyro`, `mic`, `temp`, `prs`, … |
| `belt_status` | `str`       | `OK`, `KO_HIGH_2mm`, `KO_LOW_2mm`, `KO_LOW_4mm`    |
| `condition`   | `str`       | `vel-fissa` or `no-load-cycles`                    |
| `rpm`         | `str`       | e.g. `PMS_100rpm` (or STWIN ID for no-load)        |
| `data`        | `DataFrame` | Time-series signal with normalised column names    |
| `odr`         | `float`     | Sampling rate in Hz (from `DeviceConfig.json`)     |
| `path`        | `str`       | Source acquisition folder path                     |

The loader automatically:

- Infers metadata (`belt_status`, `condition`, `rpm`) from the folder path
- Row-concatenates all acquisition chunks returned by the SDK
- Fingerprints column sets per parent chip to deduplicate SDK bugs (see §4)
- Normalises column names (`A_x [g]` → `x`, `PRESS [hPa]` → `prs`, etc.)

---

## 4. Known SDK Limitations

### Multi-subsensor chips return duplicate data

Two sensors on the STWIN board store all their sub-sensor channels in a **single `.dat` file** per acquisition:

- **LPS22HH**: pressure + temperature
- **HTS221**: temperature + humidity

The SDK exposes these as separate virtual sensor names (`lps22hh_press`, `lps22hh_temp`, `hts221_temp`, `hts221_hum`), but its `get_dataframe()` method always returns the data of the **first active sub-sensor** regardless of which virtual name is requested.

Concretely:

- Both `lps22hh_press` and `lps22hh_temp` return **pressure data**
- Both `hts221_temp` and `hts221_hum` return **temperature data**

The humidity channel of HTS221 and the temperature channel of LPS22HH are therefore **inaccessible** through this SDK version.

### How the loader handles it

The loader detects duplicates by comparing the frozenset of non-time column headers returned for each virtual name within the same parent chip. When two virtual names yield identical column fingerprints, the second is silently dropped with a warning (visible when `verbose=True`).

### Consequence for the feature matrix

The following modalities are **absent** from the final `cleaned_df.csv`:

| Missing channel                          | Reason                       |
| ---------------------------------------- | ---------------------------- |
| `hts221_hum` (humidity)                  | Duplicate of `hts221_temp`   |
| `lps22hh_temp` (temperature via LPS22HH) | Duplicate of `lps22hh_press` |

The **humidity sensor modality is entirely missing** from the dataset as a result. This is a known SDK bug, not a data collection issue, and will resolve if ST releases an updated SDK that correctly dispatches sub-sensor reads.

---

## 5. Project Folder Structure

```
Systems-Project/
├── agent/
│   ├── agent_implementation.py   # SensorDataAgent — LLM + tool wiring
│   ├── tools.py                  # LangChain Tool definitions (PlotSensor, FeatureImportance, InspectDataset)
│   ├── prompt_gen.py             # ReAct system prompt template
│   ├── debug_callback.py         # Optional LangChain callback for tool-call tracing
│   └── standalone_test.py        # Run tools directly without the agent (for debugging)
│
├── core/
│   ├── stdatalog_loader.py       # Low-level SDK wrapper — yields bag dicts from raw .dat files
│   ├── utils.py                  # Shared helpers: BagFilters, fetch_bags, filter_bags,
│   │                             #   group_by_sensor_name, get_odr_map, normalize_sensor_columns
│   ├── data_loader.py            # Client CLI: raw data → feature CSV pipeline
│   ├── feature_extraction.py     # Per-bag feature computation (time + frequency domain)
│   ├── feature_analysis.py       # Global feature selection (ANOVA + Random Forest)
│   ├── plotting.py               # Time-series and frequency-spectrum plot generation
│
|── output_dir/
│   └── processed/
│       ├── cleaned_df.csv    # ← generated by data_loader.py; used by the agent
│       ├── dictionary.json
│       └── manifest.json
│
├── interface/
│   └── chat_app.py               # Streamlit web application
│
├── data/
│   └── Sensor_STWIN/             # Raw dataset (not in repo — download separately)
│
├── models/                       # (optional) local GGUF model files
└── requirements.txt
```

### What each core module does

**`stdatalog_loader.py`** — the only file that touches the ST SDK. Iterates all HSD acquisition directories under a root path and yields one bag dict per active sub-sensor. Handles SDK quirks (multi-chunk DataFrames, duplicate sub-sensor data).

**`utils.py`** — everything shared by more than one module. Bag loading (`fetch_bags`), filtering (`filter_bags`), grouping (`group_by_sensor_name`), the `BagFilters` dataclass, sensor naming helpers, ODR parsing, and column normalisation.

**`data_loader.py`** — the user-facing pipeline. Calls `fetch_bags` from `utils`, extracts features via `feature_extraction`, and saves artifacts (`cleaned_df.csv`, `dictionary.json`, `manifest.json`). Also provides a CLI for processing and inspecting datasets.

**`feature_extraction.py`** — computes per-bag features: time-domain statistics (mean, std, RMS, skew, kurtosis, IQR, …), frequency-domain descriptors (spectral centroid, rolloff, flatness, entropy, band energies, harmonic ratio), and MFCCs for microphone data.

**`feature_analysis.py`** — global cross-sensor feature selection. Ranks features using ANOVA F-scores and Random Forest importance, then finds the optimal feature subset size by evaluating classification accuracy at different cut-offs (k = 5, 10, 15, 20, 30, …).

**`plotting.py`** — generates matplotlib figures for raw sensor signals. Supports time-domain and frequency-domain plots for all sensor types. Imports `fetch_bags`, `filter_bags`, and `group_by_sensor_name` directly from `utils.py`.

---

## 6. Generating Features with `data_loader.py`

### Process a raw dataset (first-time setup)

```bash
# From the project root, with your venv active:
python core/data_loader.py data/Sensor_STWIN --out core/output_dir/processed
```

This runs the full pipeline:

1. Walks all acquisition folders under `data/Sensor_STWIN/`
2. Loads each active sub-sensor via the SDK
3. Extracts time and frequency features from every bag
4. Saves `cleaned_df.csv`, `dictionary.json`, and `manifest.json` to the output directory

### Preview the generated dataframe

```bash
python core/data_loader.py data/Sensor_STWIN --out core/output_dir/processed --preview
```

### Load and inspect previously generated artifacts

```bash
python core/data_loader.py --load core/output_dir/processed --preview
```

### Filter while processing

```bash
# Only OK samples, accelerometer sensors
python core/data_loader.py data/Sensor_STWIN \
    --belt-status OK \
    --sensor-type acc \
    --out core/output_dir/acc_only

# Only vel-fissa condition at 100 rpm
python core/data_loader.py data/Sensor_STWIN \
    --condition vel-fissa \
    --rpm PMS_100rpm \
    --out core/output_dir/100rpm
```

### Filter an already-generated artifact

```bash
python core/data_loader.py --load core/output_dir/processed \
    --sensor-type mic \
    --preview
```

### All CLI options

```
positional argument:
  root                  Raw dataset root path (or set $STAT_AI_DATA)

options:
  --load PATH           Load existing artifacts instead of processing
  --out DIR             Output folder (default: output_dir/processed)
  --limit N             Stop after N bags (useful for quick tests)
  --include-inactive    Include inactive sensors
  --quiet               Suppress loader warnings
  --preview             Print first 5 rows of the dataframe

filters (apply in both modes):
  --sensor-type         acc | gyro | mag | mic | temp | prs
  --sensor              full name, e.g. iis3dwb_acc
  --belt-status         OK | KO_HIGH_2mm | KO_LOW_2mm | KO_LOW_4mm
  --condition           vel-fissa | no-load-cycles
  --rpm                 e.g. PMS_50rpm, PMS_100rpm
```

---

## 7. Running the Application

### Prerequisites

1. The virtual environment is active (see §2)
2. `cleaned_df.csv` has been generated and placed in `core/output_dir/processed/` (see §6)
3. **Ollama** is installed and the Qwen model is pulled:

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:7b-instruct
```

The agent uses **Qwen 2.5 7B Instruct** served locally via Ollama. It is used purely for natural-language routing — all statistical computations are performed by deterministic Python modules. No external API key is needed.

### Pointing the app at your cleaned dataframe

Open `agent/agent_implementation.py` and update the path constants near the top of the file to match your local setup:

```python
# agent/agent_implementation.py

BASE_DIR  = Path(__file__).resolve().parents[1]
CSV_PATH  = BASE_DIR / "core" / "output_dir" / "processed" / "cleaned_df.csv"
DATA_PATH = BASE_DIR / "data" / "Sensor_STWIN"
```

If you saved your artifacts to a different location (e.g. `--out my_output`), update `CSV_PATH` accordingly. `DATA_PATH` must point to the raw `Sensor_STWIN/` folder — it is used by the plot tool to load raw signals on demand.

### Starting the app

```bash
streamlit run interface/chat_app.py
```

The app will open in your browser at `http://localhost:8501`.

On first run, the agent loads `cleaned_df.csv` into memory and connects to the local Ollama instance. This may take 10–30 seconds.

### Sidebar quick actions

The sidebar provides one-click buttons for the most common operations (Dataset Info, Top Features, ACC OK Time, MIC KO Frequency) as well as a set of example query buttons. These bypass the LLM and invoke tools directly with pre-formatted inputs.

---

## 8. Sample Prompts

### Dataset inspection

```
What sensors are available in the dataset?
Show me the dataset structure
How many OK and KO samples do I have?
```

### Feature analysis

```
Show me the top features
What features best discriminate OK from KO?
Display feature importance
```

### Time-domain plots

```
acc OK time
mic KO time
gyro KO time vel-fissa
iis3dwb_acc OK time no-load-cycles STWIN_00012
mag OK time vel-fissa PMS_100rpm
```

### Frequency-domain plots

```
acc KO frequency
mic OK frequency
gyro OK frequency vel-fissa PMS_50rpm
mag KO frequency
```

### Plot command format

The `PlotSensor` tool accepts commands in the following format:

```
<sensor|sensor_type> <OK|KO> <time|frequency> [condition] [rpm|stwin_id]
```

| Argument      | Required            | Values                                       |
| ------------- | ------------------- | -------------------------------------------- |
| `sensor_type` | Yes                 | `acc`, `gyro`, `mag`, `mic`, `temp`, `prs`   |
| `sensor`      | Alternative to type | e.g. `iis3dwb_acc`, `imp34dt05_mic`          |
| `belt_status` | Yes                 | `OK`, `KO` (maps to `KO_LOW_4mm` by default) |
| `plot_type`   | Yes                 | `time`, `frequency`                          |
| `condition`   | No                  | `vel-fissa` (default), `no-load-cycles`      |
| `rpm / stwin` | No                  | e.g. `PMS_100rpm`, `STWIN_00008`             |
