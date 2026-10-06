# TOF-SIMS Driver

A pipeline for reading, processing, and interactively exploring **IONTOF TOF-SIMS** data
(`.itm`, `.ita`, `.itax`, `.itmx` files). It extracts raw spectra, 2D images, and peak
annotations from the instrument files, then serves an interactive browser for
spectrum inspection, peak picking, ROI analysis, and PCA.

## Repository layout

| File / dir | Purpose |
|---|---|
| `data_load.py` | **READ step.** Parses IONTOF files and writes `raw_*` outputs (spectrum, peaks, images, ion events, summary). |
| `data_probe.py` | Dumps the internal block tree of an IONTOF file (debugging / format exploration). |
| `probe_itm_itmx.py` | Prints the raw binary structure of `.itm` / `.itmx` files to guide reader development. |
| `sims_server.py` | **Interactive browser** (single-page app, 3 tabs: *Spectra & 2D*, *Statistical methods*, *ML / DL*). |
| `check_manualannotatedspectra.py` | Backward-compat shim — just launches `sims_server.py`. |
| `check_pipelinepeakidentification.py` | (Mostly commented out) earlier peak-identification experiment. |
| `Makefile` | Orchestration: venv setup, data loading, processing, probing, serving. |
| `pySPM/` | Nested clone of [scholi/pySPM](https://github.com/scholi/pySPM) (has its own repo; git-ignored here). |
| `data/` | Raw instrument files (git-ignored). |
| `processed/` | Pipeline output (git-ignored, regenerable). |

## Requirements

- **Python 3.10+**
- **Java 17** (required by PySpark, used in the processing step)
  ```bash
  sudo apt update && sudo apt install -y openjdk-17-jdk
  ```
- An NVIDIA GPU is **optional** — enables cuML-accelerated PCA in the *Statistical methods* tab.
  Without it the server falls back to scikit-learn, then numpy SVD.

## Quick start

```bash
# 1. Create the venv and install dependencies (pySPM, pandas, PySpark, etc.)
make venv

# 2. Load raw IONTOF files from ./data into ./processed
make data

# 3. (Optional) Run the processing / calculation step
make proc RATE=0.5        # sputter rate in nm/scan

# 4. Launch the interactive browser
make serve                 # http://0.0.0.0:8765/  (LAN-reachable)
# or restrict to localhost:
make serve HOST=127.0.0.1
```

`make all` runs `data` + `plots` + `proc` in sequence.

### Optional: GPU acceleration for PCA

```bash
make gpu                   # installs cuml-cu12 (needs NVIDIA GPU + ~2 GB disk)
```

## Make targets

| Target | Description |
|---|---|
| `venv` | Create `.venv` and install all Python dependencies. |
| `java` | Verify Java is installed and print `JAVA_HOME`. |
| `data` | Run `data_load.py` over `./data` → `./processed`. |
| `proc` | Run `proc_calc.py` on `./processed` (requires `RATE`). |
| `probe` | Dump block trees for `.itax` / `.itm` / `.itmx` files (default base: `6PPD neg1`). |
| `inspect` | List all `raw_*` / `calc_*` files under `./processed`. |
| `serve` | Start the interactive browser (default `0.0.0.0:8765`). |
| `plots` | Alias for `serve` (kept for backward compatibility). |
| `gpu` | Install cuML for GPU-accelerated PCA. |
| `clean` | Remove `./processed` entirely. |
| `clean-calc` | Remove only `calc_*` files (keeps `raw_*`). |

## Data flow

```
data/  (*.itm, *.ita, *.itax, *.itmx)
  │
  │  make data  →  data_load.py
  ▼
processed/<subsample>/<ext>/
  ├── raw_spectrum.parquet      channel, mass, counts
  ├── raw_summary.json          metadata (sf, k0, image dims, event stats)
  ├── raw_peaks.parquet         expert-annotated peaks (name, SN, lmass/cmass/umass)
  ├── raw_images.npz            2D image per peak + 'total'
  ├── raw_profiles.parquet      intensity vs scan per peak (when available)
  ├── raw_events_ch.npy         ion-event channel keys (for on-demand spectra)
  ├── raw_events_pix.npy        ion-event pixel keys (for on-demand 2D images)
  └── raw_snapshots.npz         camera images (.itax only)
```

### Format support

| Extension | Reader | Notes |
|---|---|---|
| `.itm` | `do_itm_raw` | No stored spectrum — rebuilt from raw ion events. Also builds 2D images and saves ion-event arrays for the interactive server. |
| `.ita` | `do_itm_ita` (pySPM) | Stored spectrum + images + peak list. |
| `.itax` | `do_itax` (pySPM) | Stored spectrum + mass intervals + per-scan profiles + camera snapshots. |
| `.itmx` | `do_itmx` | Peaks + metadata only; the SIMSData blob is not yet decoded. |

## Interactive browser (`sims_server.py`)

A single-page app with three tabs:

1. **Spectra & 2D** — pick polarity (pos/neg) and subsample; view the raw-count spectrum
   with a range slider; toggle expert ("true") peaks and user-created peaks; view the 2D
   image for any m/z range or peak; draw a box on the total image to get a ROI spectrum
   (down to a single pixel). User peaks persist in `processed/user_peaks.json`.

2. **Statistical methods** — add subsamples with group labels, then run PCA on their
   raw-count spectra (cuML → scikit-learn → numpy SVD fallback chain).

3. **ML / DL** — placeholder for future machine-learning workflows.

### HTTP API

| Endpoint | Description |
|---|---|
| `GET /` | Single-page app (`?pol=pos\|neg` preselects polarity). |
| `GET /api/samples` | List of samples with polarity, dimensions, event count. |
| `GET /api/truepeaks?sample=S` | Expert peaks for a sample. |
| `GET /api/spectrum?sample=S&...` | Raw-count spectrum; optional ROI box, bin width, mass max. |
| `GET /api/image?sample=S&lo=&hi=` | 2D image for an m/z range (uint32 LE). |
| `GET /api/image?sample=S&all=1` | 2D image of every ion. |
| `GET /api/peaks` | User peaks. |
| `POST /api/peaks` | Create a user peak `{name, lo, hi}`. |
| `DELETE /api/peaks?id=N` | Remove a user peak. |

## Debugging / format exploration

```bash
# Dump the block tree of a single file
make probe                          # uses PROBE_BASE="6PPD neg1"
make probe PROBE_BASE="my sample"   # different base name

# Print the raw binary structure of .itm / .itmx files
.venv/bin/python probe_itm_itmx.py "data/my_file.itm" > probe_report.txt
```

## Notes

- `pySPM/` is a nested git repository (scholi/pySPM) and is excluded from this repo via
  `.gitignore`. It is installed in editable mode (`pip install -e pySPM/`) into the venv.
- Raw instrument data (`data/`) and pipeline output (`processed/`) are git-ignored.
- The `bk.*` files are backups of earlier versions of the active scripts.
