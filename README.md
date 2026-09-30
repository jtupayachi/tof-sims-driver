# TOF-SIMS Driver

Two-stage pipeline for IONTOF TOF-SIMS data (`.itm` / `.ita` / `.itax` / `.itmx`):

1. **`data_load.py`** (READ) — extracts everything stored in the files via [pySPM](https://github.com/scholi/pySPM) into `raw_*` artifacts:
   - `raw_spectrum.parquet` — channel, mass, counts
   - `raw_summary.json` — metadata
   - `raw_peaks.parquet` — peak table (id, name, SN, masses)
   - `raw_images.npz` — summed image per peak
   - `raw_profiles.parquet` — intensity vs scan per peak
   - `raw_snapshots.npz` — camera images
2. **`proc_calc.py`** (CALC) — reads `raw_*`, writes `calc_*`:
   - `calc_spectrum_peaks.parquet` — peak detection (scipy `find_peaks`)
   - `calc_summary.json` — total counts, peak count, base peak
   - `calc_profiles.parquet` — profiles normalised to total, optional `depth_nm` via `--rate`
   - `calc_image_stats.parquet` — per-peak image statistics

## Setup

```bash
# pySPM is not vendored here — clone it next to this repo:
git clone https://github.com/scholi/pySPM.git

make venv        # creates .venv, installs pySPM (editable) + pandas/pyarrow/scipy
```

## Usage

```bash
make data        # data_load.py  "For Jose"  ->  processed/
make proc        # proc_calc.py  processed/  (add RATE=0.5 for nm/scan depth)
make all         # both
make inspect     # list all raw_*/calc_* files
```

The Makefile auto-detects its own directory; override the data folder with
`make data DATA=/path/to/raw/files` if your raw data lives elsewhere.



```

                         raw_spectrum.parquet
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
          Numerical peaks    pyOpenMS          matchms
                 |          centroid/isotope    reference
                 |                |             matching
                 +----------------+----------------+
                                  |
                                  v
                            Qwen TEXT ONLY
                                  |
              +-------------------+-------------------+
              |                   |                   |
           1–10                11–20                21–30
        peak/ion             compound/library      final
        validation            disagreement        consensus
              |                   |                   |
              +-------------------+-------------------+
                                  |
                                  v
                         final_analysis JSON
                                  |
                                  v
                         annotated PNG
```             