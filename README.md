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
make data        # data_load.py  data/  ->  processed/   (sweeps all subfolders)
make proc        # proc_calc.py  processed/  (add RATE=0.5 for nm/scan depth)
make all         # both
make inspect     # list all raw_*/calc_* files
make serve       # launch the interactive browser (see below)
```

`make data` searches **`data/` recursively**, so you can drop raw files straight in
`data/` or organise them in subfolders (e.g. `data/For Jose/…`, `data/batch2/…`) — every
`.itm/.ita/.itax/.itmx` found is processed. The processed output mirrors the subfolder
structure under `processed/`. Override the input folder with
`make data DATA=/path/to/raw/files`.

## Interactive browser

A single live app (standard library + numpy on the server, Plotly on the page) served by
`sims_server.py`. It has three tabs:

- **Spectra & 2D** — raw-count spectrum per subsample (positive / negative toggle), the
  file's own expert ("true") peaks plus peaks you add here, live 2D images for the visible
  m/z range or any selected peak, and **ROI**: drag a box on the Total image (down to a
  single pixel) to get the spectrum of just that region. Zoom with the bottom range slider;
  `reset 2D` returns to a clean state.
- **Statistical methods** — placeholder for per-peak tables, replicate statistics and group
  comparisons (coming soon).
- **ML / DL** — placeholder for dimensionality reduction, clustering and classification
  (coming soon).

```bash
make serve                      # binds 0.0.0.0:8765 -> reachable on the LAN
make serve PORT=9000            # different port
make serve HOST=127.0.0.1       # this machine only
# or directly:
.venv/bin/python sims_server.py processed --host 0.0.0.0 --port 8765
```

By default the server binds all interfaces, so others on the network can open
`http://<this-machine-ip>:8765/` (find the IP with `hostname -I`). If it is unreachable from
another machine, a host firewall is likely blocking the port — open it, e.g.
`sudo ufw allow 8765/tcp`.

The interface needs `raw_events_ch.npy` / `raw_events_pix.npy` (written by `data_load.py`
for `.itm` files) for the on-demand images and ROI spectra. Peaks you create are stored in
`processed/user_peaks.json`.

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