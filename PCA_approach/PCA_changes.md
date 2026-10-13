# PCA on identified ("true") peaks — changes & use case

Target file: `../sims_server.py` (the `check_manualannotatedspectra.py` shim just launches it).
Reference: Yang et al., *Biointerphases* 14, 051004 (2019) — see `PCA_preprocessing_Yang_2019.md`.

Status: **implemented** in `sims_server.py` (smoke-tested on `processed/`).

## What are the PCA axes?

PCA finds new axes (principal components, PC1, PC2, …) that are linear combinations of the
input features. Here the **features are the identified peaks** (one integrated intensity per
peak). So:

- **A point = one subsample** (e.g. `6PPD_pos3`).
- **PC1 (x axis)** is the direction of largest variance across the subsamples; **PC2 (y axis)**
  is the next largest and is orthogonal to PC1. The `%` in each axis label is the share of
  total variance that component explains.
- **Scores** = where each subsample sits on PC1/PC2 (the scatter plot). Samples that cluster
  together are chemically similar.
- **Loadings** = how much each peak contributes to a PC (the horizontal bar chart). Peaks with
  large positive/negative loadings on PC1 are what separate the samples along the x axis.

The axes are **not** m/z. m/z appears only in the loadings, which tell you *which peaks* build
each axis.

## Visualization (updated)

The Statistical-methods tab now shows:

1. **Scores scatter** — one marker per subsample, coloured by label (group).
2. **x axis / y axis pickers** — choose which component goes on each axis (PC1 vs PC2, PC1 vs
   PC3, …) without re-running PCA.
3. **Scree bar** — explained variance (%) per component.
4. **Loadings bar chart** — top peaks (by |weight|) driving the component on the x axis, labelled
   by peak name + m/z; green = positive, red = negative.

## Complete PCA use case

1. Load data: `make data` reads the IONTOF files into `processed/` (spectra, images,
   `raw_peaks.parquet` = the expert "true" peaks).
2. `make serve`, open the browser, go to the **Spectra & 2D** tab, pick polarity (pos/neg).
3. (Optional) **tag your own peaks**: zoom to a mass, click *from zoom*, give it a name, and add.
   The peak is stored with its `cmass` in `processed/user_peaks.json` and becomes a PCA feature.
4. Go to the **Statistical methods** tab. Add the subsamples you want to compare
   (`+ all positive` / `+ all negative`) and give each a **label** (the group you expect to
   separate, e.g. a treatment or time point).
5. Set parameters: components (how many PCs to compute), cmass tol, normalize = total-ion,
   sqrt = on. Click **Run PCA**.
6. Read the result:
   - In the **scores** plot, check whether your labelled groups separate.
   - Use the **x/y axis pickers** to look at PC1–PC2, PC1–PC3, etc.
   - In the **loadings** bar chart, read off the peaks that drive the separation (by name).
   - The **scree** plot tells you how many PCs are worth looking at.
7. Interpret: the top-loading peaks are the chemistry that distinguishes your groups. Cross-check
   them against the peak table on the first tab.

### What you need for a PCA analysis

- At least **2 subsamples** (more is better; groups need replicates to be meaningful).
- A set of **identified peaks** (the expert "true" peaks are enough; add your own for anything
  the pipeline missed).
- A **label per subsample** so the scores plot can be coloured by group.

## Context / decisions

- Data already contains **positive and negative** polarities, and the peaks are **already
  calibrated** ("true" peaks stored in `raw_peaks.parquet`). So Yang's calibration and
  interference-removal steps are not needed on our side.
- **Peak names are consistent across samples** → features are matched **by peak name**,
  with `cmass` as a tolerance-based fallback when a name is missing.
- **No medium / control subsamples yet** → the Yang "90% of max in medium" filter is
  **not implemented now** (left as a future option).
- Feature source changes from **every 1 u bin of the whole spectrum** (includes noise and
  matrix peaks) to **only the identified peaks**:
  - expert **"true"** peaks (tagged, from `raw_peaks.parquet`), plus
  - **user-labeled** peaks that I add manually.
- Because we only use identified peaks, the **PDMS (147, 221, 281, 295, 369) and Bi
  primary-ion (209, 418, 657) exclusion is no longer required** — those are not tagged as
  true peaks, so they never enter the feature matrix. (Bi = primary ion beam; PDMS = device.)
- **User-labeled peaks must carry a `cmass`** too, so they align with true peaks the same way.

## Changes to `sims_server.py`

### 1. User peaks get a `cmass`
- `POST /api/peaks` accepts `{name, lo, hi, cmass?}`; if `cmass` is missing it defaults to
  `(lo + hi) / 2`. Records in `processed/user_peaks.json` are now `{id, name, cmass, lo, hi}`.
- The "Add a peak" form has an optional **cmass** field (auto-filled to the window midpoint by
  *from zoom*). The peak table shows the stored `cmass`.

### 2. Peak-based feature vector
- `Store.peak_counts(name, windows)` — for each peak window `[lo, hi)` it integrates the raw
  counts in that m/z range straight from the sorted ion-event channels (`np.searchsorted`),
  giving **one intensity value per peak**. This is what turns a spectrum into a PCA feature row.
- `Store.feature_peaks(names, tol)` — builds the **union of identified peaks** across the
  selected subsamples: expert "true" peaks (`raw_peaks.parquet`) + the global user peaks.

#### Why the `cmass` ±tolerance fallback?
Peaks are aligned across subsamples **by name** (names are consistent). The `cmass` ±`tol`
check is only a **fallback for unnamed peaks** (e.g. a quick user peak with no label): two
unnamed peaks whose `cmass` is within `tol` (default 0.25 u) are treated as the **same feature**
instead of creating duplicate columns. Named peaks ignore the tolerance.

### 3. Preprocessing order (Yang recipe) in `compute_pca`
In order, matching Yang et al. 2019:
1. **Total-ion normalization** per subsample (over the selected peaks).
2. **Square-root transform**.
3. **Mean centering**.

Only **mean centering** is used — Yang's recipe mean-centers (it does not z-score). z-score
would give every peak equal weight and inflate weak/noisy peaks, so it was removed.

### 4. API / UI
- `/api/pca` body: `{samples, n_components, normalize, sqrt, tol, source}`.
- `source` selects the peak pool: `"both"` (true + mine, default), `"true"` (expert only),
  or `"mine"` (user peaks only).
- Statistical-methods tab controls: components, **peaks** (true + mine / true only / mine only),
  cmass tol, normalize, sqrt, and **x/y axis pickers** for the scores plot.
- Loadings report **peak name + m/z** (scatter + horizontal bar chart).

## The process from Yang et al. 2019

(Full detail in `PCA_preprocessing_Yang_2019.md`.) Their pipeline:

- **Calibration** of the mass axis (positive: Si+, Bi+, Bi2+, Bi3+; negative: Si-, SiO2-, PO2-,
  PO3-, Si2HO5-, Si3HO7-). *Our data is already calibrated, so we skip this.*
- **Peak selection** (the paper's main message): remove systematic interference (PDMS
  147/221/281/295/369, Bi primary ions 209/418/657), then a **spectral-overlay** step that
  drops peaks dominated by the culture medium (a peak is removed if the medium sample is ≥90%
  of the max across samples). Keep characteristic peaks (amino acids, lipids, water clusters,
  fatty acids). *We instead use the already-identified "true" + user peaks, so PDMS/Bi never
  enter the matrix and the 90% medium filter is not needed yet (no control samples).*
- **Preprocessing before PCA**: total-ion normalize → square-root → mean-center. *Implemented
  exactly (see section 3).*
- **PCA** on the resulting matrix, read via scores (sample separation) and loadings (which
  peaks drive each PC). *Same as our scores/loadings views.*

## Not doing now (future)
- 90% medium-matrix filter (needs medium/control subsamples).
- PDMS / Bi explicit exclusion list (unnecessary while features = identified peaks only).
