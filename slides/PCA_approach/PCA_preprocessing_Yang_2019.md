# PCA preprocessing of ToF-SIMS spectra (Yang et al., 2019)

Source: C. Yang, W. Wei, F. Liu, X.-Y. Yu, "Peak selection matters in principal component analysis: A case study of syntrophic microbes", *Biointerphases* 14, 051004 (2019), doi:10.1116/1.5118237.

Files reviewed in this folder (only two PDFs are present, and both belong to the same paper):
- `42_peak_selectrion_matters_Biointerphases_2019.pdf` (main paper)
- `42_peak_selection_matters_2019_SI-WE31923.pdf` (supporting information)

## The 3 preprocessing steps applied before spectral PCA (Sec. II.D)

Applied to the mass-calibrated spectra (MATLAB R2012a for PCA; scores/loadings plotted in Origin):

1. **Normalization to total ion intensity.** Each spectrum is divided by its total (selected) ion intensity.
2. **Square-root transformation.** Applied to the normalized intensities.
3. **Mean centering.** The mean of each peak (variable) across samples is subtracted before running PCA.

## Prerequisite: peak selection (Sec. II.E, Fig. 2)

The paper's main point is that peak selection is the most important step before PCA. Normalization above is applied to the *selected* peaks.

- **Calibration**
  - Positive mode: Si+, Bi+, Bi2+, Bi3+.
  - Negative mode: Si-, SiO2-, PO2-, PO3-, Si2HO5-, Si3HO7-.
- **Remove systematic interference peaks** ("all peak" list = everything left after this step)
  - PDMS from the device: m/z+ 147, 221, 281, 295, 369.
  - Bi primary ions: m/z+ 209, 418, 657.
- **Spectral overlay to reduce the medium matrix effect** ("selected peak" list)
  1. For each unit mass, take the maximum normalized intensity across all samples.
  2. If the medium sample reaches more than 90% of that maximum and dominates the medium spectra, remove the peak.
  3. Run PCA on the reduced list. Remove the peaks that contribute strongly to the medium samples in the scores/loadings plots.
- **Keep characteristic peaks** for further analysis: amino acids, lipids, water clusters, fatty acids.

## Key findings relevant to preprocessing

- PCA on all peaks is dominated by medium salts (K+, CaOH+, Fe+, and others). Selected peaks give higher loadings for microbial peaks (amino acids, lipids, DNA bases, water clusters).
- Sample separation is similar in both cases, but selected-peak PCA is easier to interpret.
- Both high-intensity and high-resolution data are suitable for spectral PCA. High-resolution data separates coculture planktonic from coculture aggregates better.
- Sub-set PCAs (selected peaks, amino acid peaks only, water cluster peaks only) were all useful.
- The SI contains peak assignment tables for the positive mode (Table S1) and negative mode (Table S2), and extra PC1/PC2 plots (Figs. S4-S9).
