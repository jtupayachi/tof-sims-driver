"""tof_sims_driver - read IONTOF TOF-SIMS files, explore spectra / images and run PCA.

Modules:
    data_load   READ step: IONTOF files (.itm/.ita/.itax/.itmx) -> raw_* outputs
    data_probe  dump the block tree of an IONTOF file
    store       access to the processed ion events (spectra, images, peaks)
    pca         preprocessing + PCA, with every step and formula reported
    figures     matplotlib figures of a PCA result
    server      interactive browser (HTTP server + single-page app)
"""
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("tof-sims-driver")
except PackageNotFoundError:                     # running from a source checkout
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
