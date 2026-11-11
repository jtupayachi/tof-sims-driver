#!/usr/bin/env python3
"""Backward-compatible shim: the code lives in the `tof_sims_driver` package (src/tof_sims_driver/data_load.py).
Prefer the installed console script or `python -m tof_sims_driver.data_load`."""
try:
    import tof_sims_driver  # noqa: F401
except ImportError:                                  # source checkout without `pip install -e .`
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from tof_sims_driver.data_load import *  # noqa: F401,F403,E402
from tof_sims_driver.data_load import cli  # noqa: E402

if __name__ == "__main__":
    cli()
