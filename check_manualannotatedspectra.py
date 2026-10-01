#!/usr/bin/env python3
"""check_manualannotatedspectra.py - kept for backward compatibility.

The interactive spectra/peaks/2D/ROI interface is now a live app served by sims_server.py
(one page for positive, one for negative). This shim just launches it so that older commands
and `make plots` keep working.

  python check_manualannotatedspectra.py /home/cloud/tof_sims_driver/processed
"""
from sims_server import main

if __name__ == "__main__":
    main()
