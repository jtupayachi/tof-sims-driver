
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

PYSPM ?= $(ROOT)/pySPM

DATA ?= $(ROOT)/data
OUT  ?= $(ROOT)/processed

VENV := $(ROOT)/.venv
PY   := $(VENV)/bin/python

RATE ?= 0        # sputter rate nm/scan, e.g. make proc RATE=0.5


# ------------------------------------------------------------
# Java / PySpark
# ------------------------------------------------------------

# Use JAVA_HOME if already defined; otherwise try to find Java.
JAVA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(which java) 2>/dev/null)) 2>/dev/null)

export JAVA_HOME
export PATH := $(JAVA_HOME)/bin:$(PATH)


# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

.PHONY: all venv java data proc probe inspect clean clean-calc plots serve gpu

all: data plots proc


# Check Java installation
java:
	@if [ -z "$(JAVA_HOME)" ] || [ ! -x "$(JAVA_HOME)/bin/java" ]; then \
		echo "ERROR: Java was not found."; \
		echo ""; \
		echo "Install Java with:"; \
		echo "  sudo apt update && sudo apt install -y openjdk-17-jdk"; \
		echo ""; \
		echo "Or set JAVA_HOME manually, e.g.:"; \
		echo "  export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"; \
		exit 1; \
	fi
	@echo "JAVA_HOME=$(JAVA_HOME)"
	@$(JAVA_HOME)/bin/java -version


$(VENV)/.ok: java
	python3 -m venv "$(VENV)"
	"$(PY)" -m pip install -U pip
	"$(PY)" -m pip install -e "$(PYSPM)" pandas pyarrow scipy scikit-learn parquet-tools pyspark requests pyopenms matchms
	touch $@

venv: $(VENV)/.ok

# Optional GPU acceleration for the PCA in the Statistical methods tab (needs an NVIDIA GPU +
# ~2 GB disk). Without it the server falls back to scikit-learn automatically.
gpu: venv
	"$(PY)" -m pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.*"


# Spaces in $(DATA) break make prerequisites, so the scripts loop
# over the folder themselves.

data: venv
	"$(PY)" data_load.py "$(DATA)" "$(OUT)"

proc: venv
	"$(PY)" proc_calc.py "$(OUT)" --rate $(RATE)


# ------------------------------------------------------------
# Probe
# ------------------------------------------------------------

# Dump the internal block tree of one file per format:
# make probe
# or change PROBE_BASE / extension as needed.

PROBE_BASE ?= 6PPD neg1

probe: venv
	mkdir -p "$(OUT)/_probe"
	for e in itax itm itmx; do \
		"$(PY)" data_probe.py "$(DATA)/$(PROBE_BASE).$$e" 4 "$(OUT)/_probe/$$e.txt" \
		> /dev/null 2>&1 || echo "probe $$e failed"; \
	done
	@ls -la "$(OUT)/_probe"


# ------------------------------------------------------------
# Inspect
# ------------------------------------------------------------

inspect: venv
	find "$(OUT)" \( -name 'raw_*' -o -name 'calc_*' \) -print | sort


# ------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------

clean-calc:
	find "$(OUT)" -name 'calc_*' -delete

clean:
	rm -rf "$(OUT)"

# Launch the interactive browser (3 tabs: Spectra & 2D, Statistical methods, ML/DL).
# Binds all interfaces by default so others on the LAN can reach it, e.g.
#   http://<this-machine-ip>:8765/     (find the IP with:  hostname -I)
# Restrict to this machine only with  HOST=127.0.0.1 ; change the port with PORT=9000.
HOST ?= 0.0.0.0
PORT ?= 8765

serve plots: venv
	@echo "Serving on http://$(HOST):$(PORT)/  (reachable at http://`hostname -I | awk '{print $$1}'`:$(PORT)/ )"
	"$(PY)" sims_server.py "$(OUT)" --host $(HOST) --port $(PORT)