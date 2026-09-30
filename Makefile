# # Pipeline: data_load.py (raw_*)  ->  proc_calc.py (calc_*)
# ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
# PYSPM  ?= $(ROOT)/pySPM
# DATA   ?= $(ROOT)/For Jose
# OUT    ?= $(ROOT)/processed
# VENV   := $(ROOT)/.venv
# PY     := $(VENV)/bin/python
# RATE   ?= 0        # sputter rate nm/scan, e.g. make proc RATE=0.5

# .PHONY: all venv data proc probe inspect clean clean-calc
# all: data proc

# $(VENV)/.ok:
# 	python3 -m venv "$(VENV)"
# 	"$(PY)" -m pip install -U pip
# 	"$(PY)" -m pip install -e "$(PYSPM)" pandas pyarrow scipy parquet-tools pyspark
# 	touch $@
# venv: $(VENV)/.ok

# # Spaces in $(DATA) break make prerequisites, so the scripts loop over the folder themselves
# data: venv
# 	"$(PY)" data_load.py "$(DATA)" "$(OUT)"

# proc: venv
# 	"$(PY)" proc_calc.py "$(OUT)" --rate $(RATE)

# # Dump the internal block tree of one file per format: make probe   (or FILE=... EXT to choose)
# PROBE_BASE ?= 6PPD neg1
# probe: venv
# 	mkdir -p "$(OUT)/_probe"
# 	for e in itax itm itmx; do "$(PY)" data_probe.py "$(DATA)/$(PROBE_BASE).$$e" 4 "$(OUT)/_probe/$$e.txt" > /dev/null 2>&1 || echo "probe $$e failed"; done
# 	@ls -la "$(OUT)/_probe"

# inspect: venv
# 	find "$(OUT)" -name 'raw_*' -o -name 'calc_*' | sort

# clean-calc:
# 	find "$(OUT)" -name 'calc_*' -delete
# clean:
# 	rm -rf "$(OUT)"


# Pipeline: data_load.py (raw_*) -> proc_calc.py (calc_*)

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

PYSPM ?= $(ROOT)/pySPM

DATA ?= $(ROOT)/For Jose
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

.PHONY: all venv java data proc probe inspect clean clean-calc 

all: data proc


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
	"$(PY)" -m pip install -e "$(PYSPM)" pandas pyarrow scipy parquet-tools pyspark requests pyopenms matchms
	touch $@

venv: $(VENV)/.ok


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
