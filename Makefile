# Pipeline: data_load.py (raw_*)  ->  proc_calc.py (calc_*)
ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYSPM  ?= $(ROOT)/pySPM
DATA   ?= $(ROOT)/For Jose
OUT    ?= $(ROOT)/processed
VENV   := $(ROOT)/.venv
PY     := $(VENV)/bin/python
RATE   ?= 0        # sputter rate nm/scan, e.g. make proc RATE=0.5

.PHONY: all venv data proc probe inspect clean clean-calc
all: data proc

$(VENV)/.ok:
	python3 -m venv "$(VENV)"
	"$(PY)" -m pip install -U pip
	"$(PY)" -m pip install -e "$(PYSPM)" pandas pyarrow scipy
	touch $@
venv: $(VENV)/.ok

# Spaces in $(DATA) break make prerequisites, so the scripts loop over the folder themselves
data: venv
	"$(PY)" data_load.py "$(DATA)" "$(OUT)"

proc: venv
	"$(PY)" proc_calc.py "$(OUT)" --rate $(RATE)

# Dump the internal block tree of one file per format: make probe   (or FILE=... EXT to choose)
PROBE_BASE ?= 6PPD neg1
probe: venv
	mkdir -p "$(OUT)/_probe"
	for e in itax itm itmx; do "$(PY)" data_probe.py "$(DATA)/$(PROBE_BASE).$$e" 4 "$(OUT)/_probe/$$e.txt" > /dev/null 2>&1 || echo "probe $$e failed"; done
	@ls -la "$(OUT)/_probe"

inspect: venv
	find "$(OUT)" -name 'raw_*' -o -name 'calc_*' | sort

clean-calc:
	find "$(OUT)" -name 'calc_*' -delete
clean:
	rm -rf "$(OUT)"
