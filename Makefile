PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash
.DEFAULT_GOAL := all

VENV_DIR := .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@./scripts/bootstrap_venv.sh "$(VENV_DIR)"
	@$(PIP) install -r requirements.txt

data: setup
	@$(PY) -m ucc_asr.pipeline --config "$(CONFIG)" --stage data

train: setup
	@$(PY) -m ucc_asr.pipeline --config "$(CONFIG)" --stage train

eval: setup
	@$(PY) -m ucc_asr.pipeline --config "$(CONFIG)" --stage eval

report: setup
	@$(PY) -m ucc_asr.report --results artifacts/results.json --out artifacts/report.md

all: data train eval report

clean:
	@rm -rf "$(VENV_DIR)" artifacts runs
