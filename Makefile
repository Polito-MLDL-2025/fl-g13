#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME := fl-g13
PYTHON_VERSION := 3.10
PYTHON_INTERPRETER := python3
VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV_DIR)\Scripts\activate.bat
    PYTHON := $(VENV_DIR)\Scripts\python.exe
else
    ACTIVATE := source $(VENV_DIR)/bin/activate
    PYTHON := $(VENV_DIR)/bin/python
endif

# Check if venv exists and use it, otherwise fallback to system Python
ifeq ($(shell test -d $(VENV_DIR) && echo 1),1)
    # If the venv exists, use its Python interpreter
    PYTHON := $(VENV_DIR)/bin/python
else
    # If no venv, fall back to system Python
    PYTHON := python3
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up new venv and install requirements
.PHONY: install
install:
	@echo "Creating virtual environment if it doesn't exist..."
	@test -d $(VENV_DIR) || $(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Python dependencies installed in virtual environment '$(VENV_DIR)'"

## Install only requirements (assumes venv or conda already exists)
.PHONY: requirements
requirements:
	$(PYTHON) -m pip install -r requirements.txt

## Delete all compiled Python files and venv
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Download the dataset
.PHONY: data
data: requirements
	$(PYTHON) -m fl_g13.dataset

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
