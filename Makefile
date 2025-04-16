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

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies and set up the environment
.PHONY: install
install:
	@echo "Creating virtual environment if it doesn't exist..."
	@test -d $(VENV_DIR) || $(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Python dependencies installed in virtual environment '$(VENV_DIR)'"

## Install only requirements (assumes venv already exists)
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



## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


# ## Set up Python interpreter environment
# .PHONY: create_environment
# create_environment:
# 	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
# 	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	
VENV_DIR = .venv

.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## TODO: Download dataset (does nothing now)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) fl_g13/dataset.py


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
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
