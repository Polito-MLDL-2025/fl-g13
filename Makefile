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

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	@echo ""
	@echo "🔍 ----------------------------------------------------------"
	@echo "🔍 Running Ruff Linter (checking code formatting and issues)..."
	@echo "🔍 ----------------------------------------------------------"
	@echo ""
	ruff format --check
	ruff check

	@echo ""
	@echo "✅ Linting complete. No issues detected (if any)."
	@echo ""

## Format source code with ruff
.PHONY: format
format:
	@echo ""
	@echo "⚙️ ----------------------------------------------------------"
	@echo "⚙️ Running Ruff Formatter (auto-fixing issues)..."
	@echo "⚙️ ----------------------------------------------------------"
	@echo ""
	ruff check --fix
	ruff format

	@echo ""
	@echo "✅ Formatting complete. Code is now formatted."
	@echo ""

## Set up new venv and install requirements
.PHONY: install
install:
	@echo ""
	@echo "🔧 ----------------------------------------------------------"
	@echo "🔧 Creating virtual environment (if it doesn't exist)..."
	@echo "🔧 ----------------------------------------------------------"
	@echo ""
	@test -d $(VENV_DIR) || $(PYTHON_INTERPRETER) -m venv $(VENV_DIR)

	@echo ""
	@echo "📦 Installing Python dependencies..."
	@echo "📦 ----------------------------------------------------------"
	@echo ""
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

	@echo ""
	@echo "✅ Environment setup complete in '$(VENV_DIR)'"
	@echo ""

## Install only requirements (assumes venv or conda already exists)
.PHONY: requirements
requirements:
	@echo ""
	@echo "📦 Installing dependencies from requirements.txt..."
	@echo ""
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "✅ Dependencies installed."
	@echo ""

## Delete all compiled Python files and venv
.PHONY: clean
clean:
	@echo ""
	@echo "🧹 Cleaning up __pycache__ and .pyc files..."
	@echo ""
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

	@echo ""
	@echo "🧨 Removing virtual environment directory..."
	@echo ""
	rm -rf $(VENV_DIR)

	@echo ""
	@echo "✅ Cleanup complete."
	@echo ""

## Download the dataset
.PHONY: data
data: requirements
	@echo ""
	@echo "⬇️  Downloading dataset with fl_g13.dataset..."
	@echo ""
	$(PYTHON) -m fl_g13.dataset
	@echo ""
	@echo "✅ Dataset downloaded successfully."
	@echo ""

## Export all notebooks in the notebooks/ directory
.PHONY: export_notebooks
export_notebooks: requirements
	@echo ""
	@echo "📤 Exporting all notebooks in 'notebooks/' using nbautoexport..."
	@echo ""
	$(PYTHON) -m nbautoexport export notebooks/
	@echo ""
	@echo "✅ Notebooks exported successfully."
	@echo ""


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
