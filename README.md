# fl-g13

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Federated Learning Project, Group 13

## How to Install

### Using `make`

1. Ensure you have `make` installed on your system.
2. Run the following command to set up the project environment:

    ```bash
    make install
    ```

    This will:
    - Create a new virtual enviroment (`venv`)
    - Install the required Python dependencies listed in `requirements.txt`.
    - Set up any additional configurations needed for the project.

### Using `venv`

1. Ensure you have Python installed on your system.
2. Create a virtual environment and install the required dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

    This will:
    - Create a new virtual environment named `venv`.
    - Install the required Python dependencies listed in `requirements.txt`.
    - Set up the environment for the project.

### Using `conda`

1. Ensure you have `conda` installed on your system.
2. Create a new conda environment and install the required dependencies:

    ```bash
    conda create --name fl-g13 python=3.11
    conda activate fl-g13
    pip install -r requirements.txt
    ```

    This will:
    - Create a new conda environment named `fl-g13` with Python 3.9.
    - Install the required Python dependencies listed in `requirements.txt`.
    - Set up the environment for the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fl_g13 and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── fl_g13   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fl_g13 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    └── modeling                
        ├── test.py             <- Code to test models     
        ├── train.py            <- Code to train models
        └── utils.py            <- Code with utility functions
```

---

## How to contribute

### **Jupyter Notebook Usage**

1. **Notebook Organization**
    - Notebooks must be stored in the `notebooks/` directory.
    - Naming convention: `PHASE.NOTEBOOK-INITIALS-DESCRIPTION.ipynb`
        
        Example: `0.01-pjb-data-source-1.ipynb`
        
        - `PHASE` codes:
            - `0` – Data exploration
            - `1` – Data cleaning & feature engineering
            - `2` – Visualization
            - `3` – Modeling
            - `4` – Publication
        - `pjb` – Your initials; helps identify the author and avoid conflicts.
        - `data-source-1` – Short, clear description of the notebook's purpose.

### **Code Reusability & Refactoring Regulation**

1. **Refactor Shared Code into Modules**
    - Store reusable code in the `fl_g13` package.
    - Add the following cell at the top of each notebook:

    ```python
    %load_ext autoreload
    %autoreload 2
    ```

### **Code Review & Version Control Regulation**

1. **Ensure Reviewability**
    - Commit both `.ipynb` files and their exported `.py` versions to version control.
2. **Use `nbautoexport` Tool**
    - Install with:

    ```bash
    nbautoexport install
    nbautoexport configure notebooks
    ```
    - Then, anytime you want to export a notebook in a python script, run:

    ```bash
    nbautoexport export notebooks/<notebook_name>.ipynb
    ```

#### (PyCharm only) Use a Git Hook or File Watcher

You can set up PyCharm to run `nbconvert` (or even `nbautoexport export`) every time a file is saved or committed.

 **PyCharm File Watcher**

1. Go to **Settings > Tools > File Watchers**
2. Add a new watcher with the following configuration:

- **File Type**: Jupyter Notebook (`*.ipynb`)
- **Scope**: Current project
- **Program**: Your Python interpreter path (e.g., `python`)
- **Arguments**:

  ```bash
  -m nbautoexport export $FileDir$
  ```
- **Working Directory**: `$FileDir$`
