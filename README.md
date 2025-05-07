# Machine Learning Models for Virtual Base Station Power Consumption Estimation

## Description

This repository contains machine learning models and supporting scripts for estimating the power consumption of virtual Base Stations (vBS), as presented in the article *Machine Learning Models for Virtual Base Station Power Consumption Estimation*. It includes tools for data visualization, hyperparameter tuning, model training and testing, as well as results analysis.

## Installation

To set up the project on your machine:

1. Clone or download this repository:

    ```bash
    git clone https://github.com/lasseufpa/ml-for-vbs-power.git
    ```

2. Navigate into the project folder:

    ```bash
    cd ml-for-vbs-power
    ```

3. Create a virtual environment using venv or conda and install dependencies:

    - Using venv:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    - Using Conda:
    ```bash
    conda env create -f env.yml
    conda activate env
    ```

4. Download the [`dataset_ul.csv`](https://github.com/jaayala/power_ul_dataset) file from either the [GitHub repository](https://github.com/jaayala/power_ul_dataset) or the [IEEE DataPort page](https://ieee-dataport.org/documents/o-ran-experimental-evaluation-datasets), and place it inside the `in_out_files` directory.

## Usage

| Script             | Description                                                                                   | Input            | Output                                                                  |
|--------------------|-----------------------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------|
| `experiment/density_plots.py` | Plots the density graphs of the input features and the target variable.                       | `in_out_files/dataset_ul.csv` | `in_out_files/figures/density_plot.png`                                |
| `experiment/random_search.py` | Performs hyperparameter optimization for all models and logs the results.                     | `in_out_files/dataset_ul.csv` | `in_out_files/random_search_output.txt`                                |
| `experiment/train_test.py`    | Trains and tests the models, saves evaluation results, and generates scatter plots per model. | `in_out_files/dataset_ul.csv` | `in_out_files/train_test_output.csv` and `in_out_files/figures/scatter_plot-<CPU>.png` |

You can use the following commands to automatically format, lint, and check your code for better readability and consistency:

```bash
black .
isort .
flake8 .
pyright
```

## Pre-commit Hook Setup

To enforce code quality automatically on every commit, this repository uses pre-commit. To enable it:

```bash
pre-commit install
```

This ensures formatting and linting checks run before each commit.

## Cite this work

If you use this repository in your work, please cite the corresponding publication:

```BibTeX
    @ARTICLE{<Text here>,
        author={<Text here>},
        journal={<Text here>},
        title={<Text here>},
        year={<Text here>},
        volume={<Text here>},
        number={<Text here>},
        pages={<Text here>},
        doi={<Text here>}
    }
```

```txt
<Text here>
```