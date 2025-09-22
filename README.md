# KDD-UMC-Geometric-AL

## Download
To download the code and data, clone this repository:

```bash
git clone https://github.com/username/KDD-UMC-Geometric-AL.git
cd KDD-UMC-Geometric-AL
```

Download the data from [this Google Drive](https://drive.google.com/drive/folders/132GJjjRn1ypJYsFCil8GY51zZHWUU8Ji?usp=drive_link). Unzip all of the folders in this root folder. No need for prerocessing.

## Introduction
KDD-UMC-Geometric-AL is a project focused on geometric active learning approaches for knowledge discovery in data. It implements novel geometry-based query sampling methods for the active learning of linear separators.

## Installation
Set up a Python environment and install the required dependencies.

## Usage
### Data Preprocessing
All source code now lives under the installable package root `src/gal`, while
CLI utilities are in `scripts/`.  Either install the package in editable mode
(`pip install -e .`) or export `PYTHONPATH=src` before running the commands
below.

Use the preprocessing CLI to prepare your data:

```bash
python -m scripts.preprocess --input <input_file> --output <output_directory> --normalize --fillna mean
```

For more details on preprocessing options, run:
```bash
python -m scripts.preprocess --help
```

### Running Experiments
To run the experiments from the paper:

```bash
python -m scripts.main
python -m sota.learning_to_rank.choquet_rank
```


## Layout at a Glance

```
src/gal/
  core/          # datasets & shared datatypes
  trees/         # ball-tree construction + search
  centers/       # polyhedral centre solvers
  learning/      # active-learning loop and data miners
  metrics/       # ranking and rule-quality metrics
  oracles/       # oracle implementations & priors
  utils/         # feature augmentation helpers
scripts/
  benchmarks/    # ball-tree benchmark driver
  plots/         # plotting utilities (Jaccard, metrics, trees)
  preprocess.py  # preprocessing CLI
  main.py        # experiment launcher
sota/
  learning_to_rank/choquet_rank.py  # external baselines
```

## Features
- Multiple active learning strategies based on geometric properties
- Support for various datasets
- Customizable uncertainty metrics
- Performance visualization tools

## Building the documentation

Install the optional ``docs`` dependencies and run Sphinx from the ``docs``
directory:

```bash
pip install -e .[docs]
cd docs
make html
```

The generated HTML files will be available under ``docs/_build/html``.

## Results
Experimental results show that our geometric active learning approach outperforms traditional methods on several benchmark datasets.


## License
This project is licensed under the MIT License
