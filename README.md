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
Create a virtual environment and install the package in editable mode (recommended for development):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .[docs]
```

The optional `docs` extra installs Sphinx so you can build the docs (see below).

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

### Quickstart: Experiment RQ1 (Anytime BnB)
This experiment compares kd-tree and ball-tree branch-and-bound against a random sampling baseline, and produces A@time/A@calls figures.

1) Copy the sample config and edit dataset paths:

```bash
cp experiments/rq1/config.sample.yaml my_rq1.yaml
```

2) Run the experiment as a module:

```bash
python -m experiments.rq1.run my_rq1.yaml
```

Outputs are written under the `global.output_dir` specified in the YAML (e.g., `results/rq1`). See docs page “Experiment RQ1: Anytime BnB” for configuration details.

### Other Utilities
- End-to-end driver (legacy benchmark setup):

```bash
python -m scripts.main
```

- External baseline:

```bash
python -m sota.learning_to_rank.choquet_rank
```


## Layout at a Glance

```
src/gal/
  core/          # datasets & shared datatypes
  trees/         # ball-tree construction strategies
  search/        # modular branch-and-bound search (engine, bounds, visit strategies)
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
