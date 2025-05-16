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
Use the preprocess.py script to prepare your data:

```bash
python preprocess.py --input <input_file> --output <output_directory> --normalize --fillna mean
```

For more details on preprocessing options, run:
```bash
python preprocess.py --help
```

### Running Experiments
To run the experiments from the paper:

```bash
python main.py
python choquet_rank.py
```


## Features
- Multiple active learning strategies based on geometric properties
- Support for various datasets
- Customizable uncertainty metrics
- Performance visualization tools

## Results
Experimental results show that our geometric active learning approach outperforms traditional methods on several benchmark datasets.


## License
This project is licensed under the MIT License