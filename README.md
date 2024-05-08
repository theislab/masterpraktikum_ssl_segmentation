# Build PyTorch Model

Code repository for the exercise.

## Requirements
* scanpy
* pytorch
* pathlib
* scikit-misc
* argparse
* tqdm
* numpy
* scikit-learn

## Usage Notes
### Download the Data

Data from "Spatial transcriptomics identifies pathological cell type niches in IPF" by [Mayr et al.](https://doi.org/10.1101/2023.12.13.571464) was used. 

Download the data (.h5ad) from https://zenodo.org/records/10012934 and save it in the data folder.

### Run the Model
1. Run `prepare_data.py` to reduce dataset size and generate trainval and test set.
2. Run `hyperparam_optimization.py` to train and find the best hyperparameters.
3. Run `test.py` to generate the weighted f1-score on the test-set.