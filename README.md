# A Deep Learning Approach for Predicting Function of Non-coding Genomic Variants

## Overview

A large variety of single-nucleotide polymorphisms in the genome are associated with specific diseases. Most such genomic variants occur in non-coding DNA sequences and are not directly involved in protein variation. This makes it challenging to understand their function. This project investigates a deep learning approach to predict functional variants using epigenetic markers as predictors. The models outperform previously established benchmarks on the GM12878 lymphoblastoid dataset.

## Setup

Install the conda environment:

``` conda env create -f environment.yml ```

But depending on operating system, the dependencies may not work out. To manually setup, install
numpy, pandas, scikit-learn, and keras on a Python 3.6 environment.


## Running the models

### Data
The models are trained on MPRA data from GM12878 cell line, using epigenetic markers from ENCODE as predictors.
The goal is to predict whether each candidate variant is functional. The config.py file can be modified to adjust to dataset names. The data is not included here.

### Steps
Run ```python data_setup.py``` to process the dataset and split into train/test sets.

Run ```python train_and_evaluate.py``` to fit the tuned models and evaluate their performances.

### Modifications

Random hyperparameter search can be executed using code in search_hparams.py

Some figures and analysis are available in error_analysis.py
