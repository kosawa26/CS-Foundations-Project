# MissingValueImputerCSF

A Python package for handling missing values in datasets.

## Overview

`MissingValueImputerCSF` is a Python package that provides a `MissingValueImputerCSF` class to handle missing values in datasets. It uses a combination of machine learning models and imputation techniques to impute missing values in a principled manner.

View our package on Test PyPI - https://test.pypi.org/project/MissingValueImputerCSF/

## Installation

To install `MissingValueImputerCSF`, use the following command:

```bash
pip install scikit-build
pip install numpy
pip install lightgbm
pip install -i https://test.pypi.org/simple/ MissingValueImputerCSF
```

## Usage
Here is an example of how to use MissingValueImputerCSF with IRIS dataset:

#Install the MVI package

%pip install -i https://test.pypi.org/simple/ MissingValueImputerCSF

#Import the required libraries

import seaborn as sns
import numpy as np

from MissingValueImputerCSF.MissingValueImputerCSF import MissingValueImputerCSF

#Load IRIS dataset

dataframe = sns.load_dataset("iris")

dataframe

#Add NaN values randomly

columns_with_missing = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

missing_percentage = 0.2

for column in columns_with_missing:
    mask = np.random.rand(len(dataframe)) < missing_percentage
    dataframe.loc[mask, column] = np.nan

#Check the missing value dataset

dataframe

#Verify the filled missing values

mvh = MissingValueImputerCSF()
mvh.fit(dataframe, "species", categorical=["species"])
dataframe = mvh.transform(dataframe)

dataframe