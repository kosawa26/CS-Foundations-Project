# CS-Foundations-Project

# MissingValImputerDats6450

A Python package for handling missing values in datasets.

## Overview

`MissingValImputerDats6450` is a Python package that provides a `MissingValImputerDats6450` class to handle missing values in datasets. It uses a combination of machine learning models and imputation techniques to impute missing values in a principled manner.

View our package on Test PyPI - https://test.pypi.org/project/MissingValImputerDats6450/

## Installation

To install `MissingValImputerDats6450`, use the following command:

```bash
pip install scikit-learn
pip install scikit-build
pip install numpy
pip install pandas
pip install missforest
pip install copy
pip install lightgbm
pip install -i https://test.pypi.org/simple/ MissingValImputerDats6450
```

## Usage
Here is an example of how to use MissingValImputerDats6450 with IRIS dataset:
```
#Install the package

%pip install -i https://test.pypi.org/simple/ MissingValImputerDats6450

#Import the required libraries

import seaborn as sns

from MissingValImputerDats6450.MissingValImputerDats6450 import MissingValImputerDats6450

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

mvh = MissingValImputerDats6450()
mvh.fit(dataframe, "species", categorical=["species"])
dataframe = mvh.transform(dataframe)

dataframe
```
