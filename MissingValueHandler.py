# MissingValueHandler.py

from copy import deepcopy
import numpy as np
import pandas as pd


class MissingForest:
    """
    Parameters
    ----------
    clf : estimator object, default=None.
    This object is assumed to implement the sciki-learn estimator api.
    This object is passed to MissForest

    rgr : estimator object, default=None.
    This object is assumed to implement the sciki-learn estimator api.
    This object is passed to MissForest

    max_iter : int, default=5
    Determins the number of iteration in MissForest

    initial_guess : string, callable or None, default='median'
    If ``mean``, the initial imputation in MissForest will use the mean of the features.
    If ``median``, the initial imputation in MissForest will use the median of the features.

    importance_threshold : float, default=0.5
    Determins the thresold of important features

    simple_impute_method : string, callable or None, default='median'
    If ``mean``, the imputation for non important features will use the mean of the features.
    If ``median``, the imputation for non important features will use the median of the features.
    """