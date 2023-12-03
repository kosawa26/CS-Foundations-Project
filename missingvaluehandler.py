# MissingValueHandler.py

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from missforest.missforest import MissForest
from missforest.errors import MultipleDataTypesError, NotFittedError


class MissingValueHandler:
    """
    Parameters
    ----------
    clf : estimator object, default=RandomForestClassifier().
    This object is assumed to implement the sciki-learn estimator api,
    and the class which this object belongs to is assumed to have 'feature_importances_' attribute.

    rgr : estimator object, default=RandomForestRegressor().
    This object is assumed to implement the sciki-learn estimator api,
    and the class which this object belongs to is assumed to have 'feature_importances_' attribute.

    importance_threshold : float, default=0.4
    Determins the thresold of important features

    simple_impute_method : string, callable or None, default='median'
    If ``mean``, the imputation for unimportant features will use the mean of the features.
    If ``median``, the imputation for unimportant features will use the median of the features.

    clf_miss : estimator object, default=LGBMClassifier().
    This object is assumed to implement the sciki-learn estimator api.
    This object is passed to MissForest

    rgr_miss : estimator object, default=LGBMRegressor().
    This object is assumed to implement the sciki-learn estimator api.
    This object is passed to MissForest

    max_iter_miss : int, default=5
    Determins the number of iteration in MissForest.

    initial_guess_miss : string, callable or None, default='median'
    If ``mean``, the initial imputation in MissForest will use the mean of the features.
    If ``median``, the initial imputation in MissForest will use the median of the features.
    """
    def __init__(self, clf=RandomForestClassifier(), rgr=RandomForestRegressor(),
                 importance_threshold=0.4, simple_impute_method="median",
                 clf_miss=LGBMClassifier(), rgr_miss=LGBMRegressor(),
                 max_iter_miss=5, initial_guess_miss="median"):
        # make sure 'clf' is an estimator.
        if not self._has_feature_importances(clf):
            raise ValueError("Argument 'clf' only accept estimators that has "
                             "class methods 'fit' and 'predict' and atribute 'feature_importances_'.")

        # make sure 'rgr' is an estimator.
        if not self._has_feature_importances(rgr):
            raise ValueError("Argument 'rgr' only accept estimators that has"
                             " class methods 'fit' and 'predict' and atribute 'feature_importances_'.")
        
        # make sure 'importance_threshold' is float and between 0 and 1.
        if not isinstance(importance_threshold, float):
            raise ValueError("Argument 'importance_threshold' only accept float.")
        
        # make sure 'imporance_thershold' is between 0 and 1.
        if not (0 <= importance_threshold <= 1):
            raise ValueError("Argument 'importance_threshold' only accept value between 0 and 1.")
        
        # make sure 'simple_impute_method' is str.
        if not isinstance(simple_impute_method, str):
            raise ValueError("Argument 'simple_impute_method' only accept str.")
        
        # make sure 'simple_impute_method' is either 'median' or 'mean'.
        if simple_impute_method not in ("median", "mean"):
            raise ValueError("Argument 'simple_impute_method' can only be 'median' or"
                             " 'mean'.")
        
        # make sure 'clf_miss' is an estimator.
        if not self._is_estimator(clf_miss):
            raise ValueError("Argument 'clf_miss' only accept estimators that has "
                             "class methods 'fit' and 'predict'.")

        # make sure 'rgr_miss' is an estimator.
        if not self._is_estimator(rgr_miss):
            raise ValueError("Argument 'rgr_miss' only accept estimators that has"
                             " class methods 'fit' and 'predict'.")

        # make sure 'initial_guess_miss' is str.
        if not isinstance(initial_guess_miss, str):
            raise ValueError("Argument 'initial_guess_miss' only accept str.")

        # make sure 'initial_guess_miss' is either 'median' or 'mean'.
        if initial_guess_miss not in ("median", "mean"):
            raise ValueError("Argument 'initial_guess_miss' can only be 'median' or"
                             " 'mean'.")

        # make sure 'max_iter_miss' is int.
        if not isinstance(max_iter_miss, int):
            raise ValueError("Argument 'max_iter_miss' only accept int.")
        
        self.classifier = clf
        self.regressor = rgr
        self.importance_threshold = importance_threshold
        self.simple_impute_method = simple_impute_method
        self.classifier_miss = clf_miss
        self.regressor_miss = rgr_miss
        self.initial_guess_miss = initial_guess_miss
        self.max_iter_miss = max_iter_miss
        self.missforest = None
        self.important_features = {}
        self.unimportant_features = {}
        self._simple_imputation = {}
        self._obs_row = None
        self._mappings = {}
        self._rev_mappings = {}
        self.categorical = None
        self.numerical = None
        self._is_fitted = False

    @staticmethod
    def _has_feature_importances(estimator):
        """
        Class method '_has_feature_importances' is used to check if argument
        'estimator' is an object that implement the scikit-learn estimator api
        and the class which 'estimator' belongs to has 'feature_importances_'.

        Parameters
        ----------
        estimator : estimator object
        This object is assumed to implement the scikit-learn estimator api.

        Return
        ------
        If the argument 'estimator' has class method 'fit' and 'predict' 
        and attribute 'feature_importances_' return True.

        Otherwise, return False
        """

        try:
            # get the class methods 'fit' and 'predict' of the estimator.
            is_has_fit_method = getattr(estimator, "fit")
            is_has_predict_method = getattr(estimator, "predict")

            # get the class of the estimator
            is_has_feature_importances_ = estimator.__class__

            # check if those class method are callable.
            is_has_fit_method = callable(is_has_fit_method)
            is_has_predict_method = callable(is_has_predict_method)

            # check if the class has 'feature_importances_'
            is_has_feature_importances_ = hasattr(is_has_feature_importances_, "feature_importances_")

            # assumes it is an estimator if it has 'fit' and 'predict' methods.
            return is_has_fit_method and is_has_predict_method and is_has_feature_importances_
        except AttributeError:
            return False

    @staticmethod
    def _is_estimator(estimator):
        """
        Class method '_is_estimator_or_none' is used to check if argument
        'estimator' is an object that implement the scikit-learn estimator api.

        Parameters
        ----------
        estimator : estimator object
        This object is assumed to implement the scikit-learn estimator api.

        Return
        ------
        If the argument 'estimator' is None or has class method 'fit' and
        'predict', return True.

        Otherwise, return False
        """

        try:
            # get the class methods 'fit' and 'predict' of the estimator.
            is_has_fit_method = getattr(estimator, "fit")
            is_has_predict_method = getattr(estimator, "predict")

            # check if those class method are callable.
            is_has_fit_method = callable(is_has_fit_method)
            is_has_predict_method = callable(is_has_predict_method)

            # assumes it is an estimator if it has 'fit' and 'predict' methods.
            return is_has_fit_method and is_has_predict_method
        except AttributeError:
            return False

    def _get_obs_row(self, X):
        """
        Class method '_get_obs_row' gather the rows of any rows that do not
        have any missing values.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        None
        """

        n_null = X.isnull().sum(axis=1)
        self._obs_row = X[n_null == 0].index

    def _get_map_and_rev_map(self, X, categorical):
        """
        Class method '_get_map_and_rev_map' gets the encodings and the reverse
        encodings of categorical variables.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        categorical : list
        All categorical features of X.

        Return
        ------
        None
        """

        for c in X.columns:
            if c in categorical:
                unique = X[c].dropna().unique()
                n_unique = range(X[c].dropna().nunique())

                self._mappings[c] = dict(zip(unique, n_unique))
                self._rev_mappings[c] = dict(zip(n_unique, unique))

    @staticmethod
    def _check_if_all_single_type(X):
        """
        Class method '_check_if_all_single_type' checks if all values in the
        feature belongs to the same datatype. If not, error
        'MultipleDataTypesError will be raised.'

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.
        """

        vectorized_type = np.vectorize(type)
        for c in X.columns:
            feature_no_na = X[c].dropna()
            all_type = vectorized_type(feature_no_na)
            all_unique_type = pd.unique(all_type)
            n_type = len(all_unique_type)
            if n_type > 1:
                raise MultipleDataTypesError(f"Feature {c} has more than one "
                                             f"datatype.")

    def _get_simple_impute_values(self, X, less_important_features, categorical):
        """
        Class method '_get_simple_impute_values' calculate the imputation values
        of each unimportant features.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        less_important_features : dict_keys
        All unimportant features of X.

        categorical : list
        All categorical features of X.

        Return
        ------
        None
        """
        
        for c in less_important_features:
            if c in categorical:
                self._simple_imputation[c] = X[c].mode().values[0]
            else:
                if self.simple_impute_method == "mean":
                    self._simple_imputation[c] = X[c].mean()
                elif self.simple_impute_method == "median":
                    self._simple_imputation[c] = X[c].median()
                else:
                    raise ValueError("Argument 'initial_guess' only accepts "
                                     "'mean' or 'median'.")
    
    def _simple_imputer(self, X):
        """
        Class method '_simple_imputer' imputes the values of features
        using the mean or median if they are numerical variables, else, imputes
        with mode.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        Return
        ------
        X : pd.DataFrame of shape (n_samples, n_features)
        Imputed Dataset (unimportant features only).
        """

        for c in self._simple_imputation:
            X[c].fillna(self._simple_imputation[c], inplace=True)

        return X
                
    @staticmethod
    def _label_encoding(X, mappings):
        """
        Class method '_label_encoding' performs label encoding on given
        features and the input mappings.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

        Return
        ------
        X : X : pd.DataFrame of shape (n_samples, n_features)
        Label-encoded dataset (features only).
        """

        for c in mappings:
            X[c] = X[c].map(mappings[c]).astype(int)

        return X

    @staticmethod
    def _rev_label_encoding(X, rev_mappings):
        """
        Class method '_rev_label_encoding' performs reverse label encoding on
        given features and the input reverse mappings.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset (features only) that needed to be imputed.

        rev_mappings : dict
        Dictionary that contains the categorical variables as keys and their
        corresponding encodings as values.

        Return
        ------
        X : pd.DataFrame of shape (n_samples, n_features)
        Reverse label-encoded dataset (features only).
        """

        for c in rev_mappings:
            X[c] = X[c].map(rev_mappings[c])

        return X
    
    def fit(self, X, target, categorical=None):
        """
        Class method 'fit' checks if the arguments are valid and initiates
        different class attributes.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset that needed to be imputed.

        target : string
        Name of the target feature

        categorical : list, default=None
        All categorical features of X.

        Return
        ------
        None
        """

        X = X.copy()

        # make sure 'X' is either pandas dataframe, numpy array or list of lists.
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Argument 'X' can only be pandas dataframe.")

        # make sure 'target' is str.
        if not isinstance(target, str):
            raise ValueError("Argument 'target' only accept str.")

        # make sure 'target' is in the features of X.
        if target not in X.columns:
            raise ValueError("Argument 'target' can only be the name of features in X.")

        # make sure 'categorical' is a list of str.
        if (
                categorical is not None and
                not isinstance(categorical, list) and
                not all(isinstance(elem, str) for elem in categorical)
        ):
            raise ValueError("Argument 'categorical' can only be list of "
                             "str or NoneType.")

        # make sure 'categorical' has at least one variable in it.
        if categorical is not None and len(categorical) < 1:
            raise ValueError(f"Argument 'categorical' has a len of "
                             f"{len(categorical)}.")

        # Check for +/- inf
        if (
                categorical is not None and
                np.any(np.isinf(X.drop(categorical, axis=1)))
        ):
            raise ValueError("+/- inf values are not supported.")

        # make sure there is no column with all missing values.
        if np.any(X.isnull().sum() == len(X)):
            raise ValueError("One or more columns have all rows missing.")
        
        if categorical is None:
            categorical_temp = []
        else:
            categorical_temp = categorical

        self.categorical = categorical_temp
        self.numerical = [c for c in X.columns if c not in categorical_temp]

        self._check_if_all_single_type(X)
        self._get_obs_row(X)
        self._get_map_and_rev_map(X, self.categorical)

        # get the observations without missing values and encode catogricals
        X_temp = X.loc[self._obs_row]
        X_temp = self._label_encoding(X_temp, self._mappings)

        # split the data into dependent and independent variables
        y_temp = X_temp[target]
        X_temp = X_temp.drop(columns=[target])

        # compute feature importances
        if target in self._mappings:
            estimator = deepcopy(self.classifier)
        else:
            estimator = deepcopy(self.regressor)
        feature_importances = estimator.fit(X_temp, y_temp).feature_importances_

        # separete the important features and unimportant features
        s_feature_importance = pd.Series(data=feature_importances, index=X_temp.columns)
        s_feature_importance.sort_values(ascending=False, inplace=True)
        threshold = self.importance_threshold
        for i in range(len(feature_importances)):
            if threshold > 0:
                self.important_features[s_feature_importance.index[i]] = s_feature_importance.iloc[i]
                threshold -= s_feature_importance.iloc[i]
            else:
                self.unimportant_features[X_temp.columns[i]] = feature_importances[i]
        
        # combine dependent and independent variables and reverse encode
        X_temp = pd.concat([y_temp, X_temp], axis=1)
        X_temp = self._rev_label_encoding(X_temp, self._rev_mappings)

        # get filling values for unimportant features and impute
        self._get_simple_impute_values(X_temp, self.unimportant_features.keys(), self.categorical)
        X_imp = self._simple_imputer(X)

        # changes max_iter_miss when there is no important feature to avoid error in MissForest
        if len(self.important_features) == 0:
            self.max_iter_miss = 1

        # pass the imputed data to missforest
        self.missforest = MissForest(self.classifier_miss, self.regressor_miss,
                                     self.initial_guess_miss, self.max_iter_miss)
        self.missforest.fit(X_imp, categorical)
        self._is_fitted = True


    def transform(self, X):
        """
        Class method 'transform' imputes all missing values in 'X'.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset that needed to be imputed.

        Return
        ------
        X : pd.DataFrame of shape (n_samples, n_features)
        Imputed dataset.
        """

        if not self._is_fitted:
            raise NotFittedError("MissForest is not fitted yet.")

        X = X.copy()

        # impute missing values in unimportant features
        X_imp = self._simple_imputer(X)
        if X_imp.isnull().sum().sum() != 0:
            X_imp = self.missforest.transform(X_imp)

        return X_imp
    
    def fit_transform(self, X, target, categorical=None):
        """
        Class method 'fit_transform' calls class method 'fit' and 'transform'
        on 'X'.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        Dataset that needed to be imputed.

        target : string
        Name of the target feature

        categorical : list, default=None
        All categorical features of X.

        Return
        ------
        X : pd.DataFrame of shape (n_samples, n_features)
        Imputed dataset.
        """

        self.fit(X, target, categorical)
        X = self.transform(X)

        return X
