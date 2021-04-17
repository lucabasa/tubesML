__author__ = 'lucabasa'
__version__ = '0.1.0'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class DfImputer(BaseTransformer):
    '''
    Just a wrapper for the SimpleImputer that keeps the dataframe structure.
    
    Inherits from ``BaseTransformer``
    
    :Attributes:
    ------------
        
    stragegy : str, the strategy to impute the missing values, default "mean"
              Allowed values: "mean", "median", "most_frequent", "constant"

    fill_value :  value to use to impute the missing values when the ``strategy`` is "constant"
                It is ignored by any other strategy
    '''
    def __init__(self, strategy='mean', fill_value=None, add_indicator=False):
        '''
        :Attributes:
        ------------
        
        stragegy : str, the strategy to impute the missing values, default "mean"
                  Allowed values: "mean", "median", "most_frequent", "constant"
        
        fill_value :  value to use to impute the missing values when the ``strategy`` is "constant"
                    It is ignored by any other strategy
        '''
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self._validate_input()
        self.imp = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value, add_indicator=self.add_indicator)
        self.statistics_ = None
        self.indicator_ = None
        self._to_add = []  # columns to add with the add_indicator turned on
        self._missing_cols = []
        
        
    def _validate_input(self):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies} got strategy={self.strategy}")

    
    @reset_columns
    def fit(self, X, y=None):
        '''
        Method to train the imputer.
        
        It also reset the ``columns`` attribute
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        '''
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        self.indicator_ = self.imp.indicator_
        if self.add_indicator:
            self._to_add = self.indicator_.features_
            self._missing_cols = self._get_indicator_names(X)
        return self
    
    
    def _match_columns(self, X):
        miss_train = list(set(X.columns) - set(self.columns))
        miss_test = list(set(self.columns) - set(X.columns))
        
        if len(miss_test) > 0:
            for col in miss_test:
                X[col] = 0  # insert a column for the missing indicator
        if len(miss_train) > 0:
            for col in miss_train:
                del X[col]  # delete the column of the extra indicator
            
        return X[self.columns]  # preserve original order to avoid problems with some algorithms
    

    def _get_indicator_names(self, X):
        missing = list(np.array(X.columns)[self._to_add])
        return [f'missing_{col}' for col in missing]

    
    @self_columns
    def transform(self, X, y=None):
        '''
        Method to transform the input data
        
        It populates the ``columns`` attribute with the columns of the output data
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with no missing values
        '''
        Ximp = self.imp.transform(X)
        columns = list(X.columns) + self._missing_cols
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=columns)
        if self.add_indicator:
            for col in self._missing_cols:
                Xfilled[col] = Xfilled[col].astype(int)
        return Xfilled