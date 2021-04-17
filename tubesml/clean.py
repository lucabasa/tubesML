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
        return self
    

    def _get_indicator_names(self, X):
        missing = self.indicator_.features_
        missing = list(np.array(X.columns)[missing])
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
        if self.add_indicator:
            mis_cols = self._get_indicator_names(X)
            columns = list(X.columns) + mis_cols
        else:
            columns = X.columns
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=columns)
        if self.add_indicator:
            for col in mis_cols:
                Xfilled[col] = Xfilled[col].astype(int)
        return Xfilled