__author__ = 'lucabasa'
__version__ = '0.1.1'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class DfImputer(BaseTransformer):
    '''
    Just a wrapper for the SimpleImputer that keeps the dataframe structure.
    
    Inherits from ``BaseTransformer``.
    
    :param strategy: str, the strategy to impute the missing values, default "mean".
              Allowed values: "mean", "median", "most_frequent", "constant"

    :param fill_value:  value to use to impute the missing values when the ``strategy`` is "constant".
                It is ignored by any other strategy
                
    :param add_indicator: bool, default=False. 
                    If True, a new column with binary values is created whenever missing values are found when
                    the fit method is called. The column will be called ``missing_<column_name>``
    
    :Attributes:
    
        `statistics_` : pandas Series. The statistics per column, depending on the ``strategy`` chosen.
                        The index of the series is the ``columns`` attribute of the input dataframe.

        `imp` : ``sklearn.impute.SimpleImputer``
                Core transformer. Its ``fit`` and ``transform`` methods are used here.

    '''
    def __init__(self, strategy='mean', fill_value=None, add_indicator=False):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self._validate_input()
        self.imp = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.statistics_ = None
        self._missing_cols = []
        self.data_types = None
        
        
    def _validate_input(self):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies} got strategy={self.strategy}")

    
    @fit_wrapper
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
        self.data_types = X.dtypes.to_dict()
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        if self.add_indicator:
            self._missing_cols = list(X.columns[X.isna().any()])
        return self
    
    
    @transform_wrapper
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
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        Xfilled = Xfilled.astype(self.data_types)
        if self.add_indicator:
            for col in self._missing_cols:
                Xfilled[f'missing_{col}'] = np.where(X[col].isna(), 1, 0)
        return Xfilled
