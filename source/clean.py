__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from source.base import BaseTransformer, self_columns, reset_columns

from sklearn.impute import SimpleImputer
import pandas as pd


class DfImputer(BaseTransformer):
    '''
    Just a wrapper for the SimpleImputer that keeps the dataframe structure
    '''
    def __init__(self, strategy='mean', fill_value=None):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self._validate_input()
        self.imp = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self.statistics_ = None
        
        
    def _validate_input(self):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Can only use these strategies: {allowed_strategies} got strategy={self.strategy}")

    @reset_columns
    def fit(self, X, y=None):
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    @self_columns
    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
