__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from source.base import BaseTransformer, self_columns, reset_columns
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd


class DfScaler(BaseTransformer):
    '''
    Wrapper of several sklearn scalers
    '''
    def __init__(self, method='standard'):
        super().__init__()
        self.method = method
        self._validate_input()
        self.scale_ = None
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.mean_ = None
        elif method == 'robust':
            self.scl = RobustScaler()
            self.center_ = None
    
    
    def _validate_input(self):
        allowed_methods = ["standard", 'robust']
        if self.method not in allowed_methods:
            raise ValueError(f"Can only use these methods: {allowed_methods} got method={self.method}")
    
    
    @reset_columns
    def fit(self, X, y=None):
        self.scl.fit(X)
        if self.method == 'standard':
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self
    
    
    @self_columns
    def transform(self, X):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled
    