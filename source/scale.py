__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from source.base import BaseTransformer, self_columns, reset_columns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd


class DfScaler(BaseTransformer):
    '''
    Wrapper of several sklearn scalers that keeps the dataframe structure
    '''
    def __init__(self, method='standard', feature_range=(0,1)):
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
        elif method == 'minmax':
            self.feature_range = feature_range
            self.scl = MinMaxScaler(feature_range=self.feature_range)
            self.min_ = None
            self.data_min_ = None
            self.data_max_ = None
            self.data_range_ = None
            self.n_samples_seen_ = None

            
    def _validate_input(self):
        allowed_methods = ["standard", 'robust', 'minmax']
        if self.method not in allowed_methods:
            raise ValueError(f"Can only use these methods: {allowed_methods} got method={self.method}")
    
    
    @reset_columns
    def fit(self, X, y=None):
        self.scl.fit(X)
        if self.method == 'standard':
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        elif self.method == 'minmax':
            self.min_ = pd.Series(self.scl.min_, index=X.columns)
            self.data_min_ = pd.Series(self.scl.data_min_, index=X.columns)
            self.data_max_ = pd.Series(self.scl.data_max_, index=X.columns)
            self.data_range_ = self.data_max_ - self.data_min_
            self.n_samples_seen_ = X.shape[0]
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self
    
    
    @self_columns
    def transform(self, X):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled
    