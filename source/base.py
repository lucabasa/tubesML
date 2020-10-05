__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from sklearn.base import BaseEstimator, TransformerMixin
import functools


def self_columns(func):
    @functools.wraps(func)
    def wrapped(self, X):
            X_tr = func(self, X)
            self.columns = X_tr.columns
            return X_tr
    return wrapped


class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns = [] # useful to well behave with FeatureUnion
        
        
    def fit(self, X, y=None):
        self.columns = [] # for safety if there are multiple fits
        return self
        
    @self_columns    
    def transform(self, X, y=None):
        return X
     
        
    def get_feature_names(self):
        return self.columns

