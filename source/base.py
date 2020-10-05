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


def reset_columns(func):
    @functools.wraps(func)
    def wrapped(self, X):
            func(self, X)
            self.columns = []
            return func(self, X)
    return wrapped


class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns = [] # useful to well behave with FeatureUnion
        
    @reset_columns    
    def fit(self, X, y=None):
        return self
        
    @self_columns    
    def transform(self, X, y=None):
        return X
     
        
    def get_feature_names(self):
        return self.columns

