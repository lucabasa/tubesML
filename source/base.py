__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.columns = [] # useful to well behave with FeatureUnion
        
        
    def fit(self, X, y=None):
        self.columns = [] # for safety if there are multiple fits
        return self
        
        
    def transform(self, X, y=None):
        self.columns = X.columns  # important to write this if overwritten
        return X
     
        
    def get_feature_names(self):
        return self.columns

