__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


class DfPolynomial(BaseTransformer):
    def __init__(self, degree=2, interaction_only=False, include_bias=False, to_interact='all'):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.pol = PolynomialFeatures(degree=self.degree, 
                                      include_bias=self.include_bias, 
                                      interaction_only=self.interaction_only)
        self.to_interact = to_interact
    
    @reset_columns
    def fit(self, X, y=None):
        
        if self.to_interact == 'all':
            cols = X.columns
        else:
            cols = self.to_interact
        
        self.pol.fit(X[cols])
        
        return self
        
    @self_columns
    def transform(self, X, y=None):
        
        if self.to_interact == 'all':
            X_tr = self.pol.transform(X)
            X_tr = pd.DataFrame(X_tr, columns=self.pol.get_feature_names(X.columns), index=X.index)
        else:
            X_int = self.pol.transform(X[self.to_interact])
            X_int = pd.DataFrame(X_int, columns=self.pol.get_feature_names(self.to_interact), index=X.index)
            X_tr = pd.concat([X[[col for col in X if col not in self.to_interact]], X_int], axis=1)
            
        if self.include_bias:
            X_tr.rename(columns={'1': 'BIAS_TERM'}, inplace=True)
        
        return X_tr
