__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns
from sklearn.decomposition import PCA
import pandas as pd


class DfPCA(BaseTransformer):
    '''
    Wrapper around PCA to keep the dataframe structure
    It can also return the same dataframe in a compressed form, e.g. by doing and undoing pca
    '''
    def __init__(self, n_components, svd_solver='auto', random_state=24, compress=False):
        super().__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.n_components_ = 0  # will be assigned by the fit method
        self.random_state = random_state
        self.PCA = PCA(n_components=self.n_components, svd_solver=self.svd_solver, random_state=self.random_state)
        self.compress = compress
        self.original_columns = []
        
    @reset_columns
    def fit(self, X, y=None):
        
        self.PCA.fit(X)
        self.n_components_ = self.PCA.n_components_
        
        return self
    
    @self_columns
    def transform(self, X, y=None):
                
        X_tr = self.PCA.transform(X)
        X_tr = pd.DataFrame(X_tr, columns=[f'pca_{i}' for i in range(self.n_components_)])
        
        self.original_columns = X.columns
        
        if self.compress:
            X_tr = self.inverse_transform(X_tr)
        
        return X_tr
    
    
    def inverse_transform(self, X, y=None):
        
        try:
            X_tr = self.PCA.inverse_transform(X)
        except ValueError:
            return X
        X_tr = pd.DataFrame(X_tr, columns=self.original_columns)
        
        return X_tr
