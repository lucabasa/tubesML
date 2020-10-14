__author__ = 'lucabasa'
__version__ = '0.0.3'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns
import pandas as pd
from sklearn.pipeline import FeatureUnion


class DtypeSel(BaseTransformer):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
    '''
    def __init__(self, dtype='numeric'):
        super().__init__()
        self.dtype = dtype
        self._validate_input()
      
    
    def _validate_input(self):
        allowed_dtype = ['numeric', 'category']
        if self.dtype not in allowed_dtype:
            raise ValueError(f"Can only use these dtype: {allowed_dtype} got strategy={self.dtype}")
    
    @self_columns
    def transform(self, X, y=None):
        if self.dtype == 'numeric':
            num_cols = X.columns[X.dtypes != object].tolist()
            X_tr = X[num_cols]
        elif self.dtype == 'category':
            cat_cols = X.columns[X.dtypes == object].tolist()
            X_tr = X[cat_cols]
        return X_tr
    
    
class FeatureUnionDf(BaseTransformer):
    '''
    Wrapper of FeatureUnion but returning a Dataframe, 
    the column order follows the concatenation done by FeatureUnion

    transformer_list: list of Pipelines or transformers

    '''
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        super().__init__()
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose  # these are necessary to work inside of GridSearch or similar
        self.feat_un = FeatureUnion(self.transformer_list, 
                                    n_jobs=self.n_jobs, 
                                    transformer_weights=self.transformer_weights, 
                                    verbose=self.verbose)
    @reset_columns    
    def fit(self, X, y=None):
        self.feat_un.fit(X, y)
        return self
    
    @self_columns
    def transform(self, X, y=None):
        X_tr = self.feat_un.transform(X)
        columns = []
        
        for trsnf in self.transformer_list:
            try:
                cols = trsnf[1].steps[-1][1].get_feature_names()
            except AttributeError:  # in case it is not a pipeline
                cols = trsnf[1].get_feature_names()
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        
        return X_tr

    
    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)