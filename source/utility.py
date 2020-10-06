__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from source.base import BaseTransformer, self_columns, reset_columns


class FeatSel(BaseTransformer):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
    '''
    def __init__(self, dtype='numeric'):
        self.dtype = dtype
      
    
    def _validate_input():
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