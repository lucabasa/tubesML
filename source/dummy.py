__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from source.base import BaseTransformer, self_columns
import pandas as pd
import warnings


class Dummify(BaseTransformer):
    '''
    Wrapper for pd.get_dummies
    It assures that if some column is missing or is new after the first transform, the pipeline won't break
    '''
    def __init__(self, drop_first=False, match_cols=True, verbose=False):
        super().__init__()
        self.drop_first = drop_first
        self.match_cols = match_cols
        self.verbose = verbose
        
    
    def _match_columns(self, X):
        miss_train = list(set(X.columns) - set(self.columns))
        miss_test = list(set(self.columns) - set(X.columns))
        
        err = 0
        
        if len(miss_test) > 0:
            for col in miss_test:
                X[col] = 0  # insert a column for the missing dummy
                err += 1
        if len(miss_train) > 0:
            for col in miss_train:
                del X[col]  # delete the column of the extra dummy
                err += 1
                
        if (err > 0) & (self.verbose):
            warnings.warn('The dummies in this set do not match the ones in the train set, we corrected the issue.',
                         UserWarning)
            self.verbose = False  # if called repeatedly, we only need one warning
            
        return X[self.columns]  # preserve original order to avoid problems with some algorithms
    

    def transform(self, X):
        X_tr = pd.get_dummies(X, drop_first=self.drop_first)
        if (len(self.columns) > 0):
            if self.match_cols:
                X_tr = self._match_columns(X_tr)
        else:
            self.columns = X_tr.columns
        return X_tr

