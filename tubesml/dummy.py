__author__ = 'lucabasa'
__version__ = '0.0.3'
__status__ = 'development'

from tubesml.base import BaseTransformer, transform_wrapper
import pandas as pd
import warnings


class Dummify(BaseTransformer):
    '''
    Wrapper for pd.get_dummies
    
    It assures that if some column is missing or is new after the first transform, the pipeline won't break
    
    To avoid problems with using both ``drop_first`` and ``match_cols``, specifically if the dropped category is
    missing when dummies are created after the first time, we let ``match_cols`` to have the role of ``drop_first``
    if the transformer has been ran already. See test_match_columns_drop_first_equal for an example.
    
    The fit method simply passes the data as in the ``BaseTransformer``
    
        
    :param drop_first: bool, default False.
                If True, the first dummy column is dropped

    :param match_col: bool, default False.
                If True, it makes sure that all the columns found calling the transformer the first time are found 
                every other time the transform method is called. It thus adds the missing columns (with all 0 values)
                and removes the columns not previously found.

    :param verbose: bool, default False.
             If True, it raises a UserWarning when the ``_match_columns`` method is invoked
    '''
    def __init__(self, drop_first=False, match_cols=True, verbose=False):
        super().__init__()
        self.drop_first = drop_first
        self.match_cols = match_cols
        self.verbose = verbose
        self.is_fit = False
        
    
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

    
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to transform the input data and create dummy columns. 
        If ``match_cols=True``, it also calls the ``_match_columns`` method to make sure the data shape stays consistent
        across runs. This is not done the first time the transform method is called.
        
        It populates the ``columns`` attribute with the columns of the output data. This is done only the first time
        the transformer is called and not every time it outputs new data.

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with dummified columns
        '''
        if not self.is_fit:  # if it the first time, run it as specified and populate self.columns
            X_tr = pd.get_dummies(X, drop_first=self.drop_first)
            self.columns = X_tr.columns
            self.is_fit = True
        else:  # if it is not the first time, do not use drop_first and let match_cols work
            X_tr = pd.get_dummies(X, drop_first=False) 
            if self.match_cols:
                X_tr = self._match_columns(X_tr)
        return X_tr

