__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from sklearn.base import BaseEstimator, TransformerMixin
import functools


def self_columns(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
            X_tr = func(self, *args, **kwargs)
            self.columns = X_tr.columns
            return X_tr
    return wrapped


def reset_columns(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            self.columns = []
            return res
    return wrapped


class BaseTransformer(BaseEstimator, TransformerMixin):
    '''
    This is the base class for all the transformers.
    
    :Attributes:
    
    columns: an empty list that gets reset by the fit method, populated by the transform method, 
            returned by the ``get_feature_names_out`` method
    '''
    
    def __init__(self):
        self.columns = [] # useful to well behave with FeatureUnion
        
    @reset_columns    
    def fit(self, X, y=None):
        '''
        Method to train the transformer.
        
        It also reset the ``columns`` attribute

        :param X: {array-like} of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values (class labels) as integers or strings.
        '''
        self.is_fit_ = True
        return self
        
    @self_columns    
    def transform(self, X, y=None):
        '''
        Method to transform the input data.
        
        It populates the ``columns`` attribute with the columns of the output data

        :param X: {array-like} of shape (n_samples, n_features)
            The input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values (class labels) as integers or strings.
        '''
        return X
     
        
    def get_feature_names_out(self):
        '''
        Returns the ``columns`` attribute, useful to well behave with other sklearn methods
        '''
        return self.columns
