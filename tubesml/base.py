__author__ = 'lucabasa'
__version__ = '0.1.0'
__status__ = 'development'

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import functools


def transform_wrapper(func):
    '''
    Wrapper for the transform method. It makes sure the columns are in the
    same order as when the fit method was called and it populates the ``columns`` attribute
    '''
    @functools.wraps(func)
    def wrapped(self, X, y=None, **kwargs):
        if len(self.column_order) > 0:
            X = X[self.column_order]
        X_tr = func(self, X, y, **kwargs)
        self.columns = X_tr.columns
        return X_tr
    return wrapped


def fit_wrapper(func):
    '''
    Wrapper for the fit method. It stores the column order and resets the ``columns`` attribute
    '''
    @functools.wraps(func)
    def wrapped(self, X, y=None, **kwargs):
        self.column_order = X.columns
        res = func(self, X, y, **kwargs)
        self.columns = []
        return res
    return wrapped


class BaseTransformer(BaseEstimator, TransformerMixin, ClassifierMixin):
    '''
    This is the base class for all the transformers.
    
    :Attributes:
    
    columns: an empty list that gets reset by the fit method, populated by the transform method, 
            returned by the ``get_feature_names_out`` method
    '''
    
    def __init__(self):
        self.column_order = []
        self.columns = [] # useful to well behave with FeatureUnion
        
    @fit_wrapper   
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
        
    @transform_wrapper   
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
