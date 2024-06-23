__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


class DfPolynomial(BaseTransformer):
    '''
    Wrapper around PolynomialFeatures.
    
    Inherits from ``BaseTransformer``.

    :param degree: int, default=2
              The degree of the polynomial features.

    :param interaction_only: bool, default=False.
                If True, only interaction features are produced: features that are products of at most 
                degree distinct input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
                
    :param include_bias: bool, default=False
                If True, then include a bias column, the feature in which
                all polynomial powers are zero (i.e. a column of ones - acts as an
                intercept term in a linear model). The column is added with the name BIAS_TERM
                
    :param to_interact: str or list of strings, default='all'.
                Columns to consider for the interactions. If 'all', then all the columns of the DataFrame
                will be used. If a list of columns is provided, only those columns will be used for creating
                the interactions. All the other columns will still be in the output DataFrame.
    '''
    def __init__(self, degree=2, interaction_only=False, include_bias=False, to_interact='all'):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.pol = PolynomialFeatures(degree=self.degree, 
                                      include_bias=self.include_bias, 
                                      interaction_only=self.interaction_only)
        self.to_interact = to_interact
    
    @fit_wrapper
    def fit(self, X, y=None):
        '''
        Method to train the transformer.
        
        Depending on the ``to_interact`` attribute, if fits considering different slices of the
        input DataFrame
        
        It also reset the ``columns`` attribute

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        '''
        if self.to_interact == 'all':
            cols = X.columns
        else:
            cols = self.to_interact
        
        self.pol.fit(X[cols])
        
        return self
        
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to transform the input data.
        
        It populates the ``columns`` attribute with the columns of the output data.
        
        If a bias term is inclued, it will be called ``BIAS_TERM``.

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with polynomial features
        '''
        if self.to_interact == 'all':
            X_tr = self.pol.transform(X)
            X_tr = pd.DataFrame(X_tr, columns=self.pol.get_feature_names_out(X.columns), index=X.index)
        else:
            X_int = self.pol.transform(X[self.to_interact])
            X_int = pd.DataFrame(X_int, columns=self.pol.get_feature_names_out(self.to_interact), index=X.index)
            X_tr = pd.concat([X[[col for col in X if col not in self.to_interact]], X_int], axis=1)
            
        if self.include_bias:
            X_tr.rename(columns={'1': 'BIAS_TERM'}, inplace=True)
        
        return X_tr
