__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper


class TargetEncoder(BaseTransformer):
    '''
    Heavily inspired by 
    `MaxHalford <https://github.com/MaxHalford/xam/blob/93c066990d976c7d4d74b63fb6fb3254ee8d9b48/xam/feature_extraction/encoding/bayesian_target.py#L8>`__
    
    Encodes categorical features with statistics of the target variable. For example, by using the mean target value.
    
    It allows for other aggregating functions, for now it is assumed this is provided as a string for the agg method of pandas.
    
    Inherits from ``BaseTransformer``.


    :param to_encode: str, list, None. default=None.
                (list of) column(s) to encode according to the ``agg_func``.
                If None, it will encode all the non-numerical columns.
                
    :param prior_weight: int, float. default=100.
                Value to weight the prior. The higher, the more important the prior is.
                The prior is the statistic of the target determined by ``agg_func``.
                
    :param agg_func: str, default='mean'.
                Aggregation function to use for the target encoding.
    '''

    def __init__(self, to_encode=None, prior_weight=100, agg_func='mean'):
        super().__init__()
        if isinstance(to_encode, str):
            self.to_encode = [to_encode]
        else:
            self.to_encode = to_encode
        self.prior_weight = prior_weight
        self.prior_ = None
        self.posteriors_ = None
        self.agg_func = agg_func
        
        
    @fit_wrapper
    def fit(self, X, y):
        '''
        Method to train the encoder by determining the posterior of each column
        
        If ``to_encode`` is None, it will encode all the non-numerical columns
        
        It also reset the ``columns`` attribute

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs).
            The target values (or class labels) as integers or floats.
        '''
        if self.agg_func == 'count':
            raise UserWarning('Frequency encoding not supported')  # TODO: allow this in the future
        # Encode all categorical cols by default
        if self.to_encode is None:
            self.to_encode = [c for c in X if str(X[c].dtype)=='object' or str(X[c].dtype)=='category']
        
        tmp = X.copy()
        tmp['target'] = y
        
        self.prior_ = tmp['target'].agg(self.agg_func)
        self.posteriors_ = {}
        
        for col in self.to_encode:
            
            agg = tmp.groupby(col)['target'].agg(['count', self.agg_func])
            counts = agg['count']
            data = agg[self.agg_func]
            pw = self.prior_weight
            self.posteriors_[col] = ((pw * self.prior_ + counts * data) / (pw + counts)).to_dict()
        
        del tmp
        return self
    
    
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to transform the input data
        
        It populates the ``columns`` attribute with the columns of the output data
        
        For each column to encode, it replaces each value with the posterior computed in the ``fit`` method
        If there are missing values, those are filled in with the prior (e.g. the statistic of the target 
        determined by `agg_func`)

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        
        :return: pandas DataFrame with encoded features
        '''
        X_tr = X.copy()
        
        for col in self.to_encode:
            X_tr[col] = X_tr[col].map(self.posteriors_[col]).fillna(self.prior_).astype(float)
        
        return X_tr
