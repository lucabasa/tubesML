__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns


class TargetEncoder(BaseTransformer):
    '''
    Heavily inspired by 
    https://github.com/MaxHalford/xam/blob/93c066990d976c7d4d74b63fb6fb3254ee8d9b48/xam/feature_extraction/encoding/bayesian_target.py#L8
    
    It allows for other aggregating functions, for now it is assumed this is provided as a string for the agg method of pandas
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
     
    
    @reset_columns
    def fit(self, X, y):
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
    
    
    @self_columns
    def transform(self, X, y=None):
        
        X_tr = X.copy()
        
        for col in self.to_encode:
            X_tr[col] = X_tr[col].map(self.posteriors_[col]).fillna(self.prior_).astype(float)
        
        return X_tr
