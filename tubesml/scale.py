__author__ = 'lucabasa'
__version__ = '0.0.4'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd


class DfScaler(BaseTransformer):
    '''
    Wrapper of several sklearn scalers that keeps the dataframe structure.
    
    Inherits from ``BaseTransformer``,
    
    :param method: str, the method to scale the data, default "standard"
              Allowed values: "standard", 'robust', 'minmax'
    
    :param feature_range: tuple, the range to scale the data to.
                    Relevant only if ``method=='minmax'``
    
    :Attributes:
    
        `mean_` : pandas Series with the mean of each feature in the input data.
                    It is relevant only if ``method=Standard``. 
                    The index of the series is the ``columns`` attribute of the input dataframe.

        `center_` : pandas Series with the median of each feature in the input data.
                    It is relevant only if ``method=Robust``. 
                    The index of the series is the ``columns`` attribute of the input dataframe.

        `min_` : pandas Series with the min of each feature in the input data.
                It is relevant only if ``method=minmax``. 
                The index of the series is the ``columns`` attribute of the input dataframe.

        `data_min_` : pandas Series with the min of each feature in the input data.
                It is relevant only if ``method=minmax``. 
                The index of the series is the ``columns`` attribute of the input dataframe.  

        `data_max_` : pandas Series with the max of each feature in the input data.
                It is relevant only if ``method=minmax``. 
                The index of the series is the ``columns`` attribute of the input dataframe.

        `feature_range_` : pandas Series with the difference between max and min of each feature in the input data.
                It is relevant only if ``method=minmax``. 
                The index of the series is the ``columns`` attribute of the input dataframe.
    '''
    def __init__(self, method='standard', feature_range=(0,1)):
        super().__init__()
        self.method = method
        self._validate_input()
        self.scale_ = None
        self.feature_range = feature_range
        if self.method == 'standard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None
        elif method == 'minmax':
            self.feature_range = feature_range
            self.min_ = None
            self.data_min_ = None
            self.data_max_ = None
            self.data_range_ = None
            self.n_samples_seen_ = None

            
    def _validate_input(self):
        allowed_methods = ["standard", 'robust', 'minmax']
        if self.method not in allowed_methods:
            raise ValueError(f"Can only use these methods: {allowed_methods} got method={self.method}")
    
    
    @fit_wrapper
    def fit(self, X, y=None):
        '''
        Method to train the scaler.
        
        Depending on the ``method`` attribute, it calls a different sklearn scaler
        
        It also reset the ``columns`` attribute

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        '''
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.scl.fit(X)
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.scl = RobustScaler()
            self.scl.fit(X)
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        elif self.method == 'minmax':
            self.scl = MinMaxScaler(feature_range=self.feature_range)
            self.scl.fit(X)
            self.min_ = pd.Series(self.scl.min_, index=X.columns)
            self.data_min_ = pd.Series(self.scl.data_min_, index=X.columns)
            self.data_max_ = pd.Series(self.scl.data_max_, index=X.columns)
            self.data_range_ = self.data_max_ - self.data_min_
            self.n_samples_seen_ = X.shape[0]
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self
    
    
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to transform the input data.
        
        It populates the ``columns`` attribute with the columns of the output data.

        :param X: pandas DataFrame of shape (n_samples, n_features).
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used.
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with scaled data.
        '''
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled
    