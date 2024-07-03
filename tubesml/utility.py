__author__ = 'lucabasa'
__version__ = '0.0.5'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper
import pandas as pd
from sklearn.pipeline import FeatureUnion


class DtypeSel(BaseTransformer):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
        
    :param dtype: str, the type of data to select, default='numeric'.
            Allowed values: 'numeric', 'category'
    '''
    def __init__(self, dtype='numeric'):
        super().__init__()
        self.dtype = dtype
        self._validate_input()
        
        
    def _validate_input(self):
        allowed_dtype = ['numeric', 'category']
        if self.dtype not in allowed_dtype:
            raise ValueError(f"Can only use these dtype: {allowed_dtype} got strategy={self.dtype}")
    
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to select columns based on their type.
        
        It populates the ``columns`` attribute with the columns of the output data
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with columns of the selected type
        '''
        if self.dtype == 'numeric':
            X_tr = X.select_dtypes(include='number')
        elif self.dtype == 'category':
            X_tr = X.select_dtypes(exclude='number')
        return X_tr
    
    
class FeatureUnionDf(BaseTransformer):
    '''
    Wrapper of `FeatureUnion` but returning a Dataframe, 
    the column order follows the concatenation done by FeatureUnion

    :param transformer_list: list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.
    
    :param n_jobs: int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. 
    
    :param transformer_weights: dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.
        
    :param verbose: bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.
    '''
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=False):
        super().__init__()
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose  # these are necessary to work inside of GridSearch or similar
        try:
            self.feat_un = FeatureUnion(self.transformer_list, 
                                        n_jobs=self.n_jobs, 
                                        transformer_weights=self.transformer_weights, 
                                        verbose=self.verbose,
                                        verbose_feature_names_out=verbose_feature_names_out)
        except TypeError:  # this can happen on earlier sklearn versions
            self.feat_un = FeatureUnion(self.transformer_list, 
                                        n_jobs=self.n_jobs, 
                                        transformer_weights=self.transformer_weights, 
                                        verbose=self.verbose)
    
    @fit_wrapper    
    def fit(self, X, y=None):
        '''
        Method to fit all the transformers.
        
        It also reset the ``columns`` attribute.
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs).
            The target values (class labels) as integers or strings.
        '''
        self.feat_un.fit(X, y)
        return self
    
    
    @transform_wrapper
    def transform(self, X, y=None):
        """
        Method to call all the transform methods in the ``transformer_list``
        
        It populates the ``columns`` attribute with the columns of the output data
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with all the transformation applied in the order provided
        """
        X_tr = self.feat_un.transform(X)
        columns = []
        
        for trsnf in self.transformer_list:
            try:
                cols = trsnf[1].steps[-1][1].get_feature_names_out()
            except AttributeError:  # in case it is not a pipeline
                cols = trsnf[1].get_feature_names_out()
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        
        return X_tr.convert_dtypes()

    
    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)