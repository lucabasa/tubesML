__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

from tubesml.base import BaseTransformer, fit_wrapper, transform_wrapper
from sklearn.decomposition import PCA
import pandas as pd


class DfPCA(BaseTransformer):
    '''
    Wrapper around PCA to keep the dataframe structure.
    It can also return the same dataframe in a compressed form, e.g. by doing and undoing pca.
    
    Inherits from ``BaseTransformer``.
    
    :param n_components: int, float or 'mle'.
        Number of components to keep.
        if n_components is not set all components are kept::
        n_components == min(n_samples, n_features)
        If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's
        MLE is used to guess the dimension. Use of ``n_components == 'mle'``
        will interpret ``svd_solver == 'auto'`` as ``svd_solver == 'full'``.
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.
        Hence, the None case results in::
        n_components == min(n_samples, n_features) - 1
            
    :param svd_solver: {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
        The solver is selected by a default policy based on ``X.shape`` and
        ``n_components``: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
        If full :
        run exact full SVD calling the standard LAPACK solver via
        ``scipy.linalg.svd`` and select the components by postprocessing
        If arpack :
        run SVD truncated to n_components calling ARPACK solver via
        ``scipy.sparse.linalg.svds``. It requires strictly
        0 < n_components < min(X.shape)
        If randomized :
        run randomized SVD by the method of Halko et al.
            
    :param random_state: int, RandomState instance or None, default=24
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
        
    :param compress: bool, default=False.
        If True, it reverses the PCA via ``inverse_transform`` and returns a DataFrame with the original structure
        It can be useful to remove noise from the data by compressing the information.
    '''
    def __init__(self, n_components, svd_solver='auto', random_state=24, compress=False):
        super().__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.n_components_ = 0  # will be assigned by the fit method
        self.random_state = random_state
        self.PCA = PCA(n_components=self.n_components, svd_solver=self.svd_solver, random_state=self.random_state)
        self.compress = compress
        self.original_columns = []
        
    @fit_wrapper
    def fit(self, X, y=None):
        '''
        Method to train the transformer.
        
        It also reset the ``columns`` attribute.

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        '''
        self.PCA.fit(X)
        self.n_components_ = self.PCA.n_components_

        self.original_columns = X.columns
        
        return self
    
    @transform_wrapper
    def transform(self, X, y=None):
        '''
        Method to transform the input data.
        
        It populates the ``columns`` attribute with the columns of the output data.
        
        The resulting columns will have name ``pca_{int}``. 
        
        If ``compress=True``, the ``inverse_transform`` method is called and the original
        columns are restored.

        :param X: pandas DataFrame of shape (n_samples, n_features)
            The input samples.
            
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return: pandas DataFrame with pca columns or, if ``compress=True``, pandas DataFrame
                with original columns
        '''     
        X_tr = self.PCA.transform(X)
        X_tr = pd.DataFrame(X_tr, columns=[f'pca_{i}' for i in range(self.n_components_)])
        
        self.original_columns = X.columns
        
        if self.compress:
            X_tr = self.inverse_transform(X_tr)
        
        return X_tr
    
    
    def inverse_transform(self, X, y=None):
        
        try:
            X_tr = self.PCA.inverse_transform(X)
        except ValueError:
            return X
        try:
            X_tr.columns = self.original_columns
        except AttributeError:  # FIXME: backward compatibility with Kaggle
            X_tr = pd.DataFrame(X_tr, columns=self.original_columns)
        
        return X_tr
