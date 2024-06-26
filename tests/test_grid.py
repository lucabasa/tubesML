import tubesml as tml
import pytest
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

import string
import random


def create_data():
    df, target = make_classification(n_features=10)
    
    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append(''.join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1
        
    df = pd.DataFrame(df, columns=random_names)
    df['target'] = target
    
    df.loc[df.sample(30).index, df.columns[0]] = np.nan
    
    return df

df = create_data()

pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('poly', tml.DfPolynomial()),
                     ('sca', tml.DfScaler(method='standard')),  
                     ('tarenc', tml.TargetEncoder()), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9))])
pipe = tml.FeatureUnionDf([('transf', pipe_transf)])

full_pipe = Pipeline([('pipe', pipe), 
                      ('logit', LogisticRegression(solver='lbfgs'))])


@pytest.mark.parametrize("random", [False, 20])
def test_grid_bestestimator(random):
    '''
    Test grid_search returns an estimator ready to be used with no warnings, with and without random search
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    param_grid = {'logit__C': [1, 2], 
                  'pipe__transf__sca__method': ['standard', 'robust', 'minmax'], 
                  'pipe__transf__imp__strategy': ['mean', 'median'],
                  'pipe__transf__poly__degree': [1, 2, 3], 
                  'pipe__transf__tarenc__agg_func': ['mean', 'median'],
                  'pipe__transf__dummify__drop_first': [True, False], 
                  'pipe__transf__dummify__match_cols': [True, False], 
                  'pipe__transf__pca__n_components': [0.5, 3, 5]}
    
    result, best_param, best_estimator = tml.grid_search(data=df_1, target=y, estimator=full_pipe, 
                                                         param_grid=param_grid, scoring='accuracy', cv=3, random=random)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res = best_estimator.predict(df_1)
    
    
@pytest.mark.parametrize("random", [False, 20])
def test_grid_bestestimator_proba(random):
    '''
    Test grid_search returns an estimator ready to be used with no warnings, with and without random search
    This test uses the predict_proba method
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    param_grid = {'logit__C': [1, 2], 
                  'pipe__transf__sca__method': ['standard', 'robust', 'minmax'], 
                  'pipe__transf__imp__strategy': ['mean', 'median'],
                  'pipe__transf__poly__degree': [1, 2, 3], 
                  'pipe__transf__tarenc__agg_func': ['mean', 'median'],
                  'pipe__transf__dummify__drop_first': [True, False], 
                  'pipe__transf__dummify__match_cols': [True, False], 
                  'pipe__transf__pca__n_components': [0.5, 3, 5]}
    
    result, best_param, best_estimator = tml.grid_search(data=df_1, target=y, estimator=full_pipe, 
                                                         param_grid=param_grid, scoring='neg_log_loss', cv=3, random=random)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res = best_estimator.predict(df_1)


@pytest.mark.parametrize("random, n_res", [(False, 6), (5, 5)])   
def test_gridsearch_result(random, n_res):
    '''
    Test grid_search returns a dataframe summarizing the search results
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    param_grid = {'logit__C': [1, 2], 
                  'pipe__transf__sca__method': ['standard', 'robust', 'minmax']}
    
    result, best_param, best_estimator = tml.grid_search(data=df_1, target=y, estimator=full_pipe, 
                                                         param_grid=param_grid, scoring='accuracy', cv=3, random=random)
    
    assert result.shape[0] == n_res
    assert result.shape[1] == 10


@pytest.mark.parametrize("random", [False, 5])
def test_gridsearch_params(random):
    '''
    Test grid_search returns a dictionary of parameters with the best combination of parameters
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    param_grid = {'logit__C': [1, 2], 
                  'pipe__transf__sca__method': ['standard', 'robust', 'minmax']}
    
    result, best_param, best_estimator = tml.grid_search(data=df_1, target=y, estimator=full_pipe, 
                                                         param_grid=param_grid, scoring='accuracy', cv=3, random=random)
    
    assert len(best_param.keys()) == len(param_grid.keys())


@pytest.mark.parametrize("random", [False, 5])
def test_gridsearch_nopipeline(random):
    '''
    Test grid_search when provided with a simple estimator without the pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    df_1 = tml.DfImputer('mean').fit_transform(df_1)
    
    model = LogisticRegression(solver='lbfgs')
    
    param_grid = {'C': np.arange(1, 10)}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result, best_param, best_estimator = tml.grid_search(data=df_1, target=y, estimator=model, 
                                                         param_grid=param_grid, scoring='accuracy', cv=3, random=random)
    