import tubesml as tml
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import pandas as pd

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
    
    return df

df = create_data()


def test_cvscore():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('poly', tml.DfPolynomial()),
                     ('sca', tml.DfScaler(method='standard')), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    
    full_pipe = Pipeline([('pipe', pipe), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res = tml.cv_score(df_1, y, full_pipe, cv=kfold)
    assert len(record) == 0
    assert len(res) == len(df_1)
    
    
def test_cvscore_predictproba():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('sca', tml.DfScaler(method='standard')), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    
    full_pipe = Pipeline([('pipe', pipe), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res = tml.cv_score(df_1, y, full_pipe, cv=kfold, predict_proba=True)
    assert len(record) == 0
    assert len(res) == len(df_1)


def test_cvscore_coefficients():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('sca', tml.DfScaler(method='standard')), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9, compress=True))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    
    full_pipe = Pipeline([('pipe', pipe), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res, coef = tml.cv_score(df_1, y, full_pipe, cv=kfold, imp_coef=True)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]
    

def test_cvscore_importances():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('sca', tml.DfScaler(method='standard')), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9, compress=True))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    
    full_pipe = Pipeline([('pipe', pipe), 
                          ('tree', DecisionTreeClassifier())])
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res, coef = tml.cv_score(df_1, y, full_pipe, cv=kfold, imp_coef=True)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]
