import tubesml as tml
import pytest
import warnings
import pandas as pd
import numpy as np

import string
import random

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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


@pytest.mark.parametrize("add_indicator", [True, False])
def test_transformers(add_indicator):
    '''
    Test a pipeline doesn't break when all the transformers are called in succession
    '''
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean', add_indicator=add_indicator)), 
                     ('poly', tml.DfPolynomial()), 
                     ('sca', tml.DfScaler(method='standard')), 
                     ('tarenc', tml.TargetEncoder()), 
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9, compress=True))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = pipe.fit_transform(df, df['target'])
    

@pytest.mark.parametrize("add_indicator", [True, False])
def test_predictions(add_indicator):
    '''
    Test a pipeline doesn't break when all the transformers are called in succession
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean', add_indicator=add_indicator)), 
                     ('poly', tml.DfPolynomial()), 
                     ('sca', tml.DfScaler(method='standard')),  
                     ('tarenc', tml.TargetEncoder()),
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9, compress=True))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])
    
    full_pipe = Pipeline([('pipe', pipe), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full_pipe.fit(df_1, y)
        res = full_pipe.predict(df_1)
    