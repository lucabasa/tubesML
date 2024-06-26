import tubesml
import pytest
import warnings
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification

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


def test_pca():
    '''
    Test the transformer works
    '''
    pca = tubesml.DfPCA(n_components=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res = pca.fit_transform(df)

@pytest.mark.parametrize("compress", [True, False])
def test_no_nan(compress):
    '''
    Test it returns meaningful values
    '''
    pca = tubesml.DfPCA(n_components=2, compress=compress)
    res = pca.fit_transform(df)
    assert res.isna().any().sum() == 0
    

def test_pca_columns():
    '''
    Test the transformer columns are called properly
    '''
    pca = tubesml.DfPCA(n_components=0.5)
    res = pca.fit_transform(df)
    assert 'pca_0' in res.columns
    
    
def test_pca_compression():
    '''
    Test pca with compression=True
    '''
    pca = tubesml.DfPCA(n_components=5, compress=True)
    res = pca.fit_transform(df)
    assert (res.columns == df.columns).all()
    
    
def test_inverse_transform():
    '''
    Test if the inverse transform works
    '''
    pca = tubesml.DfPCA(n_components=2)
    res = pca.fit_transform(df)
    res_2 = pca.inverse_transform(res)
    assert (res_2.columns == df.columns).all()
    assert res_2.isna().any().sum() == 0
    
    
def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DfPCA(n_components=4)
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == 'pca_0'
    assert trsf.get_feature_names_out()[1] == 'pca_1'
    