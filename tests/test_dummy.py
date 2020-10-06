from .context import source
import pytest
import pandas as pd
import numpy as np

def create_data():
    df = pd.DataFrame({'a': ['1', '2', '3', '4', '5'], 
                       'b': [1]*5 })
    return df

df = create_data()

def test_dummify():
    '''
    Test Dummify
    '''
    dummifier = source.Dummify()
    res = dummifier.fit_transform(df)
    assert res.shape[1] == df['a'].nunique() + 1


def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = source.Dummify()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names()[0] == df.columns[1]
    for i, val in enumerate(df['a'].unique()):
        assert trsf.get_feature_names()[i+1] == f'a_{val}'
