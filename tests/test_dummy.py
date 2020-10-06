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
    
    
def test_drop_first():
    '''
    Test Dummify with drop_first
    '''
    dummifier = source.Dummify(drop_first=True)
    res = dummifier.fit_transform(df)
    assert res.shape[1] == df['a'].nunique()
    
    
def test_match_columns_add():
    '''
    Test if the dummifier adds a new column
    '''
    df_2 = df.head(4)  # one category less
    dummifier = source.Dummify()
    res = dummifier.fit_transform(df)
    res_2 = dummifier.transform(df_2)
    assert res.shape[1] == res_2.shape[1]

    
def test_match_columns_remove():
    '''
    Test if the dummifier removes a new column
    '''
    df_2 = df.head(4)  # one category less
    dummifier = source.Dummify()
    res_2 = dummifier.fit_transform(df_2)
    res = dummifier.transform(df)
    assert res.shape[1] == res_2.shape[1]
    
    
def test_verbose():
    '''
    Test if the dummifier raises a warning when verbose is true
    '''
    df_2 = df.head(4)  # one category less
    dummifier = source.Dummify(verbose=True)
    res = dummifier.fit_transform(df)
    with pytest.warns(UserWarning):
        res_2 = dummifier.transform(df_2)
        

def test_verbose_change():
    '''
    Test if the dummifier does not rais a warning after the first time
    '''
    df_2 = df.head(4)  # one category less
    dummifier = source.Dummify(verbose=True)
    res = dummifier.fit_transform(df)
    with pytest.warns(UserWarning):
        res_2 = dummifier.transform(df_2)
    with pytest.warns(None) as record:
        res_2 = dummifier.transform(df_2)
    assert len(record) == 0

    
def test_scl_cols():
    '''
    Test the attribute columns is well defined
    '''
    dummifier = source.Dummify()
    res = dummifier.fit_transform(df)
    assert dummifier.columns[0] == df.columns[1]
    for i, val in enumerate(df['a'].unique()):
        assert dummifier.columns[i+1] == f'a_{val}'


def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = source.Dummify()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names()[0] == df.columns[1]
    for i, val in enumerate(df['a'].unique()):
        assert trsf.get_feature_names()[i+1] == f'a_{val}'
