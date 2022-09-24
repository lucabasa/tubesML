import pytest
import warnings
import pandas as pd
import numpy as np
import tubesml


def create_data():
    df = pd.DataFrame({'a': ['1', '2', '3', '4', '5'], 
                       'b': [1]*5 })
    return df

df = create_data()


def test_dummify():
    '''
    Test Dummify
    '''
    dummifier = tubesml.Dummify()
    res = dummifier.fit_transform(df)
    assert res.shape[1] == df['a'].nunique() + 1
    
    
def test_drop_first():
    '''
    Test Dummify with drop_first
    '''
    dummifier = tubesml.Dummify(drop_first=True)
    res = dummifier.fit_transform(df)
    assert res.shape[1] == df['a'].nunique()
    
    
def test_match_columns_add():
    '''
    Test if the dummifier adds a new column
    '''
    df_2 = df.head(4)  # one category less
    dummifier = tubesml.Dummify()
    res = dummifier.fit_transform(df)
    res_2 = dummifier.transform(df_2)
    assert res.shape[1] == res_2.shape[1]

    
def test_match_columns_remove():
    '''
    Test if the dummifier removes a new column
    '''
    df_2 = df.head(4)  # one category less
    dummifier = tubesml.Dummify()
    res_2 = dummifier.fit_transform(df_2)
    res = dummifier.transform(df)
    assert res.shape[1] == res_2.shape[1]
    
    
def test_match_columns_drop_first_add():
    '''
    Test match_columns Dummify with drop_first and a category is missing
    The 2 dataframes should eventually have the same columns
    They both drop the first value and the second one needs to have a last column of 0s
    '''
    dummifier = tubesml.Dummify(drop_first=True)
    res = dummifier.fit_transform(df)
    df_2 = df.head(4)
    res_2 = dummifier.transform(df_2)  # one category less
    assert set(res.columns) == set(res_2.columns)
    assert res_2[res_2.columns[-1]].sum() == 0

    
def test_match_columns_drop_first_remove():
    '''
    Test match_columns Dummify with drop_first and a new column is removed
    The 2 dataframes should eventually have the same columns
    They both drop the first value and the second one needs to have a column removed
    No column should be full of 0s
    '''
    df_2 = df.head(4)
    dummifier = tubesml.Dummify(drop_first=True)
    res = dummifier.fit_transform(df_2)
    res_2 = dummifier.transform(df)  # one category extra
    assert set(res.columns) == set(res_2.columns)
    assert all(res_2.sum() > 0)
    
    
def test_match_columns_drop_first_equal(): 
    '''
    Test match_columns Dummify with drop_first, the category missing is the one dropped
    The 2 dataframes should eventually have the same columns
    They both drop the first value and the second one needs to have a column of 0 with the last value
    No column should be full of 0s
    '''
    dummifier = tubesml.Dummify(drop_first=True)
    res = dummifier.fit_transform(df)
    df_2 = df.tail(4)  # the first category is missing
    res_2 = dummifier.transform(df_2) 
    assert set(res.columns) == set(res_2.columns)
    assert all(res_2.sum() > 0)
    
    
def test_verbose():
    '''
    Test if the dummifier raises a warning when verbose is true
    '''
    df_2 = df.head(4)  # one category less
    dummifier = tubesml.Dummify(verbose=True)
    res = dummifier.fit_transform(df)
    with pytest.warns(UserWarning):
        res_2 = dummifier.transform(df_2)
        

def test_verbose_change():
    '''
    Test if the dummifier does not rais a warning after the first time
    '''
    df_2 = df.head(4)  # one category less
    dummifier = tubesml.Dummify(verbose=True)
    res = dummifier.fit_transform(df)
    with pytest.warns(UserWarning):
        res_2 = dummifier.transform(df_2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res_2 = dummifier.transform(df_2)

    
def test_dummy_cols():
    '''
    Test the attribute columns is well defined
    '''
    dummifier = tubesml.Dummify()
    res = dummifier.fit_transform(df)
    assert dummifier.columns[0] == df.columns[1]
    for i, val in enumerate(df['a'].unique()):
        assert dummifier.columns[i+1] == f'a_{val}'


def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.Dummify()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == df.columns[1]
    for i, val in enumerate(df['a'].unique()):
        assert trsf.get_feature_names_out()[i+1] == f'a_{val}'
