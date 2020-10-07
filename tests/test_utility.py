from .context import tubesml
import pytest
import pandas as pd
import numpy as np


def create_data():
    df = pd.DataFrame({'a': ['1', '2', '3', '4', '5'], 
                       'b': [1]*5 })
    return df

df = create_data()


def test_dtypesel_numeric():
    '''
    Test if the numeric features are selected
    '''
    sel = tubesml.DtypeSel(dtype='numeric')
    res = sel.fit_transform(df)
    assert res.shape[1] == 1
    assert res.columns[0] == 'b'


def test_dtypesel_category():
    '''
    Test if the non-numeric features are selected
    '''
    sel = tubesml.DtypeSel(dtype='category')
    res = sel.fit_transform(df)
    assert res.shape[1] == 1
    assert res.columns[0] == 'a'
    
    
def test_dtype_othercategory():
    '''
    Test if the non-numeric features are selected if they are really category
    '''
    df_2 = df.copy()
    df_2['a'] = df_2['a'].astype('category')
    sel = tubesml.DtypeSel(dtype='category')
    res = sel.fit_transform(df)
    assert res.shape[1] == 1
    assert res.columns[0] == 'a'
    

def test_dtype_error():
    '''
    Test the scaler raises the right error 
    '''
    with pytest.raises(ValueError):
        sel = tubesml.DtypeSel(dtype='Not the right dtype')
        
        
def test_dtype_cols():
    '''
    Test the attribute columns is well defined
    '''
    trsf = tubesml.DtypeSel(dtype='numeric')
    res = trsf.fit_transform(df)
    assert trsf.columns[0] == df.columns[1]

    
    
def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DtypeSel(dtype='numeric')
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names()[0] == df.columns[1]
