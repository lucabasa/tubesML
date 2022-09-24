import pytest
import warnings
import pandas as pd
import numpy as np
import tubesml


def create_data():
    df = pd.DataFrame({'a': ['A', 'A', 'A', 'B', 'B', 'C'], 
                       'b': [14, 42, 1, 7.5, 0.4, 100] })
    return df

df = create_data()


def test_targetencoder():
    '''
    Test the encoder works without telling which columns to use
    '''
    enc = tubesml.TargetEncoder()
    res = enc.fit_transform(df, df['b'])
    
    assert res.shape[0] == df.shape[0]
    assert res.shape[1] == df.shape[1]
    
    
def test_targetencoder_toencode():
    '''
    Test the encoder works when telling to encode "a"
    '''
    enc = tubesml.TargetEncoder(to_encode='a')
    res = enc.fit_transform(df, df['b'])
    
    assert res.shape[0] == df.shape[0]
    assert res.shape[1] == df.shape[1]


@pytest.mark.parametrize("agg_func, has_warn", [("mean", None), ('median', None), ('std', None), ('min', None), ('max', None), ('sum', None), ('count', 1)])   
def test_variousencodings(agg_func, has_warn):
    '''
    Test several agg functions, including a forbidden one to check the error
    '''
    if has_warn is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            enc = tubesml.TargetEncoder(agg_func=agg_func)
            res = enc.fit_transform(df, df['b'])
    else:
        with pytest.raises(UserWarning):
            enc = tubesml.TargetEncoder(agg_func=agg_func)
            res = enc.fit_transform(df, df['b'])


def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.TargetEncoder()
    res = trsf.fit_transform(df, df['b'])
    assert trsf.get_feature_names_out()[0] == df.columns[0]
    assert trsf.get_feature_names_out()[1] == df.columns[1]
