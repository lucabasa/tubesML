import tubesml
import pytest
import pandas as pd
import numpy as np


def create_data():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 
                       'b': [1]*5 })
    return df

df = create_data()


def test_scl_standard():
    '''
    Test StandardScaler
    '''
    scaler = tubesml.DfScaler(method='standard')
    res = scaler.fit_transform(df)
    for col in df.columns:
        assert res[col].mean() == 0
    
    
def test_scl_robust():
    '''
    Test RobustScaler
    '''
    scaler = tubesml.DfScaler(method='robust')
    res = scaler.fit_transform(df)
    for col in df.columns:
        assert res[col].mean() == df[col].mean() - df[col].median()
    

def test_scl_minmax():
    '''
    Test MinMaxScaler
    '''
    scaler = tubesml.DfScaler(method='minmax')
    res = scaler.fit_transform(df)
    #todo: this test is not very robust
    assert res['a'].mean() == 0.5


def test_scl_scale():
    '''
    Test the definition of scale a series works
    '''
    scaler = tubesml.DfScaler()
    scaler.fit(df)
    #todo: this test is not very robust
    pd.testing.assert_series_equal(scaler.scale_, pd.Series([1.414214, 1], index=df.columns), check_dtype=False)
    
    
def test_scl_scale_minmax():
    '''
    Test the definition of scale a series works with minmax
    '''
    scaler = tubesml.DfScaler(method='minmax', feature_range=(0,2))
    scaler.fit(df)
    real_scale = (scaler.feature_range[1] - scaler.feature_range[0]) / (df['a'].max() - df['a'].min())
    assert scaler.scale_.iloc[0] == real_scale
    
    
def test_scl_mean():
    '''
    Test the definition of mean a series works
    '''
    scaler = tubesml.DfScaler(method='standard')
    scaler.fit(df)
    real_mean = df.mean()
    pd.testing.assert_series_equal(scaler.mean_, real_mean, check_dtype=False)
    
    
def test_scl_center():
    '''
    Test the definition of center a series works
    '''
    scaler = tubesml.DfScaler(method='robust')
    scaler.fit(df)
    real_center = df.median()
    pd.testing.assert_series_equal(scaler.center_, real_center, check_dtype=False)
    
    
def test_scl_min():
    '''
    Test the definition of min_ for minmax as a series works
    '''
    scaler = tubesml.DfScaler(method='minmax')
    scaler.fit(df)
    real_min = scaler.feature_range[0] - df.min(axis=0) * scaler.scale_
    pd.testing.assert_series_equal(scaler.min_, real_min, check_dtype=False)


def test_scl_error():
    '''
    Test the scaler raises the right error 
    '''
    with pytest.raises(ValueError):
        imputer = tubesml.DfScaler(method='Not the right method')
        
        
def test_scl_cols():
    '''
    Test the attribute columns is well defined
    '''
    scaler = tubesml.DfScaler()
    res = scaler.fit_transform(df)
    assert scaler.columns[0] == df.columns[0]
    assert scaler.columns[1] == df.columns[1]
    
    
def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DfScaler()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == df.columns[0]
    assert trsf.get_feature_names_out()[1] == df.columns[1]
    
    