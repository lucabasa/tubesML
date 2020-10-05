from .context import source
import pytest
import pandas as pd
import numpy as np


def create_data():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 
                       'b': [1]*5 })
    return df

df = create_data()


def test_scl_standard():
    scaler = source.DfScaler(method='standard')
    res = scaler.fit_transform(df)
    assert res['a'].mean() == 0
    
    
def test_scl_robust():
    scaler = source.DfScaler(method='robust')
    res = scaler.fit_transform(df)
    assert res['a'].mean() == 0
    

def test_scl_minmax():
    scaler = source.DfScaler(method='minmax')
    res = scaler.fit_transform(df)
    assert res['a'].mean() == 0.5


def test_scl_scale():
    scaler = source.DfScaler()
    scaler.fit(df)
    pd.testing.assert_series_equal(scaler.scale_, pd.Series([1.414214, 1], index=df.columns), check_dtype=False)
    
    
def test_scl_scale_minmax():
    scaler = source.DfScaler(method='minmax', feature_range=(0,2))
    scaler.fit(df)
    real_scale = (scaler.feature_range[1] - scaler.feature_range[0]) / (df['a'].max() - df['a'].min())
    assert scaler.scale_[0] == real_scale
    
    
def test_scl_mean():
    scaler = source.DfScaler(method='standard')
    scaler.fit(df)
    pd.testing.assert_series_equal(scaler.mean_, pd.Series([3, 1], index=df.columns), check_dtype=False)
    
    
def test_scl_center():
    scaler = source.DfScaler(method='robust')
    scaler.fit(df)
    pd.testing.assert_series_equal(scaler.center_, pd.Series([3, 1], index=df.columns), check_dtype=False)
    
    
def test_scl_min():
    scaler = source.DfScaler(method='minmax')
    scaler.fit(df)
    real_min = scaler.feature_range[0] - df.min(axis=0) * scaler.scale_
    pd.testing.assert_series_equal(scaler.min_, real_min, check_dtype=False)


def test_scl_error():
    with pytest.raises(ValueError):
        imputer = source.DfScaler(method='Not the right method')
        
        
def test_scl_cols():
    scaler = source.DfScaler()
    res = scaler.fit_transform(df)
    assert scaler.columns[0] == 'a'
    assert scaler.columns[1] == 'b'
    
    
def test_get_feature_names():
    scaler = source.DfScaler()
    res = scaler.fit_transform(df)
    assert scaler.get_feature_names()[0] == 'a'
    assert scaler.get_feature_names()[1] == 'b'
    
    