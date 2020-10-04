from .context import source
import pytest
import pandas as pd
import numpy as np

    
def create_data():
    df = pd.DataFrame({'a': [1, np.nan, 5], 
                       'b': [3, 2, 1] })
    return df

df = create_data()

def test_clean():
    imputer = source.DfImputer(strategy='mean')
    res = imputer.fit_transform(df)
    assert res.isna().any().any() == 0
    

def test_cl_stats():
    imputer = source.DfImputer(strategy='mean')
    imputer.fit(df)
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([3, 2], index=df.columns), check_dtype=False)
    

def test_cl_cols():
    imputer = source.DfImputer(strategy='mean')
    res = imputer.fit_transform(df)
    assert imputer.columns[0] == 'a'
    assert imputer.columns[1] == 'b'
    
    
def test_cl_strat_median():
    imputer = source.DfImputer(strategy='median')
    imputer.fit(df)
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([3, 2], index=df.columns), check_dtype=False)


def test_cl_strat_mostfreq():
    imputer = source.DfImputer(strategy='most_frequent')
    imputer.fit(df)
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([1, 1], index=df.columns), check_dtype=False)


def test_cl_strat_constant():
    imputer = source.DfImputer(strategy='constant', fill_value=5)
    imputer.fit(df)
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([5, 5], index=df.columns), check_dtype=False)

    
def test_cl_error():
    with pytest.raises(ValueError):
        imputer = source.DfImputer(strategy='not the mean')
        