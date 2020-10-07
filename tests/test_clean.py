from .context import tubesml
import pytest
import pandas as pd
import numpy as np

    
def create_data():
    df = pd.DataFrame({'a': [1, np.nan, 5], 
                       'b': [3, 2, 1] })
    return df

df = create_data()

def test_clean():
    '''
    Test the imputer actually works
    '''
    imputer = tubesml.DfImputer(strategy='mean')
    res = imputer.fit_transform(df)
    assert res.isna().any().any() == 0
    

def test_cl_stats():
    '''
    Test the imputer learns the mean
    '''
    imputer = tubesml.DfImputer(strategy='mean')
    imputer.fit(df)
    real_mean = df.mean()
    pd.testing.assert_series_equal(imputer.statistics_, real_mean, check_dtype=False)
    

def test_cl_cols():
    '''
    Test the attribute columns is well defined
    '''
    imputer = tubesml.DfImputer(strategy='mean')
    res = imputer.fit_transform(df)
    assert imputer.columns[0] == df.columns[0]
    assert imputer.columns[1] == df.columns[1]
    
    
def test_cl_stat_median():
    '''
    Test the imputer learns the median
    '''
    imputer = tubesml.DfImputer(strategy='median')
    imputer.fit(df)
    real_median = df.median()
    pd.testing.assert_series_equal(imputer.statistics_, real_median, check_dtype=False)


def test_cl_stat_mostfreq():
    '''
    Test the imputer learns the most frequent 
    '''
    imputer = tubesml.DfImputer(strategy='most_frequent')
    imputer.fit(df)
    # todo: this test is not very robust
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([1, 1], index=df.columns), check_dtype=False)


def test_cl_stat_constant():
    '''
    Test the imputer learns the most frequent 
    '''
    imputer = tubesml.DfImputer(strategy='constant', fill_value=5)
    imputer.fit(df)
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([5]*df.shape[1], index=df.columns), check_dtype=False)

    
def test_cl_error():
    '''
    Test the imputer raises the right error 
    '''
    with pytest.raises(ValueError):
        imputer = tubesml.DfImputer(strategy='not the mean')
        
        
def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DfImputer()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names()[0] == df.columns[0]
    assert trsf.get_feature_names()[1] == df.columns[1]
        