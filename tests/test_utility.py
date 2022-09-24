import tubesml
import pytest
import warnings
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline


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
    Test the selector raises the right error 
    '''
    with pytest.raises(ValueError):
        sel = tubesml.DtypeSel(dtype='Not the right dtype')

        
def test_featun():
    '''
    Test the union of features using pipelines
    '''
    num_pipe = Pipeline([('num', tubesml.DtypeSel(dtype='numeric'))])
    cat_pipe = Pipeline([('cat', tubesml.DtypeSel(dtype='category'))])
    tot_pipe = tubesml.FeatureUnionDf(transformer_list=[('cat', cat_pipe), 
                                                        ('num', num_pipe)])
    res = tot_pipe.fit_transform(df)
    assert res.columns[0] == df.columns[0]
    assert res.columns[1] == df.columns[1]
    
    
def test_featun_nowarnings():
    '''
    Test the union of features does not raises warnings
    '''
    num_pipe = Pipeline([('num', tubesml.DtypeSel(dtype='numeric'))])
    cat_pipe = Pipeline([('cat', tubesml.DtypeSel(dtype='category'))])
    tot_pipe = tubesml.FeatureUnionDf(transformer_list=[('cat', cat_pipe), 
                                                        ('num', num_pipe)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = tot_pipe.fit_transform(df)
    

def test_featun_transformers():
    '''
    Test the union of features does not raises warnings
    '''
    tot_pipe = tubesml.FeatureUnionDf(transformer_list=[('cat', tubesml.DtypeSel(dtype='category')), 
                                                        ('num', tubesml.DtypeSel(dtype='numeric'))])
    res = tot_pipe.fit_transform(df)
    assert res.columns[0] == df.columns[0]
    assert res.columns[1] == df.columns[1]
    
        
def test_dtype_cols_dtype():
    '''
    Test the attribute columns is well defined
    '''
    trsf = tubesml.DtypeSel(dtype='numeric')
    res = trsf.fit_transform(df)
    assert trsf.columns[0] == df.columns[1]

    
def test_get_feature_names_dtype():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DtypeSel(dtype='numeric')
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == df.columns[1]
    

def test_dtype_cols_featun():
    '''
    Test the attribute columns is well defined
    '''
    num_pipe = Pipeline([('num', tubesml.DtypeSel(dtype='numeric'))])
    cat_pipe = Pipeline([('cat', tubesml.DtypeSel(dtype='category'))])
    trsf = tubesml.FeatureUnionDf(transformer_list=[('cat', cat_pipe), 
                                                    ('num', num_pipe)])
    res = trsf.fit_transform(df)
    assert trsf.columns[0] == df.columns[0]


def test_get_feature_names_featun():
    '''
    Test the transformer still has get_feature_names
    '''
    num_pipe = Pipeline([('num', tubesml.DtypeSel(dtype='numeric'))])
    cat_pipe = Pipeline([('cat', tubesml.DtypeSel(dtype='category'))])
    trsf = tubesml.FeatureUnionDf(transformer_list=[('cat', cat_pipe), 
                                                    ('num', num_pipe)])
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == df.columns[0]   

