import tubesml
import pytest
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification

import string
import random


def create_data():
    df, target = make_classification(n_features=2, n_informative=2, n_redundant=0)
    
    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 2:
        random_names.append(''.join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1
        
    df = pd.DataFrame(df, columns=random_names)
    df['target'] = target
    
    return df

df = create_data()


def test_poly_all():
    '''
    Test if the interactions are created
    '''
    poly = tubesml.DfPolynomial()
    res = poly.fit_transform(df)
    
    assert len(res.columns) == 9  # 3 columns in total, all pairs of 2, inclunding squares and singles


def test_poly_interact():
    '''
    Test if the to_interact attribute works
    '''
    poly = tubesml.DfPolynomial(to_interact=[col for col in df if 'target' not in col])
    res = poly.fit_transform(df)
    
    assert len(res.columns) == 6 # do not use target in the interactions, so as above but without 3
    

def test_include_bias():
    '''
    Test if we can include a bias term
    '''
    poly = tubesml.DfPolynomial(include_bias=True)
    res = poly.fit_transform(df)
    
    assert len(res.columns) == 10 
    assert 'BIAS_TERM' in res.columns
    
    
def test_interaction_only():
    '''
    Test if we can include only the interaction
    '''
    poly = tubesml.DfPolynomial(interaction_only=True)
    res = poly.fit_transform(df)
    
    assert len(res.columns) == 6
    assert len([col for col in res if '^' in col]) == 0
    
    
def test_get_feature_names():
    '''
    Test the transformer still has get_feature_names
    '''
    trsf = tubesml.DfPolynomial()
    res = trsf.fit_transform(df)
    assert trsf.get_feature_names_out()[0] == df.columns[0]
    assert trsf.get_feature_names_out()[1] == df.columns[1]
    assert trsf.get_feature_names_out()[4] == df.columns[0] + ' ' + df.columns[1]