import tubesml
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from sklearn.datasets import make_classification
import string
import random

@pytest.mark.parametrize("verbose", [True, False])  
def test_list_missing(verbose):
    df = pd.DataFrame({'a': [1, np.nan, 5], 
                       'b': [3, 2, 1] })
    mis = tubesml.list_missing(df, verbose=verbose)
    assert len(mis) == 1
    assert mis[0] == 'a'


def test_no_missing():
    df = pd.DataFrame({'a': [1, 1, 5], 
                       'b': [3, 2, 1] })
    mis = tubesml.list_missing(df, verbose=False)
    assert len(mis) == 0

    
def create_data():
    df, target = make_classification(n_features=10)
    
    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append(''.join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1
        
    df = pd.DataFrame(df, columns=random_names)
    df['target'] = target
    
    return df

df = create_data()


@patch("matplotlib.pyplot.show")
def test_plot_correlation(_):
    corrs = tubesml.plot_correlations(df, target='target')
    assert len(corrs) == 11

    
@patch("matplotlib.pyplot.show")
def test_plot_correlation_notarget(_):
    tubesml.plot_correlations(df, target='target')
    

@pytest.mark.parametrize("estimator", [None, np.mean])    
@patch("matplotlib.pyplot.show")
def test_corr_target(_, estimator):
    tubesml.corr_target(df.sample(40), 'target', [col for col in df if 'target' not in col][:3], x_estimator=estimator)
