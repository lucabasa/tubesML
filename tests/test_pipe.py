from .context import tubesml as tml
import pytest
import pandas as pd
import numpy as np

import string
import random

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline


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

def test_transformers():
    '''
    Test a pipeline doesn't break when all the transformers are called in succession
    '''
    pipe = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('sca', tml.DfScaler(method='standard')), 
                     ('dummify', tml.Dummify())])
    with pytest.warns(None) as record:
        res = pipe.fit_transform(df)
    assert len(record) == 0
    

