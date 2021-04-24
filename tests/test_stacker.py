import tubesml
import pytest
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import string
import random


def create_data(classification=True):
    if classification:
        df, target = make_classification(n_features=10)
    else:
        df, target = make_regression(n_features=10)
    
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
df_r = create_data(classification=False)


def test_stacker():
    '''
    Test the model works for classification
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree1', DecisionTreeClassifier(max_depth=3)), 
                  ('tree2', DecisionTreeClassifier(max_depth=5))]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        stk = tubesml.Stacker(estimators=estm, 
                              final_estimator=LogisticRegression(), 
                              cv=kfold)
        stk.fit(df_1, y)
        _ = stk.predict(df_1)
    assert len(record) == 0
    

def test_stacker_pipelines():
    '''
    Test it works when pipelines are provided
    '''
    pass


def test_importances():
    '''
    test it returns the feature importances
    '''
    pass


def test_early_stopping():
    '''
    Test early stopping works and it stops earlier
    '''
    pass


def test_hybrid_params():
    '''
    Test it runs for different settings
    '''
    pass


def test_passthrough():
    '''
    Test we can have the meta model learn over more features
    '''
    pass


def test_gridsearch_stacker():
    '''
    Test tml.grid_search works on this 
    '''
    pass


def test_cv_score_stacker():
    '''
    Test tml.cv_score works on this
    '''
    pass
