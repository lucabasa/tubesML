import tubesml as tml
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import string
import random


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


def test_get_coef():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)

    with pytest.warns(None) as record:
        coef = tml.get_coef(full_pipe)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]
    
    
def test_feat_imp():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('tree', DecisionTreeClassifier())])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)

    with pytest.warns(None) as record:
        coef = tml.get_feature_importance(full_pipe)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]

    
def test_feat_imp_xgb():
    '''
    Test if the method works for XGB
    (Version 1.3.3, not in package requirements)
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('xgb', XGBClassifier(objective='binary:logistic', 
                                                use_label_encoder=False))])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)

    with pytest.warns(None) as record:
        coef = tml.get_feature_importance(full_pipe)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]
    
    
def test_feat_imp_xgb():
    '''
    Test if the method works for LGBM
    (Version 3.1.1, not in package requirements)
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('lgb', LGBMClassifier())])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)

    with pytest.warns(None) as record:
        coef = tml.get_feature_importance(full_pipe)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]
    