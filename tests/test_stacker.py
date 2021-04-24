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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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


def test_stacker_cls():
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
        _ = stk.predict_proba(df_1)
    assert len(record) == 0
    
    
def test_stacker_reg():
    '''
    Test the model works for regression
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    
    estm = [('tree1', DecisionTreeRegressor(max_depth=3)), 
            ('tree2', DecisionTreeRegressor(max_depth=5))]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        stk = tubesml.Stacker(estimators=estm, 
                              final_estimator=DecisionTreeRegressor(), 
                              cv=kfold)
        stk.fit(df_1, y)
        _ = stk.predict(df_1)
    assert len(record) == 0
    

def test_stacker_pipelines():
    '''
    Test it works when pipelines are provided
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe1 = Pipeline([('scl', tubesml.DfScaler()), ('model', DecisionTreeClassifier())])
    pipe2 = Pipeline([('scl', tubesml.DfScaler()), ('model', LogisticRegression())])
    
    estm = [('model1', pipe1), ('model2', pipe2)]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        stk = tubesml.Stacker(estimators=estm, 
                              final_estimator=pipe2, 
                              cv=kfold)
        stk.fit(df_1, y)
        _ = stk.predict(df_1)
        _ = stk.predict_proba(df_1)
    assert len(record) == 0


def test_importances():
    '''
    Test it returns the feature importances
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold)
    stk.fit(df_1, y)
    
    imps = stk.meta_importances_
    
    assert imps.shape == (2, 2)
    
    
def test_hybrid_params():
    '''
    Test it runs for different settings
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
        stk.fit(df_1, y)
        _ = stk.predict(df_1)
        _ = stk.predict_proba(df_1)
        
    assert len(record) == 0
    

def test_early_stopping():
    '''
    Test early stopping works and it stops earlier
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('xgb', XGBClassifier(n_estimators=10000, use_label_encoder=False)), 
            ('lgb', LGBMClassifier(n_estimators=10000))]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'xgb': {'predict_proba': True, 
                                                           'early_stopping': 100, 
                                                           'eval_metric': 'logloss'}, 
                                                   'lgb': {'early_stopping': 100, 
                                                           'eval_metric': 'accuracy'}})
        stk.fit(df_1, y)
        _ = stk.predict(df_1)
        _ = stk.predict_proba(df_1)
        
    assert len(record) == 0
    assert stk._estimators[0].n_estimators < 10000
    assert stk._estimators[1].n_estimators < 10000
    


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
