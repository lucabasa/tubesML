import tubesml as tml
import pytest
from unittest.mock import patch 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

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


@pytest.mark.parametrize('model', [DecisionTreeClassifier(), 
                                   XGBClassifier(use_label_encoder=False), 
                                   LGBMClassifier()])    
def test_feat_imp(model):
    '''
    Test if we can get the feature importance for various models
    XGB - Version 1.3.3, not in package requirements
    LGB - Version 3.1.1, not in package requirements
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('model', model)])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)

    with pytest.warns(None) as record:
        coef = tml.get_feature_importance(full_pipe)
    assert len(record) == 0
    assert len(coef) == df_1.shape[1]


@patch("matplotlib.pyplot.show")
def test_learning_curves(_):
    '''
    Test learning curves can be plotted
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])
    
    kfold = KFold(n_splits=3)
    with pytest.warns(None) as record:
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=None, cv=kfold,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None)
    assert len(record) == 0
    
    
# @patch("matplotlib.pyplot.show")  # todo: somewhat slow
# def test_learning_curves_xgb(mock_show):
#     '''
#     Test learning curves can be plotted with xbgboost
#     '''
#     y = df['target']
#     df_1 = df.drop('target', axis=1)
    
#     full_pipe = Pipeline([('scaler', tml.DfScaler()), 
#                           ('xgb', XGBClassifier(objective='binary:logistic', 
#                                                 use_label_encoder=False))])
    
#     kfold = KFold(n_splits=3)
#     with pytest.warns(None) as record:
#         tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=None, cv=kfold,
#                             n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None)
#     assert len(record) == 0   
    
    
@patch("matplotlib.pyplot.show")
def test_learning_curves_lgb(_):
    '''
    Test learning curves can be plotted with xbgboost
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('lgb', LGBMClassifier())])
    
    kfold = KFold(n_splits=3)
    with pytest.warns(None) as record:
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=None, cv=kfold,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None)
    assert len(record) == 0


@patch("matplotlib.pyplot.show")
def test_plot_feat_imp(_):
    '''
    Test if plot feat importance works
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('lgb', LGBMClassifier())])
    
    kfold = KFold(n_splits=3)
    full_pipe.fit(df_1, y)
    oof, coef = tml.cv_score(df_1, y, full_pipe, kfold, imp_coef=True, predict_proba=False)

    with pytest.warns(None) as record:
        tml.plot_feat_imp(coef)
    assert len(record) == 0
