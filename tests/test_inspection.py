import tubesml as tml
import pytest
import warnings
from unittest.mock import patch 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

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


def test_get_coef():
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    full_pipe.fit(df_1, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coef = tml.get_coef(full_pipe)
    assert len(coef) == df_1.shape[1]


@pytest.mark.parametrize('model', [DecisionTreeClassifier(), 
                                   XGBClassifier(n_estimators=10), 
                                   LGBMClassifier(n_estimators=10)])    
def test_feat_imp(model):
    '''
    Test if we can get the feature importance for various models
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('model', model)])
    
    full_pipe.fit(df_1, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coef = tml.get_feature_importance(full_pipe)
    assert len(coef) == df_1.shape[1]


@patch("matplotlib.pyplot.show")
def test_learning_curves(_):
    '''
    Test learning curves can be plotted
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    kfold = KFold(n_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=(0, 1), cv=kfold,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title='Title')
    
    
@patch("matplotlib.pyplot.show")  
def test_learning_curves_xgb(_):
    '''
    Test learning curves can be plotted with xbgboost
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('xgb', XGBClassifier(objective='binary:logistic', 
                                                n_estimators=2, n_jobs=-1))])
    
    kfold = KFold(n_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=(0, 1), cv=kfold,
                                n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None) 
        
        
@patch("matplotlib.pyplot.show")  
def test_learning_curves_xgb_error(_):
    '''
    Test learning curves raises an error when does not produce a result
    This happens because the sklearn method is not necessarily compatible with the
    runtime parameters of the model
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('xgb', XGBClassifier(objective='binary:logistic', 
                                                n_estimators=200, early_stopping_rounds=3, 
                                                eval_metric='accuracy', n_jobs=-1))])
    
    kfold = KFold(n_splits=3)
    with pytest.raises((RuntimeError, ValueError)):
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=(0, 1), cv=kfold,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None) 
    
    
@patch("matplotlib.pyplot.show")
def test_learning_curves_lgb(_):
    '''
    Test learning curves can be plotted with xbgboost
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('lgb', LGBMClassifier(n_estimators=10))])
    
    kfold = KFold(n_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tml.plot_learning_curve(estimator=full_pipe, X=df_1, y=y, scoring='accuracy', ylim=None, cv=kfold,
                                n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10), title=None)


@patch("matplotlib.pyplot.show")
def test_plot_feat_imp(_):
    '''
    Test if plot feat importance works
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('lgb', LGBMClassifier(n_estimators=10))])
    
    kfold = KFold(n_splits=3)
    oof, coef = tml.cv_score(df_1, y, full_pipe, kfold, imp_coef=True, predict_proba=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tml.plot_feat_imp(coef['feat_imp'])
    

def test_plot_feat_imp_warning():
    '''
    Test if plot feat throws the right error
    '''
    wrong_input = pd.DataFrame({'a': [1,2,3], 
                                'b': [2,3,4]})

    with pytest.raises(KeyError):
        tml.plot_feat_imp(wrong_input, n=10)


@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs'), 
                                   DecisionTreeRegressor(),
                                   XGBClassifier(n_estimators=10), 
                                   LGBMClassifier(n_estimators=10)])
def test_get_pdp(model):
    """
    Test basic functioning for various models
    """
    feat = df.columns[0]
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('model', model)])
    
    full_pipe.fit(df_1, y)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pdp = tml.get_pdp(full_pipe, feat, df_1)
    assert {'feat', 'x', 'x_1', 'y'} == set(pdp.columns)
    assert pdp.shape == (100, 4)
    assert pdp['x_1'].isna().all()
    

@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs'), 
                                   DecisionTreeRegressor(),
                                   XGBClassifier(n_estimators=10), 
                                   LGBMClassifier(n_estimators=10)])   
def test_get_pdp_cats(model):
    """
    Test basic functioning for various models
    XGB - Version 1.3.3, not in package requirements
    LGB - Version 3.1.1, not in package requirements
    """
    feat = 'cat'
    y = df['target']
    df_1 = df.drop('target', axis=1)
    df_1['cat'] = [1, 2] * 50
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('model', model)])
    
    full_pipe.fit(df_1, y)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pdp = tml.get_pdp(full_pipe, feat, df_1)
    assert {'feat', 'x', 'x_1', 'y'} == set(pdp.columns)
    assert pdp.shape == (2, 4)
    assert pdp['x_1'].isna().all()
    

@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs'), 
                                   DecisionTreeRegressor(),
                                   XGBClassifier(n_estimators=10), 
                                   LGBMClassifier(n_estimators=10)])    
def test_get_pdp_interaction(model):
    """
    Test if passing a tuple of features is possible
    """
    feat = (df.columns[0], df.columns[1])
    
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('model', model)])
    
    full_pipe.fit(df_1, y)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pdp = tml.get_pdp(full_pipe, feat, df_1)
    assert {'feat', 'x', 'x_1', 'y'} == set(pdp.columns)
    assert pdp.shape == (2500, 4)
    assert pdp['x_1'].notna().all()
    assert pdp['x'].nunique() == 50
    assert pdp['x_1'].nunique() == 50
    
    
def test_get_pdp_list():
    """
    Test the function raises a TypeError if a list is in the input
    """
    full_pipe = 1
    df_1 = 2
    feat = ['a', 'b']
    with pytest.raises(TypeError):
        pdp = tml.get_pdp(full_pipe, feat, df_1)


@patch("matplotlib.pyplot.show")      
def test_plot_pdp(_):
    """
    Test if plotting the partial dependence works
    """
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('imputer', tml.DfImputer()), 
                          ('lgb', LGBMClassifier(n_estimators=10))])
    pdp = df_1.columns[:3].to_list()
    
    kfold = KFold(n_splits=3)
    oof, res = tml.cv_score(df_1, y, full_pipe, kfold, pdp=pdp)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tml.plot_partial_dependence(res['pdp'])
    

@patch("matplotlib.pyplot.show")      
def test_plot_pdp_singleplot_y(_):
    """
    Test if plotting the partial dependence works
    """
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('imputer', tml.DfImputer()), 
                          ('lgb', LGBMClassifier(n_estimators=10))])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    full_pipe.fit(df_1, y)
    pdp = tml.get_pdp(full_pipe, df_1.columns[0], df_1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = tml.plot_pdp(pdp, df_1.columns[0], '', ax)
    
    
@patch("matplotlib.pyplot.show")      
def test_plot_pdp_singleplot_mean(_):
    """
    Test if plotting the partial dependence works
    """
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('imputer', tml.DfImputer()), 
                          ('lgb', LGBMClassifier(n_estimators=10))])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    
    kfold = KFold(n_splits=3)
    oof, res = tml.cv_score(df_1, y, full_pipe, kfold, pdp=df_1.columns[0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = tml.plot_pdp(res['pdp'], df_1.columns[0], '', ax)
    