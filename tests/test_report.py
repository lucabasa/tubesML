import tubesml as tml
import pytest
import warnings
from unittest.mock import patch 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

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


@patch("matplotlib.pyplot.show") 
def test_plot_regression_pred_nohue(_):
    '''
    Test plot_regression_predictions with different no hue
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('tree', DecisionTreeRegressor())])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tml.plot_regression_predictions(data=df_1, true_label=y, pred_label=oof)

    
@patch("matplotlib.pyplot.show") 
def test_plot_regression_pred_hue(_):
    '''
    Test plot_regression_predictions with different normal hue
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    df_1['cat'] = ['a']*len(df_1)
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('tree', DecisionTreeRegressor())])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tml.plot_regression_predictions(data=df_1, true_label=y, pred_label=oof, hue='cat')
    

@patch("matplotlib.pyplot.show") 
def test_plot_regression_pred_huemany(_):
    '''
    Test plot_regression_predictions with different hue that has to be ignored after a warning
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    df_1['many_cat'] = df_1[random.choice(df_1.columns)]
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('tree', DecisionTreeRegressor())])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold)
    
    with pytest.warns(UserWarning):
        tml.plot_regression_predictions(data=df_1, true_label=y, pred_label=oof, hue='many_cat')
        
@patch("matplotlib.pyplot.show") 
def test_plot_regression_features(_):
    '''
    Test plot_regression_predictions with extra plot against one feature
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    df_1['feature'] = df_1[random.choice(df_1.columns)]
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('tree', DecisionTreeRegressor())])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tml.plot_regression_predictions(data=df_1, true_label=y, pred_label=oof, feature='feature')
        
        
@patch("matplotlib.pyplot.show") 
def test_plot_regression_two_features(_):
    '''
    Test plot_regression_predictions with extra plot against more feature
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    df_1['feature'] = df_1[random.choice(df_1.columns)]
    df_1['feature2'] = df_1[random.choice(df_1.columns)]
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('tree', DecisionTreeRegressor())])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                tml.plot_regression_predictions(data=df_1, true_label=y, pred_label=oof, feature=['feature', 'feature2'])
        

@patch("matplotlib.pyplot.show")       
def test_plot_confusion_matrix_binary(_):
    '''
    Test plotting confusion matrix with ax=None and binary input
    '''
    pred = df['target']
    true = df['target']
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            tml.plot_confusion_matrix(true_label=true, pred_label=pred, ax=None)

        
@patch("matplotlib.pyplot.show")       
def test_plot_confusion_matrix_nonbinary(_):
    '''
    Test plotting confusion matrix with ax=None and nonbinary input
    '''
    pred = df_r['target']
    true = df['target']
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            tml.plot_confusion_matrix(true_label=true, pred_label=pred, ax=None)
    
    
@patch("matplotlib.pyplot.show")       
def test_plot_classification_probs(_):
    '''
    Test plotting the classification prediction without hue
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold, predict_proba=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            tml.plot_classification_probs(data=df_1, true_label=y, pred_label=oof)
    
    
@patch("matplotlib.pyplot.show")       
def test_plot_classification_probs_wronginput(_):
    '''
    Test plotting the classification prediction without hue, 
    try to plot against a non-existing feature and get a warning 
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold, predict_proba=True)
    
    with pytest.warns(UserWarning):
        tml.plot_classification_probs(data=df_1, true_label=y, pred_label=oof, feat='non_existing_feat')


@patch("matplotlib.pyplot.show")       
def test_plot_classification_probs_hue(_):
    '''
    Test plotting the classification prediction with hue
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    df_1['cat'] = ['a', 'b'] * int(len(df_1) / 2)
    
    full_pipe = Pipeline([('dummier', tml.Dummify()), 
                          ('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold, predict_proba=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            tml.plot_classification_probs(data=df_1, true_label=y, pred_label=oof, hue_feat='cat')
    

@patch("matplotlib.pyplot.show")
def test_plot_classification_probs_manyhue(_):
    '''
    Test plotting the classification prediction with hue
    Hue has many values, so it should raise a warning and ignore it
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    df_1['many_cat'] = df_1[random.choice(df_1.columns)]

    full_pipe = Pipeline([('scaler', tml.DfScaler()),
                          ('logit', LogisticRegression(solver='lbfgs'))])

    kfold = KFold(n_splits=3)

    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold, predict_proba=True)

    with pytest.warns(UserWarning):
        tml.plot_classification_probs(data=df_1, true_label=y, pred_label=oof, hue_feat='many_cat')

    
@patch("matplotlib.pyplot.show")       
def test_plot_classification_probs_wronghue(_):
    '''
    Test plotting the classification prediction with hue, 
    the hue colums is non-existing so get a warning and ignore it
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    full_pipe = Pipeline([('scaler', tml.DfScaler()), 
                          ('logit', LogisticRegression(solver='lbfgs'))])
    
    kfold = KFold(n_splits=3)
    
    oof, _ = tml.cv_score(df_1, y, full_pipe, kfold, predict_proba=True)
    
    with pytest.warns(UserWarning):
        tml.plot_classification_probs(data=df_1, true_label=y, pred_label=oof, hue_feat='non_existing_feat')  
    
        