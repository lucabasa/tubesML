import tubesml
import pytest
import warnings
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping

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
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                  final_estimator=LogisticRegression(), 
                                  cv=kfold)
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
            _ = stk.predict_proba(df_1)
    
    
def test_stacker_reg():
    '''
    Test the model works for regression
    '''
    y = df_r['target']
    df_1 = df_r.drop('target', axis=1)
    
    estm = [('tree1', DecisionTreeRegressor(max_depth=3)), 
            ('tree2', DecisionTreeRegressor(max_depth=5))]
    
    kfold = KFold(n_splits=3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                  final_estimator=DecisionTreeRegressor(), 
                                  cv=kfold)
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
    

@pytest.mark.parametrize("passthrough", [True, False])
def test_stacker_pipelines(passthrough):
    '''
    Test it works when pipelines are provided
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe1 = Pipeline([('scl', tubesml.DfScaler()), ('model', DecisionTreeClassifier())])
    pipe2 = Pipeline([('scl', tubesml.DfScaler()), ('model', LogisticRegression())])
    
    estm = [('model1', pipe1), ('model2', pipe2)]
    
    kfold = KFold(n_splits=3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                  final_estimator=pipe2, 
                                  cv=kfold, passthrough=passthrough)
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
            _ = stk.predict_proba(df_1)
        
        
@pytest.mark.parametrize("passthrough", [True, False])
def test_importances(passthrough):
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
                            cv=kfold, passthrough=passthrough)
    stk.fit(df_1, y)
    
    imps = stk.meta_importances_
    
    assert imps.shape == (2 + passthrough*10, 2)


@pytest.mark.parametrize("passthrough", [True, False])
def test_importances_pipeline(passthrough):
    '''
    Test it returns the feature importances with pipelines
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe1 = Pipeline([('scl', tubesml.DfScaler()), ('model', DecisionTreeClassifier())])
    pipe2 = Pipeline([('scl', tubesml.DfScaler()), ('model', LogisticRegression())])
    
    estm = [('model1', pipe1), ('model2', pipe2)]
    
    kfold = KFold(n_splits=3)
    stk = tubesml.Stacker(estimators=estm,
                          final_estimator=pipe2,
                          cv=kfold, passthrough=passthrough)
    stk.fit(df_1, y)
    
    imps = stk.meta_importances_
    
    assert imps.shape == (2 + passthrough*10, 2)
    
    
def test_hybrid_params():
    '''
    Test it runs for different settings
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                final_estimator=DecisionTreeClassifier(), 
                                cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
            _ = stk.predict_proba(df_1)
    

def test_early_stopping():
    '''
    Test early stopping works and it stops earlier
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('xgb', XGBClassifier(n_estimators=10000, early_stopping_rounds=5, eval_metric='logloss')), 
            ('lgb', LGBMClassifier(n_estimators=10000))]
    
    kfold = KFold(n_splits=3)
    
    callbacks = [early_stopping(10, verbose=0)]
    fit_params = {"callbacks":callbacks, 'eval_metric': 'accuracy'}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                final_estimator=DecisionTreeClassifier(), 
                                cv=kfold, lay1_kwargs={'xgb': {'predict_proba': True, 'early_stopping': True, 'fit_params': {'verbose': False}}, 
                                                       'lgb': {'early_stopping': True, 'fit_params': fit_params}})
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
            _ = stk.predict_proba(df_1)

    assert stk._estimators[0].n_estimators < 1000
    assert stk._estimators[1].n_estimators < 1000
    
    
def test_early_stopping_pipeline_estimators():
    '''
    Test early stopping is possible within a pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('xgb', Pipeline([('scl', tubesml.DfScaler()),
                              ('xgb', XGBClassifier(n_estimators=10000,
                                            early_stopping_rounds=5, eval_metric='logloss'))
                             ])
            ),
            ('lgb', Pipeline([('scl', tubesml.DfScaler()),
                              ('lgb', LGBMClassifier(n_estimators=10000))
                             ])
            )
           ]
    
    kfold = KFold(n_splits=3)
    
    callbacks = [early_stopping(10, verbose=0)]
    fit_params = {"callbacks":callbacks, 'eval_metric': 'accuracy'}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stk = tubesml.Stacker(estimators=estm, 
                                final_estimator=DecisionTreeClassifier(), 
                                cv=kfold, lay1_kwargs={'xgb': {'predict_proba': True, 'early_stopping': True, 'fit_params': {'verbose': False}}, 
                                                       'lgb': {'early_stopping': True, 'fit_params': fit_params}})
            stk.fit(df_1, y)
            _ = stk.predict(df_1)
            _ = stk.predict_proba(df_1)

    assert stk._estimators[0].steps[-1][1].n_estimators < 1000
    assert stk._estimators[1].steps[-1][1].n_estimators < 1000
    

@pytest.mark.parametrize("passthrough, n_feats", [(False, 2), (True, 12), ('hybrid', 5)])
def test_passthrough(passthrough, n_feats):
    '''
    Test we can have the meta model learn over more features
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    if passthrough == 'hybrid':
        passthrough = list(df_1.columns[:3])
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, passthrough=passthrough)
    stk.fit(df_1, y)
    _ = stk.predict(df_1)
    
    imps = stk.meta_importances_
    
    assert imps.shape == (n_feats, 2)
        
        
def test_high_correlation_warning():
    '''
    Test if the stacker raises a warning when the predictions are highly correlated
    We don't drop the target feature from the training set to be sure both classifiers
    predict perfectly
    '''
    y = df['target']
    df_1 = df 
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(UserWarning):
        stk = tubesml.Stacker(estimators=estm, 
                                final_estimator=DecisionTreeClassifier(), 
                                cv=kfold, verbose=True)
        stk.fit(df_1, y)


@pytest.mark.parametrize("scoring", ['accuracy', 'neg_log_loss'])
def test_gridsearch_stacker_simple(scoring):
    '''
    Test tml.grid_search works on this
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
    
    param_grid = {'final_estimator__max_depth': [3,4,5]}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result, best_param, best_estimator = tubesml.grid_search(data=df_1, target=y, estimator=stk, 
                                                             param_grid=param_grid, scoring=scoring, cv=3)
        
    
    
@pytest.mark.parametrize("scoring", ['accuracy', 'neg_log_loss'])   
def test_gridsearch_stacker_pipeline(scoring):
    '''
    Test tml.grid_search works on this when in a pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
    
    pipe = Pipeline([('scl', tubesml.DfScaler()), ('model', stk)])
    
    param_grid = {'model__final_estimator__max_depth': [3,4,5]}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result, best_param, best_estimator = tubesml.grid_search(data=df_1, target=y, estimator=pipe, 
                                                             param_grid=param_grid, scoring=scoring, cv=3)
        

@pytest.mark.parametrize("passthrough", [True, False, 'hybrid'])   
def test_gridsearch_stacker_passthrough(passthrough):
    '''
    Test tml.grid_search works on this when in a pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    if passthrough == 'hybrid':
        passthrough = list(df_1.columns[:3])
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}}, passthrough=passthrough)
    
    pipe = Pipeline([('scl', tubesml.DfScaler()), ('model', stk)])
    
    param_grid = {'model__final_estimator__max_depth': [3,4,5]}
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            result, best_param, best_estimator = tubesml.grid_search(data=df_1, target=y, estimator=pipe, 
                                                             param_grid=param_grid, scoring='neg_log_loss', cv=3)


    
@pytest.mark.parametrize("predict_proba", [True, False])
def test_cv_score_stacker_simple(predict_proba):
    '''
    Test tml.cv_score works on this
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=DecisionTreeClassifier(), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res, _ = tubesml.cv_score(df_1, y, stk, cv=kfold, predict_proba=predict_proba)
    
    
@pytest.mark.parametrize("predict_proba, imp_coef", [(True, False), (False, False), (True, True)])
def test_cv_score_stacker_pipeline(predict_proba, imp_coef):
    '''
    Test tml.cv_score works on this when in a pipeline and if it returns the feature importance
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=Pipeline([('scl', tubesml.DfScaler()), 
                                                      ('logit', LogisticRegression())]), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}})
    
    pipe = Pipeline([('scl', tubesml.DfScaler()), ('model', stk)])
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res, _ = tubesml.cv_score(df_1, y, stk, cv=kfold, predict_proba=predict_proba, imp_coef=imp_coef)
    
    
@pytest.mark.parametrize("passthrough", [True, False, 'hybrid'])
def test_cv_score_stacker_passthrough(passthrough):
    '''
    Test tml.cv_score works on this when in a pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    if passthrough == 'hybrid':
        passthrough = list(df_1.columns[:3])
    
    estm = [('tree', DecisionTreeClassifier(max_depth=3)), 
            ('logit', LogisticRegression())]
    
    kfold = KFold(n_splits=3)
    
    stk = tubesml.Stacker(estimators=estm, 
                            final_estimator=Pipeline([('scl', tubesml.DfScaler()), 
                                                      ('logit', LogisticRegression())]), 
                            cv=kfold, lay1_kwargs={'logit': {'predict_proba': True}}, passthrough=passthrough)
    
    pipe = Pipeline([('scl', tubesml.DfScaler()), ('model', stk)])
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            res, _ = tubesml.cv_score(df_1, y, stk, cv=kfold, predict_proba=True)
