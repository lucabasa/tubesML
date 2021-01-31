__author__ = 'lucabasa'
__version__ = '0.1.0'
__status__ = 'development'

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import clone

from tubesml.report import get_coef, get_feature_importance


def grid_search(data, target, estimator, param_grid, scoring, cv, random=False):
    '''
    Calls a grid or a randomized search over a parameter grid
    Returns a dataframe with the results for each configuration
    Returns a dictionary with the best parameters
    Returns the best (fitted) estimator
    '''
    
    if random:
        grid = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=cv, scoring=scoring, 
                                  n_iter=random, n_jobs=-1, random_state=434, return_train_score=True, error_score='raise')
    else:
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, error_score='raise',
                            cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    pd.options.mode.chained_assignment = None  # turn on and off a warning of pandas
    tmp = data.copy()
    grid = grid.fit(tmp, target)
    pd.options.mode.chained_assignment = 'warn'
    
    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', 
                                                        ascending=False).reset_index()
    
    del result['params']
    times = [col for col in result.columns if col.endswith('_time')]
    params = [col for col in result.columns if col.startswith('param_')]
    
    result = result[params + ['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score'] + times]
    
    return result, grid.best_params_, grid.best_estimator_


def cv_score(data, target, estimator, cv, imp_coef=False, predict_proba=False):
    '''
    Train and test a pipeline in kfold cross validation
    Returns the oof predictions for the entire train set and a dataframe with the
    coefficients or feature importances, averaged across the folds, with standard deviation

    todo: report on the average score and variance too
    '''
    oof = np.zeros(len(data))
    train = data.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(cv.split(train.values)):
            
        trn_data = train.iloc[train_index, :]
        val_data = train.iloc[test_index, :]
        
        trn_target = target.iloc[train_index].values.ravel()
        val_target = target.iloc[test_index].values.ravel()
        
        model = clone(estimator)  # it creates issues with match_cols in dummy otherwise
        
        model.fit(trn_data, trn_target)
        
        if predict_proba:
            oof[test_index] = model.predict_proba(val_data)[:,1]
        else:
            oof[test_index] = model.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = get_coef(model)
            except (AttributeError, KeyError):
                fold_df = get_feature_importance(model)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof