__author__ = 'lucabasa'
__version__ = '0.0.1'
__status__ = 'development'

import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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
