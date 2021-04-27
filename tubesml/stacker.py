__author__ = 'lucabasa'
__version__ = '0.0.2'
__status__ = 'development'


from tubesml.base import BaseTransformer, self_columns, reset_columns
from tubesml.report import get_coef, get_feature_importance
from tubesml.model_selection import cv_score

import pandas as pd
import numpy as np
from sklearn.base import clone


class Stacker(BaseTransformer):
    def __init__(self, estimators, final_estimator, cv, lay1_kwargs=None, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self._lay1_kwargs_input(lay1_kwargs)
        self.meta_importances_ = None
        self.passthrough = passthrough
        
        
    def _lay1_kwargs_input(self, lay1_kwargs):
        if lay1_kwargs is None:
            self.lay1_kwargs = {}
        else:
            self.lay1_kwargs = lay1_kwargs
            
        for est in self.estimators:
            if est[0] not in self.lay1_kwargs.keys():
                self.lay1_kwargs[est[0]] = {}
            for key in ['predict_proba', 'early_stopping']:
                try:
                    _ = self.lay1_kwargs[est[0]][key]
                except KeyError:
                    self.lay1_kwargs[est[0]][key] = False
                
                
    def return_feature_importances(self):
        try:
            return get_coef(self.final_estimator, self.est_names)
        except (AttributeError, KeyError):
            return get_feature_importance(self.final_estimator, self.est_names)
        
        
    def fit(self, X, y):
        self._estimators = [clone(model[1]) for model in self.estimators]
        self.est_names = [f'preds_{model[0]}' for model in self.estimators]
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self._estimators):
            oof, res = cv_score(data=X, target=y, estimator=est, cv=self.cv, 
                               **self.lay1_kwargs[self.estimators[i][0]])
            out_of_fold_predictions[:, i] = oof
            
            if self.lay1_kwargs[self.estimators[i][0]]['early_stopping']:
                self._estimators[i].set_params(**{'n_estimators': np.mean(res['iterations']).astype(int)})
                
            self._estimators[i].fit(X, y)
        
        final_train = pd.DataFrame(out_of_fold_predictions, columns=self.est_names)
        self.final_estimator.fit(final_train, y)
        
        try:  # this is useful to well behave with other sklearn methods
            self.classes_ = self.final_estimator.classes_
        except AttributeError:  # if the final_estimator does not have classes, we don't care
            pass
        
        self.meta_importances_ = self.return_feature_importances()
        
        return self
    
    
    def _make_predict_test(self, X):
        
        first_layer_predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self._estimators):
            if self.lay1_kwargs[self.estimators[i][0]]['predict_proba']:
                first_layer_predictions[:, i] = self._estimators[i].predict_proba(X)[:,1]
            else:
                first_layer_predictions[:, i] = self._estimators[i].predict(X)
        
        return pd.DataFrame(first_layer_predictions, columns=self.est_names)
    
    
    def predict(self, X, y=None):
        
        final_predict = self._make_predict_test(X)
        preds = self.final_estimator.predict(final_predict)
        return preds
    
    
    def predict_proba(self, X, y=None):
        
        final_predict = self._make_predict_test(X)
        preds = self.final_estimator.predict_proba(final_predict)
        return preds