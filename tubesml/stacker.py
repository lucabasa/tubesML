__author__ = 'lucabasa'
__version__ = '0.2.0'
__status__ = 'development'


from tubesml.base import BaseTransformer
from tubesml.model_inspection import get_coef, get_feature_importance
from tubesml.model_selection import cv_score

import pandas as pd
import numpy as np
from sklearn.base import clone

import warnings


class Stacker(BaseTransformer):
    '''
    Wrapper for stacking several estimators with a meta estimator. Each estimator creates out of fold predictions
    on the entire dataset via a KFold and it is then re-fitted against the full dataset. The predictions are generated by
    ``tubesml.cv_score`` and thus can account for early stopping and other options.
    
    The meta estimator trains on the predicitions made by the estimators. It is possible to train the meta estimator on
    other features of the initial dataset.
    
    
    :param estimators: list of tuples (name, estimator) (including pipelines).
                        These estimators will generate the first layer of predictions.
                
    :param final_estimator: estimator with a fit and a predict (or predict_proba) methods.
                            This estimator will create the final prediction.
                
    :param cv: KFold (or similar) generator.
                The CV scheme to use to generate the first layer of predictions.
        
    :param lay1_kwargs: (optional) dict,
            Dictionary of settings for the first layer predictions to be passed to ``tubesml.cv_score``.
            The keys of the dictionary refer to the names in the list of ``estimators``.
            
    :param passthrough: bool or list, default=False.
                        If True, all the features used to train the first layer of estimators will be used by the ``final_estimator``.
                        If list, only the listed features will be used.
                        If the input is incorrect, it will be set to False
            
    :param verbose: bool, default=False.
                    If True, it warns the user if the correlation of the first layer of predictions is higher than 0.9.

    :Attributes:
    
        `meta_importances_` : pandas DataFrame with the feature importances (or the coefficients) of the final estimator. 
                                Note: this estimator doesn't have a ``coef_`` or ``feature_importances_`` attribute, 
                                it won't thus work with methods that leverage these attritbutes.

        `corr_` :  pandas DataFrame.
                    Returns the correlation between the first set of estimator's predictions.
    '''
    def __init__(self, estimators, final_estimator, cv, lay1_kwargs=None, passthrough=False, verbose=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self._lay1_kwargs_input(lay1_kwargs)
        self.meta_importances_ = None
        self.coef_ = None  # this is to work well with other sklearn methods
        self.feature_importances_ = None  # this is to work well with other sklearn methods
        self.columns = None
        self.is_stacker = True
        self.passthrough = passthrough
        self.verbose = verbose
        
        if not type(self.passthrough) in [list, bool]:
            warnings.warn(f'passthrough must be a list or a boolean, we got {type(self.passthrough)} it will be ignored', UserWarning)
            self.passthrough = False
        
        
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
                    
                    
    def _get_passthrough_features(self, X, final_train):

        if self.passthrough and type(self.passthrough) is bool:
            final_train = pd.concat([final_train, X.reset_index(drop=True)], axis=1)
        elif self.passthrough and type(self.passthrough) is list:
            final_train = pd.concat([final_train, X[self.passthrough].reset_index(drop=True)], axis=1)

        return final_train
                
                
    def return_feature_importances(self, X):
        feats = X.columns
        try:
            try:
                self.feature_importances_ = self.final_estimator.steps[-1][1].coef_
                self.coef_ = self.feature_importances_
            except AttributeError:
                self.feature_importances_ = self.final_estimator.coef_
                self.coef_ = self.feature_importances_
            return get_coef(self.final_estimator, feats)
        except (AttributeError, KeyError):
            try:
                self.feature_importances_ = self.final_estimator.steps[-1][1].feature_importances_
            except AttributeError:
                self.feature_importances_ = self.final_estimator.feature_importances_
            return get_feature_importance(self.final_estimator, feats)
        
        
    def _check_correlated_predictions(self, final_train):
        if ((abs(final_train.corr()) > 0.8).sum() > 1).any():
            warnings.warn('The predictions are highly correlated, this is not ideal. Check the ``corr_`` attribute for details', UserWarning)
        
        
    def fit(self, X, y):
        """
        This method uses ``tubesml.cv_score`` to create out of fold predictions from each of the estimators provided in ``estimators``.
        
        Secondly, it fits the `final_estimator` on a dataset that contains these predictions and any feature specified by ``passthrough``.
        Each of the ``estimators`` is then refit on the entire dataset
        
        If an estimator is in the first layer of estimators was trained with early stopping, in the refit it will be trained on a number of
        iterations equal to the mean number across the folds used to generate the first layer of predictions. Be sure that the early stopping
        attribute of the estimator is ``early_stopping_round``.
        
        If ``verbose=True`` the user will be warned if any of the predictions are correlated more than 0.9. The ``corr_`` attribute is created
        by this method
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
        
        """
        self._estimators = [clone(model[1]) for model in self.estimators]
        self.est_names = [f'preds_{model[0]}' for model in self.estimators]
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self._estimators):
            oof, res = cv_score(data=X, target=y, estimator=est, cv=self.cv, 
                               **self.lay1_kwargs[self.estimators[i][0]])
            out_of_fold_predictions[:, i] = oof
            
            if self.lay1_kwargs[self.estimators[i][0]]['early_stopping']:
                try:
                    self._estimators[i].set_params(**{'n_estimators': np.mean(res['iterations']).astype(int), 
                                                      'early_stopping_rounds': None})
                except ValueError:  # if the estimator is a pipeline, setting n_estimators requires a workaround
                    self._estimators[i].steps[-1][1].set_params(**{'n_estimators': np.mean(res['iterations']).astype(int), 
                                                      'early_stopping_rounds': None})
                
            self._estimators[i].fit(X, y)
        
        final_train = pd.DataFrame(out_of_fold_predictions, columns=self.est_names)
        self.corr_ = final_train.corr()
        if self.verbose:
            self._check_correlated_predictions(final_train)
        final_train = self._get_passthrough_features(X, final_train)
        self.final_estimator.fit(final_train, y)
        
        try:  # this is useful to well behave with other sklearn methods
            self.classes_ = self.final_estimator.classes_
        except AttributeError:  # if the final_estimator does not have classes, we don't care
            pass
        
        self.meta_importances_ = self.return_feature_importances(final_train)
        self.columns = final_train.columns
        
        return self
    
    
    def _make_predict_test(self, X):
        
        first_layer_predictions = np.zeros((X.shape[0], len(self.estimators)))
        for i, est in enumerate(self._estimators):
            if self.lay1_kwargs[self.estimators[i][0]]['predict_proba']:
                first_layer_predictions[:, i] = self._estimators[i].predict_proba(X)[:,1]
            else:
                first_layer_predictions[:, i] = self._estimators[i].predict(X)
        
        return self._get_passthrough_features(X, pd.DataFrame(first_layer_predictions, columns=self.est_names))
    
    
    def predict(self, X, y=None):
        """
        Method to generate the final predictions. First, it generates the meta dataset with the predictions by the 
        ``estimators``. 
        
        If any of them was generated by a ``predict_proba`` method, it will be done again, otherwise it 
        uses the ``predict`` method of those ``estimators``.
        
        The final prediction is generated by using the ``predict`` method of the ``final_estimator``.
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return preds: ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        
        final_predict = self._make_predict_test(X)
        preds = self.final_estimator.predict(final_predict)
        return preds
    
    
    def predict_proba(self, X, y=None):
        """
        Method to generate the final predictions. First, it generates the meta dataset with the predictions by the 
        ``estimators``. 
        
        If any of them was generated by a ``predict_proba`` method, it will be done again, otherwise it 
        uses the ``predict`` method of those ``estimators``.
        
        The final prediction is generated by using the ``predict_proba`` method of the ``final_estimator``.
        
        :param X: pandas DataFrame of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), Not used
            The target values (class labels) as integers or strings.
            
        :return preds: ndarray of shape (n_samples, n_classes) or \
            list of ndarray of shape (n_output,)
            The class probabilities of the input samples.
        """
        final_predict = self._make_predict_test(X)
        preds = self.final_estimator.predict_proba(final_predict)
        return preds
