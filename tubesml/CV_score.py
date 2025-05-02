import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import clone

from tubesml.model_inspection import get_coef, get_feature_importance, get_pdp
from tubesml.base import BaseTransformer


class CrossValidate():
    def __init__(self, data,
                 target,
                 estimator,
                 cv,
                 test=None,
                 target_proc=None,
                 imp_coef=False, pdp=None, 
                 predict_proba=False, early_stopping=False, 
                 fit_params=None):
        self.train = data.copy
        self.df_test = test.copy()
        self.target = target.copy()
        self.estimator = estimator
        self.cv = cv
        self.target_proc = target_proc
        self.imp_coef = imp_coef
        self.pdp = pdp
        self.predict_proba = predict_proba
        self.early_stopping = early_stopping
        self.fit_params = fit_params
        self._initialize_loop()

    def score(self):
        for n_fold, (train_index, test_index) in enumerate(self.cv.split(self.train.values)):
            trn_data = self.train.iloc[train_index, :]
            val_data = self.train.iloc[test_index, :]

            trn_target, val_target = self._get_train_val_target(train_index, test_index)

            trn_data, val_data, test_data, model, transf_pipe = self._prepare_cv_iteration(trn_data, val_data, trn_target) 
            
            if self.early_stopping:
                # Fit the model with early stopping
                model.fit(trn_data, trn_target, eval_set=[(trn_data, trn_target),
                                                          (val_data, val_target)],
                                                          **self.fit_params)
                # store iteration used
                try:
                    self.iteration.append(model.best_iteration)
                except AttributeError:
                    self.iteration.append(model.best_iteration_)
            else:
                model.fit(trn_data, trn_target, **self.fit_params)

            if self.predict_proba:
                self.oof[test_index] = model.predict_proba(val_data)[:, 1]
                self.pred += model.predict_proba(test_data)[:, 1]
            else:
                self.oof[test_index] = model.predict(val_data).ravel()
                self.pred += model.predict(test_data).ravel()

            if self.imp_coef:
                self._fold_imp(model, trn_data, n_fold)

            if self.pdp is not None:
                self._fold_pdp(model, transf_pipe, n_fold)

        self._summarize_results()
        if self.df_test is None:
            self.pred = None
            return self.oof, self.result_dict
        else:
            self.pred /= self.cv.get_n_splits()
            return self.oof, self.pred, self.result_dict
    
    def _initialize_loop(self):
        self.oof = np.zeros(len(self.train))
        if self.df_test is not None:
            self.pred = np.zeros(len(self.df_test))
        else:
            self.pred = self.oof
        self.result_dict = {}

        self.feat_df = pd.DataFrame()
        self.iteration = []
        self.feat_pdp = pd.DataFrame()

        if self.fit_params is None:
            self.fit_params = {}

        try:  # If estimator is not a pipeline, make a pipeline
            self.estimator.steps
        except AttributeError:
            self.estimator = Pipeline([("transf", BaseTransformer()),
                                       ("model", self.estimator)])
            
    def _get_train_val_target(self, train_index, test_index):
        if self.target_proc is None:
            trn_target = pd.Series(self.target.iloc[train_index].values.ravel())
            val_target = pd.Series(self.target.iloc[test_index].values.ravel())
        else:
            trn_target, val_target = self.target_proc(self.target, train_index, test_index)
        
        return trn_target, val_target

    def _prepare_cv_iteration(self, trn_data, val_data, trn_target):
        # create model and transform pipelines
        transf_pipe = clone(Pipeline(self.estimator.steps[:-1]))
        model = clone(self.estimator.steps[-1][1])  # it creates issues with match_cols in dummy otherwise
        # Transform the data for the model
        trn_data = transf_pipe.fit_transform(trn_data, trn_target)
        val_data = transf_pipe.transform(val_data)

        if self.df_test is not None:
            test_data = transf_pipe.transform(self.test_df)
        else:
            test_data = val_data

        return trn_data, val_data, test_data, model, transf_pipe
    
    def _fold_imp(self, model, trn_data, n_fold):
        feats = trn_data.columns
        try:
            fold_df = get_coef(model, feats)
        except (AttributeError, KeyError):
            fold_df = get_feature_importance(model, feats)

        fold_df["fold"] = n_fold + 1
        self.feat_df = pd.concat([self.feat_df, fold_df], axis=0)

    def _fold_pdp(self, model, transf_pipe, n_fold):
        pdp_set = transf_pipe.transform(self.train)  # to have the same data ranges in each fold
        # The pdp will still be different by fold
        if isinstance(pdp, str):
            pdp = [pdp]
        fold_pdp = []
        for feat in pdp:
            if isinstance(feat, tuple):  # 2-way pdp is not supported as we can't take a good average
                continue
            fold_tmp = get_pdp(model, feat, pdp_set)
            fold_tmp["fold"] = n_fold + 1
            fold_pdp.append(fold_tmp)
        fold_pdp = pd.concat(fold_pdp, axis=0)
        self.feat_pdp = pd.concat([self.feat_pdp, fold_pdp], axis=0)

    def _summarize_results(self):
        if self.imp_coef:
            feat_df = self.feat_df.groupby("feat")["score"].agg(["mean", "std"])
            feat_df["abs_sco"] = abs(feat_df["mean"])
            feat_df = feat_df.sort_values(by=["abs_sco"], ascending=False)
            feat_df["std"] = feat_df["std"] / np.sqrt(self.cv.get_n_splits() - 1)  # std of the mean, unbiased
            del feat_df["abs_sco"]
            self.result_dict["feat_imp"] = feat_df

        if self.early_stopping:
            self.result_dict["iterations"] = self.iteration

        if self.pdp is not None:
            feat_pdp = self.feat_pdp.groupby(["feat", "x"])["y"].agg(["mean", "std"]).reset_index()
            feat_pdp["std"] = feat_pdp["std"] / np.sqrt(self.cv.get_n_splits() - 1)
            self.result_dict["pdp"] = feat_pdp
