import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from tubesml.base import BaseTransformer
from tubesml.model_inspection import get_coef
from tubesml.model_inspection import get_feature_importance
from tubesml.model_inspection import get_pdp
from tubesml.shap_values import get_shap_importance
from tubesml.shap_values import get_shap_values


class CrossValidate:
    """
    Train and test a pipeline in kfold cross validation

    :param data: pandas DataFrame.
           Data to tune the hyperparameters.

    :param target: numpy array or pandas Series.
            Target column.

    :param estimator: sklearn compatible estimator.
            It must have a ``predict`` method and a ``get_params`` method.
            It can be a Pipeline. If it is not a Pipeline, it will be made one for
            compatibility with other functionalities.

    :param cv: KFold object.
            For cross-validation, the estimates will be done across these folds.

    :param test: pandas DataFrame, default=None
            Data to predict on within each fold. If provided, each model trained in each fold predicts
            on this set. The predictions are then averaged across the folds. If it is a classification
            problem and we are not predicting the probabilities, the most frequent class is used. If
            there is no majority class (it can happen with an even number of folds), the class is chosen
            at random.

    :param target_proc: function, default=None.
            It must take target as input (in this context it can be one or more arrays or series) and 2
            indices for train and validation. It must return 2 series with the train and validation
            targets

    :param imp_coef: bool, default=False.
            If True, returns the feature importance or the coefficient values averaged across the folds,
            with standard deviation on the mean.

    :param pdp: string or list, default=None.
            If not None, returns the partial dependence of the given features averaged across the folds,
            with standard deviation on the mean.
            The partial dependence of 2 features simultaneously is not supported.

    :param shap: bool, default=False.
            If True, it calculates the shape values for a sample of the data in each fold. In that case
            the results will also have the shap values (concatenated) and the feature importance will have
            the one coming from the shap values.
            WARNING: if you can't guarantee the same number of features in each fold, the shap calculation
            will break.

    :param class_pos: bool, default=1.
            Position of the class of interest, relevant if using ``predict_proba`` and for some shap values
            explainers. If None, all the classes probabilities will be returned but it will conflict with
            some shap values explainers.

    :param shap_sample: int, default=700.
            Number of samples to calculate the shap values in each fold.

    :param predict_proba: bool, default=False.
            If True, calls the ``predict_proba`` method instead of the ``predict`` one.

    :param early_stopping: bool, default=False.
                        If True, uses early stopping within the folds for the estimators that support it.

    :param fit_params: dict, default=None.
                        If a dictionary is provided, it will pass it to the fit method.
                        This is useful to control the verbosity of the fit method as some packages
                        like XGBoost and LightGBM do not do that in the estimator declaration.

    :param regression: bool, default=True.
                        If True, the predictions on the test set will be averaged across folds. Set it to
                        false if the problem is a binary classification problem and you are not using
                        ``predict_proba``.

    :return oof: numpy array with the out of fold predictions for the entire train set.

    :return res_dict: A dictionary with additional results. If ``imp_coef=True``,
                    it contains a pd.DataFrame with the coefficients or
                    feature importances of the estimator, it can be found under the key ``feat_imp``.
                    If ``early_stopping=True``, it contains a list with the best iteration number per fold,
                    it can be found under the key ``iterations``. If ``pdp`` is not ``None``, it contains a
                    pd.DataFrame with the partial dependence of the given features, it can be found under
                    the key ``pdp``. If ``shap`` is true, it contains the shap values under the key ``shap_values``,
                    moreover, the feature importance will also have the average shap values.

    :return pred: (optional) numpy array with the prediction done on the test set (if provided).

    """

    def __init__(
        self,
        data,
        target,
        estimator,
        cv,
        test=None,
        target_proc=None,
        imp_coef=False,
        pdp=None,
        shap=False,
        class_pos=1,
        shap_sample=700,
        predict_proba=False,
        early_stopping=False,
        fit_params=None,
        regression=True,
    ):
        self.train = data.copy()
        if test is None:
            self.df_test = None
        else:
            self.df_test = test.copy()
        self.target = target.copy()
        self.estimator = estimator
        self.cv = cv
        self.target_proc = target_proc
        self.imp_coef = imp_coef
        self.pdp = pdp
        self.shap = shap
        self.class_pos = class_pos
        self.shap_sample = shap_sample
        self.predict_proba = predict_proba
        self.early_stopping = early_stopping
        self.fit_params = fit_params
        self.regression = regression
        self._initialize_loop()

    def score(self):
        """
        Main method to loop over the folds, train and predict. It produces out of fold predictions
        and, if provided, an average prediction on the test set. It can also produce various insights
        on the model, like feature importance and pdp's.
        """
        for n_fold, (train_index, test_index) in enumerate(self.cv.split(self.train.values)):
            trn_data = self.train.iloc[train_index, :].reset_index(drop=True)
            val_data = self.train.iloc[test_index, :].reset_index(drop=True)

            trn_target, val_target = self._get_train_val_target(train_index, test_index)

            trn_data, val_data, test_data, model, transf_pipe = self._prepare_cv_iteration(
                trn_data, val_data, trn_target
            )

            if self.early_stopping:
                # Fit the model with early stopping
                model.fit(
                    trn_data, trn_target, eval_set=[(trn_data, trn_target), (val_data, val_target)], **self.fit_params
                )
                # store iteration used
                try:
                    self.iteration.append(model.best_iteration)
                except AttributeError:
                    self.iteration.append(model.best_iteration_)
            else:
                model.fit(trn_data, trn_target, **self.fit_params)

            if self.predict_proba:
                if self.class_pos is None:
                    self.oof[test_index] = model.predict_proba(val_data)[:, :]
                else:
                    self.oof[test_index] = model.predict_proba(val_data)[:, self.class_pos]
                if self.df_test is not None:
                    if self.class_pos is None:
                        self.pred += model.predict_proba(test_data)[:, :]
                    else:
                        self.pred += model.predict_proba(test_data)[:, self.class_pos]
            else:
                self.oof[test_index] = model.predict(val_data).ravel()
                if self.df_test is not None:
                    self.pred += model.predict(test_data).ravel()

            if self.imp_coef:
                self._fold_imp(model, trn_data, n_fold)

            if self.pdp is not None:
                self._fold_pdp(model, transf_pipe, n_fold)

            if self.shap:
                self._fold_shap(model, trn_data)

        self._summarize_results()

        if self.df_test is None:
            self.pred = None
            return self.oof, self.result_dict
        else:
            self._postprocess_prediction()
            return self.oof, self.pred, self.result_dict

    def _initialize_loop(self):
        """
        Prepares everything needed to loop over the folds.
        The estimator must be a pipeline, so we make it one if it isn't
        """
        if self.class_pos is None:
            self.oof = np.zeros((len(self.train), self.target.nunique()))
        else:
            self.oof = np.zeros(len(self.train))
        if self.df_test is not None:
            if self.class_pos is None:
                self.pred = np.zeros((len(self.df_test), self.target.nunique()))
            else:
                self.pred = np.zeros(len(self.df_test))
        else:
            self.pred = self.oof
        self.result_dict = {}

        self.feat_df = pd.DataFrame()
        self.iteration = []
        self.feat_pdp = pd.DataFrame()
        self.shap_values = np.ndarray(shape=(0, 1))

        if self.fit_params is None:
            self.fit_params = {}

        try:  # If estimator is not a pipeline, make a pipeline
            self.estimator.steps
        except AttributeError:
            self.estimator = Pipeline([("transf", BaseTransformer()), ("model", self.estimator)])

    def _get_train_val_target(self, train_index, test_index):
        """
        Prepare the target for training and validation within the Fold
        """
        if self.target_proc is None:
            trn_target = pd.Series(self.target.iloc[train_index].values.ravel())
            val_target = pd.Series(self.target.iloc[test_index].values.ravel())
        else:
            trn_target, val_target = self.target_proc(self.target, train_index, test_index)

        return trn_target, val_target

    def _prepare_cv_iteration(self, trn_data, val_data, trn_target):
        """
        In each fold, make sure the data is processed without leaks.
        It separates the processing from the model in the pipeline.
        """
        # create model and transform pipelines
        transf_pipe = clone(Pipeline(self.estimator.steps[:-1]))
        model = clone(self.estimator.steps[-1][1])  # it creates issues with match_cols in dummy otherwise
        # Transform the data for the model
        trn_data = transf_pipe.fit_transform(trn_data, trn_target)
        val_data = transf_pipe.transform(val_data)

        if self.df_test is not None:
            test_data = transf_pipe.transform(self.df_test)
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
        if isinstance(self.pdp, str):
            self.pdp = [self.pdp]
        fold_pdp = []
        for feat in self.pdp:
            if isinstance(feat, tuple):  # 2-way pdp is not supported as we can't take a good average
                continue
            fold_tmp = get_pdp(model, feat, pdp_set)
            fold_tmp["fold"] = n_fold + 1
            fold_pdp.append(fold_tmp)
        fold_pdp = pd.concat(fold_pdp, axis=0)
        self.feat_pdp = pd.concat([self.feat_pdp, fold_pdp], axis=0)

    def _fold_shap(self, model, trn_data):
        shap_values = get_shap_values(trn_data, model, sample=self.shap_sample, class_pos=self.class_pos)
        if len(self.shap_values) == 0:
            self.shap_values = shap_values
        else:
            self.shap_values.values = np.append(self.shap_values.values, shap_values.values, axis=0)
            self.shap_values.data = np.append(self.shap_values.data, shap_values.data, axis=0)
            self.shap_values.base_values = np.append(self.shap_values.base_values, shap_values.base_values, axis=0)

    def _summarize_results(self):
        if self.imp_coef:
            feat_df = self.feat_df.groupby("Feature")["score"].agg(["mean", "std"])
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

        if self.shap:
            feat_df = get_shap_importance(shap_values=self.shap_values)
            if self.imp_coef:
                tmp = pd.merge(feat_df, self.result_dict["feat_imp"], on="Feature")
                self.result_dict["feat_imp"] = tmp
            else:
                self.result_dict["feat_imp"] = feat_df

            self.result_dict["shap_values"] = self.shap_values

    def _postprocess_prediction(self):
        """
        Averages the predictions on the test set across the folds.
        If it is a classification problem and we were not predicting the probabilities, the class
        most often predicted is used. Ties are solved with a random choice.
        """
        self.pred /= self.cv.get_n_splits()
        if not (self.regression or self.predict_proba):
            thr = 1 / (self.cv.get_n_splits() / 2)  # FIXME: this works only with binary classification
            self.pred = np.array([int(i + np.random.choice([thr / 10, -thr / 10]) >= thr) for i in self.pred])
