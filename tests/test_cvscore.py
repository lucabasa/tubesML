import random
import string
import warnings

import numpy as np
import pandas as pd
import pytest
from lightgbm import early_stopping
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier

import tubesml as tml


def create_data(classification=True):
    if classification:
        df, target = make_classification(n_features=10)
    else:
        df, target = make_regression(n_features=10)

    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append("".join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1

    df = pd.DataFrame(df, columns=random_names)
    df["target"] = target

    return df


df = create_data()


@pytest.mark.parametrize("predict_proba", [True, False])
def test_cvscore(predict_proba):
    """
    Test it works without warnings with both the normal prediction and the predict_proba
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    pipe_transf = Pipeline(
        [
            ("fs", tml.DtypeSel(dtype="numeric")),
            ("imp", tml.DfImputer(strategy="mean")),
            ("poly", tml.DfPolynomial()),
            ("sca", tml.DfScaler(method="standard")),
            ("tarenc", tml.TargetEncoder()),
            ("dummify", tml.Dummify()),
            ("pca", tml.DfPCA(n_components=0.9)),
        ]
    )
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    full_pipe = Pipeline([("pipe", pipe), ("logit", LogisticRegression(solver="lbfgs"))])

    kfold = KFold(n_splits=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():  # FIXME: clean before release
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1, target=y, estimator=full_pipe, cv=kfold, predict_proba=predict_proba
            )
            res, _ = cv_score.score()
    assert len(res) == len(df_1)


def test_cvscore_nopipe():
    """
    Test if the function works without a pipeline
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    kfold = KFold(n_splits=3)

    full_pipe = LogisticRegression(solver="lbfgs")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(data=df_1, target=y, estimator=full_pipe, cv=kfold, predict_proba=True)
            res, _ = cv_score.score()
    assert len(res) == len(df_1)


@pytest.mark.parametrize(
    "model", [LogisticRegression(solver="lbfgs"), DecisionTreeClassifier(), XGBClassifier(), LGBMClassifier()]
)
def test_cvscore_coef_imp(model):
    """
    Test coefficient and feature importances for a few models
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    pipe_transf = Pipeline(
        [
            ("fs", tml.DtypeSel(dtype="numeric")),
            ("imp", tml.DfImputer(strategy="mean")),
            ("poly", tml.DfPolynomial()),
            ("sca", tml.DfScaler(method="standard")),
            ("tarenc", tml.TargetEncoder()),
            ("dummify", tml.Dummify()),
            ("pca", tml.DfPCA(n_components=0.9, compress=True)),
        ]
    )
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    full_pipe = Pipeline([("pipe", pipe), ("model", model)])

    kfold = KFold(n_splits=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(data=df_1, target=y, estimator=full_pipe, cv=kfold, imp_coef=True)
            res, coef = cv_score.score()
    assert len(coef["feat_imp"]) == df_1.shape[1] * 2 + 45  # to account for the combinations


@pytest.mark.parametrize("model", [LogisticRegression(solver="lbfgs"), XGBClassifier(), LGBMClassifier()])
def test_cvscore_nopipeline(model):
    """
    Test cv score works for simple models, without being it a pipeline
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    kfold = KFold(n_splits=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(data=df_1, target=y, estimator=model, cv=kfold, imp_coef=True)
            res, coef = cv_score.score()
    assert len(res) == len(df_1)
    assert len(coef["feat_imp"]) == df_1.shape[1]


@pytest.mark.parametrize(
    "model", [LogisticRegression(solver="lbfgs"), DecisionTreeClassifier(), XGBClassifier(), LGBMClassifier(verbose=-1)]
)
def test_cvscore_pdp(model):
    """
    Test partial dependence of a few models
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    kfold = KFold(n_splits=3)

    pdp = df_1.columns[:3].to_list()

    pipe_transf = Pipeline([("fs", tml.DtypeSel(dtype="numeric")), ("imp", tml.DfImputer(strategy="mean"))])
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    full_pipe = Pipeline([("pipe", pipe), ("model", model)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(data=df_1, target=y, estimator=full_pipe, cv=kfold, pdp=pdp)
            res, pdp_res = cv_score.score()
    assert set(pdp_res["pdp"]["feat"]) == set(pdp)
    assert pdp_res["pdp"]["mean"].notna().all()
    assert pdp_res["pdp"]["std"].notna().all()


def test_fit_params():
    """
    Test that the user can provide a fit_params input
    This test is specific for Xgboost and lightgbm or any other estimator
    that allows parameters for the fit method.

    The test is not parametrized as the devs of xgboost and lightgbm can't
    agree on how to pass parameters to a function.
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    kfold = KFold(n_splits=3)

    # XGBoost
    model = XGBClassifier(early_stopping_rounds=5)
    fit_params = {"verbose": False}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1, target=y, estimator=model, cv=kfold, early_stopping=True, fit_params=fit_params
            )
            res, res_dict = cv_score.score()

    assert len(res) == len(df_1)
    assert len(res_dict["iterations"]) == 3  # one per fold

    # LightGBM
    model = LGBMClassifier()
    callbacks = [early_stopping(10, verbose=0)]
    fit_params = {"callbacks": callbacks}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1, target=y, estimator=model, cv=kfold, early_stopping=True, fit_params=fit_params
            )
            res, res_dict = cv_score.score()

    assert len(res) == len(df_1)
    assert len(res_dict["iterations"]) == 3  # one per fold


def test_fit_params_pipeline():
    """
    Test that the user can provide a fit_params input when we use a pipeline
    This test is specific for Xgboost and lightgbm or any other estimator
    that allows parameters for the fit method.

    The test is not parametrized as the devs of xgboost and lightgbm can't
    agree on how to pass parameters to a function.
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    kfold = KFold(n_splits=3)

    pipe_transf = Pipeline(
        [
            ("fs", tml.DtypeSel(dtype="numeric")),
            ("imp", tml.DfImputer(strategy="mean")),
            ("poly", tml.DfPolynomial()),
            ("sca", tml.DfScaler(method="standard")),
            ("tarenc", tml.TargetEncoder()),
            ("dummify", tml.Dummify()),
            ("pca", tml.DfPCA(n_components=0.9, compress=True)),
        ]
    )
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    # XGBoost
    model = XGBClassifier(early_stopping_rounds=5)
    full_pipe = Pipeline([("pipe", pipe), ("model", model)])
    fit_params = {"verbose": False}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1, target=y, estimator=full_pipe, cv=kfold, early_stopping=True, fit_params=fit_params
            )
            res, res_dict = cv_score.score()

    assert len(res) == len(df_1)
    assert len(res_dict["iterations"]) == 3  # one per fold

    # LightGBM
    model = LGBMClassifier()
    callbacks = [early_stopping(10, verbose=0)]
    full_pipe = Pipeline([("pipe", pipe), ("model", model)])
    fit_params = {"callbacks": callbacks}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1, target=y, estimator=full_pipe, cv=kfold, early_stopping=True, fit_params=fit_params
            )
            res, res_dict = cv_score.score()

    assert len(res) == len(df_1)
    assert len(res_dict["iterations"]) == 3  # one per fold


@pytest.mark.parametrize(
    "predict_proba, n_folds, regression", [(False, 3, False), (False, 4, False), (True, 3, False), (False, 3, True)]
)
def test_cvpredict_probaclass(predict_proba, n_folds, regression):
    """
    Test it works with a classification where we predict the probabilities
    """
    y = df["target"]
    df_1 = df.drop("target", axis=1)
    df_test = df_1.copy()

    pipe_transf = Pipeline(
        [
            ("fs", tml.DtypeSel(dtype="numeric")),
            ("imp", tml.DfImputer(strategy="mean")),
            ("poly", tml.DfPolynomial()),
            ("sca", tml.DfScaler(method="standard")),
            ("tarenc", tml.TargetEncoder()),
            ("dummify", tml.Dummify()),
            ("pca", tml.DfPCA(n_components=0.9)),
        ]
    )
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    full_pipe = Pipeline([("pipe", pipe), ("logit", LogisticRegression(solver="lbfgs"))])

    kfold = KFold(n_splits=n_folds)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(
                data=df_1,
                target=y,
                test=df_test,
                estimator=full_pipe,
                cv=kfold,
                predict_proba=predict_proba,
                regression=regression,
            )
            _, pred, _ = cv_score.score()
    assert len(pred) == len(df_1)
    assert pred.min() >= 0
    assert pred.max() <= 1
    if not (predict_proba or regression):
        assert len(np.unique(pred)) <= 2

@pytest.mark.parametrize("feat_imp", [True, False])
def test_shap_values(feat_imp):
    y = df["target"]
    df_1 = df.drop("target", axis=1)

    pipe_transf = Pipeline(
        [
            ("fs", tml.DtypeSel(dtype="numeric")),
            ("imp", tml.DfImputer(strategy="mean")),
            ("poly", tml.DfPolynomial()),
            ("sca", tml.DfScaler(method="standard")),
            ("tarenc", tml.TargetEncoder()),
            ("dummify", tml.Dummify()),
            ("pca", tml.DfPCA(n_components=15)),
        ]
    )
    pipe = tml.FeatureUnionDf([("transf", pipe_transf)])

    full_pipe = Pipeline([("pipe", pipe), ("logit", LogisticRegression(solver="lbfgs"))])

    kfold = KFold(n_splits=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            cv_score = tml.CrossValidate(data=df_1, target=y, estimator=full_pipe, cv=kfold, shap=True, shap_sample=10, imp_coef=feat_imp)
            _, res = cv_score.score()
    assert len(cv_score.shap_values.values) == 10 * 3
    assert len(cv_score.shap_values.data) == 10 * 3
    assert len(cv_score.shap_values.base_values) == 10 * 3
    assert "shap_importance" in res["feat_imp"]
    assert "shap_std" in res["feat_imp"]
    assert "feat" in res["feat_imp"]
    assert len(res["feat_imp"]) == 15
    assert len(res["shap_values"].values) == 10 * 3
    assert len(res["shap_values"].data) == 10 * 3
    assert len(res["shap_values"].feature_names) == 15
