import random
import string
import warnings
from unittest.mock import patch

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import tubesml as tml


def create_data(classification=False):
    if classification:
        df, target = make_classification(n_features=10, random_state=45)
    else:
        df, target = make_regression(n_features=10, random_state=45)

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
df_c = create_data(classification=True)

df_r_m = df[[c for c in df if c != "target"]].copy()
y_r = df["target"]
df_c_m = df_c[[c for c in df_c if c != "target"]].copy()
y_c = df_c["target"]

kfold = KFold(n_splits=5)

model = lgb.LGBMRegressor(verbose=-1)
cv_score = tml.CrossValidate(data=df_r_m, target=y_r, estimator=model, cv=kfold)
res_1_r, _ = cv_score.score()

model = lgb.LGBMClassifier(verbose=-1)
cv_score = tml.CrossValidate(data=df_c_m, target=y_c, estimator=model, cv=kfold)
res_1_c, _ = cv_score.score()

cv_score = tml.CrossValidate(data=df_c_m, target=y_c, estimator=model, cv=kfold, predict_proba=True)
res_1_c_p, _ = cv_score.score()

model = RandomForestRegressor(n_jobs=-1)
cv_score = tml.CrossValidate(data=df_r_m, target=y_r, estimator=model, cv=kfold)
res_2_r, _ = cv_score.score()

model = RandomForestClassifier(n_jobs=-1)
cv_score = tml.CrossValidate(data=df_c_m, target=y_c, estimator=model, cv=kfold)
res_2_c, _ = cv_score.score()

cv_score = tml.CrossValidate(data=df_c_m, target=y_c, estimator=model, cv=kfold, predict_proba=True)
res_2_c_p, _ = cv_score.score()


@patch("matplotlib.pyplot.show")
def test_regression_metrics_comparison(_):
    comp = tml.CompareModels(
        data=df_r_m,
        true_label=y_r,
        pred_1=res_1_r,
        pred_2=res_2_r,
        metric_func=mean_squared_error,
        kfold=kfold,
        regression=True,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_metrics()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_classification_metrics_comparison(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c,
        pred_2=res_2_c,
        metric_func=accuracy_score,
        kfold=kfold,
        regression=False,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_metrics()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_classification_metrics_comparison_probabilities(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c_p,
        pred_2=res_2_c_p,
        metric_func=roc_auc_score,
        kfold=kfold,
        regression=False,
        probabilities=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_metrics()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_regression_prediction_comparison(_):
    comp = tml.CompareModels(
        data=df_r_m,
        true_label=y_r,
        pred_1=res_1_r,
        pred_2=res_2_r,
        metric_func=mean_squared_error,
        kfold=kfold,
        regression=True,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_predictions()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_classification_prediction_comparison(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c,
        pred_2=res_2_c,
        metric_func=accuracy_score,
        kfold=kfold,
        regression=False,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_predictions()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_classification_prediction_comparison_probabilities(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c_p,
        pred_2=res_2_c_p,
        metric_func=roc_auc_score,
        kfold=kfold,
        regression=False,
        probabilities=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.compare_predictions()
        plt.close()


@patch("matplotlib.pyplot.show")
def test_regression_stat_significance(_):
    comp = tml.CompareModels(
        data=df_r_m,
        true_label=y_r,
        pred_1=res_1_r,
        pred_2=res_2_r,
        metric_func=mean_squared_error,
        kfold=kfold,
        regression=True,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.statistical_significance()


@patch("matplotlib.pyplot.show")
def test_regression_stat_significance_nokfold(_):
    comp = tml.CompareModels(
        data=df_r_m,
        true_label=y_r,
        pred_1=res_1_r,
        pred_2=res_2_r,
        metric_func=mean_squared_error,
        kfold=None,
        regression=True,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.statistical_significance()


@patch("matplotlib.pyplot.show")
def test_classification_stat_significance(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c,
        pred_2=res_2_c,
        metric_func=accuracy_score,
        regression=False,
        probabilities=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.statistical_significance()


@patch("matplotlib.pyplot.show")
def test_classification_stat_significance_probabilities(_):
    comp = tml.CompareModels(
        data=df_c_m,
        true_label=y_c,
        pred_1=res_1_c_p,
        pred_2=res_2_c_p,
        metric_func=roc_auc_score,
        regression=False,
        probabilities=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp.statistical_significance()
