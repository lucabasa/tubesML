import random
import string
import warnings

import numpy as np
import pandas as pd
import pytest
from lightgbm import early_stopping
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import tubesml as tml


def create_data(classification=True):
    if classification:
        df, target = make_classification(n_features=10, n_samples=100)
    else:
        df, target = make_regression(n_features=10, n_samples=100)

    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append("".join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1

    df = pd.DataFrame(df, columns=random_names)
    df["target"] = target

    return df


df_c = create_data()
df_r = create_data(classification=False)


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(solver="lbfgs"),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        XGBClassifier(),
        LGBMClassifier(),
    ],
)
def test_shap_values_classification(model):
    y = df_c["target"]
    df_1 = df_c.drop("target", axis=1)

    model.fit(df_1, y)

    shap_values = tml.get_shap_values(data=df_1, model=model)

    assert shap_values.values.shape == (100, 10)
    assert shap_values.data.shape == (100, 10)


@pytest.mark.parametrize(
    "model", [Ridge(), DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor(), LGBMRegressor()]
)
def test_shap_values_regression(model):
    y = df_r["target"]
    df_1 = df_r.drop("target", axis=1)

    model.fit(df_1, y)

    shap_values = tml.get_shap_values(data=df_1, model=model)

    assert shap_values.values.shape == (100, 10)
    assert shap_values.data.shape == (100, 10)


def test_shap_importance():
    y = df_c["target"]
    df_1 = df_c.drop("target", axis=1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(df_1, y)
    shap_values = tml.get_shap_values(data=df_1, model=model)
    shap_importance = tml.get_shap_importance(shap_values=shap_values)

    assert shap_importance.shape == (10, 3)
    assert "Feature" in shap_importance.columns
    assert "shap_importance" in shap_importance.columns
    assert "shap_std" in shap_importance.columns
