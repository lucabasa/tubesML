import random
import string

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_regression

from tubesml.error_analysis import ErrorAnalyzer

def create_data():
    df, target = make_regression(n_features=10)

    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append("".join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1

    df = pd.DataFrame(df, columns=random_names)
    df["target"] = target
    df["prediction"] = df["target"] * np.random.choice([1, 1.5], size=100)
    df["is_error"] = np.where(df["target"] != df["prediction"], 1, 0)

    return df


df = create_data()


def test_analyzer_regression():

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        error_column="is_error",
        true_label="target",
        fidelity_threshold=0.8,
    )

    assert "target" not in analyzer._error_train_x.columns
    assert "is_error" not in analyzer._error_train_x.columns

    analyzer.fit()

    assert len(analyzer.leaves_summary.keys()) == 3
    for key in analyzer.leaves_summary.keys():
        assert analyzer.leaves_summary[key].keys() == {
            "id",
            "n_corrects",
            "n_errors",
            "local_error",
            "total_error_fraction",
            "path_to_node",
        }

    assert len(analyzer.feature_importance) == 10

    assert len(analyzer.shap_values) == min(len(df), 1000)  # one value per sample
    assert len(analyzer.shap_values[0]) == 10  # for each sample, one value per feature


def test_analyzer_classification():
    df_c = df.copy()
    df_c["target"] = np.where(df_c["target"] > df["target"].mean(), 1, 0)
    df_c["prediction"] = np.random.choice([0, 1], size=len(df))
    del df_c["is_error"]

    analyzer = ErrorAnalyzer(
        data=df_c.copy(),
        prediction_column="prediction",
        true_label="target",
        fidelity_threshold=0.8,
        regression=False,
    )

    assert "target" not in analyzer._error_train_x.columns
    assert "is_error" not in analyzer._error_train_x.columns

    analyzer.fit()

    assert len(analyzer.leaves_summary.keys()) == 3
    for key in analyzer.leaves_summary.keys():
        assert analyzer.leaves_summary[key].keys() == {
            "id",
            "n_corrects",
            "n_errors",
            "local_error",
            "total_error_fraction",
            "path_to_node",
        }

    assert len(analyzer.feature_importance) == 10

    assert len(analyzer.shap_values) == min(len(df), 1000)  # one value per sample
    assert len(analyzer.shap_values[0]) == 10 # for each sample, one value per feature


def test_analyzer_regression_error_column():
    df_1 = df.copy()
    df_1 = df_1.drop("is_error", axis=1)

    analyzer = ErrorAnalyzer(
        data=df_1.copy(), prediction_column="prediction", true_label="target"
    )

    assert analyzer.error_column == "is_error"
    assert "is_error" in analyzer.data.columns
    assert len(analyzer._error_train_y) == len(df)
    assert hasattr(analyzer, "epsilon")


def test_analyzer_keyerror_no_pred():

    with pytest.raises(KeyError):
        _ = ErrorAnalyzer(data=df.copy(), prediction_column=None, error_column=None, true_label=None)


def test_analyzer_keyerror_wrong_col():

    with pytest.raises(KeyError):
        _ = ErrorAnalyzer(
            data=df.copy(), prediction_column="prediction", error_column="some wrong name", true_label="true_label"
        )


def test_tune_tree():

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        error_column="is_error",
        true_label="target",
        fidelity_threshold=0.8,
        param_grid={"max_depth": [3, None]},
    )

    analyzer.fit()

    assert len(analyzer.grid_result) == 2  # searching over 2 values of max_depth
    assert len(analyzer.grid_result.columns) == 3  # 1 hyperparam and 2 result columns
    assert "param_max_depth" in analyzer.grid_result.columns
    assert "mean_test_score" in analyzer.grid_result.columns
    assert "std_test_score" in analyzer.grid_result.columns
