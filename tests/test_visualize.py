import random
import string
import warnings

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from tubesml.error_analysis import ErrorAnalyzer
from tubesml.visualize_error import VisualizeError


def create_data(n_samples):
    df, target = make_regression(n_features=10, n_samples=n_samples)

    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append("".join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1

    df = pd.DataFrame(df, columns=random_names)
    df["category"] = ["A", "B"] * int(n_samples // 2)
    original_cols = df.columns
    df["target"] = target

    df.loc[df["category"] == "A", "target"] = df["target"] * 1.1
    df["prediction"] = df["target"] * 1.2

    df = pd.get_dummies(df, dtype=np.uint8)

    return df, original_cols


df, original_cols = create_data(n_samples=100)


@pytest.mark.parametrize("fit", [True, False])
def test_input_validation(fit):

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        true_label="target",
        fidelity_threshold=0.5,
    )

    if fit:
        analyzer.fit()
        _ = VisualizeError(analysis=analyzer)
    else:
        with pytest.raises(AttributeError):
            _ = VisualizeError(analysis=analyzer)


@pytest.mark.parametrize("n", [-1, 1, 10])
@pytest.mark.parametrize("imp", ["shap", "standard", "both"])
@patch("matplotlib.pyplot.show")
def test_plot_feat_importance(_, n, imp):

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        true_label="target",
        fidelity_threshold=0.5,
    )
    analyzer.fit()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            viz = VisualizeError(analysis=analyzer)
            viz.plot_feature_importance(n=n, imp=imp)

    fig = plt.gcf()
    axs = fig.get_axes()
    if imp == "both":
        assert len(axs) == 2
    else:
        assert len(axs) == 1


@pytest.mark.parametrize("n", [3, 1, 10])
@patch("matplotlib.pyplot.show")
def test_plot_pdp(_, n):

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        true_label="target",
        fidelity_threshold=0.5,
    )
    analyzer.fit()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            viz = VisualizeError(analysis=analyzer)
            viz.plot_pdp(n=n)


@pytest.mark.parametrize("n", [3, 1, 10])
@patch("matplotlib.pyplot.show")
def test_plot_error_rates(_, n):

    analyzer = ErrorAnalyzer(
        data=df.copy(),
        prediction_column="prediction",
        true_label="target",
        fidelity_threshold=0.5,
    )
    analyzer.fit()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                viz = VisualizeError(analysis=analyzer)
                viz.plot_error_rates(n=n)
