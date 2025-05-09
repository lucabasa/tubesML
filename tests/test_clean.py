import numpy as np
import pandas as pd
import pytest

import tubesml


def create_data():
    df = pd.DataFrame({"a": [1, np.nan, 5], "b": [3, 2, 1]})
    return df


df = create_data()


@pytest.mark.parametrize("imputer_type", ["simple", "knn"])
def test_clean(imputer_type):
    """
    Test the imputer actually works
    """
    df_1 = pd.DataFrame({"a": [1, np.nan, 5] * 5, "b": [3, 2, 1] * 5})
    imputer = tubesml.DfImputer(imputer_type=imputer_type, n_neighbors=2)
    res = imputer.fit_transform(df_1)
    assert res.isna().any().any() == 0


def test_add_indicator():
    """
    Test if the indicator is created appropriately
    """
    imputer = tubesml.DfImputer(strategy="mean", add_indicator=True)
    res = imputer.fit_transform(df.copy())
    assert "missing_a" in res.columns


def test_add_indicator_nomissing():
    """
    Test the indicator is not used when not necessary
    """
    imputer = tubesml.DfImputer(strategy="mean", add_indicator=True)
    res = imputer.fit_transform(df[["b"]].copy())
    assert len(res.columns) == 1


def test_add_indicator_newmissing():
    """
    Test it is not adding new columns when new missing values are found
    """
    df_1 = pd.DataFrame({"a": [3, 2, 1], "b": [1, np.nan, 5]})
    imputer = tubesml.DfImputer(strategy="mean", add_indicator=True)
    imputer.fit(df.copy())
    res = imputer.transform(df_1)
    assert "missing_a" in res.columns
    assert res["missing_a"].sum() == 0
    assert "missing_b" not in res.columns


def test_add_indicator_newmissing_inverseorder():
    """
    Test it is not adding new columns when new missing values are found
    and the column order is not the same. This test is necessary in case we try to
    use sklearn MissingIndicator, which is somewhat sensitive towards column order
    """
    df_1 = pd.DataFrame({"b": [1, np.nan, 5], "a": [3, 2, 1]})
    imputer = tubesml.DfImputer(strategy="mean", add_indicator=True)
    imputer.fit(df.copy())
    res = imputer.transform(df_1)
    assert "missing_a" in res.columns
    assert res["missing_a"].sum() == 0
    assert "missing_b" not in res.columns


def test_cl_stats():
    """
    Test the imputer learns the mean
    """
    imputer = tubesml.DfImputer(strategy="mean")
    imputer.fit(df.copy())
    real_mean = df.mean()
    pd.testing.assert_series_equal(imputer.statistics_, real_mean, check_dtype=False)


def test_cl_cols():
    """
    Test the attribute columns is well defined
    """
    imputer = tubesml.DfImputer(strategy="mean")
    res = imputer.fit_transform(df.copy())
    assert imputer.columns[0] == df.columns[0]
    assert imputer.columns[1] == df.columns[1]


def test_cl_stat_median():
    """
    Test the imputer learns the median
    """
    imputer = tubesml.DfImputer(strategy="median")
    imputer.fit(df.copy())
    real_median = df.median()
    pd.testing.assert_series_equal(imputer.statistics_, real_median, check_dtype=False)


def test_cl_stat_mostfreq():
    """
    Test the imputer learns the most frequent
    """
    imputer = tubesml.DfImputer(strategy="most_frequent")
    imputer.fit(df.copy())
    # todo: this test is not very robust
    pd.testing.assert_series_equal(imputer.statistics_, pd.Series([1, 1], index=df.columns), check_dtype=False)


def test_cl_stat_constant():
    """
    Test the imputer learns the most frequent
    """
    imputer = tubesml.DfImputer(strategy="constant", fill_value=5)
    imputer.fit(df.copy())
    pd.testing.assert_series_equal(
        imputer.statistics_, pd.Series([5] * df.shape[1], index=df.columns), check_dtype=False
    )


def test_imputer_dtype():
    """
    Test the data type is preserved after imputation
    """
    imputer = tubesml.DfImputer(strategy="mean", add_indicator=True)
    res = imputer.fit_transform(df.copy())
    assert res["b"].dtype == "int64"
    assert res["a"].dtype == "float64"


def test_cl_error():
    """
    Test the imputer raises the right error
    """
    with pytest.raises(ValueError):
        imputer = tubesml.DfImputer(strategy="not the mean")


def test_get_feature_names():
    """
    Test the transformer still has get_feature_names
    """
    trsf = tubesml.DfImputer()
    res = trsf.fit_transform(df.copy())
    assert trsf.get_feature_names_out()[0] == df.columns[0]
    assert trsf.get_feature_names_out()[1] == df.columns[1]
