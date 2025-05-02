import random
import string
import warnings

import pandas as pd

from sklearn.datasets import make_classification

import tubesml as tml


def create_data():
    df, target = make_classification(n_features=10)

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


def test_make_test():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train, test = tml.make_test(df, 0.2, 452)


def test_strat_test():
    df_1 = df.copy()
    df["cat"] = ["a"] * 50 + ["b"] * 50
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            train, test = tml.make_test(df, 0.2, 452, strat_feat="cat")
    assert len(train[train["cat"] == "a"]) == len(train) / 2
