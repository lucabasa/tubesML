import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


def grid_search(data, target, estimator, param_grid, scoring, cv, random=False):
    """
    Calls a grid or a randomized search over a parameter grid

    :param data: pandas DataFrame.
           Data to tune the hyperparameters

    :param target: numpy array or pandas Series.
            Target column

    :param estimator: sklearn compatible estimator.
            It must have a ``predict`` method and a ``get_params`` method.
            It can be a Pipeline.

    :param param_grid: dict.
            Dictionary of the parameter space to explore.
            In case the ``estimator`` is a pipeline, provide the keys in the format ``step__param``.

    :param scoring: string.
            Scoring metric for the grid search, see the sklearn documentation for the available options.

    :param cv: KFold object or int.
            For cross-validation.

    :param random: bool, default=False.
            If True, runs a RandomSearch instead of a GridSearch.

    :return: a dataframe with the results for each configuration
    :return: a dictionary with the best parameters
    :return: the best (fitted) estimator
    """

    if random:
        grid = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            cv=cv,
            scoring=scoring,
            n_iter=random,
            n_jobs=-1,
            random_state=434,
            return_train_score=True,
            error_score="raise",
        )
    else:
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            error_score="raise",
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )

    pd.options.mode.chained_assignment = None  # turn on and off a warning of pandas
    tmp = data.copy()
    grid = grid.fit(tmp, target)
    pd.options.mode.chained_assignment = "warn"

    result = pd.DataFrame(grid.cv_results_).sort_values(by="mean_test_score", ascending=False).reset_index()

    del result["params"]
    times = [col for col in result.columns if col.endswith("_time")]
    params = [col for col in result.columns if col.startswith("param_")]

    result = result[params + ["mean_train_score", "std_train_score", "mean_test_score", "std_test_score"] + times]

    return result, grid.best_params_, grid.best_estimator_


def make_test(train, test_size, random_state, strat_feat=None):
    """
    Creates a train and test, stratified on a feature or on a list of features.

    :param train: pandas DataFrame.

    :param test_size: float.
                        The size of the test set. It must be between 0 and 1.

    :param random_state: int.
                        Random state used to split the data.

    :param strat_feat: str or list, default=None.
                        The feature or features to use to stratify the split.

    :return: A train set and a test set.
    """
    if strat_feat:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in split.split(train, train[strat_feat]):
            train_set = train.iloc[train_index, :]
            test_set = train.iloc[test_index, :]

    else:
        train_set, test_set = train_test_split(train, test_size=test_size, random_state=random_state)

    return train_set, test_set
