import collections

import numpy as np
import pandas as pd
from kneed import KneeLocator
import shap
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from tubesml.base import BaseTransformer
from tubesml.model_inspection import get_coef, get_feature_importance
from tubesml.shap_values import get_shap_importance, get_shap_values


class ErrorAnalyzer(BaseTransformer):
    """
    This class trains a surrogate model to explain the error made by a generic model on tabular data.

    The user shall provide data that is ready to be used for a prediction with a sklearn model. Use the
    processing module to prepare the data if necessary. The data must contain a true label column and a
    corresponding prediction one. The user can also provide a column that flags with binary values if
    the observation is an error or not. If the error column is not provided, it will be calculated from the data.

    This class works for both regression and classification problems, but ultimately solves a classification problem
    (is this observation an error? Why is the model wrong in this case?)

    The result is a description of how the model makes wrong predictions. This comes as a summary of the most clear
    decisions the surrogate model took to predict an error and the dependency of such decisions from the provided
    features.

    :param data: pandas DataFrame with some features we think are responsible of the mistakes we want to explain, a
        true label, and a prediction. Optionally, also an error column can be in the data

    :param prediction_columns: string. Name of the column with the model predictions

    :param surrogate_model: model object, optional. If you want to use a different model to interpret the results.
        This will reduce the level of intepretability.

    :param error_column: (optional) string. Name of column signaling if the prediction was correct or not. If not
        provided, it will be calculated. The surrogate model will try to predict this column

    :param true_label: string. Name of the column with the true labels

    :param param_grid: (optional) dictionary. If provided, the surrogate model will be tuned with a GridSearch. The
        dictionary must then contain the parameters we want to try in the search for a DecisionTreeClassifier

    :param regression: boolean. Useful only if the error column is not provided. It determines if the error column
        must be determined for a regression problem or a classification one.

    :param error_class_idx: integer, default=1. Index of the classi indicating the error.

    :param fidelity_threshold: float. We trust a surrogate model that has a fidelity higher than this threshold.
        Fidelity is defined as 1 - |actual_accuracy - estimated_accuracy|

    :param probablity_threshold: float. In case we have to determine the error column for a classification problem,
        this is the threshold to consider an observation an error.

    :param n_leaves: int. Number of leaves of which we want a summary of the decisions. These are always the leaves
        with the most error predicted.

    :param random_state: int. Random state of the surrogate model. We recommend setting this to reproduce your results.
    """

    def __init__(
        self,
        data,
        prediction_column=None,
        surrogate_model=None,
        error_column=None,
        true_label=None,
        param_grid=None,
        regression=True,
        error_class_idx=1,
        fidelity_threshold=0.9,
        probability_threshold=0.5,
        n_leaves=3,
        random_state=None,
    ):

        self.data = data
        self.param_grid = param_grid
        self.regression = regression
        self.probability_threshold = probability_threshold
        self.random_state = random_state
        self.fidelity_threshold = fidelity_threshold
        self.n_leaves = n_leaves
        self.surrogate_model = surrogate_model
        self.error_class_idx = error_class_idx

        if prediction_column is None or true_label is None:
            raise KeyError(
                "Provide the column flagging the errors and the column with the predictions and the true labels"
            )

        if error_column is None:
            self._get_error_column(true_label=true_label, prediction_column=prediction_column)
            self.error_column = "is_error"
        else:
            self.error_column = error_column

        if self.error_column not in self.data.columns:
            raise KeyError("The error column is not in the data")

        self._error_tree = None
        self._error_train_x = self.data.drop(
            [self.error_column, true_label, prediction_column],
            axis=1,
        )
        self._error_train_y = self.data[self.error_column]
        self._error_train_weights = 1.0 / self.data.groupby(self.error_column)[self.error_column].transform(
            "count"
        )  # Used in shap sampling for balanced sample

    def _get_error_column(self, true_label, prediction_column):
        """
        Creates a column named `is_error` by calling a utility function that identifies the knee point
        of the Regression Error Characteristic curve.

        :param true_label: string, name of the column with the true label
        :param prediction_column: string, name of the column with the model predictions
        """
        tmp = self.data.copy()
        tmp["error"] = self.data[prediction_column] - self.data[true_label]
        if self.regression:
            self.epsilon = get_epsilon(abs(tmp["error"]))
            self.data.loc[:, "is_error"] = np.where(abs(tmp["error"]) >= self.epsilon, 1, 0)
        else:  # FIXME: works only for binary classification
            self.data.loc[:, "is_error"] = np.where(abs(tmp["error"]) >= self.probability_threshold, 1, 0)

    def fit(self, X=None, y=None):
        """
        Main method of the class that produces all the results. It prepares a surrogate model (a decision tree)
        to explain the error of the main model. Then it provides an analysis of the terminal nodes (leaves) of the
        surrogate model, feature importance, and shap values
        """
        self._set_tree()
        accepted = self._accept_tree()
        if accepted:
            self.leaves_summary = {}
            self._error_tree.fit(self._error_train_x, self._error_train_y)
            if self.surrogate_model is None:
                self._get_leaves()
                self._get_tree_structure()
                for leaf in range(min(self.n_leaves, len(self.leaf_nodes))):  # just in case the tree is too simple
                    self.leaves_summary[leaf] = self._get_leaf_summary(leaf)
            self.shap_values = get_shap_values(
                data=self._error_train_x, model=self._error_tree, check_additivity=False, sample=1000
            )
            shap_importance = get_shap_importance(shap_values=self.shap_values)
            feats = self._error_train_x.columns
            try:
                feat_imp = get_coef(self._error_tree, feats)
            except (AttributeError, KeyError):
                feat_imp = get_feature_importance(self._error_tree, feats)
            feat_df = feat_imp.groupby("Feature")["score"].agg(["mean", "std"])
            feat_df["std"] = 0
            self.feature_importance = pd.merge(shap_importance, feat_df, on="Feature")
        else:
            print(
                "The predicted model accuracy is too different from the true model accuracy."
                + "Try different data or tune the model."
            )

    def _set_tree(self):
        """
        Creates the attribute `_error_tree`, this is a DecisionTreeClassifier, either tuned via
        GridSearch, or not. It also creates the attribute `grid_result` to not create issues if the
        GridSearch is not called. To tune the tree, just provide a dictionary as `param_grid`.
        """
        if self.surrogate_model is None:
            if self.param_grid is None:
                self._error_tree = DecisionTreeClassifier(random_state=self.random_state)
                self.grid_result = pd.DataFrame()
            else:
                self._error_tree = self._tune_tree()
        else:
            self._error_tree = self.surrogate_model

    def _tune_tree(self):  # TODO: if the user provides a processing pipeline, consider
        """
        Tunes the tree and returns the best estimator. The best estimator is the one with the highest Fidelity score.
        It also creates the attribute `grid_result` with a summary of the results.
        """
        kfold = KFold(n_splits=5, random_state=self.random_state, shuffle=True)
        score = make_scorer(get_fidelity_score, greater_is_better=True)  # TODO: check that is a good metric for this
        grid = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=self.random_state),
            param_grid=self.param_grid,
            n_jobs=-1,
            cv=kfold,
            scoring=score,
        )

        grid.fit(X=self._error_train_x, y=self._error_train_y)

        result = pd.DataFrame(grid.cv_results_).sort_values(by="mean_test_score", ascending=False).reset_index()

        del result["params"]
        params = [col for col in result.columns if col.startswith("param_")]

        result = result[params + ["mean_test_score", "std_test_score"]]

        self.grid_result = result

        return grid.best_estimator_

    def _accept_tree(self):
        """
        Checks if the fidelity is higher than a threshold or rejects the tree.
        If the tree has been tuned, just use the known fidelity score. Otherwhise calculate it.
        """
        if self.param_grid:
            self.fidelity = self.grid_result["mean_test_score"][0]
        else:
            kfold = KFold(n_splits=5, random_state=self.random_state, shuffle=True)
            score = make_scorer(get_fidelity_score, greater_is_better=True)
            self.fidelity = cross_val_score(
                estimator=self._error_tree, X=self._error_train_x, y=self._error_train_y, scoring=score, cv=kfold
            ).mean()

        if self.fidelity >= self.fidelity_threshold:
            return True
        else:
            return False

    def _get_leaves(self):
        """
        Find the ids of each terminal node (leaves) and order them so that it the leaf with the highest
        error fraction comes first"""
        self.leaf_ids = np.where(self._error_tree.tree_.feature < 0)[
            0
        ]  # negative values are leaves, the node doesn"t split further
        self.error_predicted_leaves = self._error_tree.tree_.value[self.leaf_ids, 0, self.error_class_idx]
        self.total_errors = self._error_tree.tree_.value[0, 0, self.error_class_idx]
        self.total_error_fraction = self.error_predicted_leaves / self.total_errors
        self.selected_leaves = self.leaf_ids
        self.sorted_id = np.argsort(-self.total_error_fraction)
        self.leaf_nodes = self.selected_leaves.take(self.sorted_id)

    def _get_tree_structure(self):
        """
        Stores in various attributes the characteristics of the error_tree. This includes what features
        and what thresholds are used in each split"""
        self.feature_names = self._error_tree.feature_names_in_
        self.children_left = list(self._error_tree.tree_.children_left)
        self.children_right = list(self._error_tree.tree_.children_right)
        self.threshold = self._error_tree.tree_.threshold.astype("O")
        self.feature = [index if index > 0 else index for index in self._error_tree.tree_.feature]

    def _get_leaf_summary(self, idx):
        """
        Creates a dictionary with the summary of a leaf.

        :param idx: int, index of the leaf node

        :return: A dictionary with the leaf_id, the error fraction, and the path to the node
        """
        leaf_id = self.leaf_nodes[idx]
        n_errors = int(self._error_tree.tree_.value[leaf_id, 0, self.error_class_idx])
        n_samples = self._error_tree.tree_.n_node_samples[leaf_id]
        local_error = n_errors / n_samples
        total_error_fraction = n_errors / self.total_errors
        n_correct = n_samples - n_errors
        path_to_node = self._get_path_to_node(leaf_id)
        leaf_dict = {
            "id": leaf_id,
            "n_corrects": n_correct,
            "n_errors": n_errors,
            "local_error": local_error,
            "total_error_fraction": total_error_fraction,
            "path_to_node": path_to_node,
        }
        return leaf_dict

    def _get_path_to_node(self, leaf_id):
        """
        Uses the tree structure to find all the decisions that lead to a given node.
        The rules are then simplified for better readibility.
        """
        path_to_node = collections.deque()
        cur_node_id = leaf_id

        while cur_node_id > 0:
            node_is_left_child = cur_node_id in self.children_left
            if node_is_left_child:
                parent_id = self.children_left.index(cur_node_id)
            else:
                parent_id = self.children_right.index(cur_node_id)

            feat = self.feature[parent_id]
            thresh = self.threshold[parent_id]

            is_categorical = False  # TODO: sort this out
            thresh = str(thresh) if is_categorical else format_float(thresh, 2)

            decision_rule = ""
            if node_is_left_child:
                decision_rule += " <= " if not is_categorical else " is not "
            else:
                decision_rule += " > " if not is_categorical else " is "

            decision_rule = str(self.feature_names[feat]) + decision_rule + thresh
            path_to_node.appendleft(decision_rule)
            cur_node_id = parent_id

        path_to_node = self._simplify_rules(path_to_node)

        return path_to_node

    def _simplify_rules(self, path_to_node):
        """
        Many rules are repeated, like `feature_1 > 2` and `feature_1 > 4`. This function
        makes sure to give us a synthetic version of all the decision rules we had to get to a node
        In this example, it would just return `feature_1 > 4`
        """
        lefts = []
        rights = []
        for rule in path_to_node:
            if ">" in rule:
                lefts.append(rule.split(" > "))
            else:
                rights.append(rule.split(" <= "))

        max_values = {}

        for item in lefts:
            key, value = item
            value = float(value)  # Ensure the value is a float
            if key not in max_values or value > max_values[key]:
                max_values[key] = value

        max_values = [f"{key} > {value}" for key, value in max_values.items()]

        min_values = {}

        for item in rights:
            key, value = item
            value = float(value)  # Ensure the value is a float
            if key not in min_values or value < min_values[key]:
                min_values[key] = value

        min_values = [f"{key} <= {value}" for key, value in min_values.items()]

        path_to_node = max_values + min_values

        return path_to_node


def get_epsilon(difference):
    """
    Compute the threshold used to decide whether a prediction is wrong or correct (for regression tasks).

    :param difference (1D-array): The absolute differences between the true target values and the predicted ones
    (by the primary model).

    :return: epsilon (float): The value of the threshold used to decide whether the prediction for a regression task
    is wrong or correct
    """
    epsilon_range = np.linspace(min(difference), max(difference), num=50)
    cdf_error = []
    n_samples = difference.shape[0]
    for epsilon in epsilon_range:
        correct_predictions = difference <= epsilon
        cdf_error.append(np.count_nonzero(correct_predictions) / float(n_samples))
    return KneeLocator(epsilon_range, cdf_error).knee


def get_fidelity_score(y_true, y_pred):
    return 1 - abs(y_pred.mean() - y_true.mean())


def format_float(number, decimals):
    """
    Format a number to have the required number of decimals. Ensure no trailing zeros remain.

    :param number: float or integer. The number to format
    :param decimals: integer. The number of decimals required

    :return: A string with the number as a formatted string

    """
    formatted = ("{:." + str(decimals) + "f}").format(number).rstrip("0")
    if formatted.endswith("."):
        return formatted[:-1]
    return formatted
