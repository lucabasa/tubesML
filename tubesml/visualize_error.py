from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tubesml.model_inspection import plot_feat_imp, plot_shap_values


class VisualizeError:
    """
    Methods to visualize the error analysis done by the `ErrorAnalyzer` class.
    Using the feature importance of the analysis, it shows the error rates and partial
    dependency plots for the most important features to determine the error of the model
    """

    def __init__(self, analysis, original_feature_columns=None, show=False):
        self.analysis = analysis
        self.original_columns = None
        self.feature_idxs = None
        self.show = show
        self.grouped_dummified_features = defaultdict(list)
        self.dummified_to_original = {}
        self._validate_input()
        self._set_original_columns(original_feature_columns)
        self._prepare_dummies()

    def _set_original_columns(self, original_feature_columns):
        if original_feature_columns is not None:
            self.original_columns = original_feature_columns
        else:
            self.original_columns = self.analysis._error_train_x.columns

    def _validate_input(self):
        """Checks that the analysis object contains all the attributes we need"""
        attributes_needed = [
            "_error_train_x",
            "_error_train_y",
            "leaves_summary",  # TODO: no method using this for now, but we will find something
            "shap_values",
            "feature_importance",
        ]
        if not all(hasattr(self.analysis, attr) for attr in attributes_needed):
            raise AttributeError("The analysis object provided does not have all the necessary attributes")

    def _group_categorical_columns(self):
        """
        Groups dummified features by their original categorical feature.

        This method creates mappings between dummified features and their original
        categorical features. It populates two dictionaries:
        1. `self.grouped_dummified_features`: A defaultdict where each key is the
        original feature name and the value is a list of corresponding dummified
        feature names.
        2. `self.dummified_to_original`: A dictionary where each key is a dummified
        feature name and the value is its corresponding original feature name.

        If a dummified feature cannot be traced back to an original feature, it raises a ValueError.
        """
        feature_columns = self.analysis._error_train_x.columns
        original_columns_set = set(self.original_columns)
        for dummified_feature_name in feature_columns:
            if dummified_feature_name not in original_columns_set:
                # Find original categorical column name -> original name is substring of dummified name
                # If multiple original column names are substring of dummified,
                # pick most exhaustive/longest column name pattern match
                feature_name_prefix = max(
                    (
                        original_column_name
                        for original_column_name in self.original_columns
                        if original_column_name in dummified_feature_name
                    ),
                    key=len,
                    default="",
                )
                if not feature_name_prefix:
                    raise ValueError(
                        f"Dummified feature {dummified_feature_name} cannot be traced to original column. \
                                     Rename dummified feature to include original column name."
                    )
                self.grouped_dummified_features[feature_name_prefix].append(dummified_feature_name)
                self.dummified_to_original[dummified_feature_name] = feature_name_prefix

    def _sort_dummified_feature_importance(self, dummified_feature_group):
        """
        Sorts the dummified features of a categorical variable by feature importance.
        Returns sorted feature importance df for the group of dummifies features of the underlying categorical column.
        """
        feat_importance_df = self.analysis.feature_importance
        feature_importance = feat_importance_df[feat_importance_df["Feature"].isin(dummified_feature_group)]
        sorted_features = feature_importance.sort_values(by="shap_importance", ascending=False)
        return sorted_features

    def _prepare_dummies(self):
        """
        This method is meant to make the original features back from the dummies.
        This is necessary to plot_feature_importance.
        It first finds the dummified category of a categorical feature with the max importance.
        Indexes of the selected max importance dummy categories and numerical features
        are stored in self.feature_idxs.
        self.feature_idxs can be used in the plotting methods to filter the features to be displayed.
        """
        self._group_categorical_columns()
        discarded_idxs = []
        feat_importance_df = self.analysis.feature_importance
        self.sorted_feature_idxs = defaultdict(list)
        for _feature, dummified_features in self.grouped_dummified_features.items():
            sorted_features = self._sort_dummified_feature_importance(dummified_features)
            self.grouped_dummified_features[_feature] = sorted_features["Feature"].to_list()

            # Per dummified feature group, keep index of feature with max importance
            max_importance_idx = sorted_features.iloc[0].name
            all_feature_group_idxs = sorted_features.index
            unwanted_dummy_col_idxs = all_feature_group_idxs[all_feature_group_idxs != max_importance_idx]
            discarded_idxs.extend(unwanted_dummy_col_idxs)
        self.feature_idxs = ~feat_importance_df.index.isin(discarded_idxs)

    def plot_feature_importance(self, n=-1, imp="shap"):
        """
        Wrapper around ``tubesml.model_inspection.plot_feat_imp``

        :param n: integer, how many features you want to display in the plot.
        :param imp: string. It can be either shap, standard, or bool
        """
        plot_feat_imp(data=self.analysis.feature_importance.loc[self.feature_idxs], n=n, imp=imp)

    def plot_pdp(self, features=None, n=4):
        """
        Wrapper around ``tubesml.model_inspection.plot_shap_values``

        :param features: list of features to display, optional.
        :param n: int, if no feature is provided, it determines how many of the most important features
            we display.
        """
        data = pd.DataFrame(self.analysis.shap_values.data, columns=self.analysis.shap_values.feature_names)
        features = self._get_features(data=data, n=n, features=features)
        plot_shap_values(self.analysis.shap_values, features=features)

    def plot_error_rates(self, features=None, n=4, bins=20):
        """
        Plots histograms of the error rates. That is the % of observations that are flagged
        as error vs a set of features. If the features are not provided, the most important one will
        be in the plot.
        The plot adatps its size based on how many features we display

        :param features: list of features to display
        :param n: int, if no feature is provided, it determines how many of the most important features
            we display.
        :param bins: integer, how many bins to use in the histogram.
        """
        data = self.analysis._error_train_x.copy()
        data["is_error"] = self.analysis._error_train_y
        mean_error = data["is_error"].mean()

        features = self._get_features(data, n=n, features=features)
        n_feats = len(features)
        n_rows = int(n_feats / 2) + (n_feats % 2 > 0)

        _, ax = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))

        i = 0
        j = 0

        for feature in features:
            if feature in self.dummified_to_original:
                # For cat features, plot all dummified feats per histogram
                original_feature_name = self.dummified_to_original[feature]
                sorted_dummified_features = self.grouped_dummified_features[original_feature_name]
                feature = f"Dummified {original_feature_name}"
                # Per dummified feature histogram bar, plot % of error/non-error where dummified feature == 1
                dummified_data_melted = data[sorted_dummified_features + ["is_error"]].melt(
                    id_vars=["is_error"], var_name=feature, value_name="Value"
                )
                error_plot_data_dummified = dummified_data_melted[dummified_data_melted["Value"] == 1]

            plot_data = error_plot_data_dummified if "Dummified" in feature else data
            shrinkage = 0.8 if "Dummified" in feature else 1.0

            if n_rows > 1:
                sns.histplot(
                    data=plot_data,
                    x=feature,
                    hue="is_error",
                    stat="percent",
                    bins=bins,
                    multiple="fill",
                    shrink=shrinkage,
                    ax=ax[i][j],
                )
                ax[i][j].axhline(y=mean_error, color="r", linestyle="--")
                j = (j + 1) % 2
                i = i + 1 - j
            else:
                sns.histplot(
                    data=plot_data,
                    x=feature,
                    hue="is_error",
                    stat="percent",
                    bins=bins,
                    multiple="fill",
                    shrink=shrinkage,
                    ax=ax[i],
                )
                ax[i].axhline(y=mean_error, color="r", linestyle="--")
                i += 1

        if self.show:
            plt.show()

    def _get_features(self, data, features, n=4):
        if features is None:
            features = self.analysis.feature_importance.loc[self.feature_idxs].head(n)["Feature"].tolist()
        elif features == "all":
            features = self.analysis.feature_importance.loc[self.feature_idxs]["Feature"].tolist()

        features = [f for f in features if f in data]

        return features
