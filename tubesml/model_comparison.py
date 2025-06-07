import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar


class CompareModels:
    """
    This class has methods to compare the results of 2 models. The user must provide the data used for
    the predictions, the true label, and the predictions of both models on said data. The comparison is done
    based on a (user provided) metric function, statistical significance tests, and visual comparison of the
    predictions

    :param data: pandas DataFrame with the data used to create the predictions of both models

    :param true_label: pandas Series with the true label

    :param pred_1: pandas Series with the predictions of the first model

    :param pred_2: pandas Series with the predictions of the second model

    :param metric_func: function that takes `y_true` and `y_pred` or `y_score` as input. This function calculates
        the models metric

    :param regression: boolean, to flag is the problem is a regression problem. This determines the type of statistical
        test and plots

    :param probabilities: boolean, to flag if the predictions are in the form of probabilities. Relevant only if
        `regression` is False

    :param kfold: kfold object used to produce the prediction, if any. If the regression predictions were made with
        kfold, we strongly recommend providing this for a meaningful test.
    """

    def __init__(
        self,
        data,
        true_label,
        pred_1,
        pred_2,
        metric_func,
        regression=True,
        probabilities=True,
        kfold=None,
    ):
        self.data = data
        self.true_label = true_label
        self.metric_func = metric_func
        self.regression = regression
        self.probabilities = probabilities
        self.pred_1 = pred_1
        self.pred_2 = pred_2
        self.kfold = kfold

        self.pred_df = pd.DataFrame({"True Value": true_label, "Model 1": pred_1, "Model 2": pred_2})

    def compare_metrics(self):
        """
        Calculates the provided metric for both models and produces a plot to compare them visually
        """
        _, ax = plt.subplots(1, 1, figsize=(15, 5))

        try:
            eval_1 = self.metric_func(y_true=self.true_label, y_pred=self.pred_1)
            eval_2 = self.metric_func(y_true=self.true_label, y_pred=self.pred_2)
        except TypeError:
            eval_1 = self.metric_func(y_true=self.true_label, y_score=self.pred_1)
            eval_2 = self.metric_func(y_true=self.true_label, y_score=self.pred_2)
        diff = round(((eval_1 - eval_2) / eval_1) * 100, 3)
        if diff > 0:
            adj = "bigger"
        else:
            adj = "smaller"
        metric_comp = pd.DataFrame({"Model": ["Model 1", "Model 2"], "Metric": [eval_1, eval_2]}).set_index("Model")
        ax = metric_comp["Metric"].plot(kind="bar", rot=0)
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Model 1 metric is {diff}% {adj} than model 2")

        plt.show()

    def compare_predictions(self, error_margin=0.05):
        """
        Visual comparison of the model predictions. For a regression problem, the comparison is done via
        a scatter plot of the 2 sets of prediction and via a confusion matrix. The conflusion matrix is showing
        the combinations of prediction that are 'correct'. A correct prediction is defined as one with a
        percentage error below the provided `error_margin`.

        For a classification problem, if the predictions are probabilities, the result is the same as for a
        regression problem. Othewise, only a confusion matrix of the 2 models correct classifications is shown.

        :param error_margin: float, the margin of error to consider a prediction correct.
        """
        if self.regression:
            _, ax = plt.subplots(1, 2, figsize=(15, 5))
            self._regression_predictions(error_margin, ax)
        else:
            if self.probabilities:
                _, ax = plt.subplots(1, 2, figsize=(15, 5))
                self._regression_predictions(error_margin, ax)
            else:
                _, ax = plt.subplots(1, 1, figsize=(10, 5))
                self._classification_predictions(ax)

        plt.show()

    def _regression_predictions(self, error_margin, ax):
        df = self.pred_df.copy()
        if self.probabilities:
            df["M1_right"] = np.where((abs(df["True Value"] - df["Model 1"]) < error_margin), 1, 0)
            df["M2_right"] = np.where(abs((df["True Value"] - df["Model 2"]) < error_margin), 1, 0)
        else:
            df["M1_right"] = np.where((abs((df["True Value"] - df["Model 1"]) / df["True Value"]) < error_margin), 1, 0)
            df["M2_right"] = np.where((abs((df["True Value"] - df["Model 2"]) / df["True Value"]) < error_margin), 1, 0)

        both_righ = ((df["M1_right"] == 1) & (df["M2_right"] == 1)).mean()
        one_right = ((df["M1_right"] == 1) & (df["M2_right"] == 0)).mean()
        two_right = ((df["M1_right"] == 0) & (df["M2_right"] == 1)).mean()
        both_wrong = ((df["M1_right"] == 0) & (df["M2_right"] == 0)).mean()

        cm = np.array([[two_right, both_righ], [both_wrong, one_right]])

        sns.heatmap(cm, ax=ax[0], annot=True, fmt=".2%", annot_kws={"size": 15}, linewidths=0.5, cmap="coolwarm")
        ax[0] = self._plot_labels(ax[0])

        self.pred_df.plot.scatter(x="Model 1", y="Model 2", ax=ax[1])
        ax[1].set_title("Prediction comparion")

    def _classification_predictions(self, ax):
        df = self.pred_df.copy()
        df["M1_right"] = np.where(df["True Value"] == df["Model 1"], 1, 0)
        df["M2_right"] = np.where(df["True Value"] == df["Model 2"], 1, 0)

        both_righ = ((df["M1_right"] == 1) & (df["M2_right"] == 1)).mean()
        one_right = ((df["M1_right"] == 1) & (df["M2_right"] == 0)).mean()
        two_right = ((df["M1_right"] == 0) & (df["M2_right"] == 1)).mean()
        both_wrong = ((df["M1_right"] == 0) & (df["M2_right"] == 0)).mean()

        cm = np.array([[two_right, both_righ], [both_wrong, one_right]])

        sns.heatmap(cm, ax=ax, annot=True, fmt=".2%", annot_kws={"size": 15}, linewidths=0.5, cmap="coolwarm")
        ax = self._plot_labels(ax)

    def statistical_significance(self, error_margin=0.49):
        """
        Performs a statistical test to see if the differences between the 2 models are statistically significant.
        For regression problems, the test is a paired t-test on the losses of each model.
        If the predictions were produced via a Kfold process, we must perform the test on each fold in order to
        compare samples that are fairly independend. Subsequently, we can compare the mean losses of each fold
        via another paired t-test. Generally speaking, the results will be in agreement, but keep in mind that
        repeated statistical tests increase the probability of false positive.

        For classification problems, the test is the Mcnemar on the contingency table of the 2 models, showing how
        many predictions were correct from both models, and how many are misclassified by either or both models.
        If the predictions were probabilities, we consider an `error_margin` to define a correct prediction

        :param error_margin: float, margin of error for probability predictions.
        """
        loss_folds_1 = []
        loss_folds_2 = []

        if self.regression:
            if self.kfold is not None:
                for n_fold, (_, test_index) in enumerate(self.kfold.split(self.true_label.values)):
                    mod_1 = self.pred_1[test_index]
                    mod_2 = self.pred_2[test_index]
                    tar = self.true_label[test_index]

                    loss_1 = (tar - mod_1) ** 2
                    loss_2 = (tar - mod_2) ** 2
                    result = ttest_rel(loss_1, loss_2)
                    print(
                        f"Fold {n_fold + 1} - statistic={round(result.statistic, 3)}, p-value={round(result.pvalue, 3)}"
                    )

                    loss_folds_1.append(np.mean(loss_1))
                    loss_folds_2.append(np.mean(loss_2))

                result = ttest_rel(loss_folds_1, loss_folds_2)
                print(f"Across Folds - statistic={round(result.statistic, 3)}, p-value={round(result.pvalue, 3)}")
            else:
                loss_1 = (self.pred_1 - self.true_label) ** 2
                loss_2 = (self.pred_2 - self.true_label) ** 2
                result = ttest_rel(loss_1, loss_2)
                print(f"statistic={round(result.statistic, 3)}, p-value={round(result.pvalue, 3)}")

        else:
            if self.probabilities:
                m1_right = np.where(abs(self.true_label - self.pred_1) < error_margin, 1, 0)
                m2_right = np.where(abs(self.true_label - self.pred_2) < error_margin, 1, 0)
            else:
                m1_right = np.where(self.true_label == self.pred_1, 1, 0)
                m2_right = np.where(self.true_label == self.pred_2, 1, 0)
            contingency_table = pd.crosstab(m1_right, m2_right).sort_index(ascending=False)[[1, 0]].values
            result = mcnemar(contingency_table, exact=True)
            print(f"statistic={round(result.statistic, 3)}, p-value={round(result.pvalue, 3)}")

    def _plot_labels(self, ax):
        ax.set_xlabel("Model 1 Correct", fontsize=14)
        ax.set_ylabel("Model 2 Correct", fontsize=14)
        ax.set_xticklabels(["No", "Yes"], fontsize=12)
        ax.set_yticklabels(["Yes", "No"], fontsize=12)
        ax.set_title("Models agreement")
        return ax
