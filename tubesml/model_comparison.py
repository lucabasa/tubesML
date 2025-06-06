import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar


class CompareModels:
    def __init__(
        self,
        data,
        true_label,
        pred_1,
        pred_2,
        metric_func,
        regression=True,
        probabilities=True,
        fe_1=None,
        fe_2=None,
        shap_1=None,
        shap_2=None,
        kfold=None,
    ):
        self.data = data
        self.true_label = true_label
        self.metric_func = metric_func
        self.regression = regression
        self.probabilities = probabilities
        self.pred_1 = pred_1
        self.pred_2 = pred_2
        self.fe_1 = fe_1
        self.fe_2 = fe_2
        self.shap_1 = shap_1
        self.shap_2 = shap_2
        self.kfold = kfold

        self.pred_df = pd.DataFrame({"True Value": true_label, "Model 1": pred_1, "Model 2": pred_2})

    def compare_metrics(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(15, 5))
            show = True
        else:
            show = False

        eval_1 = self.metric_func(y_true=self.true_label, y_pred=self.pred_1)
        eval_2 = self.metric_func(y_true=self.true_label, y_pred=self.pred_2)
        diff = round(((eval_1 - eval_2) / eval_1) * 100, 3)
        if diff > 0:
            adj = "bigger"
        else:
            adj = "smaller"
        metric_comp = pd.DataFrame({"Model": ["Model 1", "Model 2"], "Metric": [eval_1, eval_2]}).set_index("Model")
        ax = metric_comp["Metric"].plot(kind="bar", rot=0)
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Model 1 metric is {diff}% {adj} than model 2")
        if show:
            plt.show()

    def compare_predictions(self, error_margin=0.05, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(15, 5))
            show = True
            residuals = False
        else:
            show = False
            residuals = True

        if self.regression:
            self._regression_predictions(error_margin, ax, residuals)
        else:
            self._classification_predictions(error_margin, ax)

        if show:
            plt.show()

    def _regression_predictions(self, error_margin, ax, residuals):
        df = self.pred_df.copy()
        df["M1_right"] = np.where((abs((df["True Value"] - df["Model 1"]) / df["True Value"]) < error_margin), 1, 0)
        df["M2_right"] = np.where((abs((df["True Value"] - df["Model 2"]) / df["True Value"]) < error_margin), 1, 0)

        both_righ = ((df["M1_right"] == 1) & (df["M2_right"] == 1)).mean()
        one_right = ((df["M1_right"] == 1) & (df["M2_right"] == 0)).mean()
        two_right = ((df["M1_right"] == 0) & (df["M2_right"] == 1)).mean()
        both_wrong = ((df["M1_right"] == 0) & (df["M2_right"] == 0)).mean()

        cm = np.array[[two_right, both_righ], [both_wrong, one_right]]

        sns.heatmap(cm, ax=ax[0], annot=True, fmt=".2%", annot_kws={"size": 15}, linewidths=0.5, cmap="coolwarm")
        ax[0].set_xlabel("Model 1 Correct", fontsize=14)
        ax[0].set_ylabel("Model 2 Correct", fontsize=14)
        ax[0].set_xticklabels(["No", "Yes"], fontsize=12)
        ax[0].set_yticklabels(["Yes", "No"], fontsize=12)
        ax[0].set_title("Models agreement")

        self.pred_df.plot.scatter(x="Model 1", y="Model 2", ax=ax[1])
        ax[1].set_title("Prediction comparion")

        if residuals:
            df["Residuals Model 1"] = df["True Value"] - df["Model 1"]
            df["Residuals Model 2"] = df["True Value"] - df["Model 2"]
            sns.scatterplot(x="True Value", y="Residuals Model 1", data=df, ax=ax[2], alpha=0.5, label="Model 1")
            sns.scatterplot(x="True Value", y="Residuals Model 2", data=df, ax=ax[2], alpha=0.5, label="Model 2")
            ax[2].legend()
            ax[2].set_ylabel("Residuals")
            ax[2].set_ylabel("Target vs Residuals")

    def _classification_predictions(self, error_margin, ax):
        if self.probabilities:
            self._regression_predictions(error_margin, ax, residuals=False)

        else:
            df = self.pred_df.copy()
            df["M1_right"] = np.where(df["True Value"] == df["Model 1"], 1, 0)
            df["M2_right"] = np.where(df["True Value"] == df["Model 2"], 1, 0)

            both_righ = ((df["M1_right"] == 1) & (df["M2_right"] == 1)).mean()
            one_right = ((df["M1_right"] == 1) & (df["M2_right"] == 0)).mean()
            two_right = ((df["M1_right"] == 0) & (df["M2_right"] == 1)).mean()
            both_wrong = ((df["M1_right"] == 0) & (df["M2_right"] == 0)).mean()

            cm = np.array[[two_right, both_righ], [both_wrong, one_right]]

            sns.heatmap(cm, ax=ax, annot=True, fmt=".2%", annot_kws={"size": 15}, linewidths=0.5, cmap="coolwarm")

    def statistical_significance(self, error_margin=0.49):
        loss_folds_1 = []
        loss_folds_2 = []

        if self.regression:
            if self.kfold is not None:
                for n_fold, (_, test_index) in enumerate(self.kfolds.split(self.true_label.values)):
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
