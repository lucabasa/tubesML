import pandas as pd
import matplotlib.pyplot as plt

class CompareModels():
    def __init__(self, data, true_label,
                 pred_1, pred_2,
                 metric_func,
                 fe_1=None, fe_2=None,
                 shap_1=None, shap_2=None):
        self.data = data
        self.true_label = true_label
        self.metric_func = metric_func
        self.pred_1 = pred_1
        self.pred_2 = pred_2
        self.fe_1 = fe_1
        self.fe_2 = fe_2
        self.shap_1 = shap_1
        self.shap_2 = shap_2

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
        metric_comp = pd.DataFrame({"Model": ["Model 1", "Model 2"],
                                    "Metric": [eval_1, eval_2]}).set_index("Model")
        ax = metric_comp["Metric"].plot(kind="bar", rot=0)
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Model 1 metric is {diff}% {adj} than model 2")
        if show:
            plt.show()
        
    def compare_predictions(self):
        pass

    def compare_feature_importances(self):
        pass

    def compare_pdp(self):
        pass

    def compare_error_rates(self):
        pass

    def compare(self):
        pass
