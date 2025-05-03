import pandas as pd

import shap


def get_shap_values(data, model, model_type="general", sample=2000):

    if model_type == "general":
        expl = shap.Explainer(model, data)
    elif model_type == "tree":
        expl = shap.TreeExplainer(model)
    elif model_type == "kernel":
        expl = shap.KernelExplainer(model, data)
    else:
        raise ValueError("Invalid model_type")

    n_samples = min(sample, len(data))

    shap_values = expl(data.sample(n_samples))

    return shap_values


def get_shap_importance(data, shap_values):

    shap_df = pd.DataFrame(shap_values.values, columns=data.columns).abs()

    return (
        shap_df.agg(["mean", "std"])
        .T.rename(columns={"mean": "shap_importance", "std": "shap_std"})
        .sort_values(by="shap_importance", ascending=False)
    )
