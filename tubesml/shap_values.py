import pandas as pd

import shap


def get_shap_values(data, model, sample=700, class_pos=1):

    expl = shap.Explainer(model, data)
    
    n_samples = min(sample, len(data))

    shap_values = expl(data.sample(n_samples))

    if len(shap_values.values.shape) == 3:
        shap_values = _fix_format(shap_values, class_pos)

    return shap_values


def get_shap_importance(data, shap_values):

    shap_df = pd.DataFrame(shap_values.values, columns=data.columns).abs()

    return (
        shap_df.agg(["mean", "std"])
        .T.rename(columns={"mean": "shap_importance", "std": "shap_std"})
        .sort_values(by="shap_importance", ascending=False)
    )


def _fix_format(shap_values, class_pos):

    shap_values.values = shap_values.values[:, :, class_pos]

    return shap_values