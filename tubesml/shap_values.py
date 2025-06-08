import numpy as np
import pandas as pd
import shap


def get_shap_values(data, model, sample=700, class_pos=1, check_additivity=True):
    """
    Uses the ``shap.Explainer`` to generate shap values. This is done
    on a sample of the data to save time. For some classification models,
    itwill return a shap value per class. If that's the case, the shap values
    will be the ones of the class of interest

    :param data: pandas DataFrame with the data to calculate the shap values on
        It must have the same features used to train the model

    :param model: trained model object

    :param sample: integer (default 700) with the number of data samples to use to produce the
        shap values

    :param class_pos: integer, default 1. Index of the class of interest for shap values.

    :return shap_values. It has 3 attributes: values, base values, data.
        ``values`` is an ndarray of shape (n_samples, n_features) with the shap values
    """

    expl = shap.Explainer(model, data)

    n_samples = min(sample, len(data))

    try:
        shap_values = expl(data.sample(n_samples), check_additivity=check_additivity)
    except TypeError:  # not all the explainers have check_additivity
        shap_values = expl(data.sample(n_samples))

    if len(shap_values.values.shape) == 3:
        shap_values = _fix_format(shap_values, class_pos)

    return shap_values


def get_shap_importance(shap_values):
    """
    Summarizes the shap values to get the feature importance. It takes the mean
    and standard deviation of the absolute shap values

    :param data: pandas DataFrame with the data used to calculate the shap values

    :param shap_values. It has 3 attributes: values, base values, data.
        ``values`` is an ndarray of shape (n_samples, n_features) with the shap values

    :return pandas dataframe with mean and standard deviation of the shap values of each
        feature, order by magnitude
    """

    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names).abs()
    shap_df = (
        shap_df.agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "Feature", "mean": "shap_importance", "std": "shap_std"})
        .sort_values(by="shap_importance", ascending=False)
        .reset_index(drop=True)
    )

    # std of the mean
    shap_df["shap_std"] = shap_df["shap_std"] / np.sqrt(len(shap_values.data))

    return shap_df


def _fix_format(shap_values, class_pos):
    shap_values.values = shap_values.values[:, :, class_pos]
    shap_values.base_values = shap_values.base_values[:, class_pos]
    shap_values.output_dims = ()
    try:  # this triggers an error but AFTER the replacement we need took place
        shap_values.output_names = shap_values.output_names[class_pos]
    except AttributeError:
        pass  # so yay. (This will likely break sometime in the future)

    return shap_values
