__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

import warnings


def get_coef(pipe):
    '''
    Get dataframe with coefficients of a model in Pipeline
    The step before the model has to have a get_feature_name method
    '''
    imp = pipe.steps[-1][1].coef_.ravel().tolist()
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe):
    '''
    Get dataframe with the feature importance of a model in Pipeline
    The step before the model has to have a get_feature_name method
    '''
    imp = pipe.steps[-1][1].feature_importances_.tolist() # it's a pipeline
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


def _plot_diagonal(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    scl = (high - low) / 100
    
    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line
                         'y': np.arange(low, high ,scl)})
    ax.plot(line.x, line.y, color='black', linestyle='--')
    
    return ax


def plot_regression_predictions(data, true_label, pred_label, hue=None, savename=None):
    '''
    Plot prediction vs true label and the distribution of both the label and the predictions
    Data, true_label, and pred_label must be of compatible size
    Hue is ignored when the unique values are more than 5 for readability
    '''
    
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    legend=False
    addition=''
    if hue is not None:
        if tmp[hue].nunique() > 5:
            warnings.warn(f'{hue} has more than 5 unique values, hue will be ignored', UserWarning)
            hue = None
        else:
            legend=True
            addition = f' by {hue}'

    sns.scatterplot(x='True Label', y='Prediction', data=tmp, ax=ax[0],
                         hue=hue, legend=legend, alpha=0.5)
    
    ax[0] = _plot_diagonal(ax[0])
    ax[0].set_title(f'True Label vs Prediction{addition}', fontsize=14)
        
    sns.histplot(data=tmp, x=true_label, kde=True, ax=ax[1], color='blue', label='True Label', alpha=0.4)
    sns.histplot(data=tmp, x=pred_label, kde=True, ax=ax[1], color='red', label='Prediction', alpha=0.6)
    ax[1].legend()
    ax[1].set_xlabel('Target')
    ax[1].set_title('Distribution of target and prediction', fontsize=14)
    
    if savename is not None:
        plt.savefig(savename)
        plt.show()
    else:
        plt.show()
