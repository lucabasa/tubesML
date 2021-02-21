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


def plot_learning_curve(estimator, X, y, scoring=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10), title=None):
    '''
    Plot learning curve and scalability of the model
    It may create issues when both the estimator and this function have n_jobs>1
    '''
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    train_sizes, train_scores, test_scores, fit_times, score_times = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=True)
    
    if not scoring is None:
        if 'neg' in scoring:
            train_scores = -train_scores
            test_scores = -test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1) / np.sqrt(cv.get_n_splits() - 1)  # std of the mean, unbiased
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1) / np.sqrt(cv.get_n_splits() - 1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1) / np.sqrt(cv.get_n_splits() - 1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1) / np.sqrt(cv.get_n_splits() - 1)

    # Plot learning curve
    ax[0][0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax[0][0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax[0][0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax[0][0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax[0][0].legend(loc="best")
    ax[0][0].set_title('Train and test scores', fontsize=14)
    if ylim is not None:
        ax[0][0].set_ylim(*ylim)
    ax[0][0].set_xlabel("Training examples")
    ax[0][0].set_ylabel("Score")

    # Plot n_samples vs fit_times
    ax[0][1].plot(train_sizes, fit_times_mean, 'o-')
    ax[0][1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    ax[0][1].set_xlabel("Training examples")
    ax[0][1].set_ylabel("fit_times")
    ax[0][1].set_title("Scalability of the model", fontsize=14)

    # Plot fit_time vs score
    ax[1][0].plot(fit_times_mean, test_scores_mean, 'o-')
    ax[1][0].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    ax[1][0].set_xlabel("fit_times")
    ax[1][0].set_ylabel("Score")
    ax[1][0].set_title("Fit time vs test score", fontsize=14)
    
    # Plot fit_time vs fit_score
    ax[1][1].plot(fit_times_mean, train_scores_mean, 'o-')
    ax[1][1].fill_between(fit_times_mean, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1)
    ax[1][1].set_xlabel("fit_times")
    ax[1][1].set_ylabel("Score")
    ax[1][1].set_title("Fit time vs train score", fontsize=14)
    
    if title is not None:
        fig.suptitle(f'{title}', fontsize=18)
    
    plt.show()       