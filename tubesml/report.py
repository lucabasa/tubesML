__author__ = 'lucabasa'
__version__ = '2.0.0'
__status__ = 'development'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix

from tubesml.base import BaseTransformer

import warnings


def get_coef(pipe, feats=None):
    '''
    Get dataframe with coefficients of a model in Pipeline. 
    
    The step before the model has to have a ``get_feature_names`` method. 
    
    If a simple estimator is provided, it creates a pipeline with a ``BaseTransformer``. 
    In that case, the ``feats`` input is not optional and there is no need for a ``get_feature_names`` method.
    
    :param pipe: pipeline or estimator
    
    :param feats: (optional) list of features the estimator uses.
    
    :return result: pandas DataFrame with a ``feat`` column with the feature names and a ``score`` column with the
                    coefficients values ordere by absolute magnitude.
    '''
    try:  # If estimator is not a pipeline, make a pipeline
        feats = pipe.steps[-2][1].get_feature_names()
    except AttributeError:
        pipe = Pipeline([('transf', BaseTransformer()), ('model', pipe)])
        feats = feats
    imp = pipe.steps[-1][1].coef_.ravel().tolist()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe, feats=None):
    '''
    Get dataframe with the feature importance of a model in Pipeline.
    
    The step before the model has to have a ``get_feature_names`` method.
    
    If a simple estimator is provided, it creates a pipeline with a ``BaseTransformer``. 
    In that case, the ``feats`` input is not optional and there is no need for a ``get_feature_names`` method.
    
    :param pipe: pipeline or estimator
    
    :param feats: (optional) list of features the estimator uses.
    
    :return result: pandas DataFrame with a ``feat`` column with the feature names and a ``score`` column with the
                    feature importances values ordere by magnitude.

    '''
    try:  # If estimator is not a pipeline, make a pipeline
        feats = pipe.steps[-2][1].get_feature_names()
    except AttributeError:
        pipe = Pipeline([('transf', BaseTransformer()), ('model', pipe)])
        feats = feats
    imp = pipe.steps[-1][1].feature_importances_.tolist() # it's a pipeline
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
    Plot prediction vs true label and the distribution of both the label and the predictions.
    
    :param data: pandas DataFrame. 
                Ideally, the dataframe used for training the model you are evaluating. It must have the same 
                number of rows of ``true_label`` and ``pred_label``.
                
    :param true_label: pandas Series, numpy array, or list with the true values of the target variable.
    
    :param pred_label: pandas Series, numpy array, or list with the predicted values of the target variable.
    
    :param hue: (optional) str, name of the feature to use as hue in the scatter plot. It must be in ``data`` or it will be
                ignored after a warning.
                It is ignored when the unique values are more than 5 for readability.
                
    :param savename: (optional) str with the name of the file to use to save the figure. If not provided, the function simply
                    plots the figure.
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
        plt.close()
    else:
        plt.show()


def plot_learning_curve(estimator, X, y, scoring=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10), title=None):
    '''
    Plot learning curve and scalability of the model. The estimation is an average across the folds in the
    cross validation, the uncertainty is the unbiased standard deviation of the mean.
    
    It may create issues when both the estimator and this function have n_jobs>1.
    
    :param estimator: estimator or pipeline.
    
    :param X: {array-like} of shape (n_samples, n_features)
            The training input samples.
            
    :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values (class labels) as integers or strings.
    
    :param scoring: string.
            Scoring metric for the learning curve, see the sklearn documentation for the available options.
            
    :param ylim: (optional) tuple with the limits to use in the y-axis of the plots.
    
    :param cv: int, or KFold generator. The learning curves will be computed with prediction out of folds
                generated by this cross-validation choice.
                
    :param n_jobs: int, number of jobs to run in parallel.
    
    :param train_sizes: array-like of shape (n_ticks,), \
            default=np.linspace(0.1, 1.0, 5)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        
    :param title: (optional) string for the figure title.
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
    
    
def plot_feat_imp(data, n=-1, savename=None):
    '''
    Plots a barplot with error bars of feature importance.
    It works with coefficients too.
    
    :param data: pandas DataFrame with a ``mean`` and a ``std`` column. A KeyError is raised
                if any of these columns is missing.
                
    :param n: int, default=-1. Number of features to display.
    
    :param savename: (optional) str with the name of the file to use to save the figure. If not provided, the function simply
                    plots the figure.
    '''
    
    if not set(['mean', 'std']).issubset(data.columns):
        raise KeyError('data must contain the columns feat, mean, and std')
    
    if n > 0:
        fi = data.head(n)
    else:
        fi = data
    
    fig, ax = plt.subplots(1,1, figsize=(13, int(0.3*fi.shape[0])))

    sns.barplot(x=fi['mean'], y=fi.index, xerr=fi['std'], ax=ax)
    
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()
        

def plot_pdp(data, feature, title, axes):
    '''
    Plot partial dependence of a feature in an ax.
    
    Uncertainty plotted around it.
    '''
    if not set(['feat', 'mean', 'std']).issubset(data.columns):
        raise KeyError('data must contain the columns feat, mean, and std')
    
    data[data.feat==feature].plot(ax=axes, x='x', y='mean', color='k')
    axes.fill_between(data[data.feat==feature].x, 
                      (data[data.feat==feature]['mean'] - data[data.feat==feature]['std']).astype(float),
                      (data[data.feat==feature]['mean'] + data[data.feat==feature]['std']).astype(float), 
                      alpha=0.3, color='r')
    axes.set_title(title, fontsize=14)
    axes.legend().set_visible(False)
    axes.set_xlabel('')
    return axes


def plot_partial_dependence(pdps, savename=None):
    '''
    Plot all the pdps in the dataframe in a plot with 2 columns and as many rows as necessary.
    '''
    
    if not set(['feat', 'mean', 'std']).issubset(pdps.columns):
        raise KeyError('data must contain the columns feat, mean, and std')
    
    num = pdps.feat.nunique()
    rows = int(num/2) + (num % 2 > 0)
    feats = pdps.feat.unique()
    
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
    i = 0
    j = 0
    for feat in feats:
        if (rows > 1):
            ax[i][j] = plot_pdp(pdps, feat, feat, ax[i][j])
            j = (j+1)%2
            i = i + 1 - j
        else:
            ax[i] = plot_pdp(pdps, feat, feat, ax[i])
            i = i+1

    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()
        

def plot_confusion_matrix(true_label, pred_label, ax=None, thrs=0.5):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12,12))
        end_action = True
    else:
        end_action = False
        
    if len(set(pred_label)) > 2:
        preds = pred_label > thrs
    else:
        preds = pred_label
    cm = confusion_matrix(y_true=true_label, y_pred=preds, normalize='all')
    sns.heatmap(cm, cbar=False, ax=ax,
                annot=True, fmt=".2%", annot_kws={"size":15},
                linewidths=.5, cmap="coolwarm")
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    
    if end_action:
        plt.show()
    else:
        return ax


def plot_classification_probs(data, true_label, pred_label, thrs=0.5, sample=None, feat=None, hue_feat=None, savename=None):
    '''
    Plot prediction vs true label when the prediction is a probability
    Plots also a confusion matrix, a 
    Data, true_label, and pred_label must be of compatible size
    hue_feat is ignored when the unique values are more than 5 for readability
    '''
    
    # prepare data
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    # the x axis of the third plot has to be a continuos feature
    # If missing, make one
    if feat is not None:
        if feat not in tmp.columns:
            warnings.warn(f'The feature {feat} is not in the provided data, it will be ignored', UserWarning)
            feat = None
    if feat is None:  
        tmp['DUMMY_FEAT'] = np.arange(0, len(tmp))
        feat = 'DUMMY_FEAT'
    
    fig, ax = plt.subplots(2,2, figsize=(15,12))
    
    # Plotting probability vs true label
    tmp[tmp['True Label']==1]['Prediction'].hist(bins=50, ax=ax[0][0], alpha=0.5, color='g', label='1')
    tmp[tmp['True Label']==0]['Prediction'].hist(bins=50, ax=ax[0][0], alpha=0.5, color='r', label='0')
    ax[0][0].axvline(thrs, color='k', linestyle='--')
    ax[0][0].grid(False)
    ax[0][0].legend()
    ax[0][0].set_title('Predicted Probability vs True Label', fontsize=14)
    
    # Confusion matrix
    ax[0][1] = plot_confusion_matrix(tmp['True Label'], tmp['Prediction'], ax=ax[0][1], thrs=thrs)
    
    # This is to allow for segmenting the data by a categorical feature
    addition=''
    if hue_feat is not None:
        if hue_feat not in tmp.columns:
            warnings.warn(f'{hue_feat} not in the provided data, it will be ignored', UserWarning)
            hue_feat = None
            sizes = None
        elif tmp[hue_feat].nunique() > 5:
            warnings.warn(f'{hue_feat} has more than 5 unique values, it will be ignored', UserWarning)
            hue_feat = None
            sizes = None
        else:
            addition = f' by {hue_feat}'
            # The hue_feature will be displayed with size
            n_vals = tmp[hue_feat].nunique()
            sizes = {val: (i+1)*100/n_vals for i, val in enumerate(np.sort(tmp[hue_feat].unique()))}
    else:
        sizes = None
    
    # Barplot of mean target vs mean prediction
    if hue_feat is None:
        tmp[['True Label', 'Prediction']].mean().plot(kind='bar', 
                                                      ax=ax[1][1], color=['r', 'g'], alpha=0.7)
    else:
        tmp.groupby(hue_feat)[['True Label', 'Prediction']].mean().plot(kind='bar', 
                                                                        ax=ax[1][1], color=['r', 'g'], alpha=0.7)
    
    ax[1][1].axhline(tmp['True Label'].mean(), color='r', linestyle='--')
    ax[1][1].axhline(tmp['Prediction'].mean(), color='g', linestyle='--')
    ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), rotation=0)
    ax[1][1].set_ylim((0,1))
    ax[1][1].set_title(f'Mean Label vs Mean Prediction{addition}', fontsize=14)
    
    if sample:  # to make the plot more readable
        tmp = tmp.sample(sample)
    
    # Plot continuous feature vs prediction, the true label is the hue
    # Possible to add size to also see the role of categorical feature
    sns.scatterplot(x=feat, y='Prediction', data=tmp, ax=ax[1][0], palette=['r', 'g'],
                    size=hue_feat, sizes=sizes,
                    hue='True Label', legend=True, alpha=0.3, edgecolor=None)
    
    
    ax[1][0].axhline(thrs, color='k', linestyle='--')
    ax[1][0].set_title(f'True Label vs Prediction{addition}', fontsize=14)
    
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()


def eval_classification(data, target, preds, proba=False, thrs=0.5, feats=None, plot=True, **kwargs):
    if proba:
        preds_bin = preds >= thrs
        fpr, tpr, _ = roc_curve(target, preds)
        if plot > 0:
            plot_classification_probs(data, target, preds, thrs=thrs, **kwargs)
        if plot > 1:
            plt.plot(fpr, tpr)
            plt.title('ROC', fontsize=14)
    else:
        preds_bin = preds
        if plot > 0:
            plot_confusion_matrix(target, preds)
    
    print(f'Accuracy score: \t{round(accuracy_score(target, preds_bin), 4)}')
    print(f'AUC ROC: \t\t{round(roc_auc_score(target, preds), 4)}')
    print(classification_report(target, preds_bin))
    
