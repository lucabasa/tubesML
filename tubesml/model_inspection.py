__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'


import pandas as pd
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

from tubesml.base import BaseTransformer


def get_coef(pipe, feats=None):
    '''
    Get dataframe with coefficients of a model in Pipeline. 
    
    The step before the model has to have a ``get_feature_names_out`` method. 
    
    If a simple estimator is provided, it creates a pipeline with a ``BaseTransformer``. 
    In that case, the ``feats`` input is not optional and there is no need for a ``get_feature_names_out`` method.
    
    :param pipe: pipeline or estimator
    
    :param feats: (optional) list of features the estimator uses.
    
    :return result: pandas DataFrame with a ``feat`` column with the feature names and a ``score`` column with the
                    coefficients values ordere by absolute magnitude.
    '''
    try:  # If estimator is not a pipeline, make a pipeline
        feats = pipe.steps[-2][1].get_feature_names_out()
    except AttributeError:
        pipe = Pipeline([('transf', BaseTransformer()), ('model', pipe)])
        feats = feats
    if hasattr(pipe.steps[-1][1], "is_stacker"):
        feats = pipe.steps[-1][1].get_feature_names_out()
    imp = pipe.steps[-1][1].coef_.ravel().tolist()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe, feats=None):
    '''
    Get dataframe with the feature importance of a model in Pipeline.
    
    The step before the model has to have a ``get_feature_names_out`` method.
    
    If a simple estimator is provided, it creates a pipeline with a ``BaseTransformer``. 
    In that case, the ``feats`` input is not optional and there is no need for a ``get_feature_names_out`` method.
    
    :param pipe: pipeline or estimator
    
    :param feats: (optional) list of features the estimator uses.
    
    :return result: pandas DataFrame with a ``feat`` column with the feature names and a ``score`` column with the
                    feature importances values ordere by magnitude.

    '''
    try:  # If estimator is not a pipeline, make a pipeline
        feats = pipe.steps[-2][1].get_feature_names_out()
    except AttributeError:
        pipe = Pipeline([('transf', BaseTransformer()), ('model', pipe)])
        feats = feats
    if hasattr(pipe.steps[-1][1], "is_stacker"):
        feats = pipe.steps[-1][1].get_feature_names_out()
    imp = pipe.steps[-1][1].feature_importances_.tolist() # it's a pipeline
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


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
        fi = data.head(n).copy()
    else:
        fi = data.copy()

    fi = fi.reset_index().iloc[::-1]
    
    fig, ax = plt.subplots(1,1, figsize=(13, max(1, int(0.3*fi.shape[0]))))

    ax.barh(y=fi['feat'], width=fi['mean'], xerr=fi['std'], left=0)
    
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
    
    Moreover, it doesn't behave well with early stopping, which produces no result. In that case, a
    RuntimeError is raised
    
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
    
        
    if np.isnan(train_scores).all() or np.isnan(test_scores).all():
        raise RuntimeError("No scores generated, try change the model")
    
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
    
    
def get_pdp(estimator, feature, data, grid_resolution=100):
    """
    Calculates the partial dependence of the model to a variable.
    It is a wrapper around ``sklearn.inspect.partial_dependence`` 
    
    :param estimator: model or pipeline with a predict method.
                If the ``fit`` method was not previously called, it will throw an error.
                
    :param feature: string or tuple of 2 strings.
                The feature for which to create the partial dependence. If it is a tuple, 
                a 2-way partial dependence will be created.
    
    :param data: pandas DataFrame.
                It must contain the features the ``estimator`` uses to generate predictions.
                If ``feature`` is not present in this dataframe, an error will be raised.
                
    :param grid_resolution: Integer, default 100.
                The number of equally spaced points on the grid.
                
    :return: pandas DataFrame with columns ``x`` (the ``feature`` values in the grid), 
                ``feat`` (the ``feature`` name), ``y`` (the values of the partial dependence). 
                If ``feature`` is a tuple, there is also another column ``x_1`` with the values in the 
                grid of the second feature of the tuple. If ``feature`` is a string, ``x_1`` is empty.
    """
    if isinstance(feature, tuple):
        grid_resolution = 50
    elif isinstance(feature, list):  # TODO: allow for this in the future.
        raise TypeError('This function does not support the calculation over multiple features. You can use directly the sklearn function for that.')
        
    # TODO: if the feature is not in the original data but created by the estimator, it breaks
    # In the future, it should not break
    
    pdp = partial_dependence(estimator, features=feature, 
                                   X=data, grid_resolution=grid_resolution, 
                                   kind='average')
    
    try:  # FIXME:this is for backward compatibility with kaggle
        pdp['grid_values']
        vals = 'grid_values'
    except KeyError:
        vals = 'values'
    
    if isinstance(feature, tuple):
        tmp = pd.DataFrame({'x': [pdp[vals][1]]*grid_resolution})
        tmp = pd.DataFrame(tmp.x.to_list())
        df_pdp = pd.concat([pd.DataFrame({'x': pdp[vals][0]}), tmp], axis=1)
        df_pdp = pd.melt(df_pdp, id_vars='x', value_name='x_1').drop('variable', axis=1).sort_values('x')
        df_pdp['y'] = [item for sublist in pdp['average'][0] for item in sublist]
    elif isinstance(feature, str):
        df_pdp = pd.DataFrame({'x': pdp[vals][0], 
                               'x_1': [np.nan]*len(pdp[vals][0]), 
                               'y': pdp['average'][0]})
    
    df_pdp['feat'] = [feature] * len(df_pdp)
    
    return df_pdp
            

def plot_pdp(data, feature, title, axes):
    '''
    Plot partial dependence of a feature in an ax. If available, uncertainty plotted around it.
    
    :param data: pandas Dataframe with the partial dependence
                It must contain a ``feat``, a ``x``, and either a ``y`` or a ``mean`` columns
                If there is an ``std`` column, it will be plotted as uncertainty aroudn the mean
                
    :param feature: string.
                The feature to plot as x axis in the partial dependence
    
    :param title: string.
                The title on top of the plot
                
    :param axes: matplotlib axes
                The plot will take place in this axes
                
    :return: matplotlib axes with the plot.
    
    '''
    if not {'feat', 'x'} <= set(data.columns):
        raise KeyError('data must contain the columns feat, x')
    
    if 'mean' in data.columns:
        data[data.feat==feature].plot(ax=axes, x='x', y='mean', color='k')
        pl_feat = 'mean'
    elif 'y' in data.columns:
        data[data.feat==feature].plot(ax=axes, x='x', y='y', color='k')
        pl_feat = 'y'
    else:
        raise KeyError('The input data must have either a y column or a mean column')
        
    if 'std' in data.columns:
        axes.fill_between(data[data.feat==feature].x, 
                          (data[data.feat==feature][pl_feat] - data[data.feat==feature]['std'] / 2).astype(float),
                          (data[data.feat==feature][pl_feat] + data[data.feat==feature]['std'] / 2).astype(float), 
                          alpha=0.3, color='r')
    axes.set_title(title, fontsize=14)
    axes.legend().set_visible(False)
    axes.set_xlabel('')
    return axes


def plot_two_pdp(data, feature, title, axes):
    """
    This function is still in development. Plot a 2-way partial dependence
    
    :param data: pandas Dataframe with the partial dependence
                It must contain a ``feat``, a ``x``, a ``x_1`` and a ``y`` columns.
                
    :param feature: tuple of strings.
                The features to plot as x and y axis in the partial dependence
    
    :param title: string.
                The title on top of the plot
                
    :param axes: matplotlib axes
                The plot will take place in this axes
                
    :return: matplotlib axes with the plot.
    """
    
    if not {'feat', 'x', 'x_1', 'y'} <= set(data.columns):
        raise KeyError('data must contain the columns feat, x, x_1, y')
        
    if not isinstance(feature, tuple):
        raise TypeError('feature must be a tuple for this type of plot')
        
    plt_data = data[data['feat'] == feature]

    X_axis = plt_data['x'].astype(float)
    Y_axis = plt_data['x_1'].astype(float)

    xg, yg = np.meshgrid(np.linspace(X_axis.min(), X_axis.max(), 100),
                         np.linspace(Y_axis.min(), Y_axis.max(), 100))
    
    triangles = tri.Triangulation(X_axis, Y_axis)
    tri_interp = tri.CubicTriInterpolator(triangles, plt_data['y'])
    zg = tri_interp(xg, yg)
    
    axes.contourf(xg, yg, zg, 
                   norm=plt.Normalize(vmax=plt_data['y'].max(), vmin=plt_data['y'].min()),
                   cmap=plt.cm.terrain)
    
    
    ax.set_xlabel(feature[0], fontsize=12)
    ax.set_ylabel(feature[1], fontsize=12)
    ax.set_title(title, fontsize=14)
    
    return ax


def plot_partial_dependence(pdps, savename=None):
    '''
    Plot all the pdps in the dataframe in a plot with 2 columns and as many rows as necessary.
    The function is a wrapper around ``tubesml.plot_pdp``
    
    :param pdps: pandas DataFrame with the partial dependences
                It must contain a ``feat``, a ``x``
    '''
    
    if not {'feat', 'x'} <= set(pdps.columns):
        raise KeyError('data must contain the columns feat and x')
    
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
        