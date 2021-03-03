__author__ = 'lucabasa'
__version__ = '1.1.3'
__status__ = 'development'


import pandas as pd
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns



def list_missing(data, verbose=True):
    mis_cols = [col for col in data.columns if data[col].isna().any()]
    if not verbose:
        return mis_cols
    tot_rows = len(data)
    for col in mis_cols:
        print(f'Column {col}: {round(data[col].isna().sum()*100/tot_rows, 2)}% missing')
    return mis_cols


def plot_correlations(data, target=None, limit=50, figsize=(12,10), **kwargs):
    '''
    This function  plots the correlation matrix of a dataframe
    If a target feature is provided, it will display only a certain amount of features, the ones correlated the most
    with the target. The number of features displayed is controlled by the parameter limit
    '''
    corr = data.corr()
    if target:
        corr['abs'] = abs(corr[target])
        cor_target = corr.sort_values(by='abs', ascending=False)[target]
        cor_target = cor_target[:limit]
        del corr['abs']
        corr = corr.loc[cor_target.index, cor_target.index]
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr, cmap='RdBu_r', **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()
    if target:
        return cor_target


def plot_distribution(data, column, bins=50, correlation=None):
    '''
    Plots a histogram of a given column
    If a Pandas Series is provided with the correlation values, it will be displayed in the title.
    '''
    plt.figure(figsize=(12,8))
    data[column].hist(bins=bins)
    if not correlation is None:
        value = correlation[column]
        column = column + f' - {round(value,2)}'
    plt.title(f'Distribution of {column}', fontsize=18)
    plt.grid(False)
    plt.show()


def plot_bivariate(data, x, y, hue=None, **kwargs):
    '''
    Scatterplot of the feature x vs the feature y with the possibility of adding a hue
    '''
    plt.figure(figsize=(12,8))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, **kwargs)
    if hue:
        plt.title(f'{x} vs {y}, by {hue}', fontsize=18)
    else:
        plt.title(f'{x} vs {y}', fontsize=18)
    plt.show()


def corr_target(data, target, cols, x_estimator=None):
    '''
    Scatterplot + linear regression of a list of columns against the target.
    A correlation matrix is also printed.
    It is possible to pass an estimator.
    '''
    print(data[cols+[target]].corr())
    num = len(cols)
    rows = int(num/2) + (num % 2 > 0)
    cols = list(cols)
    y = data[target]
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
    i = 0
    j = 0
    for feat in cols:
        x = data[feat]
        if (rows > 1):
            sns.regplot(x=x, y=y, ax=ax[i][j], x_estimator=x_estimator)
            j = (j+1)%2
            i = i + 1 - j
        else:
            sns.regplot(x=x, y=y, ax=ax[i], x_estimator=x_estimator)
            i = i+1
    plt.show()
    

def _ks_test(data, col, target, critical=0.05):
    '''
    It takes a categorical feature and makes dummies.
    For each dummy, it performs a Kolmogorov-Smirnov test between the distribution of the target 
    of that subset vs the rest of the population.
    ''' 
    df = pd.get_dummies(data[[col]+[target]], columns=[col])
    
    for col in df.columns:
        if col == target:
            continue
        tmp_1 = df[df[col] == 1][target]
        tmp_2 = df[df[col] == 0][target]
        ks, p = stats.ks_2samp(tmp_1, tmp_2)
        if p < critical:
            return True
    return False
    

def find_cats(data, target, thrs=0.1, agg_func='mean', critical=0.05, ks=True, frac=1):
    '''
    Finds interesting categorical features either by perfoming a Kolmogorov-Smirnov test or 
    simply be comparing the descriptive statistic of the full population versus the one obtained with the
    various subsets.
    '''
    cats = []
    tar_std = data[target].std()
    for col in data.select_dtypes(include=['object']).columns:
        counts = data[col].value_counts(dropna=False, 
                                        normalize=True)
        tmp = data.loc[data[col].isin(counts[counts > thrs].index),:]
        if ks:
            try:
                res = _ks_test(tmp, col, target, critical=critical)
            except ValueError as e:
                print(f'Column {col} throws the following error: {e}')
                continue
            if res:
                cats.append(col)
        else:
            res = tmp.groupby(col)[target].agg(agg_func).std()
            if res >= tar_std*frac:
                cats.append(col)    
    return cats


def segm_target(data, cat, target):
    '''
    Studies the target segmented by a categorical feature.
    It plots both a boxplot and a distplot for visual support
    '''
    df = data.groupby(cat)[target].agg(['count', 'mean', 'max', 
                                        'min', 'median', 'std'])
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    sns.boxplot(cat, target, data=data, ax=ax[0])
    for val in data[cat].unique():
        tmp = data[data[cat] == val]
        sns.kdeplot(tmp[target], linewidth=3, alpha=0.7,
                 label=val, ax=ax[1])  
    return df
