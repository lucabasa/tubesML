import tubesml as tml
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import pandas as pd

import string
import random


def create_data():
    df, target = make_classification(n_features=10)
    
    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < 10:
        random_names.append(''.join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1
        
    df = pd.DataFrame(df, columns=random_names)
    df['target'] = target
    
    return df

df = create_data()


@pytest.mark.parametrize("predict_proba", [True, False])
def test_cvscore(predict_proba):
    '''
    Test it works without warnings with both the normal prediction and the predict_proba
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('poly', tml.DfPolynomial()),
                     ('sca', tml.DfScaler(method='standard')),  
                     ('tarenc', tml.TargetEncoder()),
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])

    full_pipe = Pipeline([('pipe', pipe), 
                          ('logit', LogisticRegression(solver='lbfgs', multi_class='auto'))])

    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res, _ = tml.cv_score(df_1, y, full_pipe, cv=kfold, predict_proba=predict_proba)
    assert len(record) == 0
    assert len(res) == len(df_1)
    
    
def test_cvscore_nopipe():
    """
    Test if the function works without a pipeline
    """
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    kfold = KFold(n_splits=3)
    
    full_pipe = LogisticRegression(solver='lbfgs', multi_class='auto')
    
    with pytest.warns(None) as record:
        res, _ = tml.cv_score(df_1, y, full_pipe, cv=kfold, predict_proba=True)
    assert len(record) == 0
    assert len(res) == len(df_1)


@pytest.mark.parametrize('model', [XGBClassifier(use_label_encoder=False), LGBMClassifier()])
def test_earlystopping(model):
    '''
    Test early stopping for XGBoost and LGBM
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('poly', tml.DfPolynomial()),
                     ('sca', tml.DfScaler(method='standard')),  
                     ('tarenc', tml.TargetEncoder()),
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])

    full_pipe = Pipeline([('pipe', pipe), 
                          ('model', model)])

    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res, res_dict = tml.cv_score(df_1, y, full_pipe, cv=kfold, early_stopping=5, eval_metric='auc', imp_coef=True)
    assert len(record) == 0
    assert len(res) == len(df_1)
    assert len(res_dict['iterations']) == 3  # one per fold

    
@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs', multi_class='auto'), 
                                   DecisionTreeClassifier(), 
                                   XGBClassifier(use_label_encoder=False), 
                                   LGBMClassifier()])
def test_cvscore_coef_imp(model):
    '''
    Test coefficient and feature importances for a few models
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean')), 
                     ('poly', tml.DfPolynomial()),
                     ('sca', tml.DfScaler(method='standard')),  
                     ('tarenc', tml.TargetEncoder()),
                     ('dummify', tml.Dummify()), 
                     ('pca', tml.DfPCA(n_components=0.9, compress=True))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])

    full_pipe = Pipeline([('pipe', pipe), 
                          ('model', model)])

    kfold = KFold(n_splits=3)

    with pytest.warns(None) as record:
        res, coef = tml.cv_score(df_1, y, full_pipe, cv=kfold, imp_coef=True)
    assert len(record) == 0
    assert len(coef['feat_imp']) == df_1.shape[1]  * 2 + 45  # to account for the combinations


@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs', multi_class='auto'), 
                                   XGBClassifier(use_label_encoder=False), 
                                   LGBMClassifier()])   
def test_cvscore_nopipeline(model):
    '''
    Test cv score works for simple models, without being it a pipeline
    '''
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    kfold = KFold(n_splits=3)
    
    with pytest.warns(None) as record:
        res, coef = tml.cv_score(df_1, y, model, cv=kfold, imp_coef=True)
    assert len(record) == 0
    assert len(res) == len(df_1)
    assert len(coef['feat_imp']) == df_1.shape[1]
    

@pytest.mark.parametrize('model', [LogisticRegression(solver='lbfgs', multi_class='auto'), 
                                   DecisionTreeClassifier(), 
                                   XGBClassifier(use_label_encoder=False), 
                                   LGBMClassifier()])    
def test_cvscore_pdp(model):
    """
    Test partial dependence of a few models
    """
    y = df['target']
    df_1 = df.drop('target', axis=1)
    
    kfold = KFold(n_splits=3)
    
    pdp = df_1.columns[:3].to_list()
    
    pipe_transf = Pipeline([('fs', tml.DtypeSel(dtype='numeric')), 
                     ('imp', tml.DfImputer(strategy='mean'))])
    pipe = tml.FeatureUnionDf([('transf', pipe_transf)])

    full_pipe = Pipeline([('pipe', pipe), 
                          ('model', model)])
    
    with pytest.warns(None) as record:
        res, pdp_res = tml.cv_score(df_1, y, full_pipe, cv=kfold, pdp=pdp)
    assert len(record) == 0
    assert set(pdp_res['pdp']['feat']) == set(pdp)
    assert pdp_res['pdp']['mean'].notna().all()
    assert pdp_res['pdp']['std'].notna().all()

    
def test_make_test():
    with pytest.warns(None) as record:
        train, test = tml.make_test(df, 0.2, 452)
    assert len(record) == 0


def test_strat_test():
    df_1 = df.copy()
    df['cat'] = ['a']*50 + ['b']*50
    with pytest.warns(None) as record:
        train, test = tml.make_test(df, 0.2, 452, strat_feat='cat')
    assert len(record) == 0
    assert len(train[train['cat']=='a']) == len(train) / 2
    