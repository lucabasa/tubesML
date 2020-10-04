from .context import source
import pytest
import pandas as pd

    
def create_data():
    df = pd.DataFrame({'a': [3]*3, 'b': [2]*3})
    return df

def test_clean():
    df = create_data()
    imputer = source.DfImputer()
    res = imputer.fit_transform(df)

    