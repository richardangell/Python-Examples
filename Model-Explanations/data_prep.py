import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston



def get_boston_df():
    '''Load boston data from sklearn and return explanatory variables and response
    in single pd.DataFrame.'''

    boston = load_boston()

    data = pd.DataFrame(
        boston['data'], 
        columns = boston['feature_names']
    )

    data['target'] = boston['target']

    return data



def get_boston_xgb(base_margin = 0):
    '''Load boston data from sklearn and return explanatory variables and response
    in xgboost.DMatrix.'''

    boston = load_boston()

    data = xgb.DMatrix(
        data = boston['data'], 
        label = boston['target'], 
        feature_names = boston['feature_names']
    )

    data.set_base_margin([base_margin] * boston['data'].shape[0])

    return data







