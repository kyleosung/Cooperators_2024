## IMPORTS
import autograd.numpy as np # import autograd wrapped numpy
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import scale

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm

from sklearn.impute import SimpleImputer


## ------------------------------------------------------
## Functions Below

def RMSE(x, y):
    MSE = ((y - x) ** 2).mean()
    return np.sqrt(MSE)


def load_claims():
    return pd.read_csv("./Qualification_Package/Claims_Years_1_to_3.csv")


def claims_preprocess_get_xy():
    claims = pd.read_csv("./Qualification_Package/Claims_Years_1_to_3.csv")
    claims['pol_pay_freq'] = claims['pol_pay_freq'].replace( {'Biannual': 2, 'Yearly': 1, 'Monthly': 12, 'Quarterly': 4} )
    claims['pol_payd'] = claims['pol_payd'].replace( {'No': 0, 'Yes': 1} )
    claims['drv_sex1'] = claims['drv_sex1'].replace( {'M': 1, 'F': 0} )
    claims['vh_type'] = claims['vh_type'].replace( {'Tourism': 1, 'Commercial': 0} )
    claims['drv_drv2'] = claims['drv_drv2'].replace( {'No': 0, 'Yes': 1} )

    claims['vh_make_model'] = claims['vh_make_model'].apply(hash)

    objects = claims.select_dtypes(['object'])
    categorical = claims.select_dtypes(['int64'])
    continuous = claims.select_dtypes(['float64'])

    objects_filled = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(objects)
    categorical_filled = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(categorical)
    continuous_filled = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(continuous)

    objects_filled = pd.DataFrame(objects_filled, columns=objects.columns.to_list())
    categorical_filled = pd.DataFrame(categorical_filled, columns=categorical.columns.to_list())
    continuous_filled = pd.DataFrame(continuous_filled, columns=continuous.columns.to_list())
    
    design_matrix = pd.get_dummies(objects_filled, columns=['pol_usage', 'drv_sex2', 'vh_fuel'], dtype=int)
    objects_design = design_matrix.drop(columns=['id_policy'])

    df = pd.concat([objects_design, categorical_filled, continuous_filled], axis = 1)

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return x, y


def load_tests():
    return pd.read_csv("./Qualification_Package/Submission_Data.csv")


def preprocess_xy(dataframe):
    claims = dataframe.copy(deep=True)
    claims['pol_pay_freq'] = claims['pol_pay_freq'].replace( {'Biannual': 2, 'Yearly': 1, 'Monthly': 12, 'Quarterly': 4} )
    claims['pol_payd'] = claims['pol_payd'].replace( {'No': 0, 'Yes': 1} )
    claims['drv_sex1'] = claims['drv_sex1'].replace( {'M': 1, 'F': 0} )
    claims['vh_type'] = claims['vh_type'].replace( {'Tourism': 1, 'Commercial': 0} )
    claims['drv_drv2'] = claims['drv_drv2'].replace( {'No': 0, 'Yes': 1} )

    claims['vh_make_model'] = claims['vh_make_model'].apply(hash)

    objects = claims.select_dtypes(['object'])
    categorical = claims.select_dtypes(['int64'])
    continuous = claims.select_dtypes(['float64'])

    objects_filled = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(objects)
    categorical_filled = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(categorical)
    continuous_filled = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(continuous)

    objects_filled = pd.DataFrame(objects_filled, columns=objects.columns.to_list())
    categorical_filled = pd.DataFrame(categorical_filled, columns=categorical.columns.to_list())
    continuous_filled = pd.DataFrame(continuous_filled, columns=continuous.columns.to_list())
    
    design_matrix = pd.get_dummies(objects_filled, columns=['pol_usage', 'drv_sex2', 'vh_fuel'], dtype=int)
    objects_design = design_matrix.drop(columns=['id_policy'])

    try:
        categorical_filled = categorical_filled.drop(columns=['Unnamed: 0'])
    except:
        pass

    df = pd.concat([objects_design, categorical_filled, continuous_filled], axis = 1)

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y, 
        train_size = 0.8,
        test_size = 0.2, # train is 75%, test is 25% 
        random_state = 0, # stratify = y,
    )
    return x_train, x_test, y_train, y_test




def preprocess_x(dataframe):
    claims = dataframe.copy(deep=True)
    claims['pol_pay_freq'] = claims['pol_pay_freq'].replace( {'Biannual': 2, 'Yearly': 1, 'Monthly': 12, 'Quarterly': 4} )
    claims['pol_payd'] = claims['pol_payd'].replace( {'No': 0, 'Yes': 1} )
    claims['drv_sex1'] = claims['drv_sex1'].replace( {'M': 1, 'F': 0} )
    claims['vh_type'] = claims['vh_type'].replace( {'Tourism': 1, 'Commercial': 0} )
    claims['drv_drv2'] = claims['drv_drv2'].replace( {'No': 0, 'Yes': 1} )
    
    claims['vh_make_model'] = claims['vh_make_model'].apply(hash)

    objects = claims.select_dtypes(['object'])
    categorical = claims.select_dtypes(['int64'])
    continuous = claims.select_dtypes(['float64'])

    objects_filled = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(objects)
    categorical_filled = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(categorical)
    continuous_filled = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(continuous)

    objects_filled = pd.DataFrame(objects_filled, columns=objects.columns.to_list())
    categorical_filled = pd.DataFrame(categorical_filled, columns=categorical.columns.to_list())
    continuous_filled = pd.DataFrame(continuous_filled, columns=continuous.columns.to_list())

    design_matrix = pd.get_dummies(objects_filled, columns=['pol_usage', 'drv_sex2', 'vh_fuel'], dtype=int)

    ids = design_matrix[['id_policy']]
    objects_design = design_matrix.drop(columns=['id_policy'])

    try:
        categorical_filled = categorical_filled.drop(columns=['Unnamed: 0'])
    except:
        pass

    df = pd.concat([objects_design, categorical_filled, continuous_filled], axis = 1)

    return df, ids


def evaluate_scratch(claims, model):
    x_train, x_test, y_train, y_test = preprocess_xy(claims)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))



def predict(x, model):
    x_new, ids = preprocess_x(x)

    ret = pd.concat([ids, pd.DataFrame(model.predict(x_new), columns=['claim_amount'])], axis = 1)

    return ret



def test_save_model(model, filename):
    claims = load_claims()
    x_train, x_test, y_train, y_test = preprocess_xy(claims)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Training RMSE: {RMSE(y_pred, y_test)}')
    predictions = predict(load_tests(), model)
    predictions.to_csv(f'models/{filename}')