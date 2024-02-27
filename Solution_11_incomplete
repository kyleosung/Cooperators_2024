
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch 
import torch.nn as nn
import torch.optim as optim
df = pd.read_csv("../STATS3DA3/McMaster Workshop/Claims_Years_1_to_3.csv")
df
df.dtypes
X = df.iloc[:,1:-1]
Y = df.iloc[:,-1]
X
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 100, test_size = 0.8, shuffle=True)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['pol_pay_freq','pol_payd','drv_sex1','drv_drv2','drv_sex2','vh_make_model','vh_fuel','vh_type']),
        ('num', StandardScaler(),['year','pol_no_claims_discount','pol_duration','drv_age1','drv_age_lic1','drv_age2','vh_age','vh_speed','vh_value','vh_weight','population','town_surface_area'])
    ]
)
preprocessor
X_train.dropna()
Y_train.dropna()
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
severity_model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('regressor', LinearRegression())
])
severity_model.fit(X_train,Y_train)
