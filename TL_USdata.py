from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2
import pandas as pd
import sys
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras import optimizers, utils, initializers, regularizers
import keras.backend as K

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler #Importing the StandardScaler


US_df = pd.read_csv('US_Monthly_2011.csv')
# US_df = US_df.sort_values(by = ['rid'])
# print("Count values before the drop: ")
# print(US_df['rid'].value_counts())
# print(US_df.isnull().sum().sum())
US_df = US_df.dropna()
# print("Count values after the drop: ")
# print(US_df['rid'].value_counts())


Source_US_df = US_df.loc[US_df['rid'].isin(['3', '4', '5', '9'])]
Target_US_df = US_df.loc[US_df['rid'].isin(['1','2','6','7','8'])]

train_Target_US_df = Target_US_df.loc[Target_US_df['rid'].isin(['2','7','8'])]
test_Target_US_df = Target_US_df.loc[Target_US_df['rid'].isin(['1','6'])]

drop_rid = ['rid']
Source_US_df = Source_US_df.drop(drop_rid, axis =1)
train_Target_US_df = train_Target_US_df.drop(drop_rid, axis =1)
test_Target_US_df = test_Target_US_df.drop(drop_rid, axis =1)

target_column = ['pm25_value']
Source_US_df_y = Source_US_df[target_column]
Source_US_df_X = Source_US_df.drop(target_column, axis = 1)

train_Target_US_df_y = train_Target_US_df[target_column]
train_Target_US_df_X = train_Target_US_df.drop(target_column, axis = 1)

test_Target_US_df_y = test_Target_US_df[target_column]
test_Target_US_df_X = test_Target_US_df.drop(target_column, axis = 1)

TF_train_X = pd.concat([Source_US_df_X, train_Target_US_df_X], sort= False)
TF_train_y = pd.concat([Source_US_df_y, train_Target_US_df_y], sort= False)

np_TF_train_X = TF_train_X.to_numpy()
np_TF_train_y = TF_train_y.to_numpy()
np_test_Target_US_df_X = test_Target_US_df_X.to_numpy()
np_test_Target_US_df_y = test_Target_US_df_y.to_numpy()

np_TF_train_y_list = np_TF_train_y.ravel()
np_test_Target_US_df_y_list = np_test_Target_US_df_y.ravel()

print(Source_US_df_X.shape)
print(train_Target_US_df_X.shape)

sample_size = [5353, 2881]
n_estimators = 100
steps = 10
fold = 5
random_state = np.random.RandomState(1)

################################################TwoStageAdaBoostR2############################################################################
regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                      n_estimators = n_estimators, sample_size = sample_size,
                      steps = steps, fold = fold,
                      random_state = random_state)

regr_1.fit(np_TF_train_X, np_TF_train_y_list)
y_pred1 = regr_1.predict(np_test_Target_US_df_X)

mse_twostageboost = mean_squared_error(np_test_Target_US_df_y_list, y_pred1)
print("MSE of TwoStageTrAdaboostR2:", mse_twostageboost)

r2_score_twostageboost = r2_score(np_test_Target_US_df_y_list, y_pred1)
print("R^2 of TwoStageTrAdaboostR2:", r2_score_twostageboost)

###############################################Trying AdaBoost for regression####################################################################
print(train_peru_df_X.shape)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators = n_estimators)
regr_2.fit(train_Target_US_df_X, train_Target_US_df_y)
y_pred2 = regr_2.predict(test_Target_US_df_X)
mse_adaboost = mean_squared_error(test_Target_US_df_y, y_pred2)
print("MSE of regular AdaboostR2:", mse_adaboost)

r2_score_adaboost = r2_score(test_Target_US_df_y, y_pred2)
print("R^2 of regular AdaboostR2:", r2_score_adaboost)
