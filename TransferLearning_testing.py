from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2
# from newtwoStage_TrAdaBoostR2 import TradaboostRegressor
# from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2

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

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt

#Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold


sns.set_style("darkgrid")

########################################################################################################################################################
ConcreteData_df = pd.read_excel('UCI_regression/Concrete_Data.xls')
# # HousingData_df = pd.read_csv('UCI_regression/BostonHousing/BostonHousing.csv') ## 'nox' found to be correlated at 0.4 :: [0.385 - 0.871] :: 50
# dropcol_initial = ['name']
# AutoData_df = pd.read_csv('UCI_regression/MPG/Auto.csv') ## horsepower column has correlation 0.4 :: [46 - 230] :: 30
# AutoData_df = AutoData_df.drop(dropcol_initial, axis = 1)

# print("The shape of the Input data is: ", AutoData_df.shape)

## Preprocessing Data
## Finding the correlation first
# print("The correlation matrix is: ")
# print(ConcreteData_df.corr())
#
# ## Sorting datframe according to the moderately correalted.
# col = ConcreteData_df['Cement']
# ConcreteData_df = ConcreteData_df.sort_values(by=['Cement'])

# drop_col = ['Cement (component 1)(kg in a m^3 mixture)']
#
# Source_df = ConcreteData_df.loc[(ConcreteData_df['Cement (component 1)(kg in a m^3 mixture)'] < 250)]
# Source_df = Source_df.drop(drop_col, axis = 1)
#
# # print(Source_df.shape)
#
# Target_df = ConcreteData_df.loc[(ConcreteData_df['Cement (component 1)(kg in a m^3 mixture)'] > 250) & (ConcreteData_df['Cement (component 1)(kg in a m^3 mixture)'] < 350)]
# Target_df = Target_df.drop(drop_col, axis = 1)
#
# Test_df = ConcreteData_df.loc[(ConcreteData_df['Cement (component 1)(kg in a m^3 mixture)'] > 350)]
# Test_df = Test_df.drop(drop_col, axis = 1)
# print(Test_df.shape)
# #ConcreteData_df[(col < 350) & (col > 225)].count()
#
# target_column = ['Concrete compressive strength(MPa, megapascals) ']
# Source_df_y = Source_df[target_column]
# Source_df_X = Source_df.drop(target_column, axis = 1)
#
# Target_df_y = Target_df[target_column]
# Target_df_X = Target_df.drop(target_column, axis = 1)
#
# Test_df_y = Test_df[target_column]
# Test_df_X = Test_df.drop(target_column, axis = 1)
#
# # TF_Source_X = Source_df_X.to_numpy()
# # TF_Source_y = Source_df_y.to_numpy()
#
# TF_train_X = pd.concat([Target_df_X, Source_df_X], sort= False)
# TF_train_y = pd.concat([Target_df_y, Source_df_y], sort= False)
#
# np_TF_train_X = TF_train_X.to_numpy()
# np_TF_train_y = TF_train_y.to_numpy()
#
# np_TF_test_X = Test_df_X.to_numpy()
# np_TF_test_y = Test_df_y.to_numpy()
#
# np_TF_train_y_list = np_TF_train_y.ravel()
# np_TF_test_y_list = np_TF_test_y.ravel()

################################################################################################################
ConcreteData_df = pd.read_excel('UCI_regression/Concrete_Data.xls')
target_column = ['ConcreteCompressiveStrength']
ConcreteData_df_y = ConcreteData_df[target_column]
ConcreteData_df_X = ConcreteData_df.drop(target_column, axis = 1)

X_train, X_source, y_train, y_source = train_test_split(ConcreteData_df_X, ConcreteData_df_y, test_size = 0.50, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.90, random_state=1)

print(X_source.shape)
print(X_train.shape)
print(X_test.shape)

X_df = pd.concat([X_train, X_source], ignore_index=True)
y_df = pd.concat([y_train, y_source], ignore_index=True)

np_train_X = X_df.to_numpy()
np_train_y = y_df.to_numpy()

# np_train_X = X_train.to_numpy()
# np_train_y = y_train.to_numpy()

np_test_X = X_test.to_numpy()
np_test_y = y_test.to_numpy()

np_train_y_list = np_train_y.ravel()
np_test_y_list = np_test_y.ravel()

sample_size = [len(X_train), len(X_source)]
n_estimators = 100
steps = 30
fold = 20
random_state = np.random.RandomState(1)

######################################################################################################################
# drop_col = ['Cement']
#
# Train_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 225) & (ConcreteData_df['Cement'] <= 350)]
# # Train_df = HousingData_df.loc[(HousingData_df['nox'] > 0.475) & (HousingData_df['nox'] <= 0.600)]
# # Train_df = AutoData_df.loc[(AutoData_df['horsepower'] <= 80)]
# Train_df = Train_df.drop(drop_col, axis = 1)
# print(Train_df.shape)
#
# Source_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] <= 225)]
# # Source_df = HousingData_df.loc[(HousingData_df['nox'] <= 0.475)]
# # Source_df = AutoData_df.loc[(AutoData_df['horsepower'] > 110)]
# Source_df = Source_df.drop(drop_col, axis = 1)
# print(Source_df.shape)
#
# Test_df = ConcreteData_df.loc[(ConcreteData_df['Cement'] > 350)]
# # Test_df = HousingData_df.loc[(HousingData_df['nox'] > 0.600)]
# # Test_df = AutoData_df.loc[(AutoData_df['horsepower'] > 80) & (AutoData_df['horsepower'] <= 110)]
# Test_df = Test_df.drop(drop_col, axis = 1)
# print(Test_df.shape)

###################################################################################################################
# target_column = ['ConcreteCompressiveStrength']
# # target_column = ['medv']
# # target_column = ['mpg']
#
# Train_df_y = Train_df[target_column]
# Train_df_X = Train_df.drop(target_column, axis = 1)
#
# Test_df_y = Test_df[target_column]
# Test_df_X = Test_df.drop(target_column, axis = 1)
#
# Source_df_y = Source_df[target_column]
# Source_df_X = Source_df.drop(target_column, axis = 1)

## Merging the datasets
# X_df = pd.concat([Train_df_X, Test_df_X], ignore_index=True)
# y_df = pd.concat([Train_df_y, Test_df_y], ignore_index=True)
#
# np_train_X = X_df.to_numpy()
# np_train_y = y_df.to_numpy()
#
# X_df = Train_df_X
# y_df = Train_df_y
#
# np_train_X = X_df.to_numpy()
# np_train_y = y_df.to_numpy()
#
# np_source_X = Source_df_X.to_numpy()
# np_source_y = Source_df_y.to_numpy()
#
# np_train_y_list = np_train_y.ravel()

# # sample_size = [len(Target_df_X), len(Source_df_X)]
# n_estimators = 100
# steps = 30
# fold = 20
# random_state = np.random.RandomState(1)

################################################TwoStageAdaBoostR2############################################################################

# kf = KFold(n_splits = 2) ## Create no. of CV Folds
# error = []
#
# for train_idx, test_idx in kf.split(X_df):
#     X_train, X_test = np_train_X[train_idx], np_train_X[test_idx]
#     y_train, y_test = np_train_y[train_idx], np_train_y[test_idx]
#
#     sample_size = [len(Source_df_X), len(X_train)]
#
#     X_train_new = np.concatenate((np_source_X, X_train))
#     y_train_new = np.concatenate((np_source_y, y_train))
#
#     np_y_train_new_list = y_train_new.ravel()
#
#     regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
#                       n_estimators = n_estimators, sample_size = sample_size,
#                       steps = steps, fold = fold,
#                       random_state = random_state)
#     regr_1.fit(X_train_new, np_y_train_new_list)
#     y_pred2 = regr_1.predict(X_test)
#     mse_adaboost = sqrt(mean_squared_error(y_test, y_pred2))
#     print("RMSE of regular AdaboostR2:", mse_adaboost)
#     error.append(mse_adaboost)
#
# print("Mean RMSE: ", sum(error)/len(error))


# regr_1.fit(np_TF_train_X, np_TF_train_y_list)
# y_pred1 = regr_1.predict(np_TF_test_X)
#
#
# mse_twostageboost = sqrt(mean_squared_error(np_TF_test_y_list, y_pred1))
# print("RMSE of TwoStageTrAdaboostR2:", mse_twostageboost)
#
# # r2_score_twostageboost = r2_score(np_test_peru_df_y_list, y_pred1)
# r2_score_twostageboost_values = pearsonr(np_TF_test_y_list, y_pred1)
# r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
# print("R^2 of TwoStageTrAdaboostR2:", r2_score_twostageboost)

###############################################Trying AdaBoost for regression####################################################################

# kf = KFold(n_splits = 10) ## Create no. of CV Folds
# error = []
#
# for train_idx, test_idx in kf.split(X_df):
#     X_train, X_test = np_train_X[train_idx], np_train_X[test_idx]
#     y_train, y_test = np_train_y[train_idx], np_train_y[test_idx]
#
#     regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = n_estimators)
#     regr_2.fit(X_train, y_train)
#     y_pred2 = regr_2.predict(X_test)
#
#     mse_adaboost = sqrt(mean_squared_error(y_test, y_pred2))
#     print("RMSE of regular AdaboostR2:", mse_adaboost)
#     error.append(mse_adaboost)
#
# print("Mean RMSE: ", sum(error)/len(error))
# # # r2_score_adaboost = r2_score(test_peru_df_y, y_pred2)
# # r2_score_adaboost_values = pearsonr(np_TF_test_y_list, y_pred2)
# # r2_score_adaboost = (r2_score_adaboost_values[0])**2
# # print("R^2 of AdaboostR2:", r2_score_adaboost)

#####################################################################################################################################################
error = []
for x in range(0, 10):

    # regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = n_estimators)
    # regr_2.fit(X_train, y_train)
    # y_pred2 = regr_2.predict(X_test)
    #
    # mse_adaboost = sqrt(mean_squared_error(y_test, y_pred2))
    # print("RMSE of regular AdaboostR2:", mse_adaboost)
    # error.append(mse_adaboost)

    regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                      n_estimators = n_estimators, sample_size = sample_size,
                      steps = steps, fold = fold,
                      random_state = random_state)
    regr_1.fit(np_train_X, np_train_y_list)
    y_pred2 = regr_1.predict(np_test_X)
    mse_adaboost = sqrt(mean_squared_error(np_test_y_list, y_pred2))
    print("RMSE of regular AdaboostR2:", mse_adaboost)
    error.append(mse_adaboost)

print("Mean RMSE: ", sum(error)/len(error))
# r2_score_adaboost = r2_score(test_peru_df_y, y_pred2)
# r2_score_adaboost_values = pearsonr(np_TF_test_y_list, y_pred2)
# r2_score_adaboost = (r2_score_adaboost_values[0])**2
# print("R^2 of AdaboostR2:", r2_score_adaboost)
