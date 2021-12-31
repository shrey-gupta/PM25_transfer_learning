# from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ##STrAdaBoost.R2
# from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 ##two-stage TrAdaBoost.R2

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
# import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold
import matplotlib.lines as mlines

import statistics

from scipy.stats import rv_continuous
from scipy.stats import *

from statistics import mean
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.model_selection import KFold


######### Instance Transfer repositories ####################
from adapt.instance_based import TwoStageTrAdaBoostR2

print("Repositories uploaded!!")

from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.model_selection import GridSearchCV
from adapt.instance_based import KMM

print("Second Upload Completed!!")

################################### Housing ################################################################
## 'nox' found to be correlated at 0.4 :: [0.385 - 0.871] :: 50
#################################################################################################################################
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

housingData_df = pd.read_csv('UCI_regression/BostonHousing/BostonHousing.csv')
print("Housing Data")
print(housingData_df.shape)

ss = StandardScaler()

drop_col_housing = ['nox']
housing_train_df = housingData_df.loc[(housingData_df['nox'] > 0.475) & (housingData_df['nox'] <= 0.600)]
housing_train_df = housing_train_df.drop(drop_col_housing, axis = 1)
housing_train_df = housing_train_df.reset_index(drop=True)
housing_train_df = clean_dataset(housing_train_df)
housing_train_df = housing_train_df.reset_index(drop=True)
print("Target Set: ",housing_train_df.shape)

housing_source1_df = housingData_df.loc[(housingData_df['nox'] > 0.600)]
housing_source1_df = housing_source1_df.drop(drop_col_housing, axis = 1)
housing_source1_df = housing_source1_df.reset_index(drop=True)
housing_source1_df = clean_dataset(housing_source1_df)
housing_source1_df = housing_source1_df.reset_index(drop=True)
print("Source Set 1: ",housing_source1_df.shape)

housing_source2_df = housingData_df.loc[(housingData_df['nox'] <= 0.475)]
housing_source2_df = housing_source2_df.drop(drop_col_housing, axis = 1)
housing_source2_df = housing_source2_df.reset_index(drop=True)
housing_source2_df = clean_dataset(housing_source2_df)
housing_source2_df = housing_source2_df.reset_index(drop=True)
print("Source Set 2: ",housing_source2_df.shape)

housing_source_df = pd.concat([housing_source1_df, housing_source2_df], ignore_index=True)
print("Final Source Set: ",housing_source_df.shape)


#################### Splitting into features and target ####################
target_column_housing = ['medv']

housing_train_df_y = housing_train_df[target_column_housing]
housing_train_df_X = housing_train_df.drop(target_column_housing, axis = 1)
housing_cols = housing_train_df_X.columns
housing_train_df_X[housing_cols] = ss.fit_transform(housing_train_df_X[housing_cols])

housing_source_df_y = housing_source_df[target_column_housing]
housing_source_df_X = housing_source_df.drop(target_column_housing, axis = 1)
housing_cols = housing_source_df_X.columns
housing_source_df_X[housing_cols] = ss.fit_transform(housing_source_df_X[housing_cols])

print("---------------------------")

########################### Transfer Learning Housing #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_housing = []
rmselist_AdaTL_housing = []

r2scorelist_Ada_housing = []
rmselist_Ada_housing = []

r2scorelist_KMM_housing = []
rmselist_KMM_housing = []

r2scorelist_GBRTL_housing = []
rmselist_GBRTL_housing = []

r2scorelist_GBR_housing = []
rmselist_GBR_housing = []

r2scorelist_TwoTrAda_housing = []
rmselist_TwoTrAda_housing = []

r2scorelist_stradaboost_housing = []
rmselist_stradaboost_housing = []



kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(housing_train_df_X):
    ############### get data ###############
    housing_test_df_X, housing_tgt_df_X  = housing_train_df_X.iloc[train_ix], housing_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    housing_test_df_y, housing_tgt_df_y  = housing_train_df_y.iloc[train_ix], housing_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(housing_tgt_df_X.shape, housing_test_df_X.shape)

    ############### Merging the datasets ##########################################
    housing_X_df = pd.concat([housing_tgt_df_X, housing_source_df_X], ignore_index=True)
    housing_y_df = pd.concat([housing_tgt_df_y, housing_source_df_y], ignore_index=True)

    housing_np_train_X = housing_X_df.to_numpy()
    housing_np_train_y = housing_y_df.to_numpy()

    housing_np_test_X = housing_test_df_X.to_numpy()
    housing_np_test_y = housing_test_df_y.to_numpy()

    housing_np_train_y_list = housing_np_train_y.ravel()
    housing_np_test_y_list = housing_np_test_y.ravel()

    src_size_housing = len(housing_source_df_y)
    tgt_size_housing = len(housing_tgt_df_y)

    src_idx = np.arange(start = 0, stop=(src_size_housing - 1), step=1)
    tgt_idx = np.arange(start = src_size_housing, stop = ((src_size_housing + tgt_size_housing)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_housing = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_housing.fit(housing_np_train_X, housing_np_train_y_list)

    y_pred_AdaTL_housing = model_AdaTL_housing.predict(housing_np_test_X)

    mse_AdaTL_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_AdaTL_housing))
    rmselist_AdaTL_housing.append(mse_AdaTL_housing)

    r2_score_AdaTL_housing = pearsonr(housing_np_test_y_list, y_pred_AdaTL_housing)
    r2_score_AdaTL_housing = (r2_score_AdaTL_housing[0])**2
    r2scorelist_AdaTL_housing.append(r2_score_AdaTL_housing)


    ################### AdaBoost ###################
    model_Ada_housing = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_housing.fit(housing_tgt_df_X, housing_tgt_df_y)

    y_pred_Ada_housing = model_Ada_housing.predict(housing_np_test_X)

    mse_Ada_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_Ada_housing))
    rmselist_Ada_housing.append(mse_Ada_housing)

    r2_score_Ada_housing = pearsonr(housing_np_test_y_list, y_pred_Ada_housing)
    r2_score_Ada_housing = (r2_score_Ada_housing[0])**2
    r2scorelist_Ada_housing.append(r2_score_Ada_housing)


   ################### KMM ###################
    model_KMM_housing = KMM(get_estimator = get_estimator)
    model_KMM_housing.fit(housing_np_train_X, housing_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_housing = model_KMM_housing.predict(housing_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_KMM_housing))
    rmselist_KMM_housing.append(mse_KMM_housing)

    r2_score_KMM_housing = pearsonr(housing_np_test_y_list, y_pred_KMM_housing)
    r2_score_KMM_housing = (r2_score_KMM_housing[0])**2
    r2scorelist_KMM_housing.append(r2_score_KMM_housing)


    ################### GBRTL ###################
    model_GBRTL_housing = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBRTL_housing.fit(housing_np_train_X, housing_np_train_y_list)

    y_pred_GBRTL_housing = model_GBRTL_housing.predict(housing_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_GBRTL_housing))
    rmselist_GBRTL_housing.append(mse_GBRTL_housing)

    r2_score_GBRTL_housing = pearsonr(housing_np_test_y_list, y_pred_GBRTL_housing)
    r2_score_GBRTL_housing = (r2_score_GBRTL_housing[0])**2
    r2scorelist_GBRTL_housing.append(r2_score_GBRTL_housing)


    ################### GBR ###################
    model_GBR_housing = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBR_housing.fit(housing_tgt_df_X, housing_tgt_df_y)

    y_pred_GBR_housing = model_GBR_housing.predict(housing_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_GBR_housing))
    rmselist_GBR_housing.append(mse_GBR_housing)

    r2_score_GBR_housing = pearsonr(housing_np_test_y_list, y_pred_GBR_housing)
    r2_score_GBR_housing = (r2_score_GBR_housing[0])**2
    r2scorelist_GBR_housing.append(r2_score_GBR_housing)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_housing = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100 , cv = 10) #, kwargs_TwoTrAda)
    model_TwoTrAda_housing.fit(housing_np_train_X, housing_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_housing = model_TwoTrAda_housing.predict(housing_np_test_X)

    mse_TwoTrAda_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_TwoTrAda_housing))
    rmselist_TwoTrAda_housing.append(mse_TwoTrAda_housing)

    r2_score_TwoTrAda_housing = pearsonr(housing_np_test_y_list, y_pred_TwoTrAda_housing)
    r2_score_TwoTrAda_housing = (r2_score_TwoTrAda_housing[0])**2
    r2scorelist_TwoTrAda_housing.append(r2_score_TwoTrAda_housing)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(housing_tgt_df_X), len(housing_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_housing = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_housing.fit(housing_np_train_X, housing_np_train_y_list)
    y_pred_stradaboost_housing = model_stradaboost_housing.predict(housing_np_test_X)


    mse_stradaboost_housing = sqrt(mean_squared_error(housing_np_test_y, y_pred_stradaboost_housing))
    rmselist_stradaboost_housing.append(mse_stradaboost_housing)

    r2_score_stradaboost_housing = pearsonr(housing_np_test_y_list, y_pred_stradaboost_housing)
    r2_score_stradaboost_housing = (r2_score_stradaboost_housing[0])**2
    r2scorelist_stradaboost_housing.append(r2_score_stradaboost_housing)


with open('housing_rmse.txt', 'w') as housing_handle_rmse:
    housing_handle_rmse.write("AdaBoost TL:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_housing)

    housing_handle_rmse.write("\n\nAdaBoost:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_housing)

    housing_handle_rmse.write("\n\nKMM:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_housing)

    housing_handle_rmse.write("\n\nGBRT:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_housing)

    housing_handle_rmse.write("\n\nGBR:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_housing)

    housing_handle_rmse.write("\n\nTrAdaBoost:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_housing)

    housing_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    housing_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_housing)


with open('housing_r2.txt', 'w') as housing_handle_r2:
    housing_handle_r2.write("AdaBoost TL:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_housing)

    housing_handle_r2.write("\n\nAdaBoost:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_housing)

    housing_handle_r2.write("\n\nKMM:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_housing)

    housing_handle_r2.write("\n\nGBRT:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_housing)

    housing_handle_r2.write("\n\nGBR:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_housing)

    housing_handle_r2.write("\n\nTrAdaBoost:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_housing)

    housing_handle_r2.write("\n\nSTrAdaBoost:\n ")
    housing_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_housing)


######################################################################################

print("-------------------------------------------")
