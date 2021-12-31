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

######################################################## Automobile ################################################################
## horsepower column has correlation 0.4 :: [46 - 230] :: 30
#################################################################################################################################
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


dropcol_initial_auto = ['name']
autoData_df = pd.read_csv('UCI_regression/MPG/Auto.csv') ## horsepower column has correlation 0.4 :: [46 - 230] :: 30
autoData_df = autoData_df.drop(dropcol_initial_auto, axis = 1)
print("The shape of the Input data is: ", autoData_df.shape)

ss = StandardScaler()

drop_col_auto = ['horsepower']

auto_train_df = autoData_df.loc[(autoData_df['horsepower'] > 80) & (autoData_df['horsepower'] <= 110)]
auto_train_df = auto_train_df.drop(drop_col_auto, axis = 1)
auto_train_df = auto_train_df.reset_index(drop=True)
auto_train_df = clean_dataset(auto_train_df)
auto_train_df = auto_train_df.reset_index(drop=True)
print("Target Set: ",auto_train_df.shape)

auto_source1_df = autoData_df.loc[(autoData_df['horsepower'] > 110)]
auto_source1_df = auto_source1_df.drop(drop_col_auto, axis = 1)
auto_source1_df = auto_source1_df.reset_index(drop=True)
auto_source1_df = clean_dataset(auto_source1_df)
auto_source1_df = auto_source1_df.reset_index(drop=True)
print("Source Set 1: ",auto_source1_df.shape)

auto_source2_df = autoData_df.loc[(autoData_df['horsepower'] <= 80)]
auto_source2_df = auto_source2_df.drop(drop_col_auto, axis = 1)
auto_source2_df = auto_source2_df.reset_index(drop=True)
auto_source2_df = clean_dataset(auto_source2_df)
auto_source2_df = auto_source2_df.reset_index(drop=True)
print("Source Set 2: ",auto_source2_df.shape)

auto_source_df = pd.concat([auto_source1_df, auto_source2_df], ignore_index=True)
print("Final Source Set: ",auto_source_df.shape)

#################### Splitting into features and target ####################
target_column_auto = ['mpg']

auto_train_df_y = auto_train_df[target_column_auto]
auto_train_df_X = auto_train_df.drop(target_column_auto, axis = 1)
auto_cols = auto_train_df_X.columns
auto_train_df_X[auto_cols] = ss.fit_transform(auto_train_df_X[auto_cols])


auto_source_df_y = auto_source_df[target_column_auto]
auto_source_df_X = auto_source_df.drop(target_column_auto, axis = 1)
auto_cols = auto_source_df_X.columns
auto_source_df_X[auto_cols] = ss.fit_transform(auto_source_df_X[auto_cols])


########################### Transfer Learning auto #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_auto = []
rmselist_AdaTL_auto = []

r2scorelist_Ada_auto = []
rmselist_Ada_auto = []

r2scorelist_KMM_auto = []
rmselist_KMM_auto = []

r2scorelist_GBRTL_auto = []
rmselist_GBRTL_auto = []

r2scorelist_GBR_auto = []
rmselist_GBR_auto = []

r2scorelist_TwoTrAda_auto = []
rmselist_TwoTrAda_auto = []

r2scorelist_stradaboost_auto = []
rmselist_stradaboost_auto = []



kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(auto_train_df_X):
    ############### get data ###############
    auto_test_df_X, auto_tgt_df_X  = auto_train_df_X.iloc[train_ix], auto_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    auto_test_df_y, auto_tgt_df_y  = auto_train_df_y.iloc[train_ix], auto_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(auto_tgt_df_X.shape, auto_test_df_X.shape)

    ############### Merging the datasets ##########################################
    auto_X_df = pd.concat([auto_tgt_df_X, auto_source_df_X], ignore_index=True)
    auto_y_df = pd.concat([auto_tgt_df_y, auto_source_df_y], ignore_index=True)

    auto_np_train_X = auto_X_df.to_numpy()
    auto_np_train_y = auto_y_df.to_numpy()

    auto_np_test_X = auto_test_df_X.to_numpy()
    auto_np_test_y = auto_test_df_y.to_numpy()

    auto_np_train_y_list = auto_np_train_y.ravel()
    auto_np_test_y_list = auto_np_test_y.ravel()

    src_size_auto = len(auto_source_df_y)
    tgt_size_auto = len(auto_tgt_df_y)

    src_idx = np.arange(start = 0, stop=(src_size_auto - 1), step=1)
    tgt_idx = np.arange(start = src_size_auto, stop = ((src_size_auto + tgt_size_auto)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_auto = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_auto.fit(auto_np_train_X, auto_np_train_y_list)

    y_pred_AdaTL_auto = model_AdaTL_auto.predict(auto_np_test_X)

    mse_AdaTL_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_AdaTL_auto))
    rmselist_AdaTL_auto.append(mse_AdaTL_auto)

    r2_score_AdaTL_auto = pearsonr(auto_np_test_y_list, y_pred_AdaTL_auto)
    r2_score_AdaTL_auto = (r2_score_AdaTL_auto[0])**2
    r2scorelist_AdaTL_auto.append(r2_score_AdaTL_auto)


    ################### AdaBoost ###################
    model_Ada_auto = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_auto.fit(auto_tgt_df_X, auto_tgt_df_y)

    y_pred_Ada_auto = model_Ada_auto.predict(auto_np_test_X)

    mse_Ada_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_Ada_auto))
    rmselist_Ada_auto.append(mse_Ada_auto)

    r2_score_Ada_auto = pearsonr(auto_np_test_y_list, y_pred_Ada_auto)
    r2_score_Ada_auto = (r2_score_Ada_auto[0])**2
    r2scorelist_Ada_auto.append(r2_score_Ada_auto)


   ################### KMM ###################
    model_KMM_auto = KMM(get_estimator = get_estimator)
    model_KMM_auto.fit(auto_np_train_X, auto_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_auto = model_KMM_auto.predict(auto_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_KMM_auto))
    rmselist_KMM_auto.append(mse_KMM_auto)

    r2_score_KMM_auto = pearsonr(auto_np_test_y_list, y_pred_KMM_auto)
    r2_score_KMM_auto = (r2_score_KMM_auto[0])**2
    r2scorelist_KMM_auto.append(r2_score_KMM_auto)

    ################### GBRTL ###################
    model_GBRTL_auto = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_auto.fit(auto_np_train_X, auto_np_train_y_list)

    y_pred_GBRTL_auto = model_GBRTL_auto.predict(auto_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_GBRTL_auto))
    rmselist_GBRTL_auto.append(mse_GBRTL_auto)

    r2_score_GBRTL_auto = pearsonr(auto_np_test_y_list, y_pred_GBRTL_auto)
    r2_score_GBRTL_auto = (r2_score_GBRTL_auto[0])**2
    r2scorelist_GBRTL_auto.append(r2_score_GBRTL_auto)

    ################### GBR ###################
    model_GBR_auto = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBR_auto.fit(auto_tgt_df_X, auto_tgt_df_y)

    y_pred_GBR_auto = model_GBR_auto.predict(auto_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_GBR_auto))
    rmselist_GBR_auto.append(mse_GBR_auto)

    r2_score_GBR_auto = pearsonr(auto_np_test_y_list, y_pred_GBR_auto)
    r2_score_GBR_auto = (r2_score_GBR_auto[0])**2
    r2scorelist_GBR_auto.append(r2_score_GBR_auto)

    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_auto = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_auto.fit(auto_np_train_X, auto_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_auto = model_TwoTrAda_auto.predict(auto_np_test_X)

    mse_TwoTrAda_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_TwoTrAda_auto))
    rmselist_TwoTrAda_auto.append(mse_TwoTrAda_auto)

    r2_score_TwoTrAda_auto = pearsonr(auto_np_test_y_list, y_pred_TwoTrAda_auto)
    r2_score_TwoTrAda_auto = (r2_score_TwoTrAda_auto[0])**2
    r2scorelist_TwoTrAda_auto.append(r2_score_TwoTrAda_auto)

    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(auto_tgt_df_X), len(auto_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_auto = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_auto.fit(auto_np_train_X, auto_np_train_y_list)
    y_pred_stradaboost_auto = model_stradaboost_auto.predict(auto_np_test_X)


    mse_stradaboost_auto = sqrt(mean_squared_error(auto_np_test_y, y_pred_stradaboost_auto))
    rmselist_stradaboost_auto.append(mse_stradaboost_auto)

    r2_score_stradaboost_auto = pearsonr(auto_np_test_y_list, y_pred_stradaboost_auto)
    r2_score_stradaboost_auto = (r2_score_stradaboost_auto[0])**2
    r2scorelist_stradaboost_auto.append(r2_score_stradaboost_auto)


with open('auto_rmse.txt', 'w') as auto_handle_rmse:
    auto_handle_rmse.write("AdaBoost TL:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_auto)

    auto_handle_rmse.write("\n\nAdaBoost:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_auto)

    auto_handle_rmse.write("\n\nKMM:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_auto)

    auto_handle_rmse.write("\n\nGBRT:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_auto)

    auto_handle_rmse.write("\n\nGBR:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_auto)

    auto_handle_rmse.write("\n\nTrAdaBoost:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_auto)

    auto_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    auto_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_auto)


with open('auto_r2.txt', 'w') as auto_handle_r2:
    auto_handle_r2.write("AdaBoost TL:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_auto)

    auto_handle_r2.write("\n\nAdaBoost:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_auto)

    auto_handle_r2.write("\n\nKMM:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_auto)

    auto_handle_r2.write("\n\nGBRT:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_auto)

    auto_handle_r2.write("\n\nGBR:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_auto)

    auto_handle_r2.write("\n\nTrAdaBoost:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_auto)

    auto_handle_r2.write("\n\nSTrAdaBoost:\n ")
    auto_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_auto)



######################################################################################

print("-------------------------------------------")
