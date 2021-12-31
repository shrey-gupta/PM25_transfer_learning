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
###################################### elevators ###############################################
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame) #"df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

ss = StandardScaler()

target_elevators = ['Goal']
colnames_elevators = ['climbRate', 'Sgz', 'p', 'q', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SaTime1', 'SaTime2',
                      'SaTime3', 'SaTime4', 'diffSaTime1', 'diffSaTime2', 'diffSaTime3', 'diffSaTime4', 'Sa', 'Goal']
elevators_train_df = pd.read_csv('UCI_regression/Elevators/elevators.data', header = None, names = colnames_elevators)
print("Elevators Data")
print(elevators_train_df.shape)

elevators_test_df = pd.read_csv('UCI_regression/Elevators/elevators.test', header = None, names = colnames_elevators)
elevators_test_df = clean_dataset(elevators_test_df)
elevators_test_df = elevators_test_df.reset_index(drop=True)
print("Test Dataset", elevators_test_df.shape)

#################### Splitting with small target set and large source and test set #############
elevators_source_df, elevators_tgt_df = train_test_split(elevators_train_df, test_size = 0.05) ## test_size = tgt size
# print(elevators_df_tgt.shape, elevators_df_source.shape, elevators_df_test.shape)

elevators_tgt_df = elevators_tgt_df.reset_index(drop = True)
elevators_tgt_df = clean_dataset(elevators_tgt_df)
elevators_tgt_df = elevators_tgt_df.reset_index(drop=True)
print("Target Set: ", elevators_tgt_df.shape)

elevators_source_df = elevators_source_df.reset_index(drop = True)
elevators_source_df = clean_dataset(elevators_source_df)
elevators_source_df = elevators_source_df.reset_index(drop=True)
print("Source Set: ", elevators_source_df.shape)

#################### Seperate into features and target ##########################
elevators_test_df_y = elevators_test_df[target_elevators]
elevators_test_df_X = elevators_test_df.drop(target_elevators, axis = 1)
elevators_cols = elevators_test_df_X.columns
elevators_test_df_X[elevators_cols] = ss.fit_transform(elevators_test_df_X[elevators_cols])


elevators_tgt_df_y = elevators_tgt_df[target_elevators]
elevators_tgt_df_X = elevators_tgt_df.drop(target_elevators, axis = 1)
elevators_cols = elevators_tgt_df_X.columns
elevators_tgt_df_X[elevators_cols] = ss.fit_transform(elevators_tgt_df_X[elevators_cols])


elevators_source_df_y = elevators_source_df[target_elevators]
elevators_source_df_X = elevators_source_df.drop(target_elevators, axis = 1)
elevators_cols = elevators_source_df_X.columns
elevators_source_df_X[elevators_cols] = ss.fit_transform(elevators_source_df_X[elevators_cols])


############################## Merging the datasets ##########################################
elevators_X_df = pd.concat([elevators_tgt_df_X, elevators_source_df_X], ignore_index=True)
elevators_y_df = pd.concat([elevators_tgt_df_y, elevators_source_df_y], ignore_index=True)

elevators_np_train_X = elevators_X_df.to_numpy()
elevators_np_train_y = elevators_y_df.to_numpy()

elevators_np_test_X = elevators_test_df_X.to_numpy()
elevators_np_test_y = elevators_test_df_y.to_numpy()

elevators_np_train_y_list = elevators_np_train_y.ravel()
elevators_np_test_y_list = elevators_np_test_y.ravel()

src_size_elevators = len(elevators_source_df_y)
tgt_size_elevators = len(elevators_tgt_df_y)

src_idx = np.arange(start=0, stop=(src_size_elevators - 1), step=1)
tgt_idx = np.arange(start=src_size_elevators, stop=((src_size_elevators + tgt_size_elevators)-1), step=1)


print("---------------------------")

########################### Transfer Learning elevators #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_elevators = []
rmselist_AdaTL_elevators = []

r2scorelist_Ada_elevators = []
rmselist_Ada_elevators = []

r2scorelist_KMM_elevators = []
rmselist_KMM_elevators = []

r2scorelist_GBRTL_elevators = []
rmselist_GBRTL_elevators = []

r2scorelist_GBR_elevators = []
rmselist_GBR_elevators = []

r2scorelist_TwoTrAda_elevators = []
rmselist_TwoTrAda_elevators = []

r2scorelist_stradaboost_elevators = []
rmselist_stradaboost_elevators = []



kfold = KFold(n_splits = 10, random_state = 42, shuffle = False)

for x in range(0, 10):
    ################### AdaBoost Tl ###################
    model_AdaTL_elevators = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_elevators.fit(elevators_np_train_X, elevators_np_train_y_list)

    y_pred_AdaTL_elevators = model_AdaTL_elevators.predict(elevators_np_test_X)

    mse_AdaTL_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_AdaTL_elevators))
    rmselist_AdaTL_elevators.append(mse_AdaTL_elevators)

    r2_score_AdaTL_elevators = pearsonr(elevators_np_test_y_list, y_pred_AdaTL_elevators)
    r2_score_AdaTL_elevators = (r2_score_AdaTL_elevators[0])**2
    r2scorelist_AdaTL_elevators.append(r2_score_AdaTL_elevators)


    ################### AdaBoost ###################
    model_Ada_elevators = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_elevators.fit(elevators_tgt_df_X, elevators_tgt_df_y)

    y_pred_Ada_elevators = model_Ada_elevators.predict(elevators_np_test_X)

    mse_Ada_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_Ada_elevators))
    rmselist_Ada_elevators.append(mse_Ada_elevators)

    r2_score_Ada_elevators = pearsonr(elevators_np_test_y_list, y_pred_Ada_elevators)
    r2_score_Ada_elevators = (r2_score_Ada_elevators[0])**2
    r2scorelist_Ada_elevators.append(r2_score_Ada_elevators)


   ################### KMM ###################
    model_KMM_elevators = KMM(get_estimator = get_estimator)
    model_KMM_elevators.fit(elevators_np_train_X, elevators_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_elevators = model_KMM_elevators.predict(elevators_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_KMM_elevators))
    rmselist_KMM_elevators.append(mse_KMM_elevators)

    r2_score_KMM_elevators = pearsonr(elevators_np_test_y_list, y_pred_KMM_elevators)
    r2_score_KMM_elevators = (r2_score_KMM_elevators[0])**2
    r2scorelist_KMM_elevators.append(r2_score_KMM_elevators)

    ################### GBRTL ###################
    model_GBRTL_elevators = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_elevators.fit(elevators_np_train_X, elevators_np_train_y_list)

    y_pred_GBRTL_elevators = model_GBRTL_elevators.predict(elevators_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_GBRTL_elevators))
    rmselist_GBRTL_elevators.append(mse_GBRTL_elevators)

    r2_score_GBRTL_elevators = pearsonr(elevators_np_test_y_list, y_pred_GBRTL_elevators)
    r2_score_GBRTL_elevators = (r2_score_GBRTL_elevators[0])**2
    r2scorelist_GBRTL_elevators.append(r2_score_GBRTL_elevators)

    ################### GBR ###################
    model_GBR_elevators = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBR_elevators.fit(elevators_tgt_df_X, elevators_tgt_df_y)

    y_pred_GBR_elevators = model_GBR_elevators.predict(elevators_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_GBR_elevators))
    rmselist_GBR_elevators.append(mse_GBR_elevators)

    r2_score_GBR_elevators = pearsonr(elevators_np_test_y_list, y_pred_GBR_elevators)
    r2_score_GBR_elevators = (r2_score_GBR_elevators[0])**2
    r2scorelist_GBR_elevators.append(r2_score_GBR_elevators)

    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_elevators = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv = 10) #, kwargs_TwoTrAda)
    model_TwoTrAda_elevators.fit(elevators_np_train_X, elevators_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_elevators = model_TwoTrAda_elevators.predict(elevators_np_test_X)

    mse_TwoTrAda_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_TwoTrAda_elevators))
    rmselist_TwoTrAda_elevators.append(mse_TwoTrAda_elevators)

    r2_score_TwoTrAda_elevators = pearsonr(elevators_np_test_y_list, y_pred_TwoTrAda_elevators)
    r2_score_TwoTrAda_elevators = (r2_score_TwoTrAda_elevators[0])**2
    r2scorelist_TwoTrAda_elevators.append(r2_score_TwoTrAda_elevators)

    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(elevators_tgt_df_X), len(elevators_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_elevators = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_elevators.fit(elevators_np_train_X, elevators_np_train_y_list)
    y_pred_stradaboost_elevators = model_stradaboost_elevators.predict(elevators_np_test_X)


    mse_stradaboost_elevators = sqrt(mean_squared_error(elevators_np_test_y, y_pred_stradaboost_elevators))
    rmselist_stradaboost_elevators.append(mse_stradaboost_elevators)

    r2_score_stradaboost_elevators = pearsonr(elevators_np_test_y_list, y_pred_stradaboost_elevators)
    r2_score_stradaboost_elevators = (r2_score_stradaboost_elevators[0])**2
    r2scorelist_stradaboost_elevators.append(r2_score_stradaboost_elevators)


with open('elevators_rmse.txt', 'w') as elevators_handle_rmse:
    elevators_handle_rmse.write("AdaBoost TL:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_elevators)

    elevators_handle_rmse.write("\n\nAdaBoost:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_elevators)

    elevators_handle_rmse.write("\n\nKMM:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_elevators)

    elevators_handle_rmse.write("\n\nGBRT:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_elevators)

    elevators_handle_rmse.write("\n\nGBR:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_elevators)

    elevators_handle_rmse.write("\n\nTrAdaBoost:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_elevators)

    elevators_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    elevators_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_elevators)


with open('elevators_r2.txt', 'w') as elevators_handle_r2:
    elevators_handle_r2.write("AdaBoost TL:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_elevators)

    elevators_handle_r2.write("\n\nAdaBoost:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_elevators)

    elevators_handle_r2.write("\n\nKMM:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_elevators)

    elevators_handle_r2.write("\n\nGBRT:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_elevators)

    elevators_handle_r2.write("\n\nGBR:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_elevators)

    elevators_handle_r2.write("\n\nTrAdaBoost:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_elevators)

    elevators_handle_r2.write("\n\nSTrAdaBoost:\n ")
    elevators_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_elevators)



######################################################################################

print("-------------------------------------------")
