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

################################### Kinematics ################################################################
#### range: [0.04 - 1.45]
#### Mid of correlation variable: theta7
#### [0, 0.6] [0.6, 0.85], [0.6, 0.85]
#####################################################################################################################
# target_var_Kinematics = ['y']
colnames_kinematics = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8', 'y']
kinematicsData_df = pd.read_csv('UCI_regression/Kinematics/kin8nm.data', header = None,  names = colnames_kinematics)
print("Kinematics Data")
print(kinematicsData_df.shape)

drop_col_kinematics = ['theta7']

kinematics_train_df = kinematicsData_df.loc[(kinematicsData_df['theta7'] > 0.001) & (kinematicsData_df['theta7'] <= 0.600)]
kinematics_train_df = kinematics_train_df.drop(drop_col_kinematics, axis = 1)
kinematics_train_df = kinematics_train_df.reset_index(drop=True)
print("Target Set: ",kinematics_train_df.shape)


kinematics_source1_df = kinematicsData_df.loc[(kinematicsData_df['theta7'] > 0.600)]
kinematics_source1_df = kinematics_source1_df.drop(drop_col_kinematics, axis = 1)
kinematics_source1_df = kinematics_source1_df.reset_index(drop=True)
print("Source Set 1: ",kinematics_source1_df.shape)


kinematics_source2_df = kinematicsData_df.loc[(kinematicsData_df['theta7'] <= 0.001)]
kinematics_source2_df = kinematics_source2_df.drop(drop_col_kinematics, axis = 1)
kinematics_source2_df = kinematics_source2_df.reset_index(drop=True)
print("Source Set 2: ",kinematics_source2_df.shape)


kinematics_source_df = pd.concat([kinematics_source1_df, kinematics_source2_df], ignore_index=True)
print("Final Source Set: ",kinematics_source_df.shape)


#################### Splitting into features and target ####################
target_column_kinematics = ['y']

kinematics_train_df_y = kinematics_train_df[target_column_kinematics]
kinematics_train_df_X = kinematics_train_df.drop(target_column_kinematics, axis = 1)

kinematics_source_df_y = kinematics_source_df[target_column_kinematics]
kinematics_source_df_X = kinematics_source_df.drop(target_column_kinematics, axis = 1)

print("---------------------------")

########################### Transfer Learning kinematics #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}


print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_kinematics = []
rmselist_AdaTL_kinematics = []

r2scorelist_Ada_kinematics = []
rmselist_Ada_kinematics = []

r2scorelist_KMM_kinematics = []
rmselist_KMM_kinematics = []

r2scorelist_GBRTL_kinematics = []
rmselist_GBRTL_kinematics = []

r2scorelist_GBR_kinematics = []
rmselist_GBR_kinematics = []

r2scorelist_TwoTrAda_kinematics = []
rmselist_TwoTrAda_kinematics = []

r2scorelist_stradaboost_kinematics = []
rmselist_stradaboost_kinematics = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(kinematics_train_df_X):
    ############### get data ###############
    kinematics_test_df_X, kinematics_tgt_df_X  = kinematics_train_df_X.iloc[train_ix], kinematics_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    kinematics_test_df_y, kinematics_tgt_df_y  = kinematics_train_df_y.iloc[train_ix], kinematics_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(kinematics_tgt_df_X.shape, kinematics_test_df_X.shape)

    ############### Merging the datasets ##########################################
    kinematics_X_df = pd.concat([kinematics_tgt_df_X, kinematics_source_df_X], ignore_index=True)
    kinematics_y_df = pd.concat([kinematics_tgt_df_y, kinematics_source_df_y], ignore_index=True)

    kinematics_np_train_X = kinematics_X_df.to_numpy()
    kinematics_np_train_y = kinematics_y_df.to_numpy()

    kinematics_np_test_X = kinematics_test_df_X.to_numpy()
    kinematics_np_test_y = kinematics_test_df_y.to_numpy()

    kinematics_np_train_y_list = kinematics_np_train_y.ravel()
    kinematics_np_test_y_list = kinematics_np_test_y.ravel()

    src_size_kinematics = len(kinematics_source_df_y)
    tgt_size_kinematics = len(kinematics_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_kinematics - 1), step=1)
    tgt_idx = np.arange(start=src_size_kinematics, stop=((src_size_kinematics + tgt_size_kinematics)-1), step=1)



    ################### AdaBoost Tl ###################
    model_AdaTL_kinematics = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_kinematics.fit(kinematics_np_train_X, kinematics_np_train_y_list)

    y_pred_AdaTL_kinematics = model_AdaTL_kinematics.predict(kinematics_np_test_X)

    mse_AdaTL_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_AdaTL_kinematics))
    rmselist_AdaTL_kinematics.append(mse_AdaTL_kinematics)

    r2_score_AdaTL_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_AdaTL_kinematics)
    r2_score_AdaTL_kinematics = (r2_score_AdaTL_kinematics[0])**2
    r2scorelist_AdaTL_kinematics.append(r2_score_AdaTL_kinematics)


    ################### AdaBoost ###################
    model_Ada_kinematics = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_kinematics.fit(kinematics_tgt_df_X, kinematics_tgt_df_y)

    y_pred_Ada_kinematics = model_Ada_kinematics.predict(kinematics_np_test_X)

    mse_Ada_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_Ada_kinematics))
    rmselist_Ada_kinematics.append(mse_Ada_kinematics)

    r2_score_Ada_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_Ada_kinematics)
    r2_score_Ada_kinematics = (r2_score_Ada_kinematics[0])**2
    r2scorelist_Ada_kinematics.append(r2_score_Ada_kinematics)


   ################### KMM ###################
    model_KMM_kinematics = KMM(get_estimator = get_estimator)
    model_KMM_kinematics.fit(kinematics_np_train_X, kinematics_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_kinematics = model_KMM_kinematics.predict(kinematics_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_KMM_kinematics))
    rmselist_KMM_kinematics.append(mse_KMM_kinematics)

    r2_score_KMM_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_KMM_kinematics)
    r2_score_KMM_kinematics = (r2_score_KMM_kinematics[0])**2
    r2scorelist_KMM_kinematics.append(r2_score_KMM_kinematics)


    ################### GBRTL ###################
    model_GBRTL_kinematics = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBRTL_kinematics.fit(kinematics_np_train_X, kinematics_np_train_y_list)

    y_pred_GBRTL_kinematics = model_GBRTL_kinematics.predict(kinematics_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_GBRTL_kinematics))
    rmselist_GBRTL_kinematics.append(mse_GBRTL_kinematics)

    r2_score_GBRTL_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_GBRTL_kinematics)
    r2_score_GBRTL_kinematics = (r2_score_GBRTL_kinematics[0])**2
    r2scorelist_GBRTL_kinematics.append(r2_score_GBRTL_kinematics)


    ################### GBR ###################
    model_GBR_kinematics = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_kinematics.fit(kinematics_tgt_df_X, kinematics_tgt_df_y)

    y_pred_GBR_kinematics = model_GBR_kinematics.predict(kinematics_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_GBR_kinematics))
    rmselist_GBR_kinematics.append(mse_GBR_kinematics)

    r2_score_GBR_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_GBR_kinematics)
    r2_score_GBR_kinematics = (r2_score_GBR_kinematics[0])**2
    r2scorelist_GBR_kinematics.append(r2_score_GBR_kinematics)


    ################### Two-TrAdaBoost ###################
    model_TwoTrAda_kinematics = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_kinematics.fit(kinematics_np_train_X, kinematics_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_kinematics = model_TwoTrAda_kinematics.predict(kinematics_np_test_X)

    mse_TwoTrAda_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_TwoTrAda_kinematics))
    rmselist_TwoTrAda_kinematics.append(mse_TwoTrAda_kinematics)

    r2_score_TwoTrAda_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_TwoTrAda_kinematics)
    r2_score_TwoTrAda_kinematics = (r2_score_TwoTrAda_kinematics[0])**2
    r2scorelist_TwoTrAda_kinematics.append(r2_score_TwoTrAda_kinematics)


    ################### STrAdaBoost ###################
    sample_size = [len(kinematics_tgt_df_X), len(kinematics_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    model_stradaboost_kinematics = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_kinematics.fit(kinematics_np_train_X, kinematics_np_train_y_list)
    y_pred_stradaboost_kinematics = model_stradaboost_kinematics.predict(kinematics_np_test_X)


    mse_stradaboost_kinematics = sqrt(mean_squared_error(kinematics_np_test_y, y_pred_stradaboost_kinematics))
    rmselist_stradaboost_kinematics.append(mse_stradaboost_kinematics)

    r2_score_stradaboost_kinematics = pearsonr(kinematics_np_test_y_list, y_pred_stradaboost_kinematics)
    r2_score_stradaboost_kinematics = (r2_score_stradaboost_kinematics[0])**2
    r2scorelist_stradaboost_kinematics.append(r2_score_stradaboost_kinematics)


with open('kinematics_rmse.txt', 'w') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_AdaTL_kinematics)

with open('kinematics_r2.txt', 'w') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_kinematics)

######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_Ada_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_Ada_kinematics)

######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_KMM_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_KMM_kinematics)


######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_GBRTL_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_kinematics)


######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_GBR_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_GBR_kinematics)


######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_kinematics)

######################################################################################

with open('kinematics_rmse.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in rmselist_stradaboost_kinematics)

with open('kinematics_r2.txt', 'a') as kinematics_handle:
    kinematics_handle.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_kinematics)


######################################################################################

print("-------------------------------------------")
