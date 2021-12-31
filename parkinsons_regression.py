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
################################### parkinsons ###########################################################################################################



################################### Parkinsons ###########################################################################################################
## Target Data: motor_UPDRS, total_UPDRS: choose one --> Let's go with total_UPDRS and remove motor UPDRS
## Correlation col: Shimmer
## Cuts at: 0.021 and 0.035
##########################################################################################################################################################

parkinsons_df = pd.read_csv("Scientific_data/Parkinsons/parkinsons_updrs.data")
motor_col = ['motor_UPDRS']
parkinsons_df = parkinsons_df.drop(motor_col, axis = 1)

print("Parkinsons Data")
print("-------------------------------------------")
print(parkinsons_df.shape)

# print("The correlation matrix is: ")
# Parkinsons_df.corr()['total_UPDRS'].abs().sort_values()

drop_col_parkinsons = ['Shimmer']

parkinsons_train_df = parkinsons_df.loc[(parkinsons_df['Shimmer'] >= 0.021) & (parkinsons_df['Shimmer'] < 0.035)]
parkinsons_train_df = parkinsons_train_df.drop(drop_col_parkinsons, axis = 1)
parkinsons_train_df = parkinsons_train_df.reset_index(drop = True)
print("Training Set: ", parkinsons_train_df.shape)

parkinsons_source1_df = parkinsons_df.loc[(parkinsons_df['Shimmer'] < 0.021)]
parkinsons_source1_df = parkinsons_source1_df.drop(drop_col_parkinsons, axis = 1)
parkinsons_source1_df = parkinsons_source1_df.reset_index(drop = True)
print("Source Set 1: ", parkinsons_source1_df.shape)

parkinsons_source2_df = parkinsons_df.loc[(parkinsons_df['Shimmer'] >= 0.035)]
parkinsons_source2_df = parkinsons_source2_df.drop(drop_col_parkinsons, axis = 1)
parkinsons_source2_df = parkinsons_source2_df.reset_index(drop = True)
print("Source Set 2: ",parkinsons_source2_df.shape)


parkinsons_source_df = pd.concat([parkinsons_source1_df, parkinsons_source2_df], ignore_index=True)
print("Final Source Set: ",parkinsons_source_df.shape)

#################### Splitting into features and target ####################
target_column_parkinsons = ['total_UPDRS']

parkinsons_train_df_y = parkinsons_train_df[target_column_parkinsons]
parkinsons_train_df_X = parkinsons_train_df.drop(target_column_parkinsons, axis = 1)

parkinsons_source_df_y = parkinsons_source_df[target_column_parkinsons]
parkinsons_source_df_X = parkinsons_source_df.drop(target_column_parkinsons, axis = 1)


########################### Transfer Learning parkinsons #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_parkinsons = []
rmselist_AdaTL_parkinsons = []

r2scorelist_Ada_parkinsons = []
rmselist_Ada_parkinsons = []

r2scorelist_KMM_parkinsons = []
rmselist_KMM_parkinsons = []

r2scorelist_GBRTL_parkinsons = []
rmselist_GBRTL_parkinsons = []

r2scorelist_GBR_parkinsons = []
rmselist_GBR_parkinsons = []

r2scorelist_TwoTrAda_parkinsons = []
rmselist_TwoTrAda_parkinsons = []

r2scorelist_stradaboost_parkinsons = []
rmselist_stradaboost_parkinsons = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(parkinsons_train_df_X):
    ############### get data ###############
    parkinsons_test_df_X, parkinsons_tgt_df_X  = parkinsons_train_df_X.iloc[train_ix], parkinsons_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    parkinsons_test_df_y, parkinsons_tgt_df_y  = parkinsons_train_df_y.iloc[train_ix], parkinsons_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(parkinsons_tgt_df_X.shape, parkinsons_test_df_X.shape)

    ############### Merging the datasets ##########################################
    parkinsons_X_df = pd.concat([parkinsons_tgt_df_X, parkinsons_source_df_X], ignore_index=True)
    parkinsons_y_df = pd.concat([parkinsons_tgt_df_y, parkinsons_source_df_y], ignore_index=True)

    parkinsons_np_train_X = parkinsons_X_df.to_numpy()
    parkinsons_np_train_y = parkinsons_y_df.to_numpy()

    parkinsons_np_test_X = parkinsons_test_df_X.to_numpy()
    parkinsons_np_test_y = parkinsons_test_df_y.to_numpy()

    parkinsons_np_train_y_list = parkinsons_np_train_y.ravel()
    parkinsons_np_test_y_list = parkinsons_np_test_y.ravel()

    src_size_parkinsons = len(parkinsons_source_df_y)
    tgt_size_parkinsons = len(parkinsons_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_parkinsons - 1), step=1)
    tgt_idx = np.arange(start=src_size_parkinsons, stop=((src_size_parkinsons + tgt_size_parkinsons)-1), step=1)


    ################### AdaBoost Tl ###################
    model_AdaTL_parkinsons = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_parkinsons.fit(parkinsons_np_train_X, parkinsons_np_train_y_list)

    y_pred_AdaTL_parkinsons = model_AdaTL_parkinsons.predict(parkinsons_np_test_X)

    mse_AdaTL_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_AdaTL_parkinsons))
    rmselist_AdaTL_parkinsons.append(mse_AdaTL_parkinsons)

    r2_score_AdaTL_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_AdaTL_parkinsons)
    r2_score_AdaTL_parkinsons = (r2_score_AdaTL_parkinsons[0])**2
    r2scorelist_AdaTL_parkinsons.append(r2_score_AdaTL_parkinsons)


    ################### AdaBoost ###################
    model_Ada_parkinsons = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_parkinsons.fit(parkinsons_tgt_df_X, parkinsons_tgt_df_y)

    y_pred_ada_parkinsons = model_Ada_parkinsons.predict(parkinsons_np_test_X)

    mse_Ada_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_ada_parkinsons))
    rmselist_Ada_parkinsons.append(mse_Ada_parkinsons)

    r2_score_Ada_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_ada_parkinsons)
    r2_score_Ada_parkinsons = (r2_score_Ada_parkinsons[0])**2
    r2scorelist_Ada_parkinsons.append(r2_score_Ada_parkinsons)


   ################### KMM ###################
    model_KMM_parkinsons = KMM(get_estimator = get_estimator)
    model_KMM_parkinsons.fit(parkinsons_np_train_X, parkinsons_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_parkinsons = model_KMM_parkinsons.predict(parkinsons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_KMM_parkinsons))
    rmselist_KMM_parkinsons.append(mse_KMM_parkinsons)

    r2_score_KMM_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_KMM_parkinsons)
    r2_score_KMM_parkinsons = (r2_score_KMM_parkinsons[0])**2
    r2scorelist_KMM_parkinsons.append(r2_score_KMM_parkinsons)


    ################### GBRTL ###################
    model_GBRTL_parkinsons = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 4, n_estimators = 1000, subsample = 0.5)
    model_GBRTL_parkinsons.fit(parkinsons_np_train_X, parkinsons_np_train_y_list)

    y_pred_GBRTL_parkinsons = model_GBRTL_parkinsons.predict(parkinsons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_GBRTL_parkinsons))
    rmselist_GBRTL_parkinsons.append(mse_GBRTL_parkinsons)

    r2_score_GBRTL_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_GBRTL_parkinsons)
    r2_score_GBRTL_parkinsons = (r2_score_GBRTL_parkinsons[0])**2
    r2scorelist_GBRTL_parkinsons.append(r2_score_GBRTL_parkinsons)


    ################### GBR ###################
    model_GBR_parkinsons = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_parkinsons.fit(parkinsons_tgt_df_X, parkinsons_tgt_df_y)

    y_pred_GBR_parkinsons = model_GBR_parkinsons.predict(parkinsons_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_GBR_parkinsons))
    rmselist_GBR_parkinsons.append(mse_GBR_parkinsons)

    r2_score_GBR_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_GBR_parkinsons)
    r2_score_GBR_parkinsons = (r2_score_GBR_parkinsons[0])**2
    r2scorelist_GBR_parkinsons.append(r2_score_GBR_parkinsons)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_parkinsons = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_parkinsons.fit(parkinsons_np_train_X, parkinsons_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_parkinsons = model_TwoTrAda_parkinsons.predict(parkinsons_np_test_X)

    mse_TwoTrAda_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_TwoTrAda_parkinsons))
    rmselist_TwoTrAda_parkinsons.append(mse_TwoTrAda_parkinsons)

    r2_score_TwoTrAda_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_TwoTrAda_parkinsons)
    r2_score_TwoTrAda_parkinsons = (r2_score_TwoTrAda_parkinsons[0])**2
    r2scorelist_TwoTrAda_parkinsons.append(r2_score_TwoTrAda_parkinsons)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(parkinsons_tgt_df_X), len(parkinsons_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_parkinsons = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_parkinsons.fit(parkinsons_np_train_X, parkinsons_np_train_y_list)
    y_pred_stradaboost_parkinsons = model_stradaboost_parkinsons.predict(parkinsons_np_test_X)


    mse_stradaboost_parkinsons = sqrt(mean_squared_error(parkinsons_np_test_y, y_pred_stradaboost_parkinsons))
    rmselist_stradaboost_parkinsons.append(mse_stradaboost_parkinsons)

    r2_score_stradaboost_parkinsons = pearsonr(parkinsons_np_test_y_list, y_pred_stradaboost_parkinsons)
    r2_score_stradaboost_parkinsons = (r2_score_stradaboost_parkinsons[0])**2
    r2scorelist_stradaboost_parkinsons.append(r2_score_stradaboost_parkinsons)



with open('parkinsons_rmse.txt', 'w') as parkinsons_handle_rmse:
    parkinsons_handle_rmse.write("AdaBoost TL:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_parkinsons)

    parkinsons_handle_rmse.write("\n\nAdaBoost:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_parkinsons)

    parkinsons_handle_rmse.write("\n\nKMM:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_parkinsons)

    parkinsons_handle_rmse.write("\n\nGBRT:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_parkinsons)

    parkinsons_handle_rmse.write("\n\nGBR:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_parkinsons)

    parkinsons_handle_rmse.write("\n\nTrAdaBoost:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_parkinsons)

    parkinsons_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    parkinsons_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_parkinsons)


with open('parkinsons_r2.txt', 'w') as parkinsons_handle_r2:
    parkinsons_handle_r2.write("AdaBoost TL:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_parkinsons)

    parkinsons_handle_r2.write("\n\nAdaBoost:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_parkinsons)

    parkinsons_handle_r2.write("\n\nKMM:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_parkinsons)

    parkinsons_handle_r2.write("\n\nGBRT:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_parkinsons)

    parkinsons_handle_r2.write("\n\nGBR:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_parkinsons)

    parkinsons_handle_r2.write("\n\nTrAdaBoost:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_parkinsons)

    parkinsons_handle_r2.write("\n\nSTrAdaBoost:\n ")
    parkinsons_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_parkinsons)


######################################################################################


# print("RMSE of Adaboost.R2(TL):", statistics.mean(rmselist_AdaTL_parkinsons))
# print("R^2 of AdaboostR2(TL):", statistics.mean(r2scorelist_AdaTL_parkinsons))
# print("\n")
# print("RMSE of Adaboost.R2(TL):", rmselist_AdaTL_parkinsons)
# print("R^2 of AdaboostR2(TL):", r2scorelist_AdaTL_parkinsons)


print("-------------------------------------------")
