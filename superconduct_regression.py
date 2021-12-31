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
################################### superconduct ###########################################################################################################


################################### Superconductivity ###########################################################################################################
## Target Data: critical_temp
## Correlation col: gmean_ElectronAffinity
## Cuts at: 40.0 and 60.0
##########################################################################################################################################################

superconduct_df = pd.read_csv("Scientific_data/superconduct/train.csv")
superconduct_unq_df = pd.read_csv("Scientific_data/superconduct/train.csv")

print("Superconductivity Data")
print("-------------------------------------------")
print(superconduct_df.shape)

# print("The correlation matrix is: ")
# superconduct_df_corr = superconduct_df.corr()['critical_temp'].abs().sort_values()
# print(superconduct_df_corr.to_string()) ### Helps to print the entire series

# to find where to split the data
# print('Min: ', superconduct_df['gmean_ElectronAffinity'].min())
# print('Max: ', superconduct_df['gmean_ElectronAffinity'].max())

ss = StandardScaler()

drop_col_superconduct = ['gmean_ElectronAffinity']
# superconduct_df['gmean_ElectronAffinity'].sort_values()

superconduct_train_df = superconduct_df.loc[(superconduct_df['gmean_ElectronAffinity'] > 40.0) & (superconduct_df['gmean_ElectronAffinity'] <= 60.0)]
superconduct_train_df = superconduct_train_df.drop(drop_col_superconduct, axis = 1)
superconduct_train_df = superconduct_train_df.reset_index(drop = True)
print("Training Set: ", superconduct_train_df.shape)

superconduct_source1_df = superconduct_df.loc[(superconduct_df['gmean_ElectronAffinity'] > 60.0)]
superconduct_source1_df = superconduct_source1_df.drop(drop_col_superconduct, axis = 1)
superconduct_source1_df = superconduct_source1_df.reset_index(drop = True)
print("Source Set 1: ", superconduct_source1_df.shape)

superconduct_source2_df = superconduct_df.loc[(superconduct_df['gmean_ElectronAffinity'] <= 40.0)]
superconduct_source2_df = superconduct_source2_df.drop(drop_col_superconduct, axis = 1)
superconduct_source2_df = superconduct_source2_df.reset_index(drop = True)
print("Source Set 2: ",superconduct_source2_df.shape)


superconduct_source_df = pd.concat([superconduct_source1_df, superconduct_source2_df], ignore_index=True)
superconduct_source_df = superconduct_source_df.sample(frac = 0.3, replace = True, random_state = 1)

print("Final Source Set: ",superconduct_source_df.shape)

#################### Splitting into features and target ####################
target_column_superconduct = ['critical_temp']

superconduct_train_df_y = superconduct_train_df[target_column_superconduct]
superconduct_train_df_X = superconduct_train_df.drop(target_column_superconduct, axis = 1)
superconduct_cols = superconduct_train_df_X.columns
superconduct_train_df_X[superconduct_cols] = ss.fit_transform(superconduct_train_df_X[superconduct_cols])

superconduct_source_df_y = superconduct_source_df[target_column_superconduct]
superconduct_source_df_X = superconduct_source_df.drop(target_column_superconduct, axis = 1)
superconduct_cols = superconduct_source_df_X.columns
superconduct_source_df_X[superconduct_cols] = ss.fit_transform(superconduct_source_df_X[superconduct_cols])

########################### Transfer Learning superconduct #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_superconduct = []
rmselist_AdaTL_superconduct = []

r2scorelist_Ada_superconduct = []
rmselist_Ada_superconduct = []

r2scorelist_KMM_superconduct = []
rmselist_KMM_superconduct = []

r2scorelist_GBRTL_superconduct = []
rmselist_GBRTL_superconduct = []

r2scorelist_GBR_superconduct = []
rmselist_GBR_superconduct = []

r2scorelist_TwoTrAda_superconduct = []
rmselist_TwoTrAda_superconduct = []

r2scorelist_stradaboost_superconduct = []
rmselist_stradaboost_superconduct = []



kfold = KFold(n_splits = 20, random_state = 42, shuffle=False)

for train_ix, test_ix in kfold.split(superconduct_train_df_X):
    ############### get data ###############
    superconduct_test_df_X, superconduct_tgt_df_X  = superconduct_train_df_X.iloc[train_ix], superconduct_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    superconduct_test_df_y, superconduct_tgt_df_y  = superconduct_train_df_y.iloc[train_ix], superconduct_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(superconduct_tgt_df_X.shape, superconduct_test_df_X.shape)

    ############### Merging the datasets ##########################################
    superconduct_X_df = pd.concat([superconduct_tgt_df_X, superconduct_source_df_X], ignore_index=True)
    superconduct_y_df = pd.concat([superconduct_tgt_df_y, superconduct_source_df_y], ignore_index=True)

    superconduct_np_train_X = superconduct_X_df.to_numpy()
    superconduct_np_train_y = superconduct_y_df.to_numpy()

    superconduct_np_test_X = superconduct_test_df_X.to_numpy()
    superconduct_np_test_y = superconduct_test_df_y.to_numpy()

    superconduct_np_train_y_list = superconduct_np_train_y.ravel()
    superconduct_np_test_y_list = superconduct_np_test_y.ravel()

    src_size_superconduct = len(superconduct_source_df_y)
    tgt_size_superconduct = len(superconduct_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_superconduct - 1), step=1)
    tgt_idx = np.arange(start=src_size_superconduct, stop=((src_size_superconduct + tgt_size_superconduct)-1), step=1)


    ################### AdaBoost Tl ###################
    model_AdaTL_superconduct = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_superconduct.fit(superconduct_np_train_X, superconduct_np_train_y_list)

    y_pred_AdaTL_superconduct = model_AdaTL_superconduct.predict(superconduct_np_test_X)

    mse_AdaTL_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_AdaTL_superconduct))
    rmselist_AdaTL_superconduct.append(mse_AdaTL_superconduct)

    r2_score_AdaTL_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_AdaTL_superconduct)
    r2_score_AdaTL_superconduct = (r2_score_AdaTL_superconduct[0])**2
    r2scorelist_AdaTL_superconduct.append(r2_score_AdaTL_superconduct)


    ################### AdaBoost ###################
    model_Ada_superconduct = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_superconduct.fit(superconduct_tgt_df_X, superconduct_tgt_df_y)

    y_pred_ada_superconduct = model_Ada_superconduct.predict(superconduct_np_test_X)

    mse_Ada_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_ada_superconduct))
    rmselist_Ada_superconduct.append(mse_Ada_superconduct)

    r2_score_Ada_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_ada_superconduct)
    r2_score_Ada_superconduct = (r2_score_Ada_superconduct[0])**2
    r2scorelist_Ada_superconduct.append(r2_score_Ada_superconduct)


   ################### KMM ###################
    model_KMM_superconduct = KMM(get_estimator = get_estimator)
    model_KMM_superconduct.fit(superconduct_np_train_X, superconduct_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_superconduct = model_KMM_superconduct.predict(superconduct_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_KMM_superconduct))
    rmselist_KMM_superconduct.append(mse_KMM_superconduct)

    r2_score_KMM_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_KMM_superconduct)
    r2_score_KMM_superconduct = (r2_score_KMM_superconduct[0])**2
    r2scorelist_KMM_superconduct.append(r2_score_KMM_superconduct)


    ################### GBRTL ###################
    model_GBRTL_superconduct = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 4, n_estimators = 1000, subsample = 0.5)
    model_GBRTL_superconduct.fit(superconduct_np_train_X, superconduct_np_train_y_list)

    y_pred_GBRTL_superconduct = model_GBRTL_superconduct.predict(superconduct_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_GBRTL_superconduct))
    rmselist_GBRTL_superconduct.append(mse_GBRTL_superconduct)

    r2_score_GBRTL_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_GBRTL_superconduct)
    r2_score_GBRTL_superconduct = (r2_score_GBRTL_superconduct[0])**2
    r2scorelist_GBRTL_superconduct.append(r2_score_GBRTL_superconduct)


    ################### GBR ###################
    model_GBR_superconduct = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_superconduct.fit(superconduct_tgt_df_X, superconduct_tgt_df_y)

    y_pred_GBR_superconduct = model_GBR_superconduct.predict(superconduct_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_GBR_superconduct))
    rmselist_GBR_superconduct.append(mse_GBR_superconduct)

    r2_score_GBR_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_GBR_superconduct)
    r2_score_GBR_superconduct = (r2_score_GBR_superconduct[0])**2
    r2scorelist_GBR_superconduct.append(r2_score_GBR_superconduct)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_superconduct = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_superconduct.fit(superconduct_np_train_X, superconduct_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_superconduct = model_TwoTrAda_superconduct.predict(superconduct_np_test_X)

    mse_TwoTrAda_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_TwoTrAda_superconduct))
    rmselist_TwoTrAda_superconduct.append(mse_TwoTrAda_superconduct)

    r2_score_TwoTrAda_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_TwoTrAda_superconduct)
    r2_score_TwoTrAda_superconduct = (r2_score_TwoTrAda_superconduct[0])**2
    r2scorelist_TwoTrAda_superconduct.append(r2_score_TwoTrAda_superconduct)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(superconduct_tgt_df_X), len(superconduct_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_superconduct = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_superconduct.fit(superconduct_np_train_X, superconduct_np_train_y_list)
    y_pred_stradaboost_superconduct = model_stradaboost_superconduct.predict(superconduct_np_test_X)


    mse_stradaboost_superconduct = sqrt(mean_squared_error(superconduct_np_test_y, y_pred_stradaboost_superconduct))
    rmselist_stradaboost_superconduct.append(mse_stradaboost_superconduct)

    r2_score_stradaboost_superconduct = pearsonr(superconduct_np_test_y_list, y_pred_stradaboost_superconduct)
    r2_score_stradaboost_superconduct = (r2_score_stradaboost_superconduct[0])**2
    r2scorelist_stradaboost_superconduct.append(r2_score_stradaboost_superconduct)



with open('superconduct_rmse.txt', 'w') as superconduct_handle_rmse:
    superconduct_handle_rmse.write("AdaBoost TL:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_superconduct)

    superconduct_handle_rmse.write("\n\nAdaBoost:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_superconduct)

    superconduct_handle_rmse.write("\n\nKMM:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_superconduct)

    superconduct_handle_rmse.write("\n\nGBRT:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_superconduct)

    superconduct_handle_rmse.write("\n\nGBR:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_superconduct)

    superconduct_handle_rmse.write("\n\nTrAdaBoost:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_superconduct)

    superconduct_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    superconduct_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_superconduct)


with open('superconduct_r2.txt', 'w') as superconduct_handle_r2:
    superconduct_handle_r2.write("AdaBoost TL:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_superconduct)

    superconduct_handle_r2.write("\n\nAdaBoost:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_superconduct)

    superconduct_handle_r2.write("\n\nKMM:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_superconduct)

    superconduct_handle_r2.write("\n\nGBRT:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_superconduct)

    superconduct_handle_r2.write("\n\nGBR:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_superconduct)

    superconduct_handle_r2.write("\n\nTrAdaBoost:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_superconduct)

    superconduct_handle_r2.write("\n\nSTrAdaBoost:\n ")
    superconduct_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_superconduct)


######################################################################################

print("-------------------------------------------")
