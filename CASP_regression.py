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


################################### CASP ###########################################################################################################
## Target Data: RMSD
## Correlation col: F6
## Cuts at: 105.0 and 160.0
##########################################################################################################################################################

casp_df = pd.read_csv("Scientific_data/Casp/CASP.csv")

print("CASP Data")
print("-------------------------------------------")
print(casp_df.shape)

# print("The correlation matrix is: ")
# casp_df.corr()['RMSD'].abs().sort_values()

drop_col_casp = ['F6']
# casp_df['F6'].sort_values()


casp_train_df = casp_df.loc[(casp_df['F6'] >= 105.0) & (casp_df['F6'] < 160.0)]
casp_train_df = casp_train_df.drop(drop_col_casp, axis = 1)
casp_train_df = casp_train_df.reset_index(drop = True)
print("Training Set: ", casp_train_df.shape)

casp_source1_df = casp_df.loc[(casp_df['F6'] < 105.0)]
casp_source1_df = casp_source1_df.drop(drop_col_casp, axis = 1)
casp_source1_df = casp_source1_df.reset_index(drop = True)
print("Source Set 1: ", casp_source1_df.shape)

casp_source2_df = casp_df.loc[(casp_df['F6'] >= 160.0)]
casp_source2_df = casp_source2_df.drop(drop_col_casp, axis = 1)
casp_source2_df = casp_source2_df.reset_index(drop = True)
print("Source Set 2: ",casp_source2_df.shape)


casp_source_df = pd.concat([casp_source1_df, casp_source2_df], ignore_index=True)
print("Final Source Set: ",casp_source_df.shape)

#################### Splitting into features and target ####################
target_column_casp = ['RMSD']

casp_train_df_y = casp_train_df[target_column_casp]
casp_train_df_X = casp_train_df.drop(target_column_casp, axis = 1)

casp_source_df_y = casp_source_df[target_column_casp]
casp_source_df_X = casp_source_df.drop(target_column_casp, axis = 1)


########################### Transfer Learning casp #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_casp = []
rmselist_AdaTL_casp = []

r2scorelist_Ada_casp = []
rmselist_Ada_casp = []

r2scorelist_KMM_casp = []
rmselist_KMM_casp = []

r2scorelist_GBRTL_casp = []
rmselist_GBRTL_casp = []

r2scorelist_GBR_casp = []
rmselist_GBR_casp = []

r2scorelist_TwoTrAda_casp = []
rmselist_TwoTrAda_casp = []

r2scorelist_stradaboost_casp = []
rmselist_stradaboost_casp = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(casp_train_df_X):
    ############### get data ###############
    casp_test_df_X, casp_tgt_df_X  = casp_train_df_X.iloc[train_ix], casp_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    casp_test_df_y, casp_tgt_df_y  = casp_train_df_y.iloc[train_ix], casp_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(casp_tgt_df_X.shape, casp_test_df_X.shape)

    ############### Merging the datasets ##########################################
    casp_X_df = pd.concat([casp_tgt_df_X, casp_source_df_X], ignore_index=True)
    casp_y_df = pd.concat([casp_tgt_df_y, casp_source_df_y], ignore_index=True)

    casp_np_train_X = casp_X_df.to_numpy()
    casp_np_train_y = casp_y_df.to_numpy()

    casp_np_test_X = casp_test_df_X.to_numpy()
    casp_np_test_y = casp_test_df_y.to_numpy()

    casp_np_train_y_list = casp_np_train_y.ravel()
    casp_np_test_y_list = casp_np_test_y.ravel()

    src_size_casp = len(casp_source_df_y)
    tgt_size_casp = len(casp_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_casp - 1), step=1)
    tgt_idx = np.arange(start=src_size_casp, stop=((src_size_casp + tgt_size_casp)-1), step=1)


    ################### AdaBoost Tl ###################
    model_AdaTL_casp = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_casp.fit(casp_np_train_X, casp_np_train_y_list)

    y_pred_AdaTL_casp = model_AdaTL_casp.predict(casp_np_test_X)

    mse_AdaTL_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_AdaTL_casp))
    rmselist_AdaTL_casp.append(mse_AdaTL_casp)

    r2_score_AdaTL_casp = pearsonr(casp_np_test_y_list, y_pred_AdaTL_casp)
    r2_score_AdaTL_casp = (r2_score_AdaTL_casp[0])**2
    r2scorelist_AdaTL_casp.append(r2_score_AdaTL_casp)


    ################### AdaBoost ###################
    model_Ada_casp = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_casp.fit(casp_tgt_df_X, casp_tgt_df_y)

    y_pred_ada_casp = model_Ada_casp.predict(casp_np_test_X)

    mse_Ada_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_ada_casp))
    rmselist_Ada_casp.append(mse_Ada_casp)

    r2_score_Ada_casp = pearsonr(casp_np_test_y_list, y_pred_ada_casp)
    r2_score_Ada_casp = (r2_score_Ada_casp[0])**2
    r2scorelist_Ada_casp.append(r2_score_Ada_casp)


   ################### KMM ###################
    model_KMM_casp = KMM(get_estimator = get_estimator)
    model_KMM_casp.fit(casp_np_train_X, casp_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_casp = model_KMM_casp.predict(casp_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_KMM_casp))
    rmselist_KMM_casp.append(mse_KMM_casp)

    r2_score_KMM_casp = pearsonr(casp_np_test_y_list, y_pred_KMM_casp)
    r2_score_KMM_casp = (r2_score_KMM_casp[0])**2
    r2scorelist_KMM_casp.append(r2_score_KMM_casp)


    ################### GBRTL ###################
    model_GBRTL_casp = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 4, n_estimators = 1000, subsample = 0.5)
    model_GBRTL_casp.fit(casp_np_train_X, casp_np_train_y_list)

    y_pred_GBRTL_casp = model_GBRTL_casp.predict(casp_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_GBRTL_casp))
    rmselist_GBRTL_casp.append(mse_GBRTL_casp)

    r2_score_GBRTL_casp = pearsonr(casp_np_test_y_list, y_pred_GBRTL_casp)
    r2_score_GBRTL_casp = (r2_score_GBRTL_casp[0])**2
    r2scorelist_GBRTL_casp.append(r2_score_GBRTL_casp)


    ################### GBR ###################
    model_GBR_casp = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_casp.fit(casp_tgt_df_X, casp_tgt_df_y)

    y_pred_GBR_casp = model_GBR_casp.predict(casp_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_GBR_casp))
    rmselist_GBR_casp.append(mse_GBR_casp)

    r2_score_GBR_casp = pearsonr(casp_np_test_y_list, y_pred_GBR_casp)
    r2_score_GBR_casp = (r2_score_GBR_casp[0])**2
    r2scorelist_GBR_casp.append(r2_score_GBR_casp)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_casp = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_casp.fit(casp_np_train_X, casp_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_casp = model_TwoTrAda_casp.predict(casp_np_test_X)

    mse_TwoTrAda_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_TwoTrAda_casp))
    rmselist_TwoTrAda_casp.append(mse_TwoTrAda_casp)

    r2_score_TwoTrAda_casp = pearsonr(casp_np_test_y_list, y_pred_TwoTrAda_casp)
    r2_score_TwoTrAda_casp = (r2_score_TwoTrAda_casp[0])**2
    r2scorelist_TwoTrAda_casp.append(r2_score_TwoTrAda_casp)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(casp_tgt_df_X), len(casp_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_casp = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_casp.fit(casp_np_train_X, casp_np_train_y_list)
    y_pred_stradaboost_casp = model_stradaboost_casp.predict(casp_np_test_X)


    mse_stradaboost_casp = sqrt(mean_squared_error(casp_np_test_y, y_pred_stradaboost_casp))
    rmselist_stradaboost_casp.append(mse_stradaboost_casp)

    r2_score_stradaboost_casp = pearsonr(casp_np_test_y_list, y_pred_stradaboost_casp)
    r2_score_stradaboost_casp = (r2_score_stradaboost_casp[0])**2
    r2scorelist_stradaboost_casp.append(r2_score_stradaboost_casp)



with open('casp_rmse.txt', 'w') as casp_handle_rmse:
    casp_handle_rmse.write("AdaBoost TL:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_casp)

    casp_handle_rmse.write("\n\nAdaBoost:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_casp)

    casp_handle_rmse.write("\n\nKMM:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_casp)

    casp_handle_rmse.write("\n\nGBRT:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_casp)

    casp_handle_rmse.write("\n\nGBR:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_casp)

    casp_handle_rmse.write("\n\nTrAdaBoost:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_casp)

    casp_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    casp_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_casp)


with open('casp_r2.txt', 'w') as casp_handle_r2:
    casp_handle_r2.write("AdaBoost TL:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_casp)

    casp_handle_r2.write("\n\nAdaBoost:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_casp)

    casp_handle_r2.write("\n\nKMM:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_casp)

    casp_handle_r2.write("\n\nGBRT:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_casp)

    casp_handle_r2.write("\n\nGBR:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_casp)

    casp_handle_r2.write("\n\nTrAdaBoost:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_casp)

    casp_handle_r2.write("\n\nSTrAdaBoost:\n ")
    casp_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_casp)


######################################################################################


# print("RMSE of Adaboost.R2(TL):", statistics.mean(rmselist_AdaTL_casp))
# print("R^2 of AdaboostR2(TL):", statistics.mean(r2scorelist_AdaTL_casp))
# print("\n")
# print("RMSE of Adaboost.R2(TL):", rmselist_AdaTL_casp)
# print("R^2 of AdaboostR2(TL):", r2scorelist_AdaTL_casp)


print("-------------------------------------------")
