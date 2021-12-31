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

################################### predict ###########################################################################################################


################################### YearPrediction ###########################################################################################################
## Target Data: year
## Correlation col: timbre_cov_009
## Cuts at: 220 to 340
##########################################################################################################################################################

predict_df = pd.read_csv("Scientific_data/YearPrediction/YearPredictionMSD.csv")

print("YearPredictionMSD Data")
print("-------------------------------------------")
print(predict_df.shape)

# print("The correlation matrix is: ")
# predict_df_corr = predict_df.corr()['year'].abs().sort_values()
# print(predict_df_corr.to_string())

# print('Min: ', predict_df['timbre_cov_009'].min())
# print('Max: ', predict_df['timbre_cov_009'].max())

drop_col_predict = ['timbre_cov_009']

predict_train_df = predict_df.loc[(predict_df['timbre_cov_009'] >= 220) & (predict_df['timbre_cov_009'] < 340)]
predict_train_df = predict_train_df.drop(drop_col_predict, axis = 1)
predict_train_df = predict_train_df.reset_index(drop = True)
print("Training Set: ", predict_train_df.shape)

predict_source1_df = predict_df.loc[(predict_df['timbre_cov_009'] >= 340)]
predict_source1_df = predict_source1_df.drop(drop_col_predict, axis = 1)
predict_source1_df = predict_source1_df.reset_index(drop = True)
print("Source Set 1: ", predict_source1_df.shape)

predict_source2_df = predict_df.loc[(predict_df['timbre_cov_009'] < 220)]
predict_source2_df = predict_source2_df.drop(drop_col_predict, axis = 1)
predict_source2_df = predict_source2_df.reset_index(drop = True)
print("Source Set 2: ",predict_source2_df.shape)


predict_source_df = pd.concat([predict_source1_df, predict_source2_df], ignore_index=True)
print("Final Source Set: ",predict_source_df.shape)

#################### Splitting into features and target ####################
target_column_predict = ['year']

predict_train_df_y = predict_train_df[target_column_predict]
predict_train_df_X = predict_train_df.drop(target_column_predict, axis = 1)

predict_source_df_y = predict_source_df[target_column_predict]
predict_source_df_X = predict_source_df.drop(target_column_predict, axis = 1)


########################### Transfer Learning predict #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_predict = []
rmselist_AdaTL_predict = []

r2scorelist_Ada_predict = []
rmselist_Ada_predict = []

r2scorelist_KMM_predict = []
rmselist_KMM_predict = []

r2scorelist_GBRTL_predict = []
rmselist_GBRTL_predict = []

r2scorelist_GBR_predict = []
rmselist_GBR_predict = []

r2scorelist_TwoTrAda_predict = []
rmselist_TwoTrAda_predict = []

r2scorelist_stradaboost_predict = []
rmselist_stradaboost_predict = []



kfold = KFold(n_splits = 10, random_state=42, shuffle=False)

for train_ix, test_ix in kfold.split(predict_train_df_X):
    ############### get data ###############
    predict_test_df_X, predict_tgt_df_X  = predict_train_df_X.iloc[train_ix], predict_train_df_X.iloc[test_ix] #### Make it opposite, so target size is small.
    predict_test_df_y, predict_tgt_df_y  = predict_train_df_y.iloc[train_ix], predict_train_df_y.iloc[test_ix] #### Make it opposite, so target size is small.

    print(predict_tgt_df_X.shape, predict_test_df_X.shape)

    ############### Merging the datasets ##########################################
    predict_X_df = pd.concat([predict_tgt_df_X, predict_source_df_X], ignore_index=True)
    predict_y_df = pd.concat([predict_tgt_df_y, predict_source_df_y], ignore_index=True)

    predict_np_train_X = predict_X_df.to_numpy()
    predict_np_train_y = predict_y_df.to_numpy()

    predict_np_test_X = predict_test_df_X.to_numpy()
    predict_np_test_y = predict_test_df_y.to_numpy()

    predict_np_train_y_list = predict_np_train_y.ravel()
    predict_np_test_y_list = predict_np_test_y.ravel()

    src_size_predict = len(predict_source_df_y)
    tgt_size_predict = len(predict_tgt_df_y)

    src_idx = np.arange(start=0, stop=(src_size_predict - 1), step=1)
    tgt_idx = np.arange(start=src_size_predict, stop=((src_size_predict + tgt_size_predict)-1), step=1)


    ################### AdaBoost Tl ###################
    model_AdaTL_predict = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_AdaTL_predict.fit(predict_np_train_X, predict_np_train_y_list)

    y_pred_AdaTL_predict = model_AdaTL_predict.predict(predict_np_test_X)

    mse_AdaTL_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_AdaTL_predict))
    rmselist_AdaTL_predict.append(mse_AdaTL_predict)

    r2_score_AdaTL_predict = pearsonr(predict_np_test_y_list, y_pred_AdaTL_predict)
    r2_score_AdaTL_predict = (r2_score_AdaTL_predict[0])**2
    r2scorelist_AdaTL_predict.append(r2_score_AdaTL_predict)


    ################### AdaBoost ###################
    model_Ada_predict = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 8), learning_rate=0.01, n_estimators=500)
    model_Ada_predict.fit(predict_tgt_df_X, predict_tgt_df_y)

    y_pred_ada_predict = model_Ada_predict.predict(predict_np_test_X)

    mse_Ada_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_ada_predict))
    rmselist_Ada_predict.append(mse_Ada_predict)

    r2_score_Ada_predict = pearsonr(predict_np_test_y_list, y_pred_ada_predict)
    r2_score_Ada_predict = (r2_score_Ada_predict[0])**2
    r2scorelist_Ada_predict.append(r2_score_Ada_predict)


   ################### KMM ###################
    model_KMM_predict = KMM(get_estimator = get_estimator)
    model_KMM_predict.fit(predict_np_train_X, predict_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_predict = model_KMM_predict.predict(predict_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_KMM_predict))
    rmselist_KMM_predict.append(mse_KMM_predict)

    r2_score_KMM_predict = pearsonr(predict_np_test_y_list, y_pred_KMM_predict)
    r2_score_KMM_predict = (r2_score_KMM_predict[0])**2
    r2scorelist_KMM_predict.append(r2_score_KMM_predict)


    ################### GBRTL ###################
    model_GBRTL_predict = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 4, n_estimators = 1000, subsample = 0.5)
    model_GBRTL_predict.fit(predict_np_train_X, predict_np_train_y_list)

    y_pred_GBRTL_predict = model_GBRTL_predict.predict(predict_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_GBRTL_predict))
    rmselist_GBRTL_predict.append(mse_GBRTL_predict)

    r2_score_GBRTL_predict = pearsonr(predict_np_test_y_list, y_pred_GBRTL_predict)
    r2_score_GBRTL_predict = (r2_score_GBRTL_predict[0])**2
    r2scorelist_GBRTL_predict.append(r2_score_GBRTL_predict)


    ################### GBR ###################
    model_GBR_predict = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=1000, subsample=0.5)
    model_GBR_predict.fit(predict_tgt_df_X, predict_tgt_df_y)

    y_pred_GBR_predict = model_GBR_predict.predict(predict_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_GBR_predict))
    rmselist_GBR_predict.append(mse_GBR_predict)

    r2_score_GBR_predict = pearsonr(predict_np_test_y_list, y_pred_GBR_predict)
    r2_score_GBR_predict = (r2_score_GBR_predict[0])**2
    r2scorelist_GBR_predict.append(r2_score_GBR_predict)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_predict = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 1000, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_predict.fit(predict_np_train_X, predict_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_predict = model_TwoTrAda_predict.predict(predict_np_test_X)

    mse_TwoTrAda_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_TwoTrAda_predict))
    rmselist_TwoTrAda_predict.append(mse_TwoTrAda_predict)

    r2_score_TwoTrAda_predict = pearsonr(predict_np_test_y_list, y_pred_TwoTrAda_predict)
    r2_score_TwoTrAda_predict = (r2_score_TwoTrAda_predict[0])**2
    r2scorelist_TwoTrAda_predict.append(r2_score_TwoTrAda_predict)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(predict_tgt_df_X), len(predict_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_predict = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_predict.fit(predict_np_train_X, predict_np_train_y_list)
    y_pred_stradaboost_predict = model_stradaboost_predict.predict(predict_np_test_X)


    mse_stradaboost_predict = sqrt(mean_squared_error(predict_np_test_y, y_pred_stradaboost_predict))
    rmselist_stradaboost_predict.append(mse_stradaboost_predict)

    r2_score_stradaboost_predict = pearsonr(predict_np_test_y_list, y_pred_stradaboost_predict)
    r2_score_stradaboost_predict = (r2_score_stradaboost_predict[0])**2
    r2scorelist_stradaboost_predict.append(r2_score_stradaboost_predict)



with open('predict_rmse.txt', 'w') as predict_handle_rmse:
    predict_handle_rmse.write("AdaBoost TL:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_predict)

    predict_handle_rmse.write("\n\nAdaBoost:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_predict)

    predict_handle_rmse.write("\n\nKMM:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_predict)

    predict_handle_rmse.write("\n\nGBRT:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_predict)

    predict_handle_rmse.write("\n\nGBR:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_predict)

    predict_handle_rmse.write("\n\nTrAdaBoost:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_predict)

    predict_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    predict_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_predict)


with open('predict_r2.txt', 'w') as predict_handle_r2:
    predict_handle_r2.write("AdaBoost TL:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_predict)

    predict_handle_r2.write("\n\nAdaBoost:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_predict)

    predict_handle_r2.write("\n\nKMM:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_predict)

    predict_handle_r2.write("\n\nGBRT:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_predict)

    predict_handle_r2.write("\n\nGBR:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_predict)

    predict_handle_r2.write("\n\nTrAdaBoost:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_predict)

    predict_handle_r2.write("\n\nSTrAdaBoost:\n ")
    predict_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_predict)


######################################################################################


# print("RMSE of Adaboost.R2(TL):", statistics.mean(rmselist_AdaTL_predict))
# print("R^2 of AdaboostR2(TL):", statistics.mean(r2scorelist_AdaTL_predict))
# print("\n")
# print("RMSE of Adaboost.R2(TL):", rmselist_AdaTL_predict)
# print("R^2 of AdaboostR2(TL):", r2scorelist_AdaTL_predict)


print("-------------------------------------------")
