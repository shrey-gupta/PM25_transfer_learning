######################## Header files ################################################
# from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2 ##For STrAdaBoost.R2
# from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 ## For two-stage TrAdaBoost.R2

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
#import geopandas as gdp
#from matplotlib.colors import ListedColormap
#import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold
import matplotlib.lines as mlines
import folium
import glob

import statistics
from sklearn.cluster import KMeans
from scipy.spatial import distance

pd.options.display.max_columns = None

from adapt.instance_based import (TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2)


print("Done uploading repositories")

from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2
from sklearn.model_selection import GridSearchCV
from adapt.instance_based import KMM

print("Second Upload Completed!!")

############################## UCI Italian dataset #######################################################
#################### Dataset Information: 2 years and single terrain #####################################
## Predictors: T, Ah, Rh, NMHC_GT, NOx_GT, CO_GT, C6H6_GT ,Target: O3
###########################################################################################################
#AQI_datasets/UCI_AQI
aqi_df = pd.read_csv('AQI_datasets/UCI_AQI/AirQualityUCI.csv', sep=',', delimiter=";", decimal=",", index_col = None, header=0)
print(aqi_df.shape)

def remove_outlier(col):
    aqi_df[col] = aqi_df.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))

################ Pre-processing ############################################
aqi_df.dropna(how = 'all', inplace = True) ## drop end rows with NaN values
drop_unamed = ['Unnamed: 15', 'Unnamed: 16']
aqi_df = aqi_df.drop(drop_unamed, axis = 1) ## drop unamed columns

drop_uw = ['Time', 'PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 'NO2_GT', 'PT08_S4_NO2']
aqi_df = aqi_df.drop(drop_uw, axis = 1) ## Drop unwanted columns

aqi_df.replace(to_replace = -200, value = np.NaN, inplace = True) ## Replace the -200 values seen in the dataset with NaN

## Replace the NaN values with the column mean
col_list = aqi_df.columns[1:]
for i in col_list:
    remove_outlier(i)

aqi_df.fillna(method ='ffill', inplace= True)
aqi_df.dropna(axis = 0)

## Convert 'Date' column to datetime and then seperate out year and month into different columns.
aqi_df.Date = pd.to_datetime(aqi_df.Date)
aqi_df['Year'] = aqi_df['Date'].dt.year
aqi_df['Month'] = aqi_df['Date'].dt.month
drop_date = ['Date']
aqi_df = aqi_df.drop(drop_date, axis = 1)
aqi_df = aqi_df.reset_index(drop=True)

print("Dataset after pre-processing: ")
print(aqi_df.shape)

################ Observing data statistics ################################
# print(aqi_df.describe())

#Split the dataset according to the year.
drop_cols = ['Year', 'Month']
aqi_df_2004 = aqi_df[aqi_df['Year'] == 2004]
aqi_df_2004 = aqi_df_2004.drop(drop_cols, axis = 1)

aqi_df_2005 = aqi_df[aqi_df['Year'] == 2005]
aqi_df_2005 = aqi_df_2005.drop(drop_cols, axis = 1)

aqi_df_2004 = aqi_df_2004.reset_index(drop=True)
aqi_df_2005 = aqi_df_2005.reset_index(drop=True)

aqi_df_2004

################ Divide the dataframe into target and the predictors. ################
target_uci_col = ['PT08_S5_O3']
aqi_df_2004_target = aqi_df_2004[target_uci_col]
aqi_df_2004_target.columns = ['O3']
aqi_df_2004_predictors = aqi_df_2004.drop(target_uci_col, axis = 1)
aqi_df_2004_predictors = aqi_df_2004_predictors.reset_index(drop=True)

aqi_df_2005_target = aqi_df_2005[target_uci_col]
aqi_df_2005_target.columns = ['O3']
aqi_df_2005_predictors = aqi_df_2005.drop(target_uci_col, axis = 1)
aqi_df_2005_predictors = aqi_df_2005_predictors.reset_index(drop=True)

################### 2004: Source Dataset, 2005: Training Set [Training and Testing Set] ###################
################ Standardize the dataset ################
columns_uci = aqi_df_2004_predictors.columns

ss = StandardScaler()
aqi_df_2004_predictors[columns_uci] = ss.fit_transform(aqi_df_2004_predictors[columns_uci])
aqi_df_2005_predictors[columns_uci] = ss.fit_transform(aqi_df_2005_predictors[columns_uci])

#################### Renaming features and target ####################

italianAQ_train_df_y = aqi_df_2005_target
italianAQ_train_df_X = aqi_df_2005_predictors

italianAQ_source_df_y = aqi_df_2004_target
italianAQ_source_df_X = aqi_df_2004_predictors

################## Split into target and test dataset ###################
def TimeSeriesTrainTestSplit(X, y, test_size):

        test_index = int(len(X)*(1-test_size))

        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]
        return X_train, y_train, X_test, y_test

italianAQ_test_df_X, italianAQ_test_df_y, italianAQ_tgt_df_X, italianAQ_tgt_df_y = TimeSeriesTrainTestSplit(italianAQ_train_df_X,italianAQ_train_df_y, 0.02)

print(italianAQ_tgt_df_X.shape)
print(italianAQ_test_df_X.shape)

############### Merging the datasets ##########################################
italianAQ_X_df = pd.concat([italianAQ_tgt_df_X, italianAQ_source_df_X], ignore_index=True)
italianAQ_y_df = pd.concat([italianAQ_tgt_df_y, italianAQ_source_df_y], ignore_index=True)

italianAQ_np_train_X = italianAQ_X_df.to_numpy()
italianAQ_np_train_y = italianAQ_y_df.to_numpy()

italianAQ_np_test_X = italianAQ_test_df_X.to_numpy()
italianAQ_np_test_y = italianAQ_test_df_y.to_numpy()

italianAQ_np_train_y_list = italianAQ_np_train_y.ravel()
italianAQ_np_test_y_list = italianAQ_np_test_y.ravel()

src_size_italianAQ = len(italianAQ_source_df_y)
tgt_size_italianAQ = len(italianAQ_tgt_df_y)

src_idx = np.arange(start=0, stop=(src_size_italianAQ - 1), step=1)
tgt_idx = np.arange(start=src_size_italianAQ, stop=((src_size_italianAQ + tgt_size_italianAQ) - 1), step=1)



########################### Transfer Learning Italian AQ #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_italianAQ = []
rmselist_AdaTL_italianAQ = []

r2scorelist_Ada_italianAQ = []
rmselist_Ada_italianAQ = []

r2scorelist_KMM_italianAQ = []
rmselist_KMM_italianAQ = []

r2scorelist_GBRTL_italianAQ = []
rmselist_GBRTL_italianAQ = []

r2scorelist_GBR_italianAQ = []
rmselist_GBR_italianAQ = []

r2scorelist_TwoTrAda_italianAQ = []
rmselist_TwoTrAda_italianAQ = []

r2scorelist_stradaboost_italianAQ = []
rmselist_stradaboost_italianAQ = []


kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for x in range(0, 10):
    ################### AdaBoost Tl ###################
    model_AdaTL_italianAQ = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_italianAQ.fit(italianAQ_np_train_X, italianAQ_np_train_y_list)

    y_pred_AdaTL_italianAQ = model_AdaTL_italianAQ.predict(italianAQ_np_test_X)

    mse_AdaTL_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_AdaTL_italianAQ))
    rmselist_AdaTL_italianAQ.append(mse_AdaTL_italianAQ)

    r2_score_AdaTL_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_AdaTL_italianAQ)
    r2_score_AdaTL_italianAQ = (r2_score_AdaTL_italianAQ[0])**2
    r2scorelist_AdaTL_italianAQ.append(r2_score_AdaTL_italianAQ)


    ################### AdaBoost ###################
    model_Ada_italianAQ = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_italianAQ.fit(italianAQ_tgt_df_X, italianAQ_tgt_df_y)

    y_pred_ada_italianAQ = model_Ada_italianAQ.predict(italianAQ_np_test_X)

    mse_Ada_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_ada_italianAQ))
    rmselist_Ada_italianAQ.append(mse_Ada_italianAQ)

    r2_score_Ada_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_ada_italianAQ)
    r2_score_Ada_italianAQ = (r2_score_Ada_italianAQ[0])**2
    r2scorelist_Ada_italianAQ.append(r2_score_Ada_italianAQ)


   ################### KMM ###################
    model_KMM_italianAQ = KMM(get_estimator = get_estimator)
    model_KMM_italianAQ.fit(italianAQ_np_train_X, italianAQ_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_italianAQ = model_KMM_italianAQ.predict(italianAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_KMM_italianAQ))
    rmselist_KMM_italianAQ.append(mse_KMM_italianAQ)

    r2_score_KMM_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_KMM_italianAQ)
    r2_score_KMM_italianAQ = (r2_score_KMM_italianAQ[0])**2
    r2scorelist_KMM_italianAQ.append(r2_score_KMM_italianAQ)


    ################### GBRTL ###################
    model_GBRTL_italianAQ = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_italianAQ.fit(italianAQ_np_train_X, italianAQ_np_train_y_list)

    y_pred_GBRTL_italianAQ = model_GBRTL_italianAQ.predict(italianAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_GBRTL_italianAQ))
    rmselist_GBRTL_italianAQ.append(mse_GBRTL_italianAQ)

    r2_score_GBRTL_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_GBRTL_italianAQ)
    r2_score_GBRTL_italianAQ = (r2_score_GBRTL_italianAQ[0])**2
    r2scorelist_GBRTL_italianAQ.append(r2_score_GBRTL_italianAQ)


    ################### GBR ###################
    model_GBR_italianAQ = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBR_italianAQ.fit(italianAQ_tgt_df_X, italianAQ_tgt_df_y)

    y_pred_GBR_italianAQ = model_GBR_italianAQ.predict(italianAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_GBR_italianAQ))
    rmselist_GBR_italianAQ.append(mse_GBR_italianAQ)

    r2_score_GBR_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_GBR_italianAQ)
    r2_score_GBR_italianAQ = (r2_score_GBR_italianAQ[0])**2
    r2scorelist_GBR_italianAQ.append(r2_score_GBR_italianAQ)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_italianAQ = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_italianAQ.fit(italianAQ_np_train_X, italianAQ_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_italianAQ = model_TwoTrAda_italianAQ.predict(italianAQ_np_test_X)

    mse_TwoTrAda_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_TwoTrAda_italianAQ))
    rmselist_TwoTrAda_italianAQ.append(mse_TwoTrAda_italianAQ)

    r2_score_TwoTrAda_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_TwoTrAda_italianAQ)
    r2_score_TwoTrAda_italianAQ = (r2_score_TwoTrAda_italianAQ[0])**2
    r2scorelist_TwoTrAda_italianAQ.append(r2_score_TwoTrAda_italianAQ)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(italianAQ_tgt_df_X), len(italianAQ_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_italianAQ = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_italianAQ.fit(italianAQ_np_train_X, italianAQ_np_train_y_list)
    y_pred_stradaboost_italianAQ = model_stradaboost_italianAQ.predict(italianAQ_np_test_X)


    mse_stradaboost_italianAQ = sqrt(mean_squared_error(italianAQ_np_test_y, y_pred_stradaboost_italianAQ))
    rmselist_stradaboost_italianAQ.append(mse_stradaboost_italianAQ)

    r2_score_stradaboost_italianAQ = pearsonr(italianAQ_np_test_y_list, y_pred_stradaboost_italianAQ)
    r2_score_stradaboost_italianAQ = (r2_score_stradaboost_italianAQ[0])**2
    r2scorelist_stradaboost_italianAQ.append(r2_score_stradaboost_italianAQ)



with open('italianAQ_rmse.txt', 'w') as italianAQ_handle_rmse:
    italianAQ_handle_rmse.write("AdaBoost TL:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_italianAQ)

    italianAQ_handle_rmse.write("\n\nAdaBoost:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_italianAQ)

    italianAQ_handle_rmse.write("\n\nKMM:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_italianAQ)

    italianAQ_handle_rmse.write("\n\nGBRT:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_italianAQ)

    italianAQ_handle_rmse.write("\n\nGBR:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_italianAQ)

    italianAQ_handle_rmse.write("\n\nTrAdaBoost:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_italianAQ)

    italianAQ_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    italianAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_italianAQ)


with open('italianAQ_r2.txt', 'w') as italianAQ_handle_r2:
    italianAQ_handle_r2.write("AdaBoost TL:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_italianAQ)

    italianAQ_handle_r2.write("\n\nAdaBoost:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_italianAQ)

    italianAQ_handle_r2.write("\n\nKMM:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_italianAQ)

    italianAQ_handle_r2.write("\n\nGBRT:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_italianAQ)

    italianAQ_handle_r2.write("\n\nGBR:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_italianAQ)

    italianAQ_handle_r2.write("\n\nTrAdaBoost:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_italianAQ)

    italianAQ_handle_r2.write("\n\nSTrAdaBoost:\n ")
    italianAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_italianAQ)


######################################################################################

print("-------------------------------------------")
