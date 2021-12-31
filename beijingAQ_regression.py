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

################# UCI Beijing dataset (2013 - 2017) #####################################################################
############### Spacio-temporal dataset. (multi-year and multi-terrain) #################################################

# beijing_aqi_df = pd.read_csv('AQI_datasets/Beijing_AQI/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv')
# drop_index = ['No']
# beijing_aqi_df = beijing_aqi_df.drop(drop_index, axis=1)
# #print(beijing_aqi_df.head())
# print(beijing_aqi_df.isnull().sum())

#AQI_datasets/Beijing_AQI/PRSA_Data_20130301-20170228
path_beijing = r'AQI_datasets/Beijing_AQI/PRSA_Data_20130301-20170228/' ## Path for all the files
allFiles = glob.glob(path_beijing + "/*.csv")
beijing_aqi_df = pd.DataFrame()
list_beijing = []

for file_ in allFiles:
    temp_df = pd.read_csv(file_, index_col = None, header=0)
    list_beijing.append(temp_df)
beijing_aqi_df = pd.concat(list_beijing)

cols = ['No', 'year', 'month', 'day', 'hour', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP',
       'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station']

beijing_aqi_df = beijing_aqi_df[cols]

drop_index = ['No']
beijing_aqi_df = beijing_aqi_df.drop(drop_index, axis=1)

beijing_aqi_df = beijing_aqi_df.sort_values(['station', 'year'])

######################### Seperate the dataset into predictors and target variable. ############

predictors = ['year', 'month', 'SO2', 'NO2', 'CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM', 'station', 'O3']
beijing_df = beijing_aqi_df[predictors]
wd_codes = {'N':1, 'E': 2, 'W': 3, 'S': 4, 'NE': 5, 'NW': 6, 'SE': 7, 'SW': 8, 'NNE': 9, 'NNW': 10, 'SSE': 11,
            'SSW': 12, 'WNW': 13, 'WSW': 14, 'ENE': 15, 'ESE': 16 }


beijing_df.replace(wd_codes, inplace=True)
beijing_df.wd.value_counts()
beijing_df = beijing_df.dropna()

######################### Drop stations ########################################
drop_stations = ['station']
beijing_df = beijing_df.drop(drop_stations, axis=1)

################ Splitting the dataset by the year #############################
beijing_df.year.value_counts()
beijing_df_target = beijing_df[beijing_df['year'].isin([2014, 2015])]
beijing_df_source = beijing_df[beijing_df['year'].isin([2016])]

drop_cols = ['year', 'month']
beijing_df_target = beijing_df_target.drop(drop_cols, axis = 1)
beijing_df_source = beijing_df_source.drop(drop_cols, axis = 1)

beijing_df_target = beijing_df_target.reset_index(drop=True)
beijing_df_source = beijing_df_source.reset_index(drop=True)

#### Select first 10k source and first 50k train instances
beijing_df_source = beijing_df_source.iloc[:10000]
beijing_df_target = beijing_df_target.iloc[:50000]

print(beijing_df_source.shape, beijing_df_target.shape)
beijing_df_target

########################## Standardize the dataset. ############################

cols_to_norm = ['SO2', 'NO2', 'CO', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

ss = StandardScaler()
beijing_df_target[cols_to_norm] = ss.fit_transform(beijing_df_target[cols_to_norm])
beijing_df_source[cols_to_norm] = ss.fit_transform(beijing_df_source[cols_to_norm])

##################### Splitting the dataset into train and test set ################

target_beijing_col = ['O3']
beijingAQ_train_df_y = beijing_df_target[target_beijing_col]
beijingAQ_train_df_X = beijing_df_target.drop(target_beijing_col, axis =1)


beijingAQ_source_df_y = beijing_df_source[target_beijing_col]
beijingAQ_source_df_X = beijing_df_source.drop(target_beijing_col, axis =1)


################## Split into target and test dataset ###################
def TimeSeriesTrainTestSplit(X, y, test_size):

        test_index = int(len(X)*(1-test_size))

        X_train = X.iloc[:test_index]
        y_train = y.iloc[:test_index]
        X_test = X.iloc[test_index:]
        y_test = y.iloc[test_index:]
        return X_train, y_train, X_test, y_test

beijingAQ_test_df_X, beijingAQ_test_df_y, beijingAQ_tgt_df_X, beijingAQ_tgt_df_y = TimeSeriesTrainTestSplit(beijingAQ_train_df_X, beijingAQ_train_df_y, 0.008)

print(beijingAQ_tgt_df_X.shape)
print(beijingAQ_test_df_X.shape)


############### Merging the datasets ##########################################
beijingAQ_X_df = pd.concat([beijingAQ_tgt_df_X, beijingAQ_source_df_X], ignore_index=True)
beijingAQ_y_df = pd.concat([beijingAQ_tgt_df_y, beijingAQ_source_df_y], ignore_index=True)

beijingAQ_np_train_X = beijingAQ_X_df.to_numpy()
beijingAQ_np_train_y = beijingAQ_y_df.to_numpy()

beijingAQ_np_test_X = beijingAQ_test_df_X.to_numpy()
beijingAQ_np_test_y = beijingAQ_test_df_y.to_numpy()

beijingAQ_np_train_y_list = beijingAQ_np_train_y.ravel()
beijingAQ_np_test_y_list = beijingAQ_np_test_y.ravel()

src_size_beijingAQ = len(beijingAQ_source_df_y)
tgt_size_beijingAQ = len(beijingAQ_tgt_df_y)

src_idx = np.arange(start=0, stop=(src_size_beijingAQ - 1), step=1)
tgt_idx = np.arange(start=src_size_beijingAQ, stop=((src_size_beijingAQ + tgt_size_beijingAQ) - 1), step=1)


########################### Transfer Learning Italian AQ #####################################################
from sklearn.ensemble import AdaBoostRegressor

def get_estimator(**kwargs):
    return DecisionTreeRegressor(max_depth = 6)

kwargs_TwoTrAda = {'steps': 30,
                    'fold': 10,
                  'learning_rate': 0.1}



print("Adaboost.R2 Transfer Learning (M + H, L)")
print("-------------------------------------------")

r2scorelist_AdaTL_beijingAQ = []
rmselist_AdaTL_beijingAQ = []

r2scorelist_Ada_beijingAQ = []
rmselist_Ada_beijingAQ = []

r2scorelist_KMM_beijingAQ = []
rmselist_KMM_beijingAQ = []

r2scorelist_GBRTL_beijingAQ = []
rmselist_GBRTL_beijingAQ = []

r2scorelist_GBR_beijingAQ = []
rmselist_GBR_beijingAQ = []

r2scorelist_TwoTrAda_beijingAQ = []
rmselist_TwoTrAda_beijingAQ = []

r2scorelist_stradaboost_beijingAQ = []
rmselist_stradaboost_beijingAQ = []


kfold = KFold(n_splits = 10, random_state = 42, shuffle=False)

for x in range(0, 10):
    ################### AdaBoost Tl ###################
    model_AdaTL_beijingAQ = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_AdaTL_beijingAQ.fit(beijingAQ_np_train_X, beijingAQ_np_train_y_list)

    y_pred_AdaTL_beijingAQ = model_AdaTL_beijingAQ.predict(beijingAQ_np_test_X)

    mse_AdaTL_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_AdaTL_beijingAQ))
    rmselist_AdaTL_beijingAQ.append(mse_AdaTL_beijingAQ)

    r2_score_AdaTL_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_AdaTL_beijingAQ)
    r2_score_AdaTL_beijingAQ = (r2_score_AdaTL_beijingAQ[0])**2
    r2scorelist_AdaTL_beijingAQ.append(r2_score_AdaTL_beijingAQ)


    ################### AdaBoost ###################
    model_Ada_beijingAQ = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6), learning_rate = 0.1, n_estimators = 100)
    model_Ada_beijingAQ.fit(beijingAQ_tgt_df_X, beijingAQ_tgt_df_y)

    y_pred_ada_beijingAQ = model_Ada_beijingAQ.predict(beijingAQ_np_test_X)

    mse_Ada_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_ada_beijingAQ))
    rmselist_Ada_beijingAQ.append(mse_Ada_beijingAQ)

    r2_score_Ada_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_ada_beijingAQ)
    r2_score_Ada_beijingAQ = (r2_score_Ada_beijingAQ[0])**2
    r2scorelist_Ada_beijingAQ.append(r2_score_Ada_beijingAQ)


   ################### KMM ###################
    model_KMM_beijingAQ = KMM(get_estimator = get_estimator)
    model_KMM_beijingAQ.fit(beijingAQ_np_train_X, beijingAQ_np_train_y_list, src_idx, tgt_idx)

    y_pred_KMM_beijingAQ = model_KMM_beijingAQ.predict(beijingAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_KMM_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_KMM_beijingAQ))
    rmselist_KMM_beijingAQ.append(mse_KMM_beijingAQ)

    r2_score_KMM_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_KMM_beijingAQ)
    r2_score_KMM_beijingAQ = (r2_score_KMM_beijingAQ[0])**2
    r2scorelist_KMM_beijingAQ.append(r2_score_KMM_beijingAQ)


    ################### GBRTL ###################
    model_GBRTL_beijingAQ = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample = 0.5)
    model_GBRTL_beijingAQ.fit(beijingAQ_np_train_X, beijingAQ_np_train_y_list)

    y_pred_GBRTL_beijingAQ = model_GBRTL_beijingAQ.predict(beijingAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBRTL_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_GBRTL_beijingAQ))
    rmselist_GBRTL_beijingAQ.append(mse_GBRTL_beijingAQ)

    r2_score_GBRTL_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_GBRTL_beijingAQ)
    r2_score_GBRTL_beijingAQ = (r2_score_GBRTL_beijingAQ[0])**2
    r2scorelist_GBRTL_beijingAQ.append(r2_score_GBRTL_beijingAQ)


    ################### GBR ###################
    model_GBR_beijingAQ = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 100, subsample=0.5)
    model_GBR_beijingAQ.fit(beijingAQ_tgt_df_X, beijingAQ_tgt_df_y)

    y_pred_GBR_beijingAQ = model_GBR_beijingAQ.predict(beijingAQ_test_df_X) ##Using dataframe instead of the numpy matrix

    mse_GBR_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_GBR_beijingAQ))
    rmselist_GBR_beijingAQ.append(mse_GBR_beijingAQ)

    r2_score_GBR_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_GBR_beijingAQ)
    r2_score_GBR_beijingAQ = (r2_score_GBR_beijingAQ[0])**2
    r2scorelist_GBR_beijingAQ.append(r2_score_GBR_beijingAQ)


    ################### Two-TrAdaBoost ###################
    from adapt.instance_based import TrAdaBoost, TrAdaBoostR2, TwoStageTrAdaBoostR2

    model_TwoTrAda_beijingAQ = TwoStageTrAdaBoostR2(get_estimator = get_estimator, n_estimators = 100, cv=10) #, kwargs_TwoTrAda)
    model_TwoTrAda_beijingAQ.fit(beijingAQ_np_train_X, beijingAQ_np_train_y_list, src_idx, tgt_idx)

    y_pred_TwoTrAda_beijingAQ = model_TwoTrAda_beijingAQ.predict(beijingAQ_np_test_X)

    mse_TwoTrAda_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_TwoTrAda_beijingAQ))
    rmselist_TwoTrAda_beijingAQ.append(mse_TwoTrAda_beijingAQ)

    r2_score_TwoTrAda_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_TwoTrAda_beijingAQ)
    r2_score_TwoTrAda_beijingAQ = (r2_score_TwoTrAda_beijingAQ[0])**2
    r2scorelist_TwoTrAda_beijingAQ.append(r2_score_TwoTrAda_beijingAQ)


    ################### STrAdaBoost ###################
    from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2

    sample_size = [len(beijingAQ_tgt_df_X), len(beijingAQ_source_df_X)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)


    model_stradaboost_beijingAQ = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth = 6),
                        n_estimators = n_estimators, sample_size = sample_size,
                        steps = steps, fold = fold, random_state = random_state)


    model_stradaboost_beijingAQ.fit(beijingAQ_np_train_X, beijingAQ_np_train_y_list)
    y_pred_stradaboost_beijingAQ = model_stradaboost_beijingAQ.predict(beijingAQ_np_test_X)


    mse_stradaboost_beijingAQ = sqrt(mean_squared_error(beijingAQ_np_test_y, y_pred_stradaboost_beijingAQ))
    rmselist_stradaboost_beijingAQ.append(mse_stradaboost_beijingAQ)

    r2_score_stradaboost_beijingAQ = pearsonr(beijingAQ_np_test_y_list, y_pred_stradaboost_beijingAQ)
    r2_score_stradaboost_beijingAQ = (r2_score_stradaboost_beijingAQ[0])**2
    r2scorelist_stradaboost_beijingAQ.append(r2_score_stradaboost_beijingAQ)



with open('beijingAQ_rmse.txt', 'w') as beijingAQ_handle_rmse:
    beijingAQ_handle_rmse.write("AdaBoost TL:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_AdaTL_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nAdaBoost:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_Ada_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nKMM:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_KMM_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nGBRT:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBRTL_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nGBR:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_GBR_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nTrAdaBoost:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_TwoTrAda_beijingAQ)

    beijingAQ_handle_rmse.write("\n\nSTrAdaBoost:\n ")
    beijingAQ_handle_rmse.writelines("%s\n" % ele for ele in rmselist_stradaboost_beijingAQ)


with open('beijingAQ_r2.txt', 'w') as beijingAQ_handle_r2:
    beijingAQ_handle_r2.write("AdaBoost TL:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_AdaTL_beijingAQ)

    beijingAQ_handle_r2.write("\n\nAdaBoost:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_Ada_beijingAQ)

    beijingAQ_handle_r2.write("\n\nKMM:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_KMM_beijingAQ)

    beijingAQ_handle_r2.write("\n\nGBRT:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBRTL_beijingAQ)

    beijingAQ_handle_r2.write("\n\nGBR:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_GBR_beijingAQ)

    beijingAQ_handle_r2.write("\n\nTrAdaBoost:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_TwoTrAda_beijingAQ)

    beijingAQ_handle_r2.write("\n\nSTrAdaBoost:\n ")
    beijingAQ_handle_r2.writelines("%s\n" % ele for ele in r2scorelist_stradaboost_beijingAQ)


######################################################################################

print("-------------------------------------------")
