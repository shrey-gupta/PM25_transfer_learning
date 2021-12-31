from two_TrAdaBoostR2 import TwoStageTrAdaBoostR2
# from newtwoStage_TrAdaBoostR2 import TradaboostRegressor

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
from sklearn.preprocessing import MinMaxScaler

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt

#Geo plotting libraries
import geopandas as gdp
from matplotlib.colors import ListedColormap
import geoplot as glpt

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import GradientBoostingRegressor

from statistics import mean

# sns.set_style("darkgrid")
# ax.set_facecolor('white')
#######################################################################################################################
#
# def remove_outlier(col):
#     aqi_df[col] = aqi_df.groupby('Date')[col].transform(lambda x: x.fillna(x.mean()))
#
#
#
#
# aqi_df = pd.read_csv('AQI_datasets/UCI_AQI/AirQualityUCI.csv', sep=',', delimiter=";",decimal=",")
# aqi_df = aqi_df.drop(["Unnamed: 15","Unnamed: 16"], axis=1)
# aqi_df.dropna(inplace=True)
# aqi_df.set_index("Date", inplace=True)
# aqi_df.index = pd.to_datetime(aqi_df.index)
# type(aqi_df.index)
# aqi_df['Time'] = pd.to_datetime(aqi_df['Time'],format= '%H.%M.%S').dt.hour
# type(aqi_df['Time'][0])
#
#
# # aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])
# # aqi_df_2004 = aqi_df[aqi_df['Date'].dt.year == 2004]
# # aqi_df_2005 = aqi_df[aqi_df['Date'].dt.year == 2005]
#
#
# aqi_df.drop('NMHC_GT', axis=1, inplace=True)
# aqi_df.replace(to_replace= -200, value= np.NaN, inplace= True)
# aqi_df.replace(to_replace= -200, value= np.NaN, inplace= True)
#
# col_list = aqi_df.columns[1:]
# for i in col_list:
#     remove_outlier(i)
#
# aqi_df.fillna(method='ffill', inplace= True)
# aqi_df.dropna(axis = 0)
# print(aqi_df.shape)
#
# # col = aqi_df['PT08.S3(NOx)']
# # AutoData_df = AutoData_df.sort_values(by=['PT08.S3(NOx)'])
# # aqi_df = aqi_df.reset_index(drop=True, inplace=True)
#
# aqi_df['YEAR'] = aqi_df.index.year
#
# aqi_df_2004 = aqi_df[aqi_df['YEAR'] == 2004]
# aqi_df_2005 = aqi_df[aqi_df['YEAR'] == 2005]
#
# drop_year = ['YEAR']
# aqi_df_2004 = aqi_df_2004.drop(drop_year, axis=1)
# aqi_df_2005 = aqi_df_2005.drop(drop_year, axis=1)
#
#
# X_2004 = aqi_df_2004.drop(['NOx_GT','T','Time'], axis=1)
# y_2004 = aqi_df_2004['NOx_GT'] ##NOx_GT
#
# X_2005 = aqi_df_2005.drop(['NOx_GT','T','Time'], axis=1)
# y_2005 = aqi_df_2005['NOx_GT']
#
# columns_uci = X_2004.columns
# index_uci_2004 = X_2004.index
# index_uci_2005 = X_2005.index
#
# ss = StandardScaler()
# X_2004_std = ss.fit_transform(X_2004)
# X_2005_std = ss.fit_transform(X_2005)
#
# X_2004 = pd.DataFrame(X_2004_std, columns = columns_uci)
# X_2004 = X_2004.set_index(index_uci_2004)
# X_2005 = pd.DataFrame(X_2005_std, columns = columns_uci)
# X_2005 = X_2005.set_index(index_uci_2005)
# # print(X_2004)
# # print(X_2005)
#
# X_2005_train, X_2005_test, y_2005_train, y_2005_test = train_test_split(X_2005, y_2005, test_size = 0.92, random_state=1)
# print("Train: ",X_2005_train.shape)
# print("Test: ", X_2005_test.shape)
# print(X_2005_train.columns)
# # print("Source: ",X_source.shape)
# # print("Training data is: ", X_train.columns)

###############################################################################################################################3
target_column = ['NOx_GT']

X_train = pd.read_csv('ActiveSampling/UCI_NOx_activesampling_train.csv')
X_test = pd.read_csv('ActiveSampling/UCI_NOx_activesampling_test.csv')
X_source = pd.read_csv('ActiveSampling/UCI_NOx_activesampling_source.csv')

print(X_train.shape)
print(X_test.shape)
print(X_source.shape)

y_2004 = X_source[target_column]
X_2004 = X_source.drop(target_column, axis = 1)

y_2005_train = X_train[target_column]
X_2005_train = X_train.drop(target_column, axis = 1)

y_2005_test = X_test[target_column]
X_2005_test = X_test.drop(target_column, axis = 1)

################################################################# Bullshit #############################################################3

# col=['DATE','TIME','CO_GT','PT08_S1_CO','NMHC_GT','C6H6_GT','PT08_S2_NMHC',
#      'NOX_GT','PT08_S3_NOX','NO2_GT','PT08_S4_NO2','PT08_S5_O3','T','RH','AH']
#
# #define number of columns from csv
# use=list(np.arange(len(col)))
#
# #read the data from csv
# df_air = pd.read_csv('AQI_datasets/UCI_AQI/AirQualityUCI.csv', header=None, skiprows=1, names=col, na_filter=True, na_values=-200, usecols=use)
# df_air.head()
#
# #drop end rows with NaN values
# df_air.dropna(how='all',inplace=True)
# #drop RH NAN rows
# df_air.dropna(thresh=10,axis=0,inplace=True)
#
# #Split hour from time into new column
# df_air['HOUR'] = df_air['TIME'].apply(lambda x: int(x.split(':')[0]))
# df_air.HOUR.head()
#
# df_air['DATE'] = pd.to_datetime(df_air.DATE, format='%m/%d/%Y')   #Format date column
#
# # set the index as date
# df_air.set_index('DATE',inplace=True)
#
# df_air['MONTH'] = df_air.index.month     #Create month column (Run once)
# df_air.reset_index(inplace=True)
#
# df_air.drop('NMHC_GT',axis=1,inplace=True)    #drop col
#
# df_air['CO_GT'] = df_air['CO_GT'].fillna(df_air.groupby(['MONTH','HOUR'])['CO_GT'].transform('mean'))
# df_air['NOX_GT'] = df_air['NOX_GT'].fillna(df_air.groupby(['MONTH','HOUR'])['NOX_GT'].transform('mean'))
# df_air['NO2_GT'] = df_air['NO2_GT'].fillna(df_air.groupby(['MONTH','HOUR'])['NO2_GT'].transform('mean'))
#
# df_air['CO_GT'] = df_air['CO_GT'].fillna(df_air.groupby(['HOUR'])['CO_GT'].transform('mean'))
# df_air['NOX_GT'] = df_air['NOX_GT'].fillna(df_air.groupby(['HOUR'])['NOX_GT'].transform('mean'))
# df_air['NO2_GT'] = df_air['NO2_GT'].fillna(df_air.groupby(['HOUR'])['NO2_GT'].transform('mean'))
#
# ptint(df_air)
######################################################################################################################################################

predictionlist = []
r2scorelist = []
rmselist = []

print("S-TrAdaBoost::::: NOx")

for x in range(0, 10):

    TF_train_X = pd.concat([X_2004, X_2005_train], sort= False)
    TF_train_y = pd.concat([y_2004, y_2005_train], sort= False)

    np_TF_train_X = TF_train_X.to_numpy()
    np_TF_train_y = TF_train_y.to_numpy()

    # np_TF_train_X = X_2005_train.to_numpy()
    # np_TF_train_y = y_2005_train.to_numpy()

    np_test_X = X_2005_test.to_numpy()
    np_test_y = y_2005_test.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_test_y_list = np_test_y.ravel()

    sample_size = [len(X_2004), len(X_2005_train)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    ################################################TwoStageAdaBoostR2############################################################################
    regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6), #xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.0001,max_depth = 6,n_estimators = 2000),
                          n_estimators = n_estimators, sample_size = sample_size,
                          steps = steps, fold = fold,
                          random_state = random_state)

    regr_1.fit(np_TF_train_X, np_TF_train_y_list)
    y_pred1 = regr_1.predict(np_test_X)
    predictionlist.append(y_pred1)

    mse_twostageboost = np.sqrt(mean_squared_error(np_test_y_list, y_pred1))
    print("RMSE of TrAdaboostR2:", mse_twostageboost)
    rmselist.append(mse_twostageboost)

    r2_score_twostageboost_values = pearsonr(np_test_y_list, y_pred1)
    r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
    print("R^2 of TrAdaboostR2:", r2_score_twostageboost)
    r2scorelist.append(r2_score_twostageboost)

    # regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = n_estimators)
    #
    # regr_2.fit(np_TF_train_X, np_TF_train_y_list)
    # y_pred2 = regr_2.predict(np_test_X)
    # predictionlist.append(y_pred2)
    #
    # rmse_adaboost = np.sqrt(mean_squared_error(np_test_y_list, y_pred2))
    # print("RMSE of AdaboostR2:", rmse_adaboost)
    # rmselist.append(rmse_adaboost)
    #
    # r2_score_adaboost_values = pearsonr(np_test_y_list, y_pred2)
    # r2_score_adaboost = (r2_score_adaboost_values[0])**2
    # print("R^2 of AdaboostR2:", r2_score_adaboost)
    # r2scorelist.append(r2_score_adaboost)


predict_adaboost = np.mean(predictionlist, axis=0)
plt.scatter(predict_adaboost, np_test_y_list,  c ="purple", alpha=0.3)

print("mean RMSE of TrAdaboostR2:", mean(rmselist))
print("mean R^2 of TrAdaboostR2:", mean(r2scorelist))

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('AdaBoost.R2 Predicted vs Actual')
# plt.savefig("AQI_datasets/UCI_AQI_Results/NOx/AdaBoostR2_Transfer.png")
plt.show()

# with open('AQI_datasets/UCI_AQI_Results/CO/AdaBoost_Output_UCI_R2.txt', 'w') as filehandle1:
#     for listitem in r2scorelist:
#         filehandle1.write('%s\n' % listitem)
#
# with open('AQI_datasets/UCI_AQI_Results/CO/AdaBoost_Output_UCI_RMSE.txt', 'w') as filehandle2:
#     for listitem in rmselist:
#         filehandle2.write('%s\n' % listitem)
#
# with open('AQI_datasets/UCI_AQI_Results/CO/AdaBoost_Output_UCI_prediction.txt', 'w') as filehandle2:
#     for listitem in predict_adaboost:
#         filehandle2.write('%s\n' % listitem)
