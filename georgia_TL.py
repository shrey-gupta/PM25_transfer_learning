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

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr
from math import sqrt

################# Geo plotting libraries
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
from sklearn.cluster import KMeans
from scipy.spatial import distance


sns.set_style("darkgrid")
################################################################################################################

us_df = pd.read_csv('us.csv')
print(us_df.shape)
us_df = us_df.sort_values(by=['rid'])

droplist = ['Latitude', 'Longitude','cmaq_id', 'lon', 'lat', 'day', 'year', 'month']
us_df = us_df.drop(droplist, axis = 1)
# print(us_df.columns)

Target_df  = us_df.loc[us_df['rid'] == 9]
Target_df = Target_df.reset_index(drop = True)

Source_df = us_df.loc[us_df['rid'] == 5]
Source_df = Source_df.reset_index(drop = True)
# print(Target_df.shape)

drop_rid = ['rid']
Target_df  = Target_df.drop(drop_rid, axis = 1)
Source_df = Source_df.drop(drop_rid, axis = 1)

target_column = ['pm25_value']
Target_df_y = Target_df[target_column]
Target_df_X = Target_df.drop(target_column, axis = 1)

y_source = Source_df[target_column]
X_source = Source_df.drop(target_column, axis = 1)
X_source['pm25_value']=  y_source

X_train, X_test, y_train, y_test = train_test_split(Target_df_X, Target_df_y, test_size=0.97, random_state=1)
X_train['pm25_value'] =  y_train
X_test['pm25_value'] =  y_test

X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)

print(X_train.shape)
print(X_source.shape)
print(X_test.shape)
print(X_train.columns)

#######################################################################################################################################
# target_column = ['pm25_value']
#
# X_train = pd.read_csv('ActiveSampling/us_activesampling_train.csv')
# X_test = pd.read_csv('ActiveSampling/us_activesampling_test.csv')
# X_source = pd.read_csv('ActiveSampling/us_activesampling_source.csv')

y_source = X_source[target_column]
X_source = X_source.drop(target_column, axis = 1)

y_train = X_train[target_column]
X_train = X_train.drop(target_column, axis = 1)

y_test = X_test[target_column]
X_test = X_test.drop(target_column, axis = 1)


predictionlist = []
r2scorelist = []
rmselist = []
print(len(X_source))
print(len(X_train))
for x in range(0, 10):

    # TF_train_X = pd.concat([X_source, X_train], sort= False)
    # TF_train_y = pd.concat([y_source, y_train], sort= False)
    #
    # np_TF_train_X = TF_train_X.to_numpy()
    # np_TF_train_y = TF_train_y.to_numpy()

    np_TF_train_X = X_train.to_numpy()
    np_TF_train_y = y_train.to_numpy()

    np_test_X = X_test.to_numpy()
    np_test_y = y_test.to_numpy()

    np_TF_train_y_list = np_TF_train_y.ravel()
    np_test_y_list = np_test_y.ravel()

    sample_size = [len(X_source), len(X_train)]
    n_estimators = 100
    steps = 30
    fold = 10
    random_state = np.random.RandomState(1)

    ################################################TwoStageAdaBoostR2############################################################################
    print(x)
    # regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6), #xgb.XGBRegressor(objective ='reg:squarederror',learning_rate = 0.0001,max_depth = 6,n_estimators = 2000),
    #                       n_estimators = n_estimators, sample_size = sample_size,
    #                       steps = steps, fold = fold,
    #                       random_state = random_state)
    #
    # regr_1.fit(np_TF_train_X, np_TF_train_y_list)
    # y_pred1 = regr_1.predict(np_test_X)
    # mse_twostageboost = sqrt(mean_squared_error(np_test_y_list, y_pred1))
    # print("RMSE of S-TrAdaboostR2:", mse_twostageboost)
    # rmselist.append(mse_twostageboost)
    #
    #
    # r2_score_twostageboost_values = pearsonr(np_test_y_list, y_pred1)
    # r2_score_twostageboost = (r2_score_twostageboost_values[0])**2
    # print("R^2 of S-TrAdaboostR2:", r2_score_twostageboost)
    # r2scorelist.append(r2_score_twostageboost)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators = n_estimators)

    regr_2.fit(np_TF_train_X, np_TF_train_y_list)
    y_pred2 = regr_2.predict(np_test_X)
    predictionlist.append(y_pred2)

    rmse_adaboost = sqrt(mean_squared_error(np_test_y_list, y_pred2))
    print("RMSE of AdaboostR2:", rmse_adaboost)
    rmselist.append(rmse_adaboost)

    r2_score_adaboost_values = pearsonr(np_test_y_list, y_pred2)
    r2_score_adaboost = (r2_score_adaboost_values[0])**2
    print("R^2 of AdaboostR2:", r2_score_adaboost)
    r2scorelist.append(r2_score_adaboost)

################### Scatterplot prediction list ########################################################
predict_adaboost = np.mean(predictionlist, axis=0)


print("mean RMSE of AdaboostR2:", mean(rmselist))
print("mean R^2 of AdaboostR2:", mean(r2scorelist))

# plt.scatter(np_test_y_list, predict_adaboost, c="purple", alpha=0.3)
sns.regplot(np_test_y_list, predict_adaboost, color="purple")
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Adaboost.R2 Predicted vs Actual')
plt.show()

with open('ActiveSampling/AdaBoost_NoSource_Output_US_R2.txt', 'w') as filehandle1:
    for listitem in r2scorelist:
        filehandle1.write('%s\n' % listitem)

with open('ActiveSampling/AdaBoost_NoSource_Output_US_RMSE.txt', 'w') as filehandle2:
    for listitem in rmselist:
        filehandle2.write('%s\n' % listitem)
